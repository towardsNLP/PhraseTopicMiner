"""
phrase_miner.py

PhraseMiner: grammar-based, Markdown-aware phrase extraction for
phrase-driven topic modeling and downstream NLP.

Main features
-------------
- Optional Markdown cleanup (footnotes, reference-style links, citation
  markers) before tagging.
- Dual backends: spaCy or NLTK for tokenization + POS (and NER with spaCy).
- Noun phrase mining via a simple coarse-POS grammar (BaseNP, NP+PP,
  NP+multiPP).
- Optional verb-argument patterns (VerbObj, VerbPP, SubjVerb) for
  relational / argumentative phrases.
- Smart canonicalization: common phrases are lowercased, while named
  entities keep their capitalization.
- Optional PhraseRecord metadata for each extracted phrase (NP/VP type,
  pattern, doc/sentence index, surface vs canonical form).

Quick usage
-----------
    from phrase_miner import PhraseMiner

    texts = ["# Aspects of the mind\\n\\nThought is when we absorb what happens..."]

    miner = PhraseMiner(
        method="spacy",               # or "nltk"
        spacy_model="en_core_web_sm",
        include_verb_phrases=True,
        clean_markdown=True,
    )

    np_counter, vp_counters, phrase_records, sentences_by_doc = miner.mine_phrases_with_types(
        texts,
        include_metadata=True,
    )

    print(np_counter.most_common(10))
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterable, Tuple, Optional, Set, Dict
from collections import Counter
import re


@dataclass
class PhraseRecord:
    phrase: str        # canonicalized phrase (for counting)
    surface: str       # cleaned surface text (no citations, punctuation stripped)
    kind: str          # "NP" or "VP"
    pattern: str       # e.g. "BaseNP", "NP+PP", "VerbObj", "VerbPP", "SubjVerb"
    doc_index: int
    sent_index: int



# ---------------------------------------------------------------------------
# ---------- Main PhraseMiner class – grammar-based phrase extraction ----------
# ---------------------------------------------------------------------------


class PhraseMiner:
    """
    Grammar-based phrase mining for English text.

    This class is responsible for:
      * Tokenization and POS-tagging (SpaCy or NLTK backend)
      * Converting fine-grained POS tags to a coarse 5-tag set
      * Applying a simple but expressive noun-phrase grammar
      * (Optionally) extracting verb-argument patterns
      * Collapsing subsumed phrases (shorter phrases fully contained in
        longer ones) by re-weighting their counts.
      * Cleaning Wikipedia-style footnote markers (e.g. [4], [1][2][4]) from
        extracted phrases.

    The output of :meth:`mine_phrases` is a cleaned ``Counter`` from
    phrase → frequency which can be fed directly into :class:`TopicModeler`.

    The design deliberately keeps this component lightweight and deterministic
    so that phrase extraction is easy to reason about and debug.
    """

    def __init__(
        self,
        method: str = "spacy",
        spacy_model: str = "en_core_web_sm",
        include_verb_phrases: bool = False,
        clean_markdown: bool = False,  # optional Markdown cleaning
    ) -> None:
        """
        Parameters
        ----------
        method:
            Either ``"spacy"`` (default) or ``"nltk"`` – selects the underlying
            tokenizer + POS-tagger implementation.
        spacy_model:
            Name of the SpaCy model to use if ``method="spacy"``.
            Common choices are ``"en_core_web_sm"`` and ``"en_core_web_lg"``.
        include_verb_phrases:
            If ``True``, mined phrases will also include simple
            Verb-Argument patterns (verb–object, verb–PP, subject–verb).
        clean_markdown:
            If ``True``, each input document is lightly cleaned from Markdown
            to plain text before tagging (reference/footnote lines removed,
            links flattened, etc.).
        """
        self.method = method.lower()
        self.spacy_model = spacy_model
        self.include_verb_phrases = include_verb_phrases
        self.clean_markdown = clean_markdown  # NEW

        # Load the chosen tagging backend lazily.
        if self.method == "spacy":
            self._nlp = self._load_spacy_model(spacy_model)
            self._tokenizer = None
            self._tagger = None
        elif self.method == "nltk":
            self._tokenizer, self._tagger = self._load_nltk_models()
            self._nlp = None
        else:
            raise ValueError("method must be 'spacy' or 'nltk'")

        # Coarse POS map for the phrase grammar.
        # We use the same 5-tag abstraction as in Handler et al. (2016):
        #   A: adjectives / numeric modifiers
        #   D: determiners
        #   P: prepositions
        #   N: nouns / proper nouns / foreign words
        #   V: verbs
        self.coarsemap: Dict[str, List[str]] = {
            "A": ["JJ", "JJR", "JJS", "ADJ", "CD", "A", "CoarseADJ", "CoarseNUM"],
            "D": ["DT", "DET", "D", "CoarseDET"],
            "P": ["IN", "TO", "ADP", "P", "CoarseADP"],
            "N": ["NN", "NNS", "NNP", "NNPS", "FW", "NOUN", "PROPN", "N", "CoarseNOUN"],
            "V": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            "O": [],  # Other tags fall back to "O"
        }

        # Noun-phrase grammars (in terms of A, D, P, N)
        # BaseNP:     (Adj | Noun)* Noun
        # PP:         Prep Determiner* BaseNP
        # Full NP:    BaseNP (PP)*
        self.base_np_grammar = r"(A|N)*N"
        self.pp_grammar = r"P(D*(A|N)*N)"
        self.np_grammar = rf"{self.base_np_grammar}({self.pp_grammar})*"

        # Simple verb-argument patterns.
        # These are deliberately conservative – the goal is not perfect
        # syntactic coverage but high-precision relational phrases.
        #
        # Note: "SubjVerb" and "SubjCopula" are handled together in
        # _extract_verb_phrases: we use this generic N(V)+ pattern and then
        # split matches into:
        #   - SubjVerb   → subject + NON-copular verb(s)
        #   - SubjCopula → subject + only copular "be" verbs
        self.verb_arg_grammar: Dict[str, str] = {
            # Verb followed by optional adjectives/nouns and a final noun.
            "VerbObj": r"V(N|A)*N",
            # Verb followed by a PP headed by a preposition.
            "VerbPP": r"V(PD*(A|N)*N)",
            # Raw subject-verb pattern (N + one or more verbs).
            # We will refine this into SubjVerb vs SubjCopula downstream.
            "SubjVerb": r"N(V)+",
        }

        # Lexicon of copular "be" forms used to distinguish SubjCopula
        # from contentful SubjVerb patterns.
        self._copula_forms: Set[str] = {
            "be", "is", "are", "was", "were", "been", "being",
            "am",  # rarely tagged in N-subject patterns, but harmless to include
        }

        # Lightweight patterns for Markdown footnotes / references
        self._md_footnote_ref_re = re.compile(r"\[\^?[0-9a-zA-Z_-]+\]")
        self._md_footnote_def_re = re.compile(
            r"^\[\^?[0-9a-zA-Z_-]+\]:\s+.*$", re.MULTILINE
        )
        self._md_reference_def_re = re.compile(
            r"^\[[^\]]+\]:\s+.*$", re.MULTILINE
        )

        # Entity labels considered "proper" (used with spaCy backend)
        self._proper_entity_labels: Set[str] = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "NORP",
            "FAC",
            "EVENT",
            "WORK_OF_ART",
            "LAW",
            "LANGUAGE",
        }
        

    def mine_phrases(self, texts: List[str]) -> Counter:
        """
        Mine phrases from a list of documents and return a simple frequency Counter.

        This is a convenience API for users who only need aggregate noun/verb
        phrases and do not care about per-occurrence metadata.

        Pipeline:
        1. (Optionally) clean Markdown artifacts.
        2. POS-tag each document.
        3. Extract noun phrases (and optionally verb-argument patterns).
        4. Aggregates raw counts across all documents.
        5. Removes subsumed phrases by subtracting counts of longer phrases
           from their contained substrings.
        6. Clean Wikipedia-style citation markers from phrase strings.

        Parameters
        ----------
        texts:
            List of raw text documents (each can be a sentence, paragraph,
            meeting transcript, etc.).

        Returns
        -------
        Counter
            Cleaned frequency distribution mapping phrase → count.
        """
        phrase_counter: Counter = Counter()

        for doc_index, raw_doc in enumerate(texts):
            # Optional Markdown → plain text preprocessing
            doc = self._preprocess_document_text(raw_doc)

            # Mine this document (sentence-aware for both spaCy and NLTK)
            np_counts, vp_counts, _ = self._mine_single_document(
                doc,
                doc_index=doc_index,
                collect_phrase_records=False,  # no PhraseRecord instances
                phrase_records=None,
            )

            phrase_counter.update(np_counts)

            if self.include_verb_phrases:
                for pattern_counts in vp_counts.values():
                    phrase_counter.update(pattern_counts)

        cleaned = self._remove_subsumed_phrases(phrase_counter)
        return cleaned

                
    def _mine_single_document(
        self,
        text: str,
        doc_index: int,
        collect_phrase_records: bool,
        phrase_records: Optional[List[PhraseRecord]],
    ) -> Tuple[Counter, Dict[str, Counter], List[str]]:
        """
        Mine phrases from a single *preprocessed* document string.

        This method is sentence-aware (both for spaCy and NLTK backends) and
        returns:

        - local NP counts,
        - local VP counts (per pattern),
        - the list of sentence texts for this document, in the exact order and
          indexing that PhraseRecord.doc_index / sent_index refer to.

        Parameters
        ----------
        text:
            Preprocessed document text (Markdown cleaned if enabled).
        doc_index:
            Index of this document in the original input list.
        collect_phrase_records:
            If True, NP/VP occurrences will be appended to the provided
            phrase_records list.
        phrase_records:
            List that will collect PhraseRecord instances if
            collect_phrase_records is True.

        Returns
        -------
        np_counts:
            Counter of noun phrases in this document.
        vp_counts:
            Dict from pattern name → Counter of verb phrases.
        doc_sentences:
            List of sentence strings for this document; their indices
            are exactly the sent_index values used in PhraseRecord.
        """
        np_counts: Counter = Counter()
        vp_counts: Dict[str, Counter] = {
            "VerbObj": Counter(),
            "VerbPP": Counter(),
            "SubjVerb": Counter(),
            "SubjCopula": Counter(),
        }
        doc_sentences: List[str] = []


        if self.method == "spacy":
            # Use spaCy's sentence boundaries so headings and sentences are separate.
            doc = self._nlp(text)  # type: ignore[operator]

            for sent_index, sent in enumerate(doc.sents):
                sent_text = sent.text.strip()
                doc_sentences.append(sent_text)
                
                tokens = [token.text for token in sent]
                pos_tags = [token.tag_ for token in sent]

                tokens, pos_tags = self._merge_possessives(tokens, pos_tags)

                record_list = phrase_records if collect_phrase_records else None

                # Noun phrases for this sentence
                sent_np = self._extract_noun_phrases(
                    tokens,
                    pos_tags,
                    doc_index=doc_index,
                    sent_index=sent_index,
                    phrase_records=record_list,
                )
                np_counts.update(sent_np)

                # Optional verb phrases for this sentence
                if self.include_verb_phrases:
                    sent_vp = self._extract_verb_phrases(
                        tokens,
                        pos_tags,
                        doc_index=doc_index,
                        sent_index=sent_index,
                        phrase_records=record_list,
                    )
                    for name, counter in sent_vp.items():
                        vp_counts[name].update(counter)

        elif self.method == "nltk":
            import nltk

            # 1) Use Punkt to get sentence-like chunks
            raw_sents = nltk.sent_tokenize(text)

            # 2) Further split each chunk on literal newlines so that
            #    block-level lines (e.g. headings, list items) do not get
            #    glued to the following sentence.
            sentences: List[str] = []
            for s in raw_sents:
                for line in s.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    sentences.append(line)

            for sent_index, sent_text in enumerate(sentences):
                doc_sentences.append(sent_text)
                
                tokens = self._tokenizer.tokenize(sent_text)
                tagged = self._tagger.tag(tokens)
                pos_tags = [t for _, t in tagged]
                tokens = [w for w, _ in tagged]

                tokens, pos_tags = self._merge_possessives(tokens, pos_tags)

                record_list = phrase_records if collect_phrase_records else None

                # Noun phrases for this sentence/line
                sent_np = self._extract_noun_phrases(
                    tokens,
                    pos_tags,
                    doc_index=doc_index,
                    sent_index=sent_index,
                    phrase_records=record_list,
                )
                np_counts.update(sent_np)

                # Optional verb phrases
                if self.include_verb_phrases:
                    sent_vp = self._extract_verb_phrases(
                        tokens,
                        pos_tags,
                        doc_index=doc_index,
                        sent_index=sent_index,
                        phrase_records=record_list,
                    )
                    for name, counter in sent_vp.items():
                        vp_counts[name].update(counter)
        else:
            raise ValueError("method must be 'spacy' or 'nltk'")

        return np_counts, vp_counts, doc_sentences



    def mine_phrases_with_types(
        self,
        texts: List[str],
    ) -> Tuple[Counter, Dict[str, Counter], List[PhraseRecord], List[List[str]]]:
        """
        Extended variant of :meth:`mine_phrases` that:

        - keeps noun phrases and verb phrases separate (NP vs each VP pattern),
        - ALWAYS returns per-phrase metadata (PhraseRecord),
        - returns the sentence grid used to define doc_index / sent_index.

        Parameters
        ----------
        texts:
            List of raw text documents.

        Returns
        -------
        np_counter:
            Cleaned frequency distribution for noun phrases only
            (after subsumed-phrase removal).
        vp_counters:
            Mapping from verb-pattern name (e.g. "VerbObj", "VerbPP",
            "SubjVerb", "SubjCopula") to its own Counter of phrases.
            If ``include_verb_phrases`` is False, this will be an empty dict.
        phrase_records:
            List of PhraseRecord objects for *all* NP/VP occurrences
            across all documents.
        sentences_by_doc:
            Nested list of sentence strings, aligned with PhraseRecord
            indices, i.e.:

                sentences_by_doc[doc_index][sent_index] -> sentence text
        """
        total_np: Counter = Counter()
        total_vp: Dict[str, Counter] = {
            "VerbObj": Counter(),
            "VerbPP": Counter(),
            "SubjVerb": Counter(),
            "SubjCopula": Counter(),
        }
        phrase_records: List[PhraseRecord] = []
        sentences_by_doc: List[List[str]] = []

        for doc_index, raw_doc in enumerate(texts):
            doc = self._preprocess_document_text(raw_doc)

            np_counts, vp_counts, doc_sentences = self._mine_single_document(
                doc,
                doc_index=doc_index,
                collect_phrase_records=True,
                phrase_records=phrase_records,
            )

            sentences_by_doc.append(doc_sentences)
            total_np.update(np_counts)

            if self.include_verb_phrases:
                for name, counter in vp_counts.items():
                    total_vp[name].update(counter)

        # Apply subsumed-phrase removal only to noun phrases
        total_np = self._remove_subsumed_phrases(total_np)

        if not self.include_verb_phrases:
            total_vp = {}

        return total_np, total_vp, phrase_records, sentences_by_doc



    # -------------------------
    # Internal Tagging Methods
    # -------------------------
    def _tag_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenize and POS-tag text, then merge possessives.

        Returns
        -------
        tokens:
            Final token list after merging possessive constructions.
            Possessive endings ( ’s / 's / ’ ) are glued to the preceding noun
            so “Kant’s moral philosophy” is treated as one noun-phrase.
        
        pos_tags:
            Fine-grained POS tags aligned with ``tokens``.
        """
        if self.method == "spacy":
            doc = self._nlp(text)  # type: ignore[operator]
            tokens = [token.text for token in doc]
            pos_fine = [token.tag_ for token in doc]  # Fine-grained tags
        else:
            # NLTK approach
            import nltk
            tokens = nltk.word_tokenize(text)
            # CHANGED: use self._tagger (the attribute set in __init__)
            pos_fine = [t for _, t in self._tagger.tag(tokens)]
            tokens = [w for w, _ in self._tagger.tag(tokens)]
            
        tokens, pos_fine = self._merge_possessives(tokens, pos_fine)
        return tokens, pos_fine

    # -------------------------
    # Phrase Extraction Methods
    # -------------------------
    def _extract_noun_phrases(
        self,
        tokens: List[str],
        pos_tags: List[str],
        doc_index: int = -1,
        sent_index: int = -1,
        phrase_records: Optional[List[PhraseRecord]] = None,
    ) -> Counter:
        """
        Extract noun phrases (NPs) using the NP grammar and optionally
        record each occurrence as a PhraseRecord.

        Parameters
        ----------
        tokens, pos_tags:
            Token sequence and corresponding fine-grained POS tags for
            a single sentence.
        doc_index, sent_index:
            Where this sentence lives in the original input (used only
            for PhraseRecord metadata).
        phrase_records:
            If provided, each NP occurrence will append a PhraseRecord
            to this list.

        Returns
        -------
        Counter
            Phrase → local frequency within this sentence.
        """
        coarse_tags = self._to_coarse_tags(pos_tags)
        tag_sequence = "".join(coarse_tags)

        # Match noun phrases with the NP grammar over the coarse tag string.
        matches = re.finditer(self.np_grammar, tag_sequence)

        phrases: List[str] = []
        for m in matches:
            raw_phrase = " ".join(tokens[m.start() : m.end()])
            cleaned = self._clean_phrase(raw_phrase)
            if not cleaned:
                continue

            # Smart canonicalization (preserve proper names, lowercase common phrases).
            canonical = self._canonicalize_phrase_text(cleaned)
            if not canonical:
                continue

            phrases.append(canonical)

            # Optional metadata capture
            if phrase_records is not None:
                span_tags = coarse_tags[m.start() : m.end()]
                pattern = self._classify_np_pattern(span_tags)
                phrase_records.append(
                    PhraseRecord(
                        phrase=canonical,
                        surface=cleaned,
                        kind="NP",
                        pattern=pattern,
                        doc_index=doc_index,
                        sent_index=sent_index,
                    )
                )

        return Counter(phrases)

    def _classify_np_pattern(self, span_tags: List[str]) -> str:
        """
        Classify an NP span based on its coarse tag sequence.

        Returns one of:
          - "BaseNP"      → no prepositions (P) in the span
          - "NP+PP"       → exactly one P (single PP attached)
          - "NP+multiPP"  → multiple P (NP with multiple PP attachments)
        """
        num_p = span_tags.count("P")
        if num_p == 0:
            return "BaseNP"
        elif num_p == 1:
            return "NP+PP"
        else:
            return "NP+multiPP"

    
    def _extract_verb_phrases(
        self,
        tokens: List[str],
        pos_tags: List[str],
        doc_index: int = -1,
        sent_index: int = -1,
        phrase_records: Optional[List[PhraseRecord]] = None,
    ) -> Dict[str, Counter]:
        """
        Extract verb-argument patterns according to configured grammars.

        We currently support:
          - "VerbObj"    : V (A|N)* N
          - "VerbPP"     : V (P D* (A|N)* N)
          - "SubjVerb"   : N + NON-copular verb(s)
          - "SubjCopula" : N + only copular 'be' verbs (is/are/was/...)

        Internally, we first find all N(V)+ spans, then split them into
        SubjVerb vs SubjCopula by inspecting the actual verb tokens.

        Parameters
        ----------
        tokens, pos_tags:
            Token sequence and corresponding fine-grained POS tags for
            a single sentence.
        doc_index, sent_index:
            Where this sentence lives in the original input (used only
            for PhraseRecord metadata).
        phrase_records:
            If provided, each VP occurrence will append a PhraseRecord
            to this list.

        Returns
        -------
        Dict[str, Counter]
            Mapping from pattern name (e.g. "VerbObj", "VerbPP",
            "SubjVerb", "SubjCopula") to a local Counter of phrases.
        """
        patterns: Dict[str, Counter] = {
            "VerbObj": Counter(),
            "VerbPP": Counter(),
            "SubjVerb": Counter(),
            "SubjCopula": Counter(),
        }

        coarse_tags = self._to_coarse_tags(pos_tags)
        tag_sequence = "".join(coarse_tags)

        # 1) Handle VerbObj and VerbPP using their regex grammars directly.
        for name in ("VerbObj", "VerbPP"):
            grammar = self.verb_arg_grammar[name]
            for m in re.finditer(grammar, tag_sequence):
                raw_phrase = " ".join(tokens[m.start() : m.end()])
                cleaned = self._clean_phrase(raw_phrase)
                if not cleaned:
                    continue
                canonical = self._canonicalize_phrase_text(cleaned)
                if not canonical:
                    continue

                patterns[name][canonical] += 1

                if phrase_records is not None:
                    phrase_records.append(
                        PhraseRecord(
                            phrase=canonical,
                            surface=cleaned,
                            kind="VP",
                            pattern=name,
                            doc_index=doc_index,
                            sent_index=sent_index,
                        )
                    )

        # 2) Handle SubjVerb vs SubjCopula using a shared N(V)+ pattern.
        subj_verb_pattern = self.verb_arg_grammar["SubjVerb"]
        for m in re.finditer(subj_verb_pattern, tag_sequence):
            span_start, span_end = m.start(), m.end()
            raw_phrase = " ".join(tokens[span_start:span_end])
            cleaned = self._clean_phrase(raw_phrase)
            if not cleaned:
                continue
            canonical = self._canonicalize_phrase_text(cleaned)
            if not canonical:
                continue

            # Identify which positions in the span are verbs (coarse "V")
            verb_indices: List[int] = [
                idx for idx in range(span_start, span_end)
                if coarse_tags[idx] == "V"
            ]
            verb_tokens = [tokens[i].lower() for i in verb_indices]

            # Decide whether this is a pure copular pattern or not.
            if verb_tokens and all(v in self._copula_forms for v in verb_tokens):
                pattern_name = "SubjCopula"
            else:
                pattern_name = "SubjVerb"

            patterns[pattern_name][canonical] += 1

            if phrase_records is not None:
                phrase_records.append(
                    PhraseRecord(
                        phrase=canonical,
                        surface=cleaned,
                        kind="VP",
                        pattern=pattern_name,  # "SubjVerb" or "SubjCopula"
                        doc_index=doc_index,
                        sent_index=sent_index,
                    )
                )

        return patterns


    # ------------------------------------------------------------------
    # Phrase cleaning: remove citations like [4], [1][2][4], etc.
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_phrase(phrase: str) -> str:
        """
        Normalize and denoise a mined phrase string.

        Specifically tailored to Wikipedia-style text where phrases may be
        contaminated by footnote markers such as::

            "relation to the physical body.[4]"
            "yoga schools of hindu philosophy,[9]"
            "behaviorism.[1][2][4]"

        This function:
          * trims outer whitespace
          * removes patterns like "[4]" or ".[4]" / ",[9]" (with optional
            commas / spaces inside the brackets)
          * collapses repeated whitespace
          * strips leading/trailing punctuation
          * drops phrases that end up empty or purely numeric
        """
        # CHANGED: only normalize outer whitespace here (no global lowercasing).
        phrase = phrase.strip()

        # Remove citation markers of the form:
        #   [4]
        #   [1, 2]
        #   .[4] / ,[9] (we also drop the dot/comma just before the bracket)
        #
        # The regex:
        #   (\s*[.,])?     → optional spaces + '.' or ','
        #   \[ [0-9,\s]+ ] → bracketed digits, commas, spaces
        phrase = re.sub(r"(\s*[.,])?\[[0-9,\s]+\]", "", phrase)

        # Collapse excess internal whitespace created by removals.
        phrase = re.sub(r"\s+", " ", phrase)

        # Strip generic leading/trailing punctuation: periods, commas,
        # brackets, quotes, etc.
        phrase = phrase.strip(" .,!?:;[]()\"'")

        # If after cleaning we only have digits (e.g. "2020", "4"), treat it
        # as noise and discard.
        if phrase.isdigit():
            return ""

        return phrase
    
    # ------------------
    # Utility • Tag maps
    # ------------------
    def _to_coarse_tags(self, pos_tags: List[str]) -> List[str]:
        """Convert fine-grained tags to coarse categories."""
        return [self._map_tag(tag) for tag in pos_tags]

    def _map_tag(self, tag: str) -> str:
        for coarse, tag_list in self.coarsemap.items():
            if tag in tag_list:
                return coarse
        return 'O'

    # -------------------------------------
    # Canonicalization (case strategy)
    # -------------------------------------
    def _canonicalize_phrase_text(self, phrase: str) -> str:
        """
        Compute a canonical form for a phrase string.

        - With SpaCy backend:
            * Preserve capitalization for tokens that are:
              - part of PERSON/ORG/GPE/... entities, or
              - tagged as PROPN
            * Lowercase all other tokens (sentence-initial caps, titles, etc.).
        - With NLTK backend:
            * Fall back to full lowercasing.
        """
        phrase = phrase.strip()
        if not phrase:
            return ""

        # NLTK backend (no NER available here) → conservative lowercasing.
        if self.method != "spacy" or self._nlp is None:
            return phrase.lower()

        # SpaCy backend: use a short pass over the phrase to mix kept + lowered tokens
        doc = self._nlp(phrase)

        # Map token indices that belong to "proper" named entities
        proper_ent_token_idxs: Set[int] = set()
        for ent in doc.ents:
            if ent.label_ in self._proper_entity_labels:
                proper_ent_token_idxs.update(range(ent.start, ent.end))

        canonical_tokens: List[str] = []
        for token in doc:
            if token.pos_ == "PROPN" or token.i in proper_ent_token_idxs:
                canonical_tokens.append(token.text)
            else:
                canonical_tokens.append(token.text.lower())

        canonical = " ".join(canonical_tokens)
        return canonical.strip()

    # ---------------------------------------
    # Document-level Markdown preprocessing
    # ---------------------------------------
    def _preprocess_document_text(self, text: str) -> str:
        """
        Preprocess a raw document before POS tagging.

        Currently:
        - If ``self.clean_markdown`` is True, clean Markdown to plain text
          while stripping reference/footnote artifacts.
        - Otherwise, return the text unchanged.
        """
        if not self.clean_markdown:
            return text
        return self._clean_markdown_text(text)

    def _clean_markdown_text(self, text: str) -> str:
        """
        Convert Markdown-ish text to plain text suitable for NLP.

        This is intentionally lightweight and robust:

        - Strips reference-style link and footnote definition lines, e.g.:
            [ref]: https://example.com
            [^1]: Some note
        - Removes inline footnote markers: [^1], [1], [note-id]
        - Uses 'markdown' + 'beautifulsoup4' if available to remove formatting
          and code blocks; otherwise falls back to a simple regex-only cleanup.

        IMPORTANT: we preserve newlines (paragraph / heading boundaries)
        so that later POS-tagging and NP grammar don't glue together headings
        and sentences into junk like "psychology psychology".
        """
        # Remove definition lines (link refs, footnote defs)
        without_defs = self._md_reference_def_re.sub("", text)
        without_defs = self._md_footnote_def_re.sub("", without_defs)

        # Remove inline footnote/reference markers like [^1], [1], [note-id]
        without_defs = self._md_footnote_ref_re.sub("", without_defs)

        # Normalize *excessive* blank lines (3+ → 2), but keep paragraph breaks
        without_defs = re.sub(r"\n{3,}", "\n\n", without_defs)

        # Try full Markdown → HTML → plain text pipeline
        try:
            import markdown as _markdown
            html = _markdown.markdown(
                without_defs,
                extensions=[],
                output_format="html5",
            )
        except Exception:
            # Fallback: crude Markdown cleaning without extra dependencies
            crude = without_defs

            # Strip heading / emphasis markers etc. but keep line breaks.
            crude = re.sub(r"[#*_`~>-]+", " ", crude)

            # Inline links: [text](url) → text
            crude = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", crude)

            # Collapse spaces and tabs, but *not* newlines
            crude = re.sub(r"[ \t]+", " ", crude)
            crude = re.sub(r"\n{3,}", "\n\n", crude)

            return crude.strip()

        # If we successfully produced HTML, try BeautifulSoup
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Drop code/pre blocks (often noise for NLP)
            for tag in soup.find_all(["code", "pre"]):
                tag.decompose()

            # NEW: keep block boundaries as newlines, not spaces
            text_out = soup.get_text(separator="\n")

            # Collapse spaces and tabs inside lines, preserve newlines
            text_out = re.sub(r"[ \t]+", " ", text_out)
            text_out = re.sub(r"\n{3,}", "\n\n", text_out)

            return text_out.strip()
        except Exception:
            # If BeautifulSoup is missing, strip HTML tags with a newline
            html_no_tags = re.sub(r"<[^>]+>", "\n", html)
            html_no_tags = re.sub(r"[ \t]+", " ", html_no_tags)
            html_no_tags = re.sub(r"\n{3,}", "\n\n", html_no_tags)
            return html_no_tags.strip()


    # -------------------------------
    # Subsumed-Phrase Removal Method
    # -------------------------------
    def _remove_subsumed_phrases(self, phrase_counter: Counter) -> Counter:
        """
        If shorter_phrase is a contiguous sublist of a longer_phrase,
        reduce the shorter_phrase's frequency by the longer_phrase's frequency.
        Remove if freq <= 0.

        Example
        -------
        If we have::
            'university of california' : 5
            'university of california berkeley' : 3

        then we subtract 3 from the shorter phrase, yielding::
            'university of california' : 2
            'university of california berkeley' : 3

        If a shorter phrase's frequency drops to zero or below, it is removed.
        """
        phrases = phrase_counter.copy()
        # Sort phrases by descending length (in tokens) so we always subtract
        # from shorter phrases after processing longer ones.
        sorted_phrases = sorted(
            phrases.keys(),
            key=lambda x: len(x.split()),
            reverse=True,
        )

        for i in range(len(sorted_phrases)):
            if sorted_phrases[i] not in phrases:
                continue
                
            longer_phrase = sorted_phrases[i]
            longer_freq = phrases[longer_phrase]
            if longer_freq <= 0:
                continue
                
            # Tokenize once
            longer_tokens = longer_phrase.split()

            # Check shorter phrases
            for j in range(i+1, len(sorted_phrases)):
                if sorted_phrases[j] not in phrases:
                    continue

                shorter_phrase = sorted_phrases[j]
                shorter_freq = phrases[shorter_phrase]
                if shorter_freq <= 0:
                    continue

                # Check token-based sublist
                if self._is_token_sublist(shorter_phrase, longer_tokens):
                    new_freq = shorter_freq - longer_freq
                    if new_freq <= 0:
                        del phrases[shorter_phrase]
                    else:
                        phrases[shorter_phrase] = new_freq

        # Drop any phrase whose final count is non-positive <= 0
        return Counter({p: c for p, c in phrases.items() if c > 0})

    @staticmethod
    def _is_token_sublist(shorter_phrase: str, longer_tokens: List[str]) -> bool:
        """
        Return True if 'shorter_phrase' is a contiguous token sublist of 'longer_tokens'.
        Example:
            shorter_phrase = "university of california"
            longer_tokens = ["university", "of", "california", "berkeley"] -> True
        """
        shorter_tokens = shorter_phrase.split()
        len_shorter = len(shorter_tokens)
        len_longer = len(longer_tokens)
        if len_shorter == 0 or len_shorter > len_longer:
            return False

        for start_idx in range(len_longer - len_shorter + 1):
            if longer_tokens[start_idx : start_idx + len_shorter] == shorter_tokens:
                return True
        return False

    # ------------------
    # POS possessive merge
    # ------------------
    @staticmethod
    def _merge_possessives(tokens: List[str], tags: List[str]):
        """
        Merge [NOUN, POS] or [NOUN, '’'] into a single possessive-noun token.
        * Do NOT merge when the first token is a pronoun (PRP, PRP$) so that forms like "she's" remain separate.
        * Works for both 's and trailing apostrophe (plural possessive).
        """
        merged_tok, merged_tag = [], []
        i = 0
        while i < len(tokens):
            if (i + 1 < len(tokens)
                and tags[i+1] == "POS"                         # SpaCy & NLTK use POS for ’s
                and tags[i] not in {"PRP", "PRP$"}             # skip he’s / she’s
                and tags[i] not in {"WP", "WP$"}):             # skip who’s etc.
                # combine tokens[i] + tokens[i+1]  ➜ keep noun tag
                merged_tok.append(tokens[i] + tokens[i+1])    # Kant + 's  →  Kant's
                merged_tag.append(tags[i])                    # keep original noun tag
                i += 2
            # plural possessive: token endswith 's'  +  next token is "'"
            elif (i + 1 < len(tokens)
                  and tokens[i+1] in {"'", "’"}
                  and tags[i] not in {"PRP", "PRP$", "WP", "WP$"}):
                merged_tok.append(tokens[i] + tokens[i+1])    # philosophers + ' → philosophers'
                merged_tag.append(tags[i])
                i += 2
            else:
                merged_tok.append(tokens[i])
                merged_tag.append(tags[i])
                i += 1
        return merged_tok, merged_tag


    # ---------------------------------------------------------------------
    # Lazy back-end loaders (keep heavy imports optional)
    # ---------------------------------------------------------------------
    @staticmethod
    def _load_spacy_model(model_name: str):
        """
        Load a SpaCy model, downloading it on-the-fly if necessary.

        This keeps installation friction low for end-users: they can simply
        install ``spacy`` and let the library fetch the appropriate language
        model the first time it is used.
        """
        import importlib
        import subprocess
        import sys

        try:
            import spacy

            return spacy.load(model_name)  # type: ignore[return-value]
        except OSError:
            # Model not downloaded yet → auto-download.
            print(f"[PhraseMiner] spaCy model '{model_name}' not found. Downloading…")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            import spacy  # re-import after download

            return spacy.load(model_name)  # type: ignore[return-value]
        except ImportError as e:  # spaCy not installed
            raise ImportError(
                "spaCy is required for method='spacy'. Install with 'pip install spacy'."
            ) from e

    @staticmethod
    def _load_nltk_models():
        """
        Load NLTK's tokenizer and PerceptronTagger.

        We download the minimal required resources automatically (punkt +
        averaged_perceptron_tagger) in a quiet mode.
        """
        import nltk

        from nltk.tag import PerceptronTagger
        from nltk.tokenize import TreebankWordTokenizer

        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("punkt", quiet=True)
        return TreebankWordTokenizer(), PerceptronTagger()
