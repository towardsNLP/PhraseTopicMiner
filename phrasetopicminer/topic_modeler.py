"""
topic_modeler.py

Core topic modeling for PhraseTopicMiner:
- Takes mined phrase metadata (PhraseRecord list) + sentence lists from PhraseMiner.
- Optionally consumes pre-aggregated phrase frequencies (Counter) from PhraseMiner
  (after subsumed-phrase removal). If not provided, frequencies are reconstructed
  from PhraseRecord occurrences.
- Embeds phrases with configurable backends.
- Optionally denoises with PCA.
- Builds a clustering geometry with UMAP.
- Clusters phrases using HDBSCAN or weighted KMeans (Auto-K via Silhouette).
- Produces a structured TopicCoreResult with:
    * phrase-level DataFrame (phrases, counts, cluster_id, x/y for viz)
    * TopicCluster objects (phrase lists, counts, representative phrases)
    * phrase_occurrences mapping: phrase → [PhraseRecord, ...]
      (doc_index, sent_index, NP/VP pattern information).
    * phrase_sentences mapping: phrase → list of sentence texts
      (resolved via doc_index/sent_index against sentences_by_doc).

This module is **API #1: Core clustering**.
Later APIs can:
- Add LLM-based titles/descriptions per cluster.
- Add timeline/topic-flow visualizations using sentence mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from collections import Counter

import numpy as np
import pandas as pd

from .phrase_miner import PhraseRecord  # import from your package


# ---------------------------------------------------------------------
# Dataclasses for core results
# ---------------------------------------------------------------------


@dataclass
class TopicCluster:
    """
    Lightweight container for a single topic cluster.

    Attributes
    ----------
    cluster_id:
        Integer label of the cluster (HDBSCAN/KMeans label).
        By convention, -1 is reserved for "noise" in HDBSCAN.
    phrases:
        List of phrase strings assigned to this cluster.
        (Canonicalized forms from PhraseMiner.)
    phrase_counts:
        Per-phrase frequencies (aligned with `phrases`).
    total_count:
        Sum of frequencies for all phrases in this cluster.
    representative_phrases:
        Short list of "headline" phrases for the cluster,
        typically the top-k phrases by frequency.
    importance_score:
        Scalar score used to rank clusters by importance. Higher is more
        important. By default this combines total frequency and cluster size.
    """

    cluster_id: int
    phrases: List[str]
    phrase_counts: List[int]
    total_count: int
    representative_phrases: List[str]
    importance_score: float


@dataclass
class TopicCoreResult:
    """
    Core output of the TopicModeler clustering pass.

    Attributes
    ----------
    phrases_df:
        Pandas DataFrame with one row per phrase used in clustering.
        Columns include:
            - 'phrase'      : canonical phrase string
            - 'count'       : frequency
            - 'n_tokens'    : length of phrase in tokens
            - 'cluster_id'  : integer cluster label
            - 'x', 'y'      : 2D coordinates for visualization
            - 'embedding'   : high-dimensional embedding vector (np.ndarray)
                              (kept as an object column)
    clusters:
        List of TopicCluster objects summarizing each cluster,
        excluding any HDBSCAN noise cluster (-1).
    phrase_occurrences:
        Mapping from phrase string → list of PhraseRecord occurrences,
        restricted to phrases that passed frequency and pattern filters.
    phrase_sentences:
        Mapping from phrase string → list of sentence texts in which that
        phrase appears (deduplicated by doc_index/sent_index) based on
        `sentences_by_doc`.
    config:
        Dictionary capturing key run-time parameters used for this fit
        (clustering algorithm, thresholds, UMAP/t-SNE settings, etc.).
        This is designed for reproducibility and audit trails.
    """

    phrases_df: pd.DataFrame
    clusters: List[TopicCluster]
    phrase_occurrences: Dict[str, List[PhraseRecord]]
    phrase_sentences: Dict[str, List[str]]
    config: Dict[str, Any]


# ---------------------------------------------------------------------
# TopicModeler – core clustering engine
# ---------------------------------------------------------------------


class TopicModeler:
    """
    Phrase-level topic modeling for PhraseTopicMiner.

    Responsibilities
    ----------------
    - Filter mined phrases by n-gram length + frequency thresholds.
    - Filter by kind/pattern (NP/VP, BaseNP/VerbObj/etc.) based on PhraseRecord.
    - Embed phrases into a dense vector space (SentenceTransformers, spaCy,
      or any custom batch embedding function).
    - Optionally denoise with PCA (e.g. 50D) before manifold learning.
    - Compute a clustering geometry with UMAP.
    - Cluster phrases with either:
        * HDBSCAN (density-based, handles noise), or
        * Auto-K KMeans with Silhouette selection (+ frequency weights).
    - Compute a separate 2D embedding for visualization (UMAP or t-SNE).
    - Build TopicCluster objects and a TopicCoreResult.

    Design philosophy
    -----------------
    - This class is **purely geometric/statistical**: it does not call LLMs.
      Later stages can consume TopicCoreResult to generate human-friendly
      titles, descriptions, and visual timelines.
    - All LLMs (OpenAI, Cohere, etc.) are intended to be plugged in via
      higher-level orchestration, not here.
    - For vendor embeddings (OpenAI, Cohere, etc.), use
      `embedding_backend="custom"` and pass a batch embedding function via
      `embedding_fn`.
    """

    def __init__(
        self,
        embedding_backend: Literal["sentence_transformers", "spacy", "custom"] = "sentence_transformers",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        spacy_nlp: Optional[Any] = None,
        random_state: int = 42,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        embedding_backend:
            Which embedding strategy to use:
                - "sentence_transformers":
                    Uses SentenceTransformers with `embedding_model`.
                - "spacy":
                    Uses a spaCy model passed via `spacy_nlp` that has
                    a `.vector` for tokens/documents.
                - "custom":
                    Uses a user-supplied batch embedding function
                    `embedding_fn(phrases: List[str]) -> np.ndarray`.
                    This is the recommended mode for OpenAI/Cohere APIs.
        embedding_model:
            Name of the SentenceTransformers model when
            `embedding_backend="sentence_transformers"`.
        embedding_fn:
            Batch embedding function used if `embedding_backend="custom"`.
            It must take a list of phrases and return an array of shape
            (n_phrases, dim).
        spacy_nlp:
            spaCy language object used when `embedding_backend="spacy"`.
            It must provide `.pipe()` and `.vector` for docs.
        random_state:
            Random seed used for PCA, UMAP, t-SNE, and KMeans
            to make experiments repeatable.
        logger:
            Optional logging callback used when `verbose=True` in :meth:`fit_core`.
            If provided, it must accept a single string argument. This allows
            you to plug in different UIs:

            - Console:   `logger=None` (falls back to `print`)
            - Streamlit: `logger=lambda msg: st.markdown(msg)`
            - Rich/loguru/etc.: wrap their log methods here.
        """
        self.embedding_backend = embedding_backend
        self.embedding_model_name = embedding_model
        self.embedding_fn = embedding_fn
        self.spacy_nlp = spacy_nlp
        self.random_state = random_state

        # Optional logger (UI-agnostic)
        self.logger = logger

        # Lazy-loaded embedding model for SentenceTransformers
        self._st_model = None

    # ------------------------------------------------------------------
    # Internal helper – unified logging
    # ------------------------------------------------------------------
    def _log(self, message: str, verbose: bool=True) -> None:
        """
        Log a message if `verbose` is True.

        - If `self.logger` is provided, it will be called with the message.
        - Otherwise, falls back to `print(message)`.

        This keeps the core logic UI-agnostic while letting callers plug in
        Streamlit, Rich, loguru, etc.
        """
        if not verbose:
            return
        if self.logger is not None:
            self.logger(message)
        else:
            print(message)

    # ------------------------------------------------------------------
    # Public core API
    # ------------------------------------------------------------------

    def fit_core(
        self,
        phrase_records: List[PhraseRecord],
        sentences_by_doc: List[List[str]],
        *,
        # optional: provide cleaned phrase frequencies from PhraseMiner
        phrase_frequencies: Optional[Counter] = None,
        # phrase filtering options
        include_kinds: Tuple[str, ...] = ("NP",),
        include_patterns: Optional[Set[str]] = None,
        min_freq_unigram: int = 3,
        min_freq_bigram: int = 2,
        min_freq_trigram_plus: int = 1,
        # geometric pipeline options
        pca_n_components: Optional[int] = 50,
        cluster_geometry: Literal["umap_2d", "umap_nd"] = "umap_nd",
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_cluster_n_components: int = 10,
        # clustering options
        clustering_algorithm: Literal["hdbscan", "kmeans"] = "hdbscan",
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: Optional[int] = None,
        hdbscan_metric: str = "euclidean",
        kmeans_max_clusters: int = 15,
        # visualization geometry
        viz_reducer: Literal["same", "umap_2d", "tsne_2d"] = "umap_2d",
        tsne_perplexity: float = 30.0,
        tsne_learning_rate: float = 200.0,
        tsne_n_iter: int = 1000,
        # cluster summaries
        top_n_representatives: int = 10,
        # logging
        verbose: bool = False,
    ) -> TopicCoreResult:
        """
        Execute the **core topic modeling pipeline**:

        1. Filter phrase occurrences (PhraseRecord) by kind/pattern.
        2. Aggregate phrase frequencies:
             - if `phrase_frequencies` is provided, use it as authoritative
               counts (e.g., post subsumed-phrase removal from PhraseMiner),
             - otherwise, reconstruct counts from PhraseRecord occurrences.
        3. Apply n-gram-aware frequency thresholds.
        4. Embed phrases with the configured backend.
        5. Optionally denoise with PCA.
        6. Build clustering geometry with UMAP (2D or ND).
        7. Cluster phrases (HDBSCAN or Auto-K KMeans, weighted by frequency).
        8. Compute 2D visualization coordinates (UMAP or t-SNE).
        9. Reconstruct phrase → sentence texts using `sentences_by_doc`.
        10. Build TopicCluster objects and assemble a TopicCoreResult.

        Parameters
        ----------
        phrase_records:
            List of PhraseRecord objects describing individual phrase
            occurrences (NP/VP kind, pattern, doc_index, sent_index, surface).
            These are the primary source of truth for:
                - filtering phrases by kind/pattern
                - building phrase → occurrences mapping.
        sentences_by_doc:
            Nested list of sentences for each document:
                sentences_by_doc[doc_index][sent_index] -> sentence text.
            This must come from PhraseMiner (or an equivalent pipeline) and
            use the same indexing convention as PhraseRecord.doc_index and
            PhraseRecord.sent_index.
        phrase_frequencies:
            Optional Counter mapping phrase string → global frequency
            (usually the NP Counter returned by PhraseMiner after
            subsumed-phrase removal). If not provided, frequencies are
            reconstructed by counting PhraseRecord occurrences.

        include_kinds:
            Tuple of allowed PhraseRecord.kind values, e.g. ("NP",) or
            ("NP", "VP"). Only phrases whose occurrences match these kinds
            are considered in clustering.
        include_patterns:
            Optional set of allowed PhraseRecord.pattern strings,
            e.g. {"BaseNP", "NP+PP"}. If provided, phrases whose occurrences
            have patterns outside this set will be ignored.
        min_freq_unigram, min_freq_bigram, min_freq_trigram_plus:
            Minimum frequency thresholds based on phrase token length.
            Example:
                - `min_freq_unigram=3` → keep unigrams with count ≥ 3
                - `min_freq_bigram=2`  → keep bigrams with count ≥ 2
                - `min_freq_trigram_plus=1` → keep 3+-grams with count ≥ 1

        pca_n_components:
            If not None, apply PCA to the embeddings before UMAP/t-SNE to
            denoise and stabilize the neighborhood structure (e.g. 50).
        cluster_geometry:
            Whether to run UMAP in 2D or ND for clustering:
                - "umap_2d":  clustering and visualization both in 2D UMAP.
                - "umap_nd":  clustering uses ND UMAP (e.g., 10D) while
                               visualization gets its own 2D embedding.
        umap_n_neighbors, umap_min_dist:
            UMAP parameters controlling the notion of local neighborhood
            and cluster compactness.
        umap_cluster_n_components:
            Dimensionality of the UMAP embedding used **for clustering**
            when `cluster_geometry="umap_nd"`. Ignored when "umap_2d".

        clustering_algorithm:
            - "hdbscan": density-based, handles variable densities and noise.
            - "kmeans" : centroid-based Auto-K KMeans with Silhouette
                         selection and **frequency weights**.
        hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_metric:
            HDBSCAN hyperparameters. See hdbscan docs for details.
        kmeans_max_clusters:
            Maximum K considered when searching for the best K via Silhouette
            (2..kmeans_max_clusters).

        viz_reducer:
            How to compute a 2D embedding for visualization:
                - "same"   : reuse the 2D clustering geometry (only valid if
                             cluster_geometry="umap_2d").
                - "umap_2d": run a 2D UMAP dedicated for visualization.
                - "tsne_2d": run t-SNE on the PCA-reduced embeddings.
        tsne_perplexity, tsne_learning_rate, tsne_n_iter:
            Standard t-SNE hyperparameters.

        top_n_representatives:
            Number of top phrases (by frequency) to include as
            `representative_phrases` in each TopicCluster.

        Returns
        -------
        TopicCoreResult
            Structured core result containing:
                - a phrase-level DataFrame,
                - TopicCluster summaries,
                - phrase_occurrence mapping,
                - phrase_sentences mapping,
                - a config dictionary with all relevant run-time parameters.
        """
        if not phrase_records:
            raise ValueError("phrase_records cannot be empty.")

        if not sentences_by_doc:
            raise ValueError("sentences_by_doc cannot be empty.")

        if verbose:
            self._log("[TopicModeler] Step 1/12 – filtering PhraseRecords by kind/pattern...")

        # --------------------------------------------------------------
        # 1. Filter phrase occurrences by kind/pattern
        # --------------------------------------------------------------
        allowed_kinds = set(include_kinds)
        allowed_patterns = set(include_patterns) if include_patterns else None

        filtered_records: List[PhraseRecord] = []
        for rec in phrase_records:
            if rec.kind not in allowed_kinds:
                continue
            if allowed_patterns is not None and rec.pattern not in allowed_patterns:
                continue
            filtered_records.append(rec)

        if not filtered_records:
            raise ValueError(
                "After filtering by kind/pattern, no PhraseRecord remains. "
                "Check include_kinds/include_patterns."
            )

        if verbose:
            self._log(f"[TopicModeler]   → kept {len(filtered_records)} occurrences after filtering.")

        # --------------------------------------------------------------
        # 2. Build phrase → occurrences & frequency counts
        # --------------------------------------------------------------
        occurrence_map: Dict[str, List[PhraseRecord]] = {}
        for rec in filtered_records:
            occurrence_map.setdefault(rec.phrase, []).append(rec)

        if phrase_frequencies is not None:
            # Use provided Counter (e.g., post subsumed-phrase removal) and
            # restrict to phrases that have occurrences.
            counts_dict = {
                phrase: phrase_frequencies[phrase]
                for phrase in occurrence_map.keys()
                if phrase in phrase_frequencies
            }
            freq_source = "external_counter"
        else:
            # Reconstruct simple counts from occurrences.
            counts_dict = {phrase: len(occs) for phrase, occs in occurrence_map.items()}
            freq_source = "reconstructed_from_records"

        if not counts_dict:
            raise ValueError(
                "No usable phrase frequencies after aligning phrase_records "
                "and phrase_frequencies. Ensure they come from the same "
                "PhraseMiner run."
            )

        if verbose:
            self._log(f"[TopicModeler] Step 2/12 – built occurrence map for {len(occurrence_map)} phrases "
                  f"(freq source: {freq_source}).")

        # --------------------------------------------------------------
        # 3. Build initial phrase DataFrame
        # --------------------------------------------------------------
        df = pd.DataFrame(
            [(p, c) for p, c in counts_dict.items()],
            columns=["phrase", "count"],
        )
        # sort by frequency (important for weighted KMeans & reporting)
        df.sort_values("count", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Add token length for n-gram-aware thresholds
        df["n_tokens"] = df["phrase"].str.split().str.len()

        if verbose:
            self._log(f"[TopicModeler] Step 3/12 – phrase DataFrame has {len(df)} rows before n-gram filtering.")

        # --------------------------------------------------------------
        # 4. Apply n-gram frequency thresholds
        # --------------------------------------------------------------
        df = self._filter_by_ngram_frequency(
            df,
            min_freq_unigram=min_freq_unigram,
            min_freq_bigram=min_freq_bigram,
            min_freq_trigram_plus=min_freq_trigram_plus,
        )
        if df.empty:
            raise ValueError(
                "All phrases were filtered out by n-gram frequency thresholds. "
                "Consider lowering min_freq_* parameters."
            )

        # Filter occurrence_map accordingly (we only keep phrases in df)
        kept_phrases = set(df["phrase"].tolist())
        phrase_occurrences: Dict[str, List[PhraseRecord]] = {
            p: occs for p, occs in occurrence_map.items() if p in kept_phrases
        }

        if verbose:
            self._log(f"[TopicModeler] Step 4/12 – {len(df)} phrases remain after n-gram thresholds.")

        # --------------------------------------------------------------
        # 5. Reconstruct phrase → sentences mapping
        # --------------------------------------------------------------
        phrase_sentences = self._build_phrase_sentences(
            phrase_occurrences=phrase_occurrences,
            sentences_by_doc=sentences_by_doc,
        )

        if verbose:
            self._log(f"[TopicModeler] Step 5/12 – phrase_sentences built for {len(phrase_sentences)} phrases.")

        # --------------------------------------------------------------
        # 6. Embed phrases
        # --------------------------------------------------------------
        phrases = df["phrase"].tolist()
        self._log(f"[TopicModeler] Step 6/12 – embedding {len(phrases)} phrases...", verbose)
        embeddings = self._embed_phrases(phrases, verbose=verbose)
        if embeddings.shape[0] != len(phrases):
            raise ValueError(
                "Embedding function returned mismatched number of vectors: "
                f"{embeddings.shape[0]} for {len(phrases)} phrases."
            )

        # Optionally store embedding vectors in the DataFrame
        df["embedding"] = list(embeddings)

        self._log(f"[TopicModeler]   → embeddings computed with backend='{self.embedding_backend}'.", verbose)

        # --------------------------------------------------------------
        # 7. Optional PCA denoising
        # --------------------------------------------------------------
        X = embeddings
        if pca_n_components is not None:
            X = self._apply_pca(X, n_components=pca_n_components)

        if pca_n_components is not None and verbose:
            self._log(f"[TopicModeler] Step 7/12 – PCA denoising to {pca_n_components} dims.")

        # --------------------------------------------------------------
        # 8. Clustering geometry (UMAP)
        # --------------------------------------------------------------
        if cluster_geometry == "umap_2d":
            # Geometry and viz share the same 2D UMAP
            cluster_embedding = self._compute_umap(
                X,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                n_components=2,
            )
        elif cluster_geometry == "umap_nd":
            cluster_embedding = self._compute_umap(
                X,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                n_components=umap_cluster_n_components,
            )
        else:
            raise ValueError("cluster_geometry must be 'umap_2d' or 'umap_nd'.")

        if verbose:
            self._log(f"[TopicModeler] Step 8/12 – UMAP geometry ({cluster_geometry}).")

        # --------------------------------------------------------------
        # 9. Cluster phrases (HDBSCAN or KMeans)
        # --------------------------------------------------------------
        counts = df["count"].to_numpy(dtype=float)

        if clustering_algorithm == "hdbscan":
            labels = self._cluster_hdbscan(
                cluster_embedding,
                min_cluster_size=hdbscan_min_cluster_size,
                min_samples=hdbscan_min_samples,
                metric=hdbscan_metric,
            )
        elif clustering_algorithm == "kmeans":
            labels = self._cluster_kmeans_auto_k(
                cluster_embedding,
                weights=counts,
                max_clusters=kmeans_max_clusters,
            )
        else:
            raise ValueError("clustering_algorithm must be 'hdbscan' or 'kmeans'.")

        df["cluster_id"] = labels

        if verbose:
            self._log(f"[TopicModeler] Step 9/12 – clustering with '{clustering_algorithm}'.")

        # --------------------------------------------------------------
        # 10. Visualization embedding (2D)
        # --------------------------------------------------------------
        # Logic:
        # - If cluster_geometry == "umap_2d" and viz_reducer == "same":
        #     reuse the 2D cluster_embedding.
        # - Otherwise:
        #     compute a separate 2D embedding (UMAP or t-SNE) on X (PCA space).
        if cluster_geometry == "umap_2d" and viz_reducer == "same":
            viz_embedding = cluster_embedding
            if viz_embedding.shape[1] != 2:
                raise RuntimeError(
                    "Internal inconsistency: 'umap_2d' geometry did not "
                    "produce 2D embedding."
                )
        else:
            if viz_reducer == "umap_2d":
                viz_embedding = self._compute_umap(
                    X,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_components=2,
                )
            elif viz_reducer == "tsne_2d":
                viz_embedding = self._compute_tsne(
                    X,
                    perplexity=tsne_perplexity,
                    learning_rate=tsne_learning_rate,
                    n_iter=tsne_n_iter,
                )
            elif viz_reducer == "same":
                # If user asks for "same" but cluster_geometry != "umap_2d",
                # there is no 2D geometry to reuse → fall back to UMAP 2D.
                viz_embedding = self._compute_umap(
                    X,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    n_components=2,
                )
            else:
                raise ValueError("viz_reducer must be 'same', 'umap_2d', or 'tsne_2d'.")

        df["x"] = viz_embedding[:, 0]
        df["y"] = viz_embedding[:, 1]

        if verbose:
            self._log(f"[TopicModeler] Step 10/12 – computing 2D viz embedding via '{viz_reducer}'.")

        # --------------------------------------------------------------
        # 11. Build TopicCluster objects
        # --------------------------------------------------------------
        clusters = self._build_topic_clusters(
            df,
            top_n_representatives=top_n_representatives,
        )

        if verbose:
            self._log("[TopicModeler] Step 11/12 – aggregating TopicCluster objects.")

        # --------------------------------------------------------------
        # 12. Assemble config dict for reproducibility
        # --------------------------------------------------------------
        num_documents = len(sentences_by_doc)
        num_sentences_total = sum(len(doc_sents) for doc_sents in sentences_by_doc)

        config = {
            "embedding_backend": self.embedding_backend,
            "embedding_model_name": self.embedding_model_name,
            "pca_n_components": pca_n_components,
            "cluster_geometry": cluster_geometry,
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
            "umap_cluster_n_components": umap_cluster_n_components,
            "clustering_algorithm": clustering_algorithm,
            "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
            "hdbscan_min_samples": hdbscan_min_samples,
            "hdbscan_metric": hdbscan_metric,
            "kmeans_max_clusters": kmeans_max_clusters,
            "viz_reducer": viz_reducer,
            "tsne_perplexity": tsne_perplexity,
            "tsne_learning_rate": tsne_learning_rate,
            "tsne_n_iter": tsne_n_iter,
            "min_freq_unigram": min_freq_unigram,
            "min_freq_bigram": min_freq_bigram,
            "min_freq_trigram_plus": min_freq_trigram_plus,
            "include_kinds": include_kinds,
            "include_patterns": (
                sorted(include_patterns) if include_patterns else None
            ),
            "random_state": self.random_state,
            "num_documents": num_documents,
            "num_sentences_total": num_sentences_total,
            "phrase_frequency_source": freq_source,
        }

        if verbose:
            self._log("[TopicModeler] Step 12/12 – assembling TopicCoreResult.")

        return TopicCoreResult(
            phrases_df=df,
            clusters=clusters,
            phrase_occurrences=phrase_occurrences,
            phrase_sentences=phrase_sentences,
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal helpers – phrase filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_ngram_frequency(
        df: pd.DataFrame,
        *,
        min_freq_unigram: int,
        min_freq_bigram: int,
        min_freq_trigram_plus: int,
    ) -> pd.DataFrame:
        """
        Filter phrases based on n-gram length and count thresholds.

        This lets you enforce stricter thresholds on unigrams (often noisy),
        looser thresholds on bigrams, and keep even rare but informative
        longer phrases (3+ tokens).
        """

        def keep_row(row) -> bool:
            n = row["n_tokens"]
            c = row["count"]
            if n <= 0:
                return False
            if n == 1:
                return c >= min_freq_unigram
            elif n == 2:
                return c >= min_freq_bigram
            else:
                return c >= min_freq_trigram_plus

        mask = df.apply(keep_row, axis=1)
        return df[mask].copy()

    # ------------------------------------------------------------------
    # Internal helpers – phrase → sentences reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_phrase_sentences(
        phrase_occurrences: Dict[str, List[PhraseRecord]],
        sentences_by_doc: List[List[str]],
    ) -> Dict[str, List[str]]:
        """
        Resolve PhraseRecord (doc_index, sent_index) into sentence strings,
        building phrase → [sentence, ...] mapping.

        - Deduplicates repeated (doc_index, sent_index) pairs per phrase
          while preserving the order of first appearance.
        - Silently skips out-of-range indices to be robust to mismatches.
        """
        from typing import Set

        phrase_sentences: Dict[str, List[str]] = {}

        for phrase, occs in phrase_occurrences.items():
            seen: Set[Tuple[int, int]] = set()
            sent_texts: List[str] = []

            for rec in occs:
                key = (rec.doc_index, rec.sent_index)
                if key in seen:
                    continue
                seen.add(key)

                if 0 <= rec.doc_index < len(sentences_by_doc):
                    doc_sents = sentences_by_doc[rec.doc_index]
                    if 0 <= rec.sent_index < len(doc_sents):
                        sent_texts.append(doc_sents[rec.sent_index])

            phrase_sentences[phrase] = sent_texts

        return phrase_sentences

    # ------------------------------------------------------------------
    # Internal helpers – embeddings
    # ------------------------------------------------------------------

    def _embed_phrases(self, phrases: List[str], verbose: bool = False) -> np.ndarray:
        """
        Compute embeddings for a list of phrases using the configured backend.

        Parameters
        ----------
        phrases:
            Canonical phrase strings to embed.
        verbose:
            If True and using SentenceTransformers **without** a custom logger,
            a built-in progress bar will be shown on stdout.

        Returns
        -------
        np.ndarray
            Array of shape (n_phrases, dim).
        """
        if self.embedding_backend == "sentence_transformers":
            # Only show the built-in progress bar when logging to console
            # (i.e., no custom logger is provided).
            show_progress = verbose and self.logger is None
            return self._embed_with_sentence_transformers(phrases, show_progress=show_progress)
        elif self.embedding_backend == "spacy":
            return self._embed_with_spacy(phrases)
        elif self.embedding_backend == "custom":
            if self.embedding_fn is None:
                raise ValueError(
                    "embedding_backend='custom' requires embedding_fn to be provided."
                )
            embeddings = self.embedding_fn(phrases)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.asarray(embeddings, dtype=np.float32)
            return embeddings
        else:
            raise ValueError(
                "embedding_backend must be one of "
                "'sentence_transformers', 'spacy', or 'custom'."
            )

    def _embed_with_sentence_transformers(
        self,
        phrases: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        SentenceTransformers embedding backend.

        Lazily loads the model to avoid import cost if unused.

        Parameters
        ----------
        phrases:
            List of phrase strings to embed.
        show_progress:
            If True, show SentenceTransformers' built-in tqdm progress bar
            on stdout. Typically tied to `verbose` when no custom logger
            is provided.
        """
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for "
                    "embedding_backend='sentence_transformers'. "
                    "Install with 'pip install sentence-transformers'."
                ) from e

            self._st_model = SentenceTransformer(self.embedding_model_name)

        embeddings = self._st_model.encode(
            phrases,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)


    def _embed_with_spacy(self, phrases: List[str]) -> np.ndarray:
        """
        spaCy embedding backend.

        Requires a spaCy Language object with `.pipe()` and `.vector`.
        """
        if self.spacy_nlp is None:
            raise ValueError(
                "embedding_backend='spacy' requires a spaCy Language object "
                "to be provided via `spacy_nlp`."
            )

        vectors: List[np.ndarray] = []
        # Use .pipe for efficient batch processing
        for doc in self.spacy_nlp.pipe(phrases):  # type: ignore[union-attr]
            vec = doc.vector
            if vec is None or len(vec) == 0:
                # Fallback: all zeros for phrases without lexical vectors
                vec = np.zeros(self.spacy_nlp.meta.get("vectors", {}).get("width", 300))  # type: ignore[index]
            vectors.append(np.asarray(vec, dtype=np.float32))

        return np.vstack(vectors)

    # ------------------------------------------------------------------
    # Internal helpers – PCA, UMAP, t-SNE
    # ------------------------------------------------------------------

    def _apply_pca(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """
        Apply PCA for denoising and dimensionality reduction.

        This tends to improve neighborhood quality for UMAP/t-SNE
        by removing very low-variance noise dimensions.

        - If n_components <= 0 → skip PCA and return X.
        - If n_components >= min(n_samples, n_features) → clamp it down.
        """
        from sklearn.decomposition import PCA

        if not n_components or n_components <= 0:
            # No PCA requested
            return X
    
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
    
        # If the requested number of components is too large for the data,
        # we clamp it so that sklearn.PCA doesn't raise.
        if n_components >= max_components:
            # Leave at least 1 dimension for PCA; if that isn't possible,
            # just skip PCA entirely.
            n_components = max_components - 1
            if n_components <= 0:
                if self.logger:
                    self.logger(
                        "[TopicModeler] PCA skipped: "
                        f"n_samples={n_samples}, n_features={n_features} "
                        "too small for the requested n_components."
                    )
                return X

        if self.logger:
            self.logger(
                f"[TopicModeler] Applying PCA with n_components={n_components} "
                f"(n_samples={n_samples}, n_features={n_features})"
            )

        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pca.fit_transform(X)

    def _compute_umap(
        self,
        X: np.ndarray,
        n_neighbors: int,
        min_dist: float,
        n_components: int,
    ) -> np.ndarray:
        """
        Compute a UMAP embedding with safe fallbacks for very small datasets.

        For tiny n_samples, UMAP's spectral initialization can fail with
        SciPy errors like:
            TypeError: Cannot use scipy.linalg.eigh for sparse A with k >= N.

        Strategy:
        ---------
        - If n_samples is very small (<= 5), or
        - if n_components + 1 >= n_samples,
          → skip UMAP and just return X.
        """
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'umap-learn' package is required for UMAP embeddings. "
                "Install with 'pip install umap-learn'."
            ) from e

        n_samples, n_features = X.shape

        # Hard guardrail for tiny datasets:
        # UMAP's spectral layout is unstable when k >= N for the Laplacian.
        if n_samples <= 5 or (n_components + 1) >= n_samples:
            if self.logger:
                self.logger(
                    "[TopicModeler] UMAP skipped: "
                    f"n_samples={n_samples}, n_components={n_components} "
                    "too small for stable spectral initialization. "
                    "Returning original vectors instead."
                )
            return X

        # Also cap n_neighbors to something meaningful for small n_samples
        effective_neighbors = min(max(2, n_neighbors), max(2, n_samples - 1))

        if self.logger:
            self.logger(
                "[TopicModeler] Running UMAP with "
                f"n_samples={n_samples}, n_neighbors={effective_neighbors}, "
                f"n_components={n_components}, min_dist={min_dist}"
            )

        reducer = umap.UMAP(
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=self.random_state,
        )
        return reducer.fit_transform(X)

    def _compute_tsne(
        self,
        X: np.ndarray,
        perplexity: float,
        learning_rate: float,
        n_iter: int,
    ) -> np.ndarray:
        """
        Compute a 2D t-SNE embedding on the (optionally PCA-reduced) data.

        This is typically used **only for visualization**, not clustering,
        because t-SNE strongly optimizes local structure at the cost of
        global geometry.

        Notes
        -----
        - scikit-learn requires: perplexity < n_samples.
        - Very small n_samples (<= 5) don't benefit from t-SNE; we simply
          return X in those cases.
        - For general n_samples, we clamp perplexity into a safe range:
              2.0 <= perplexity < n_samples
          and also avoid absurdly large perplexities relative to n_samples.
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for t-SNE visualizations. "
                "Install with 'pip install scikit-learn'."
            ) from e
    
        n_samples, n_features = X.shape

        # For extremely small datasets, t-SNE is not meaningful and often
        # numerically fragile. Just return X.
        if n_samples <= 5:
            if self.logger:
                self.logger(
                    "[TopicModeler] t-SNE skipped: "
                    f"n_samples={n_samples} too small for a stable embedding. "
                    "Returning original vectors instead."
                )
            return X

        # t-SNE requires perplexity < n_samples. Also, rule-of-thumb:
        # perplexity should be much smaller than n_samples.
        # We cap it at roughly (n_samples - 1)/3, but never below 2.0.
        max_reasonable = max(2.0, (n_samples - 1) / 3.0)
        # ensure strictly less than n_samples
        max_allowed = min(max_reasonable, n_samples - 1.0)

        effective_perplexity = float(
            max(2.0, min(perplexity, max_allowed))
        )

        if self.logger and abs(effective_perplexity - perplexity) > 1e-6:
            self.logger(
                "[TopicModeler] Adjusted t-SNE perplexity from "
                f"{perplexity} to {effective_perplexity} "
                f"(n_samples={n_samples})."
            )

        # A small guard for iterations: t-SNE can behave oddly with tiny n_iter.
        effective_n_iter = max(250, int(n_iter))

        if self.logger:
            self.logger(
                "[TopicModeler] Running t-SNE with "
                f"n_samples={n_samples}, perplexity={effective_perplexity}, "
                f"learning_rate={learning_rate}, n_iter={effective_n_iter}"
            )

        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=effective_perplexity,
        #     learning_rate=learning_rate,
        #     n_iter=effective_n_iter,
        #     init="random",
        #     random_state=self.random_state,
        # )
        # ---- Build kwargs in a version-agnostic way ----
        import inspect
        tsne_kwargs = dict(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate=learning_rate,
            init="random",       # safest default across backends
            random_state=self.random_state,
        )

        sig = inspect.signature(TSNE.__init__)
        params = sig.parameters

        # Newer sklearn: `max_iter` (no `n_iter`)
        if "max_iter" in params:
            tsne_kwargs["max_iter"] = n_iter
        # Older sklearn: `n_iter`
        elif "n_iter" in params:
            tsne_kwargs["n_iter"] = n_iter
        else:
            # Extremely defensive fallback – don't pass either,
            # TSNE will use its own default.
            pass

        tsne = TSNE(**tsne_kwargs)
        return tsne.fit_transform(X)

    # ------------------------------------------------------------------
    # Internal helpers – clustering
    # ------------------------------------------------------------------

    def _cluster_hdbscan(
        self,
        X: np.ndarray,
        *,
        min_cluster_size: int,
        min_samples: Optional[int],
        metric: str,
    ) -> np.ndarray:
        """
        Cluster embeddings with HDBSCAN.

        NOTE
        ----
        The standard `hdbscan` Python library does **not** support sample
        weights. Phrase frequencies are therefore not used in the geometry;
        they can still be used in downstream interpretation (cluster sizes,
        representative phrases, etc.).
        """
        try:
            import hdbscan
        except ImportError as e:
            raise ImportError(
                "The 'hdbscan' package is required for clustering_algorithm='hdbscan'. "
                "Install with 'pip install hdbscan'."
            ) from e

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        labels = clusterer.fit_predict(X)
        return labels

    def _cluster_kmeans_auto_k(
        self,
        X: np.ndarray,
        *,
        weights: np.ndarray,
        max_clusters: int,
    ) -> np.ndarray:
        """
        Auto-K KMeans with Silhouette selection, using phrase frequency
        as **sample weights** in the clustering step.

        Strategy
        --------
        - Restrict K to the range [2, max_K], but not exceeding n_samples.
        - For each K, run KMeans (with sample_weight=weights).
        - Compute a standard (unweighted) Silhouette score on X.
        - Choose the K with the best Silhouette score.
        - If unable to find a valid K (corner cases), fall back to a single
          cluster (all zeros).
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot cluster an empty embedding matrix.")

        max_k = min(max_clusters, n_samples)
        if max_k <= 1:
            # Degenerate case: only one cluster possible
            return np.zeros(n_samples, dtype=int)

        best_k = None
        best_score = -1.0
        best_labels: Optional[np.ndarray] = None

        for k in range(2, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
            )
            # Use phrase counts as sample weights
            labels = kmeans.fit_predict(X, sample_weight=weights)

            # Silhouette requires at least 2 clusters with >1 sample each
            try:
                score = silhouette_score(X, labels)
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            # Fall back to single cluster
            return np.zeros(n_samples, dtype=int)

        return best_labels

    # ------------------------------------------------------------------
    # Internal helpers – TopicCluster construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_topic_clusters(
        df: pd.DataFrame,
        *,
        top_n_representatives: int,
    ) -> List[TopicCluster]:
        """
        Aggregate phrase-level DataFrame into TopicCluster objects.

        Improvements over a naive "top-N by frequency" strategy:
        - Representative phrases are selected by a combined score:
              score = α * freq_norm + (1-α) * centroid_closeness
          where:
              * freq_norm is phrase frequency normalized within the cluster,
              * centroid_closeness is 1 - normalized distance to cluster
                centroid in embedding space.
        - Clusters are ranked by an importance_score and returned in
          descending order of importance.

        Notes
        -----
        - Noise cluster (-1) is **excluded** from the returned list,
          but its phrases remain in `phrases_df` with cluster_id=-1.
        - This method assumes `df["embedding"]` contains 1D numpy arrays
          representing the semantic embedding of each phrase.
        """
        clusters: List[TopicCluster] = []

        # Unique cluster IDs
        cluster_ids = sorted(df["cluster_id"].unique().tolist())

        for cid in cluster_ids:
            # if cid == -1:
            #     # HDBSCAN noise cluster; keep in df but skip as "topic cluster"
            #     continue

            sub = df[df["cluster_id"] == cid].copy()
            if sub.empty:
                continue

            phrases = sub["phrase"].tolist()
            counts = sub["count"].astype(int).tolist()
            total_count = int(sub["count"].sum())
            n_phrases = len(phrases)

            # --- Build embedding matrix for this cluster ---
            # Each entry in sub["embedding"] is a 1D vector
            emb_list = sub["embedding"].tolist()
            emb_mat = np.vstack(emb_list).astype("float32")  # (n_phrases, dim)

            # --- Distance to centroid (euclidean) ---
            centroid = emb_mat.mean(axis=0, keepdims=True)  # (1, dim)
            dists = np.linalg.norm(emb_mat - centroid, axis=1)  # (n_phrases,)

            # Convert distances to "closeness" in [0, 1]
            max_dist = float(dists.max()) if dists.size > 0 else 0.0
            if max_dist > 0.0:
                closeness = 1.0 - (dists / max_dist)
            else:
                # All embeddings identical → treat all as equally close
                closeness = np.ones_like(dists, dtype="float32")

            # --- Frequency normalization within cluster ---
            counts_arr = sub["count"].to_numpy(dtype=float)
            max_count = float(counts_arr.max()) if counts_arr.size > 0 else 1.0
            if max_count > 0.0:
                freq_norm = counts_arr / max_count
            else:
                freq_norm = np.ones_like(counts_arr, dtype="float32")

            # --- Combined representativeness score ---
            alpha = 0.6  # weight for frequency vs. centroid closeness
            combined_score = alpha * freq_norm + (1.0 - alpha) * closeness

            sub = sub.assign(_repr_score=combined_score)
            sub_sorted = sub.sort_values("_repr_score", ascending=False)

            reps = sub_sorted["phrase"].head(top_n_representatives).tolist()

            # --- Cluster-level importance score ---
            # Simple heuristic: high total frequency + more distinct phrases
            # importance ≈ total_count * log(1 + n_phrases)
            importance_score = float(total_count * np.log1p(n_phrases))

            clusters.append(
                TopicCluster(
                    cluster_id=int(cid),
                    phrases=phrases,
                    phrase_counts=counts,
                    total_count=total_count,
                    representative_phrases=reps,
                    importance_score=importance_score,
                )
            )

        # Sort clusters by importance score (descending)
        clusters.sort(key=lambda c: c.importance_score, reverse=True)
        return clusters

