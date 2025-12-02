"""
phrasetopicminer.smoke_test

Minimal end-to-end smoke test for PhraseTopicMiner.

Usage (from your project root or any directory where the env is active):

    python -m phrasetopicminer.smoke_test

What it does:
- Creates a tiny in-memory corpus (3 short English sentences).
- Runs PhraseMiner to extract phrases.
- Runs TopicModeler.fit_core to embed + cluster phrases.
- Builds a reading-time topic timeline.
- Prints a short summary to stdout.

If a spaCy model such as "en_core_web_sm" is not installed, you may see
an error from PhraseMiner. In that case, install it with:

    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

from typing import Any, Dict

from .phrase_miner import PhraseMiner
from .topic_modeler import TopicModeler
from .topic_timeline import TopicTimelineBuilder


def run_smoke_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Run a small end-to-end test of the main pipeline.

    Returns
    -------
    result : dict
        A dictionary containing:
        - "docs"
        - "phrase_records"
        - "sentences_by_doc"
        - "core"     (TopicCoreResult)
        - "timeline" (TopicTimelineResult)
    """
    docs = [
        # Doc 1 – phrase-centric topic modeling basics
        """
        Phrase-based topic modeling treats noun phrases and verb phrases as the
        main carriers of meaning in a document collection. Instead of working at
        the level of single tokens, we mine phrases such as "neural topic model",
        "customer feedback", or "research pipeline". This phrase-centric view makes
        clusters easier to interpret, because each topic is anchored in human
        readable expressions rather than abstract word distributions.
        """,
    
        # Doc 2 – applications to meeting notes
        """
        In recurring team meetings, the same themes appear again and again:
        roadmap decisions, technical debt, customer pain points, and hiring plans.
        PhraseTopicMiner can mine key phrases from the transcripts, cluster them
        into topics, and then project those phrase clusters into a two-dimensional
        map. Each cluster becomes a labeled island of discussion, helping product
        and engineering leaders see which themes dominate the conversation over time.
        """,
    
        # Doc 3 – research literature exploration
        """
        When exploring a new research field, we often read dozens of papers without
        a clear overview of the main conceptual structure. By extracting phrases
        such as "contrastive learning objective", "causal inference", or "human
        evaluation protocol" from abstracts and introductions, PhraseTopicMiner
        builds a geometric map of ideas. The resulting clusters highlight families
        of methods, evaluation strategies, and application domains in a way that is
        visually intuitive and analytically useful.
        """,
    
        # Doc 4 – product discovery & user interviews
        """
        User interview transcripts are full of recurring expressions: people
        describe friction, workarounds, and desired outcomes in surprisingly
        consistent language. A phrase-centric topic model can surface patterns like
        "manual spreadsheet export", "notification overload", or "difficult onboarding
        experience". Clustering those phrases reveals coherent themes in the voice
        of the user, which can then be prioritized and tracked across releases.
        """,
    
        # Doc 5 – educational content analysis
        """
        Educators working with large collections of lecture notes, assignments, and
        discussion forum posts often struggle to see which concepts confuse students
        the most. Mining phrases such as "backpropagation intuition", "regularization
        trade-off", or "evaluation metric" and grouping them into topics provides a
        living map of conceptual difficulty. This can guide revision of teaching
        materials and the design of targeted practice exercises.
        """,
    
        # Doc 6 – monitoring conceptual drift over time
        """
        Over time, the language of a project, product, or research field evolves.
        New phrases appear while others gradually disappear. PhraseTopicMiner can
        track phrase clusters as timelines, showing when ideas emerge, stabilize,
        or fade out. This temporal view helps teams notice conceptual drift early
        and decide whether it reflects healthy innovation or a loss of focus.
        """,
    
        # Doc 7 – History of Ideas / Intellectual History
        """
        In the history of ideas and intellectual history, we often track how key
        concepts are articulated, contested, and transformed across different
        genres of writing: pamphlets, newspaper articles, treatises, and speeches.
        Instead of counting single words like "freedom" or "despotism", a
        phrase-centric topic model focuses on richer expressions such as
        "freedom under law", "arbitrary royal power", "constitutional limits",
        "rights of the people", or "religious authority".
    
        By mining and clustering these multi-word phrases, PhraseTopicMiner can
        surface distinct conceptual constellations that correspond to competing
        vocabularies of freedom, authority, and community. Each cluster becomes a
        map of how authors link key ideas together in practice, not just in theory.
        When we add a temporal dimension, these phrase clusters can be followed
        across years or decades, revealing when certain constellations emerge,
        overlap, or decline. This complements close reading: the historian still
        interprets texts line by line, but now against a geometric overview of
        conceptual change in the archive.
        """,
    ]

    if verbose:
        print("[smoke_test] Starting PhraseTopicMiner smoke test...")
        print(f"[smoke_test] Using {len(docs)} small demo documents.")

    # -------------------------------------------------------
    # 1) Phrase mining
    # -------------------------------------------------------
    if verbose:
        print("[smoke_test] Creating PhraseMiner and mining phrases...")

    miner = PhraseMiner()
    total_np, total_vp, phrase_records, sentences_by_doc = miner.mine_phrases_with_types(docs)

    if verbose:
        print(
            f"[smoke_test] Mined {len(phrase_records)} phrase records "
            f"across {len(sentences_by_doc)} document(s)."
        )
        print(f"[smoke_test] Total NP: {total_np}, total VP: {total_vp}")

    # -------------------------------------------------------
    # 2) Topic modeling (core clustering)
    # -------------------------------------------------------
    if verbose:
        print("[smoke_test] Creating TopicModeler and fitting core model...")

    modeler = TopicModeler(
        embedding_backend="sentence_transformers",   # or "spacy"
        embedding_model="all-MiniLM-L6-v2",
    )

    core = modeler.fit_core(
        phrase_records=phrase_records,
        sentences_by_doc=sentences_by_doc,
        # --- phrase filtering options (low thresholds for tiny demo) ---
        include_kinds={"NP", "VP"},
        include_patterns={
            "BaseNP", "NP+PP", "NP+multiPP",
            "VerbObj", "VerbPP", "SubjVerb",
        },
        min_freq_unigram=2,            # threshold for 1-word phrases
        min_freq_bigram=1,             # threshold for 2-word phrases
        min_freq_trigram_plus=1,       # threshold for >=3-word phrases
    
        # --- geometric pipeline options ---
        pca_n_components=10,         # internal fallback already caps this if data is tiny
        cluster_geometry="umap_nd",  # "umap_nd" or "umap_2d" - cluster in higher dim, then project to 2D
        umap_n_neighbors=10,         # higher → a bit more global structure 
        umap_min_dist=0.05,          # smaller  → tighter clusters, more separation between them
        umap_cluster_n_components=10,# target dim for clustering (if using umap_nd)
    
    
        # --- clustering options (fewer / cleaner topics) ---
        clustering_algorithm="hdbscan",   # "hdbscan" or "kmeans"
        hdbscan_min_cluster_size=10,      # larger min cluster size → fewer topics
        hdbscan_min_samples=3,            # a little robustness without over-fragmenting
        hdbscan_metric="euclidean",
        kmeans_max_clusters=15,          # used only if clustering_algorithm="kmeans"
    
        # --- visualization geometry ---
        viz_reducer="tsne_2d",           # "same", "umap_2d", or "tsne_2d"
        tsne_perplexity=30.0,
        tsne_learning_rate=200.0,
        tsne_n_iter=1000,
    
        # --- cluster representatives ---
        top_n_representatives=20,
    
        verbose=True,
    )

    if verbose:
        print(
            f"[smoke_test] TopicCoreResult: {len(core.phrases_df)} phrases, "
            f"{len(core.clusters)} clusters, "
            f"embedding dimention = {len(core.phrases_df['embedding'][0])}"
        )

        print(f" # --- config ---\n{core.config}")

    # -------------------------------------------------------
    # 3) Topic timeline
    # -------------------------------------------------------
    if verbose:
        print("[smoke_test] Building topic timeline...")

    timeline_builder = TopicTimelineBuilder(
        timeline_mode="reading_time",
        speech_rate_wpm=200,
        log_fn=print if verbose else (lambda *_args, **_kwargs: None),
    )

    timeline = timeline_builder.build(
        core_result=core,
        sentences_by_doc=sentences_by_doc
    )

    if verbose:
        print(
            f"[smoke_test] Timeline built for {len(timeline.phrase_sentence_df)} phrases "
            f"in {len(timeline.cluster_sentence_df)} topic(s)."
        )
        print("[smoke_test] Smoke test completed successfully ✅")

    return {
        "docs": docs,
        "phrase_records": phrase_records,
        "sentences_by_doc": sentences_by_doc,
        "core": core,
        "timeline": timeline,
    }
def main() -> None:
    """
    CLI entrypoint for: python -m phrasetopicminer.smoke_test
    """
    try:
        run_smoke_test(verbose=True)
    except OSError as e:
        # Common case: spaCy model not installed
        msg = str(e)
        print("\n[smoke_test] ERROR during PhraseTopicMiner smoke test.")
        print(f"[smoke_test] Underlying error: {msg}\n")

        if "Can't find model" in msg or "can't find model" in msg:
            print(
                "[smoke_test] It looks like a spaCy language model is missing.\n"
                "Try installing a small English model with:\n\n"
                "    python -m spacy download en_core_web_sm\n"
            )
        # Re-raise so CI / scripts still see a failure
        raise


if __name__ == "__main__":
    main()
