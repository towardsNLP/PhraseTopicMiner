# visualization_datamap.py

"""
visualization_datamap.py

DataMapPlot-based visualizations for PhraseTopicMiner.

These helpers take a TopicCoreResult (from TopicModeler.fit_core) and
produce either:

- a static Matplotlib figure using `datamapplot.create_plot`, or
- an interactive HTML plot using `datamapplot.create_interactive_plot`.

The only hard dependency is the external `datamapplot` package, which is
imported lazily so that PhraseTopicMiner can be used without it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Sequence

import html
import re

import numpy as np
import pandas as pd


import datamapplot

from .topic_modeler import TopicCoreResult
from .phrase_miner import PhraseRecord



def build_phrase_sentence_examples_from_occurrences(
    core_result: TopicCoreResult,
    sentences_by_doc: Sequence[Sequence[str]],
    *,
    max_sentences_per_phrase: int = 1,
    min_sentence_chars: int = 0,
) -> Dict[str, List[str]]:
    """
    Build a mapping phrase -> example sentences using PhraseRecord
    metadata and a list of sentences per document.

    Parameters
    ----------
    core_result
        Result from TopicModeler.fit_core. We use its
        `phrase_occurrences` mapping.
    sentences_by_doc
        Nested list of raw sentence texts as returned by PhraseMiner,
        i.e. sentences_by_doc[doc_index][sent_index] -> sentence string.
    max_sentences_per_phrase
        Maximum number of distinct sentences to keep per phrase.
        (For hover text, 1–3 is usually enough.)
    min_sentence_chars
        If > 0, skip sentences whose length is below this many characters
        to avoid tiny fragments in the hover text.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from phrase (canonical form) to a list of example
        sentences in which the phrase appears.
    """
    phrase_examples: Dict[str, List[str]] = {}

    for phrase, occs in core_result.phrase_occurrences.items():
        # We may already have enough examples for this phrase
        examples: List[str] = []
        seen: set = set()  # avoid duplicate sentences

        for rec in occs:
            # Defensive: ensure indices are in range
            if (
                rec.doc_index < 0
                or rec.doc_index >= len(sentences_by_doc)
            ):
                continue

            doc_sents = sentences_by_doc[rec.doc_index]
            if (
                rec.sent_index < 0
                or rec.sent_index >= len(doc_sents)
            ):
                continue

            sent = doc_sents[rec.sent_index]
            if min_sentence_chars and len(sent) < min_sentence_chars:
                continue

            if sent in seen:
                continue

            seen.add(sent)
            examples.append(sent)

            if len(examples) >= max_sentences_per_phrase:
                break

        if examples:
            phrase_examples[phrase] = examples

    return phrase_examples


# ---------------------------------------------------------------------
# Some small helpers
# ---------------------------------------------------------------------

def _build_phrase_sentence_map(
    phrase_occurrences: Dict[str, List[PhraseRecord]],
    sentences_by_doc: Sequence[Sequence[str]],
) -> Dict[str, str]:
    """
    Map each phrase -> a single representative sentence string.

    Strategy: for each phrase we take the first occurrence whose
    (doc_index, sent_index) can be resolved into a sentence.
    """
    mapping: Dict[str, str] = {}

    for phrase, occ_list in phrase_occurrences.items():
        for rec in occ_list:
            try:
                sent = sentences_by_doc[rec.doc_index][rec.sent_index]
            except (IndexError, TypeError):
                continue
            if isinstance(sent, str) and sent.strip():
                mapping[phrase] = sent.strip()
                break

    return mapping


def _highlight_phrase_in_sentence(phrase: str, sentence: str) -> str:
    """
    Return an HTML-safe sentence string with the phrase highlighted.

    - Escapes the sentence and phrase.
    - Wraps the first occurrence of the phrase in <mark> … </mark>.
    """
    escaped_sentence = html.escape(sentence)
    escaped_phrase = html.escape(phrase)

    # Case-insensitive match in the escaped sentence
    pattern = re.compile(re.escape(escaped_phrase), flags=re.IGNORECASE)
    highlighted = pattern.sub(r"<mark>\g<0></mark>", escaped_sentence, count=1)

    return highlighted



def _get_datamapplot():
    """
    Lazy-import helper so `datamapplot` is only required if the user calls
    these visualization helpers.
    """
    try:
        import datamapplot  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "The 'datamapplot' package is required for DataMapPlot visualizations.\n"
            "Install it with:  pip install datamapplot\n"
            "or as an extras dependency for PhraseTopicMiner."
        ) from e
    return datamapplot


# ---------------------------------------------------------------------
# 1) Static topic map (PNG / notebook figure)
# ---------------------------------------------------------------------


def make_datamapplot_static(
    core_result: TopicCoreResult,
    cluster_name_map: Optional[Dict[int, str]] = None,
    *,
    noise_label: str = "Unlabelled",
    title: str = "PhraseTopicMiner – Topic Map",
    sub_title: str = "Phrases clustered in 2D space",
    save_path: Optional[str] = None,
    **datamap_kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Create a **static** DataMapPlot topic map from a TopicCoreResult.

    Parameters
    ----------
    core_result:
        Output of `TopicModeler.fit_core`.
    cluster_name_map:
        Optional mapping `{cluster_id: human_title}`.
        Typically produced by an LLM-based labeling stage.
        If omitted, default labels like "Cluster 0", "Cluster 1" are used.
    noise_label:
        Label used for HDBSCAN noise points (cluster_id == -1).
    title, sub_title:
        Figure title and subtitle passed to DataMapPlot.
    save_path:
        Optional path for saving the figure. If provided, `fig.savefig`
        is called with this path (PNG / PDF / etc., depending on the
        extension).
    datamap_kwargs:
        Extra keyword arguments forwarded to `datamapplot.create_plot`
        (e.g. `label_font_size=11`, `use_medoids=True`, etc.).

    Returns
    -------
    (fig, ax):
        Matplotlib figure and axes objects returned by DataMapPlot.
    """
    datamapplot = _get_datamapplot()

    df: pd.DataFrame = core_result.phrases_df

    # 2D coordinates from TopicModeler (UMAP / t-SNE)
    data_coords = df[["x", "y"]].to_numpy(dtype="float32")  # shape (n_phrases, 2)

    # Raw numeric cluster IDs
    cluster_ids = df["cluster_id"].to_numpy()

    # Normalize the cluster_name_map to integer keys, if provided
    cluster_name_map = cluster_name_map or {}
    normalized_name_map: Dict[int, str] = {}
    for k, v in cluster_name_map.items():
        try:
            normalized_name_map[int(k)] = v
        except (TypeError, ValueError):
            # If user passed string keys (e.g. "0"), try to coerce
            continue

    # Build string labels for DataMapPlot: one label per point
    labels = []
    for cid in cluster_ids:
        if cid == -1:
            labels.append(noise_label)
        else:
            labels.append(normalized_name_map.get(int(cid), f"Cluster {cid}"))
    labels = np.asarray(labels, dtype=object)

    # Create the static DataMapPlot
    fig, ax = datamapplot.create_plot(
        data_coords,
        labels,
        noise_label=noise_label,
        title=title,
        sub_title=sub_title,
        **datamap_kwargs,
    )

    # Optional save
    if save_path is not None:
        # Matplotlib-style save
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


# ---------------------------------------------------------------------
# 2) Interactive topic map (HTML)
# ---------------------------------------------------------------------


def make_datamapplot_interactive(
    core_result: TopicCoreResult,
    sentences_by_doc: Optional[Sequence[Sequence[str]]] = None,
    cluster_name_map: Optional[Dict[int, str]] = None,
    noise_label: str = "Unlabelled",
    title: str = "PhraseTopicMiner – Interactive Topic Map",
    sub_title: Optional[str] = "Phrases clustered in 2D space",
    save_html_path: Optional[str] = None,
    hover_text_fn: Optional[Callable[[pd.Series], str]] = None,
    **datamap_kwargs: Any,
):
    """
    Interactive DataMapPlot map for PhraseTopicMiner clusters.

    Parameters
    ----------
    core_result:
        Output of TopicModeler.fit_core.
    sentences_by_doc:
        Optional nested list of original sentences as returned by PhraseMiner:
        ``List[List[str]]`` with shape [num_docs][num_sentences_per_doc].
        If provided, tooltips will show a representative sentence with the
        phrase highlighted.
    cluster_name_map:
        Optional mapping {cluster_id: human_title}. If not provided, simple
        "Cluster k" labels will be used.
    noise_label:
        String label used for noise / unclustered points (cluster_id == -1).
    title, sub_title:
        Title texts for the interactive map.
    save_html_path:
        If not None, save the interactive HTML file to this path.
    hover_text_fn:
        Optional function mapping a row of phrases_df (pd.Series) to a
        plain-text hover string. If None, the phrase text is used.
    **datamap_kwargs:
        Extra keyword arguments forwarded to DataMapPlot. For convenience, a
        ``point_size`` kwarg will be translated into both
        ``point_radius_min_pixels`` and ``point_radius_max_pixels``.
    """
    df = core_result.phrases_df.copy()

    # 1) 2D coordinates from TopicModeler (UMAP/t-SNE)
    coords = df[["x", "y"]].to_numpy(dtype="float32")

    # 2) Labels for DataMapPlot (cluster ids -> "Cluster k" / noise_label)
    cluster_ids = df["cluster_id"].to_numpy()
    unique_cids = np.unique(cluster_ids)

    if cluster_name_map is None:
        cluster_name_map = {
            int(cid): f"Cluster {int(cid)}"
            for cid in unique_cids
            if cid != -1
        }

    labels_str = np.where(
        cluster_ids == -1,
        noise_label,
        np.array(
            [cluster_name_map.get(int(cid), f"Cluster {int(cid)}") for cid in cluster_ids],
            dtype=object,
        ),
    )

    # 3) Base hover_text (plain text) – used also for search
    if hover_text_fn is None:
        hover_text = df["phrase"].tolist()
    else:
        hover_text = [hover_text_fn(row) for _, row in df.iterrows()]

    # 4) Optional sentence / highlight context
    extra_df = None
    hover_template = None

    if sentences_by_doc is not None:
        # Build phrase -> representative sentence map
        phrase_to_sentence = _build_phrase_sentence_map(
            core_result.phrase_occurrences,
            sentences_by_doc,
        )

        sentences = [phrase_to_sentence.get(p, "") for p in df["phrase"]]
        sentences_highlighted = [
            _highlight_phrase_in_sentence(p, s) if s else ""
            for p, s in zip(df["phrase"], sentences)
        ]

        extra_df = pd.DataFrame(
            {
                "phrase": df["phrase"].tolist(),
                "count": df["count"].tolist(),
                "cluster_id": df["cluster_id"].tolist(),
                "sentence": sentences,
                "sentence_highlighted": sentences_highlighted,
            }
        )

        # HTML template for the tooltip – DataMapPlot will not escape this.
        hover_template = """
        <div style="max-width: 420px;">
          <div style="font-weight:600; margin-bottom:4px;">
            {phrase}
          </div>
          <div style="font-size:13px; line-height:1.4; margin-bottom:4px;">
            {sentence_highlighted}
          </div>
          <div style="font-size:11px; color:#555;">
            count={count}, cluster={cluster_id}
          </div>
        </div>
        """

    # 5) Convenience: map point_size -> point_radius_*_pixels
    point_size = datamap_kwargs.pop("point_size", None)
    if point_size is not None:
        ps = float(point_size)
        datamap_kwargs.setdefault("point_radius_min_pixels", ps)
        datamap_kwargs.setdefault("point_radius_max_pixels", ps)

    # 6) Create interactive DataMapPlot
    fig = datamapplot.create_interactive_plot(
        coords,
        labels_str,
        hover_text=hover_text,
        noise_label=noise_label,
        title=title,
        sub_title=sub_title,
        extra_point_data=extra_df,
        hover_text_html_template=hover_template,
        **datamap_kwargs,
    )

    if save_html_path is not None:
        fig.save(save_html_path)

    return fig




