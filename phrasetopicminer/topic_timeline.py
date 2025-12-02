# topic_timeline.py

"""
topic_timeline.py

Sentence / timeline reconstruction for PhraseTopicMiner.

Given:
- TopicCoreResult from TopicModeler (phrases_df, clusters, phrase_occurrences)
- sentences_by_doc from PhraseMiner (nested list: docs → sentences)

this module builds:

- sentence_df:
    One row per sentence with:
        doc_index, sent_index, sentence_text,
        word_count, timeline_idx, start_ms, end_ms, start_min, duration_ms.

- phrase_sentence_df:
    One row per (phrase occurrence, sentence) with:
        phrase, cluster_id, global_count, occurrence_index,
        doc_index, sent_index, sentence_text, start_ms, start_min, duration_ms,
        kind, pattern.

- cluster_sentence_df:
    One row per cluster with:
        cluster_id,
        sentence_indices: list[(doc_index, sent_index)],
        sentence_start_ms_list,
        sentence_start_min_list,
        sentence_text_list,
        num_sentences.

This is API #2: temporal / contextual reconstruction.
Visualization and LLM-based labeling can build on these DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .topic_modeler import TopicCoreResult
from .phrase_miner import PhraseRecord


# ---------------------------------------------------------------------
# Dataclass for timeline result
# ---------------------------------------------------------------------


@dataclass
class TopicTimelineResult:
    """
    Container for sentence / timeline reconstruction.

    Attributes
    ----------
    sentences_df:
        One row per sentence across all documents.

        Columns:
            - doc_index          : index of the source document
            - sent_index         : sentence index within that document
            - sentence_text      : raw sentence string (as returned by PhraseMiner)
            - word_count         : number of whitespace-delimited 
            - timeline_idx       : int  (0..N-1 across all docs)
            - start_ms           : global start time in milliseconds
            - end_ms             : float
            - duration_ms        : approximate duration in milliseconds
            - start_min          : start time in minutes (start_ms / 60000.0)

    phrase_sentence_df:
        One row per phrase–sentence pair **that survived clustering**.

        Columns:
            - phrase             : canonical phrase string
            - cluster_id         : topic cluster label (can be -1 for noise)
            - global_count       : int  (total count of phrase in phrases_df)
            - occurrence_index   : int  (within this phrase)
            - doc_index          : document index
            - sent_index         : sentence index within document
            - sentence_text      : sentence string
            - sentence_start_ms  : sentence start time (ms)
            - sentence_duration_ms : sentence duration (ms)
            - sentence_start_min : sentence start time (minutes)
            - timeline_idx       : int
            - kind               : str  ("NP" or "VP")
            - pattern            : str  (e.g. "BaseNP", "VerbObj", ...)


    cluster_sentence_df:
        One row per cluster, aggregating sentences that contain any phrase
        from that cluster.

        Columns:
            - cluster_id                : topic cluster label
            - sentence_indices        : List[Tuple[int, int]]
            - sentence_start_ms_list  : List[float]
            - sentence_start_min_list : List[float]
            - sentence_text_list      : List[str]
            - num_sentences           : int
    config:
        Configuration dictionary capturing timeline parameters used for
        this reconstruction (timeline_mode, speech_rate_wpm, etc.).
    """

    sentence_df: pd.DataFrame
    phrase_sentence_df: pd.DataFrame
    cluster_sentence_df: pd.DataFrame
    config: Dict[str, Any]


# ---------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------


class TopicTimelineBuilder:
    """
    Build sentence-level timelines and phrase→sentence mappings
    from TopicCoreResult and sentences_by_doc.

    This class is intentionally separate from TopicModeler so that:
    - TopicModeler stays purely geometric/statistical.
    - You can recompute timelines with different parameters (e.g. reading
      speed) without re-running clustering.
    """

    def __init__(
        self,
        *,
        timeline_mode: Literal["reading_time", "index"] = "reading_time",
        speech_rate_wpm: float = 160.0,
        reset_time_per_document: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        timeline_mode:
            - "reading_time":
                Assign sentence start times based on an estimated reading
                duration derived from word counts and `speech_rate_wpm`.
            - "index":
                Use a simple integer index as the "time" axis
                (sentence order). Duration is set to 1 for all sentences.
        speech_rate_wpm:
            Words-per-minute used in "reading_time" mode.
            Effective milliseconds per word = 60000 / speech_rate_wpm.
        reset_time_per_document:
            If True, each document's timeline starts at 0 independently.
            If False (default), time is cumulative across documents.
        log_fn:
            Optional logging function. If provided, it is called with
            short log messages instead of using print(). This allows you
            to plug Streamlit (`st.write`/`st.markdown`), a logger, etc.
        """
        self.timeline_mode = timeline_mode
        self.speech_rate_wpm = float(speech_rate_wpm)
        self.reset_time_per_document = reset_time_per_document
        self.log_fn = log_fn or (lambda _msg: None)

        if self.timeline_mode not in {"reading_time", "index"}:
            raise ValueError("timeline_mode must be 'reading_time' or 'index'.")

    # ----------------------------- #
    # Public main entry point       #
    # ----------------------------- #

    def build(
        self,
        core_result: TopicCoreResult,
        sentences_by_doc: Sequence[Sequence[str]],
    ) -> TopicTimelineResult:
        """
        Construct sentence_df, phrase_sentence_df, and cluster_sentence_df
        from a TopicCoreResult and nested list of sentences.

        Parameters
        ----------
        core_result:
            Output of TopicModeler.fit_core.
        sentences_by_doc:
            Nested list where `sentences_by_doc[doc_index][sent_index]`
            is the original sentence text corresponding to PhraseRecord
            indices.

        Returns
        -------
        TopicTimelineResult
            Structured sentence and cluster timelines, ready to feed
            into visualization or LLM-labeling layers.
        """
        self._log("▶ Building sentence-level timeline…")
        sentence_df = self._build_sentence_df(sentences_by_doc)

        self._log("▶ Linking phrases to sentences…")
        phrase_sentence_df = self._build_phrase_sentence_df(
            core_result=core_result,
            sentence_df=sentence_df,
        )

        self._log("▶ Aggregating sentences per cluster…")
        cluster_sentence_df = self._build_cluster_sentence_df(
            phrase_sentence_df=phrase_sentence_df
        )

        config = {
            "timeline_mode": self.timeline_mode,
            "speech_rate_wpm": self.speech_rate_wpm,
            "reset_time_per_document": self.reset_time_per_document,
        }

        self._log("✅ TopicTimelineBuilder finished.")
        return TopicTimelineResult(
            sentence_df=sentence_df,
            phrase_sentence_df=phrase_sentence_df,
            cluster_sentence_df=cluster_sentence_df,
            config=config,
        )

    # ----------------------------- #
    # Internal helpers              #
    # ----------------------------- #

    def _build_sentence_df(
        self,
        sentences_by_doc: Sequence[Sequence[str]],
    ) -> pd.DataFrame:
        """
        Flatten nested sentences into a single DataFrame with timestamps.

        We honour the doc_index / sent_index structure used in PhraseRecord,
        so we can join later without ambiguity.
        """
        rows: List[Dict[str, Any]] = []

        # milliseconds per word if using reading_time
        ms_per_word = 60000.0 / self.speech_rate_wpm if self.timeline_mode == "reading_time" else 0.0

        global_t_ms = 0.0  # global timeline start
        global_idx = 0     # optional integer index for ordering

        for doc_index, sent_list in enumerate(sentences_by_doc):
            if self.reset_time_per_document and doc_index > 0:
                # Restart timeline for each document
                global_t_ms = 0.0

            for sent_index, raw_sent in enumerate(sent_list):
                text = (raw_sent or "").strip()
                if not text:
                    # We still keep an empty sentence so indices line up,
                    # but it gets zero duration.
                    word_count = 0
                else:
                    word_count = len(text.split())

                if self.timeline_mode == "index":
                    start_ms = float(global_idx)
                    duration_ms = 1.0
                    end_ms = start_ms + duration_ms
                    global_t_ms = end_ms
                else:
                    # reading_time mode
                    duration_ms = float(word_count) * ms_per_word
                    start_ms = global_t_ms
                    end_ms = start_ms + duration_ms
                    global_t_ms = end_ms

                rows.append(
                    {
                        "doc_index": doc_index,
                        "sent_index": sent_index,
                        "sentence_text": text,
                        "word_count": int(word_count),
                        "timeline_idx": global_idx,
                        "start_ms": float(start_ms),
                        "end_ms": float(end_ms),
                        "duration_ms": float(duration_ms),
                        "start_min": float(start_ms) / 60000.0,
                    }
                )

                global_idx += 1

        sentence_df = pd.DataFrame(rows)
        return sentence_df

    def _build_phrase_sentence_df(
        self,
        *,
        core_result: TopicCoreResult,
        sentence_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join phrases/clusters to their sentence contexts using PhraseRecord
        indices and the sentence_df timeline.
        """
        phrase_df = core_result.phrases_df

        # Lookups: phrase → global_count / cluster_id
        phrase_to_count: Dict[str, int] = {
            p: int(c) for p, c in zip(phrase_df["phrase"], phrase_df["count"])
        }
        phrase_to_cluster: Dict[str, int] = {
            p: int(cid) for p, cid in zip(phrase_df["phrase"], phrase_df["cluster_id"])
        }

        # Fast lookup for sentence metadata by (doc_index, sent_index)
        sentence_index_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for row in sentence_df.itertuples(index=False):
            key = (row.doc_index, row.sent_index)
            sentence_index_map[key] = {
                "sentence_text": row.sentence_text,
                "timeline_idx": int(row.timeline_idx),
                "start_ms": float(row.start_ms),
                "start_min": float(row.start_min),
                "duration_ms": float(row.duration_ms),
            }

        rows: List[Dict[str, Any]] = []

        for phrase, occurrences in core_result.phrase_occurrences.items():
            cluster_id = int(phrase_to_cluster.get(phrase, -1))
            global_count = int(phrase_to_count.get(phrase, len(occurrences)))

            for occ_idx, rec in enumerate(occurrences):
                key = (rec.doc_index, rec.sent_index)
                sent_meta = sentence_index_map.get(key)

                if sent_meta is None:
                    # This should not happen if PhraseMiner and TimelineBuilder
                    # use the same sentence segmentation, but we guard anyway.
                    continue

                rows.append(
                    {
                        "phrase": phrase,
                        "cluster_id": cluster_id,
                        "global_count": global_count,
                        "occurrence_index": occ_idx,
                        "doc_index": rec.doc_index,
                        "sent_index": rec.sent_index,
                        "sentence_text": sent_meta["sentence_text"],
                        "sentence_start_ms": sent_meta["start_ms"],
                        "sentence_start_min": sent_meta["start_min"],
                        "sentence_duration_ms": sent_meta["duration_ms"],
                        "timeline_idx": sent_meta["timeline_idx"],
                        "kind": rec.kind,
                        "pattern": rec.pattern,
                    }
                )

        phrase_sentence_df = pd.DataFrame(rows)
        return phrase_sentence_df

    def _build_cluster_sentence_df(
        self,
        *,
        phrase_sentence_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate phrase_sentence_df at the cluster level.

        For each cluster:
        - Collect unique sentences (doc_index, sent_index).
        - Sort them by start_ms.
        - Save lists of sentence indices, times, and texts.
        """
        rows: List[Dict[str, Any]] = []

        if phrase_sentence_df.empty:
            return pd.DataFrame(
                columns=[
                    "cluster_id",
                    "sentence_indices",
                    "sentence_start_ms_list",
                    "sentence_start_min_list",
                    "sentence_text_list",
                    "num_sentences",
                ]
            )

        # Group by cluster_id, including -1 noise; UI can filter later.
        for cluster_id, sub in phrase_sentence_df.groupby("cluster_id"):
            # Unique sentences per cluster
            unique = (
                sub[["doc_index", "sent_index", "sentence_start_ms", "sentence_start_min", "sentence_text"]]
                .drop_duplicates()
            )

            # Sort by start_ms (timeline order)
            unique = unique.sort_values("sentence_start_ms", ascending=True)

            sentence_indices = list(
                zip(unique["doc_index"].tolist(), unique["sent_index"].tolist())
            )
            start_ms_list = unique["sentence_start_ms"].tolist()
            start_min_list = unique["sentence_start_min"].tolist()
            text_list = unique["sentence_text"].tolist()

            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "sentence_indices": sentence_indices,
                    "sentence_start_ms_list": start_ms_list,
                    "sentence_start_min_list": start_min_list,
                    "sentence_text_list": text_list,
                    "num_sentences": len(sentence_indices),
                }
            )

        cluster_sentence_df = pd.DataFrame(rows)
        return cluster_sentence_df

    # ----------------------------- #
    # Logging helper                #
    # ----------------------------- #

    def _log(self, msg: str) -> None:
        """Send a short log message via the configured log function."""
        self.log_fn(msg)