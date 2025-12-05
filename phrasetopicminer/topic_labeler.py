# topic_labeler.py

"""
topic_labeler.py

LLM-backed topic labeling for PhraseTopicMiner.

This module takes the geometric/statistical output of TopicModeler
(TopicCoreResult + phrase_occurrences) plus raw sentence text and uses
an LLM to generate a short title and description for each phrase cluster.

It is deliberately *LLM- and framework-agnostic*. You can plug in:

1. A simple LLM callable (recommended default)
   - Pass `llm=` as a sync or async callable: `prompt: str -> str`.
   - TopicLabeler handles asyncio under the hood; user code stays simple.

2. A LangChain / chat model wrapper
   - Wrap your `ChatOpenAI` (or similar) in a tiny adapter that
     turns `prompt: str` into a `str` response, and pass that as `llm=`.

3. The OpenAI Agents SDK
   - Pass an `Agent` instance via `agent=...`.
   - TopicLabeler will call `Runner.run(...)` under a `trace(...)` context
     so you get full traces in the OpenAI Platform when labeling topics.

Pipeline
--------
- Input: TopicCoreResult (from TopicModeler.fit_core) + sentences_by_doc
- Build per-cluster inputs (phrases + counts + example sentences)
- Call the configured LLM backend for each cluster
- Parse / validate JSON into TopicLabelModel objects
- Output: TopicLabelingResult with:
    * labeled_clusters (full audit trail per topic),
    * labels_by_cluster (cluster_id → TopicLabelModel),
    * cluster_name_map (cluster_id → title string for plots/treemaps).
"""


from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
import json
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydantic import BaseModel, Field, ValidationError

from agents import Agent, Runner, trace  # OpenAI Agents SDK

from .topic_modeler import TopicCoreResult
from .phrase_miner import PhraseRecord


# -------------------------------------------------------------------
# Pydantic models and small containers
# -------------------------------------------------------------------


class TopicLabelModel(BaseModel):
    """
    Structured label for a single topic / cluster.

    Attributes
    ----------
    title:
        Short human-readable name for the topic (3–8 words is ideal).
    description:
        Concise 2–6 sentence explanation of what this topic is about.
    """

    title: str = Field(
        ...,
        description="Short, 3–8 word title naming the topic.",
    )
    description: str = Field(
        ...,
        description="2–6 sentence description of the topic.",
    )


@dataclass
class ClusterLabelingInput:
    """
    Material for labeling a single cluster.

    This is distilled into a prompt for the LLM.
    """

    cluster_id: int
    phrases_with_counts: List[Tuple[str, int]]
    representative_phrases: List[str]
    example_sentences: List[str]


@dataclass
class ClusterLabelingResult:
    """
    Raw LLM response for a single cluster.

    Attributes
    ----------
    cluster_id:
        ID of the cluster that was labeled.
    label:
        Parsed TopicLabelModel returned by the Agent.
    raw_response:
        Optional unstructured text from the Agent (for debugging/audit).
    """
    cluster_id: int
    label: TopicLabelModel
    raw_response: Optional[str] = None

@dataclass
class LabeledTopicCluster:
    """
    Full, audit-friendly view of a single labeled topic cluster.

    This object keeps **both**:
      - the LLM-produced label (title + description)
      - the evidence used to produce it (phrases, counts, sentences).

    Attributes
    ----------
    cluster_id:
        ID of the cluster (matches TopicModeler cluster_id).
    label:
        TopicLabelModel containing title and description.
    phrases_with_counts:
        List of (phrase, count) tuples actually sent to the LLM.
    representative_phrases:
        Headline phrases for the topic (from TopicModeler).
    example_sentences:
        Sentences used as contextual evidence for this topic.
    importance_score:
        Optional importance score from TopicModeler (if available).
    rank:
        Optional rank among clusters (1 = most important), if available.
    raw_response:
        Optional raw LLM text used to construct `label`.
    """
    cluster_id: int
    label: TopicLabelModel
    phrases_with_counts: List[Tuple[str, int]]
    representative_phrases: List[str]
    example_sentences: List[str]
    importance_score: Optional[float] = None
    rank: Optional[int] = None
    raw_response: Optional[str] = None


@dataclass
class TopicLabelingResult:
    """
    Aggregate result for a full labeling run.

    Attributes
    ----------
    labeled_clusters:
        List of LabeledTopicCluster objects, one per labeled cluster.
    labels_by_cluster:
        Convenience mapping: cluster_id → TopicLabelModel.
    cluster_name_map:
        Convenience mapping: cluster_id → title string.
        (This plugs directly into DataMapPlot, treemaps, etc.)
    config:
        Dictionary with run-time configuration for this labeling step,
        including a reference to the originating TopicModeler config.
    """
    labeled_clusters: List[LabeledTopicCluster]
    labels_by_cluster: Dict[int, TopicLabelModel]
    cluster_name_map: Dict[int, str]
    config: Dict[str, Any]


# -------------------------------------------------------------------
# TopicLabeler – main public API
# -------------------------------------------------------------------


class TopicLabeler:
    """
    LLM-backed topic labeling helper.

    Two ways to use it
    ------------------
    1) With the OpenAI Agents SDK (original design):

        from agents import Agent
        topic_agent = Agent(...)
        labeler = TopicLabeler(agent=topic_agent, ...)

    2) With a generic LLM object or callable (LLM-agnostic):

        # simplest: pass a function that returns a string
        def simple_llm(prompt: str) -> str:
            ...

        labeler = TopicLabeler(llm=simple_llm, ...)

        # or pass a LangChain ChatOpenAI, which has .invoke() / .ainvoke():
        from langchain_openai import ChatOpenAI
        lc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        labeler = TopicLabeler(llm=lc_llm, ...)

    Exactly one of `agent` or `llm` must be provided.
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        *,
        llm: Optional[Any] = None,
        max_phrases_per_cluster: int = 25,
        max_sentences_per_cluster: int = 40,
        include_noise: bool = False,
        labeler_name: str = "default_topic_labeler",
        log_fn: Optional[Callable[[str], None]] = print,
    ) -> None:
        """
        Parameters
        ----------
        agent:
            Optional `Agent` instance from the OpenAI Agents SDK.
            If provided, TopicLabeler will call `Runner.run(agent, prompt)`.
        llm:
            Optional generic LLM backend. Can be either:

            - a callable:  `str -> str` or `str -> Awaitable[str]`
            - or an object with `.invoke(prompt)` / `.ainvoke(prompt)`
              (e.g. LangChain's `ChatOpenAI`).

            If `agent` is None and `llm` is provided, TopicLabeler
            will use this backend instead of Agents.
        max_phrases_per_cluster:
            Maximum number of phrases (sorted by frequency) to send
            to the LLM for each cluster.
        max_sentences_per_cluster:
            Maximum number of example sentences to include per cluster.
        include_noise:
            If True, also label the noise cluster (cluster_id == -1).
        labeler_name:
            Free-form identifier for this labeler configuration, e.g.
            the underlying LLM/model or agent name. Stored in `config`.
        log_fn:
            Callable used for progress logging, e.g.:

                - `print` (default, console / Jupyter)
                - `st.write` / `st.markdown` (Streamlit)
                - any custom logger you like
        """
        if (agent is None) == (llm is None):
            raise ValueError(
                "TopicLabeler expects exactly one of `agent` or `llm`.\n"
                "Pass an OpenAI Agents `Agent` via `agent=`, or a plain "
                "LLM callable / object via `llm=`."
            )

        self.agent: Optional[Agent] = agent
        self.llm: Optional[Any] = llm

        self.max_phrases_per_cluster = max_phrases_per_cluster
        self.max_sentences_per_cluster = max_sentences_per_cluster
        self.include_noise = include_noise
        self.labeler_name = labeler_name
        self._log_fn = log_fn or (lambda _msg: None)

    # ------------- small logger wrapper -------------

    def _log(self, msg: str) -> None:
        """Internal logging helper."""
        try:
            self._log_fn(msg)
        except Exception:
            # Never let logging break the pipeline
            pass

    # ------------------------------------------------------------------
    # Public sync entrypoint (for scripts / CLIs)
    # ------------------------------------------------------------------

    def label_topics(
        self,
        core_result: TopicCoreResult,
        sentences_by_doc: Sequence[Sequence[str]],
        *,
        cluster_ids: Optional[Sequence[int]] = None,
    ) -> TopicLabelingResult:
        """
        Synchronous convenience wrapper around `label_topics_async`.

        In non-async environments (plain Python scripts), you can call this directly.
        In notebooks / async contexts, prefer:

            result = await labeler.label_topics_async(...)

        Raises a clear error if called from an already-running
        event loop (common in Jupyter) to avoid silent hangs.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "TopicLabeler.label_topics() was called from an async "
                "context (e.g. Jupyter). Please use "
                "`await label_topics_async(...)` instead."
            )

        return asyncio.run(
            self.label_topics_async(
                core_result,
                sentences_by_doc,
                cluster_ids=cluster_ids,
            )
        )

    # ------------------------------------------------------------------
    # Public async API (for notebooks / advanced usage)
    # ------------------------------------------------------------------

    async def label_topics_async(
        self,
        core_result: TopicCoreResult,
        sentences_by_doc: Sequence[Sequence[str]],
        *,
        cluster_ids: Optional[Sequence[int]] = None,
    ) -> TopicLabelingResult:
        """
        Asynchronously label clusters in a TopicCoreResult using `self.agent`.

        Parameters
        ----------
        core_result:
            Output of TopicModeler.fit_core (TopicCoreResult).
        sentences_by_doc:
            Nested list of original sentences, as returned from PhraseMiner:
            sentences_by_doc[doc_index][sent_index] -> sentence text.
        cluster_ids:
            Optional explicit list of cluster_ids to label. If None, all
            clusters in `core_result.phrases_df` are considered, with optional
            exclusion of noise (-1) depending on `include_noise`.

        Returns
        -------
        TopicLabelingResult
            Rich result object containing one LabeledTopicCluster per cluster,
            convenience mappings, and a config dictionary.
        """
        df = core_result.phrases_df

        # Map cluster_id → TopicCluster metadata (if available)
        cluster_meta_map = {c.cluster_id: c for c in core_result.clusters}

        # Decide which cluster IDs to label
        all_cluster_ids = sorted(df["cluster_id"].unique().tolist())
        self._log(
            f"[TopicLabeler] Found cluster_ids in core_result: {all_cluster_ids}"
        )

        # Filter cluster ids according to user request and noise flag
        if cluster_ids is None:
            cluster_ids_to_use = [
                cid for cid in all_cluster_ids
                if (self.include_noise or cid != -1)
            ]
        else:
            cluster_ids_to_use = [
                cid for cid in cluster_ids
                if (self.include_noise or cid != -1)
            ]

        if not cluster_ids_to_use:
            raise ValueError(
                "No clusters selected for labeling. "
                "Check include_noise / cluster_ids."
            )

        self._log(
            f"[TopicLabeler] Labeling {len(cluster_ids_to_use)} cluster(s): {cluster_ids_to_use}"
        )

        # --------------------------------------------------
        # Build ClusterLabelingInput for each cluster
        # --------------------------------------------------
        inputs: List[ClusterLabelingInput] = []

        for cid in cluster_ids_to_use:
            sub = df[df["cluster_id"] == cid].copy()
            if sub.empty:
                continue

            # Phrases sorted by frequency (descending)
            sub_sorted = sub.sort_values("count", ascending=False)
            phrases_with_counts: List[Tuple[str, int]] = [
                (p, int(c))
                for p, c in zip(sub_sorted["phrase"], sub_sorted["count"])
            ]

            # Limit phrases passed to the LLM
            phrases_with_counts = phrases_with_counts[: self.max_phrases_per_cluster]

            # Representative phrases from TopicModeler, if available
            rep_phrases: List[str] = []
            meta = cluster_meta_map.get(cid)
            if meta is not None and meta.representative_phrases:
                rep_phrases = meta.representative_phrases
            else:
                # Fallback: top few phrases by frequency
                rep_phrases = [p for p, _ in phrases_with_counts[:10]]

            # Example sentences for this cluster
            example_sentences = self._collect_example_sentences_for_cluster(
                phrases_with_counts=phrases_with_counts,
                phrase_occurrences=core_result.phrase_occurrences,
                sentences_by_doc=sentences_by_doc,
            )

            inputs.append(
                ClusterLabelingInput(
                    cluster_id=int(cid),
                    phrases_with_counts=phrases_with_counts,
                    representative_phrases=rep_phrases,
                    example_sentences=example_sentences,
                )
            )

        if inputs:
            self._log(
                "[TopicLabeler] Prepared cluster inputs "
                f"(phrases + sentences) for {len(inputs)} cluster(s)."
            )
        else:
            raise ValueError("No ClusterLabelingInput objects could be built.")

        # --------------------------------------------------
        # Call the agent concurrently for all clusters under a top-level trace
        # --------------------------------------------------
        with trace(f"PhraseTopicMiner TopicLabeler (clusters={len(inputs)})"):
            tasks = [self._label_single_cluster(inp) for inp in inputs]
            raw_results: List[ClusterLabelingResult] = await asyncio.gather(*tasks)
        
        # --------------------------------------------------
        # Assemble LabeledTopicCluster objects and mappings
        # --------------------------------------------------
        labeled_clusters: List[LabeledTopicCluster] = []
        labels_by_cluster: Dict[int, TopicLabelModel] = {}
        cluster_name_map: Dict[int, str] = {}

        # Build a quick lookup for inputs by cluster_id
        input_by_cid: Dict[int, ClusterLabelingInput] = {
            inp.cluster_id: inp for inp in inputs
        }

        for res in raw_results:
            cid = res.cluster_id
            label = res.label
            inp = input_by_cid[cid]

            labels_by_cluster[cid] = label
            cluster_name_map[cid] = label.title

            meta = cluster_meta_map.get(cid)
            importance = getattr(meta, "importance_score", None)
            rank = getattr(meta, "rank", None)

            labeled_clusters.append(
                LabeledTopicCluster(
                    cluster_id=cid,
                    label=label,
                    phrases_with_counts=inp.phrases_with_counts,
                    representative_phrases=inp.representative_phrases,
                    example_sentences=inp.example_sentences,
                    importance_score=importance,
                    rank=rank,
                    raw_response=res.raw_response,
                )
            )

        # Sort clusters by rank (if available), then by importance, then id
        labeled_clusters.sort(
            key=lambda lc: (
                float("inf") if lc.rank is None else lc.rank,
                -1.0 if lc.importance_score is None else -lc.importance_score,
                lc.cluster_id,
            )
        )

        # --------------------------------------------------
        # Build config dictionary (includes TopicModeler config)
        # --------------------------------------------------
        config: Dict[str, Any] = {
            "labeler_name": self.labeler_name,
            "max_phrases_per_cluster": self.max_phrases_per_cluster,
            "max_sentences_per_cluster": self.max_sentences_per_cluster,
            "include_noise": self.include_noise,
            "num_clusters_available": len(all_cluster_ids),
            "num_clusters_labeled": len(labeled_clusters),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            # full reference to the originating topic model run:
            "topic_modeler_config": core_result.config,
        }

        self._log("[TopicLabeler] ✅  Labeling complete.")
        return TopicLabelingResult(
            labeled_clusters=labeled_clusters,
            labels_by_cluster=labels_by_cluster,
            cluster_name_map=cluster_name_map,
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal helpers – building cluster inputs
    # ------------------------------------------------------------------

    def _collect_example_sentences_for_cluster(
        self,
        phrases_with_counts: List[Tuple[str, int]],
        phrase_occurrences: Dict[str, List["PhraseRecord"]],
        sentences_by_doc: List[List[str]],
    ) -> List[str]:
        """
        Collect up to `max_sentences_per_cluster` example sentences for a cluster.

        Strategy
        --------
        - Iterate phrases in descending frequency order.
        - For each phrase, collect sentences from phrase_occurrences.
        - Use (doc_index, sent_index) + text string to deduplicate.
        """
        max_sents = self.max_sentences_per_cluster
        examples: List[str] = []
        seen_keys = set()

        for phrase, _count in phrases_with_counts:
            occ_list = phrase_occurrences.get(phrase, [])
            for occ in occ_list:
                try:
                    sent_text = sentences_by_doc[occ.doc_index][occ.sent_index]
                except (IndexError, KeyError):
                    continue

                key = (occ.doc_index, occ.sent_index, sent_text)
                if key in seen_keys:
                    continue

                seen_keys.add(key)
                examples.append(sent_text)

                if len(examples) >= max_sents:
                    return examples

        return examples

    
    def _build_cluster_labeling_input(
        self,
        *,
        cluster_id: int,
        core_result: TopicCoreResult,
        sentences_by_doc: Sequence[Sequence[str]],
        cluster_obj: Optional[Any] = None,
    ) -> ClusterLabelingInput:
        """
        Collect phrases + counts + example sentences for a single cluster.
        """
        df = core_result.phrases_df

        sub = df[df["cluster_id"] == cluster_id].copy()
        if sub.empty:
            raise ValueError(f"No phrases found for cluster_id={cluster_id}")

        # Sort phrases by frequency (descending)
        sub.sort_values("count", ascending=False, inplace=True)

        phrases_with_counts: List[Tuple[str, int]] = list(
            zip(
                sub["phrase"].tolist(),
                sub["count"].astype(int).tolist(),
            )
        )

        if self.max_phrases_per_cluster is not None:
            phrases_with_counts = phrases_with_counts[: self.max_phrases_per_cluster]

        # Representative phrases – prefer TopicCluster, fall back to top phrases
        if cluster_obj is not None and getattr(
            cluster_obj, "representative_phrases", None
        ):
            representative_phrases = list(
                cluster_obj.representative_phrases[: self.max_phrases_per_cluster]
            )
        else:
            representative_phrases = [
                p for p, _ in phrases_with_counts[: min(10, len(phrases_with_counts))]
            ]

        # Example sentences from phrase_occurrences
        phrase_occurrences: Dict[str, List[PhraseRecord]] = core_result.phrase_occurrences

        example_sentences: List[str] = []
        seen_indices: Set[Tuple[int, int]] = set()

        for phrase, _cnt in phrases_with_counts:
            for rec in phrase_occurrences.get(phrase, []):
                doc_idx = int(rec.doc_index)
                sent_idx = int(rec.sent_index)
                key = (doc_idx, sent_idx)

                if key in seen_indices:
                    continue

                try:
                    sent_text = sentences_by_doc[doc_idx][sent_idx]
                except (IndexError, TypeError):
                    continue

                sent_text = sent_text.strip()
                if not sent_text:
                    continue

                example_sentences.append(sent_text)
                seen_indices.add(key)

                if len(example_sentences) >= self.max_sentences_per_cluster:
                    break

            if len(example_sentences) >= self.max_sentences_per_cluster:
                break

        self._log(
            f"[TopicLabeler] Cluster {cluster_id}: "
            f"{len(phrases_with_counts)} phrases sent, "
            f"{len(example_sentences)} example sentences."
        )

        return ClusterLabelingInput(
            cluster_id=cluster_id,
            phrases_with_counts=phrases_with_counts,
            representative_phrases=representative_phrases,
            example_sentences=example_sentences,
        )

    # ------------------------------------------------------------------
    # Internal helpers – LLM call & prompt
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str, cluster_id: int) -> str:
        """
        Unified LLM call.

        - If `self.agent` is set, use the OpenAI Agents Runner.
        - Otherwise, use `self.llm`:

            * callable(prompt)  -> text or awaitable
            * or object with .ainvoke(prompt) / .invoke(prompt)
              (LangChain-style).
        """
        # --- 1) Agents SDK path -----------------------------------------
        if self.agent is not None:
            if Runner is None:
                raise RuntimeError(
                    "TopicLabeler was configured with `agent=...`, but the "
                    "`agents` package is not installed."
                )

            with trace(
                f"label_cluster_{cluster_id}_{datetime.utcnow().isoformat()}"
            ):
                result = await Runner.run(self.agent, prompt)

            raw = getattr(result, "final_output", "") or ""
            return raw.strip()

        # --- 2) Generic LLM path ---------------------------------------
        if self.llm is None:
            raise RuntimeError(
                "TopicLabeler has neither `agent` nor `llm` configured."
            )

        backend = self.llm

        # (a) Bare callable: func(prompt) -> text or awaitable
        if callable(backend) and not hasattr(backend, "invoke") and not hasattr(backend, "ainvoke"):
            out = backend(prompt)
            if inspect.isawaitable(out):
                out = await out

        # (b) LangChain-style: has .ainvoke(prompt)
        elif hasattr(backend, "ainvoke"):
            out = await backend.ainvoke(prompt)

        # (c) LangChain-style: has .invoke(prompt)
        elif hasattr(backend, "invoke"):
            out = backend.invoke(prompt)

        else:
            raise TypeError(
                "llm= must be either a callable, or an object with "
                "an `.invoke(prompt)` or `.ainvoke(prompt)` method."
            )

        # --- 3) Normalise various outputs to a plain string ------------

        # simple string
        if isinstance(out, str):
            return out.strip()

        # LangChain messages (have `.content`)
        content = getattr(out, "content", None)
        if content is not None:
            return str(content).strip()

        # OpenAI-style responses with .choices[0].message.content
        choices = getattr(out, "choices", None)
        if choices:
            first = choices[0]
            msg = getattr(first, "message", None) or getattr(first, "delta", None)
            if msg is not None and getattr(msg, "content", None) is not None:
                return str(msg.content).strip()

        # dict-like { "text": "..."} or {"content": "..."}
        if isinstance(out, dict):
            for key in ("text", "content", "output"):
                if key in out:
                    return str(out[key]).strip()

        # last-resort fallback
        return str(out).strip()
        

    async def _label_single_cluster(
        self,
        cluster_input: ClusterLabelingInput,
    ) -> ClusterLabelingResult:
        """
        Label a single cluster using either an Agents `Agent` or a generic LLM.

        The actual backend is decided by `_call_llm`.

        Returns
        -------
        ClusterLabelingResult
            Structured result with cluster_id, TopicLabelModel, and
            the raw text output from the agent (for audit/debugging).
        """
        prompt = self._build_prompt(cluster_input)

        self._log(
            f"[TopicLabeler] Calling LLM for cluster {cluster_input.cluster_id}..."
        )

        raw_output = await self._call_llm(prompt, cluster_input.cluster_id)

        self._log(
            f"[TopicLabeler] LLM output for cluster {cluster_input.cluster_id}: "
            f"{raw_output[:120]}{'...' if len(raw_output) > 120 else ''}"
        )

        # Prefer strict JSON → TopicLabelModel
        label = self._parse_label_output(raw_output, cluster_input.cluster_id)

        return ClusterLabelingResult(
            cluster_id=cluster_input.cluster_id,
            label=label,
            raw_response=raw_output,
        )

    @staticmethod
    def _build_prompt(cluster_input: ClusterLabelingInput) -> str:
        """
        Construct a concise, model-agnostic prompt that asks for a
        strict JSON response.
        """
        ph_lines = "\n".join(
            f"- {phrase} [{count}]"
            for phrase, count in cluster_input.phrases_with_counts
        )
        sent_lines = "\n".join(
            f"- {s}" for s in cluster_input.example_sentences
        )

        return f"""
                You are a topic labeling assistant for a phrase-based topic model.
                
                You are given:
                - A set of key phrases for a single topic, each with its frequency.
                - A small set of example sentences where these phrases occur.
                
                Your task for this ONE topic is to:
                1. Propose a short, 3–8 word title that best names the topic.
                2. Write a concise, 2–6 sentence description of what this topic is about.
                
                Guidelines:
                - Focus ONLY on the phrases and sentences provided.
                - Do NOT mention 'cluster', 'topic number', or 'phrases' in the title.
                - Avoid generic titles like "Miscellaneous" unless the content is truly incoherent.
                
                Return your answer as a SINGLE JSON object with EXACTLY these keys:
                
                {{
                  "title": "short title here",
                  "description": "2–6 sentence description here"
                }}
                
                Do not include any additional text, commentary, markdown, or code fences.
                Topic id: {cluster_input.cluster_id}
                
                Representative phrases (phrase [count]):
                {ph_lines}
                
                Example sentences:
                {sent_lines}
                """.strip()

    @staticmethod
    def _parse_label_output(
        raw_output: str,
        cluster_id: int,
    ) -> TopicLabelModel:
        """
        Parse the agent's output into a TopicLabelModel, with robust
        fallbacks if the model drifts away from strict JSON.
        """
        # First attempt: direct JSON → TopicLabelModel
        try:
            return TopicLabelModel.model_validate_json(raw_output)
        except (ValidationError, json.JSONDecodeError):
            pass

        # Second attempt: json.loads, then TopicLabelModel(**data)
        try:
            data = json.loads(raw_output)
            return TopicLabelModel(**data)
        except Exception:
            pass

        # Last-resort: heuristic fallback – first line = title, rest = description
        lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
        if not lines:
            return TopicLabelModel(
                title=f"Topic {cluster_id}",
                description="No description could be parsed from the model output.",
            )

        title = lines[0][:80]
        description = "\n".join(lines[1:]) or lines[0]
        return TopicLabelModel(title=title, description=description)