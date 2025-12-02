"""
PhraseTopicMiner

Phrase-centric topic mining + clustering + timelines + visualization.

High-level API
--------------
- PhraseMiner        → extract NP/VP phrases and sentence grid
- TopicModeler       → embed + cluster phrases into topics
- TopicTimelineBuilder → reconstruct sentence-level timelines per topic
- TopicLabeler       → LLM-backed labels for each topic cluster
- Visualization helpers:
    * plot_topic_timeline, plot_phrase_bubble_map, plot_phrase_treemap
    * make_datamapplot_static / make_datamapplot_interactive
"""

from importlib.metadata import PackageNotFoundError, version



# Core APIs
from .phrase_miner import PhraseMiner, PhraseRecord
from .topic_modeler import TopicModeler, TopicCoreResult, TopicCluster
from .topic_timeline import TopicTimelineBuilder, TopicTimelineResult
from .topic_labeler import (
    TopicLabeler,
    ClusterLabelingInput,
    ClusterLabelingResult,
    TopicLabelModel,
    TopicLabelingResult,
    LabeledTopicCluster,
)

# Visualization APIs
from .topic_viz import (
    plot_topic_timeline,
    plot_phrase_bubble_map,
    plot_phrase_treemap,
)

from .visualization_datamap import (
    make_datamapplot_static,
    make_datamapplot_interactive,
    build_phrase_sentence_examples_from_occurrences,
)


# ---------------------------------------------------------------------
# Runtime version (single source of truth = pyproject.toml)
# ---------------------------------------------------------------------
try:
    __version__ = version("phrasetopicminer")
except PackageNotFoundError:
    # Fallback when running directly from a clone without installation
    __version__ = "0.0.0"

__all__ = [
    "PhraseMiner",
    "PhraseRecord",
    "TopicModeler",
    "TopicCoreResult",
    "TopicCluster",
    "TopicTimelineResult",
    "TopicTimelineBuilder",
    "TopicLabeler",
    "ClusterLabelingInput",
    "ClusterLabelingResult",
    "TopicLabelModel",
    "TopicLabelingResult",
    "LabeledTopicCluster",
    "plot_topic_timeline",
    "plot_phrase_bubble_map",
    "plot_phrase_treemap",
    "make_datamapplot_static",
    "make_datamapplot_interactive",
    "build_phrase_sentence_examples_from_occurrences",
    "__version__",
]
