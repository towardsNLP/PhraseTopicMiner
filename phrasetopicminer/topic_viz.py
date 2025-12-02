# topic_viz.py

"""
topic_viz.py

Visualization helpers for PhraseTopicMiner.

This module is intentionally thin and UI-agnostic. It:

- Renders topic timelines with Plotly from TopicTimelineResult.
- Renders 2D phrase maps (bubble charts) from TopicCoreResult.phrases_df.
- Renders phrase treemaps from phrases_df.
- Produces a DataFrame suitable for use with DataMapPlot.

All functions return Plotly Figure objects or pandas DataFrames, so they can
be used in notebooks, Streamlit, Dash, etc.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .topic_modeler import TopicCoreResult
from .topic_timeline import TopicTimelineResult


# ---------------------------------------------------------------------
# Topic timeline (cluster → vertical bars over time)
# ---------------------------------------------------------------------


def plot_topic_timeline(
    timeline_result: TopicTimelineResult,
    cluster_id: int,
    *,
    time_unit: str = "min",
    width: int = 800,
    height: int = 120,
    title_prefix: str = "Topic Timeline",
) -> go.Figure:
    """
    Plot a simple topic timeline: vertical bars for sentences that
    belong to a given cluster.

    Parameters
    ----------
    timeline_result:
        Output of TopicTimelineBuilder.build().
    cluster_id:
        Cluster to visualize.
    time_unit:
        "min"  → use sentence_start_min_list on the x-axis.
        "ms"   → use sentence_start_ms_list on the x-axis.
    width, height:
        Figure size in pixels.
    title_prefix:
        Title prefix; the cluster id and sentence count are appended.

    Returns
    -------
    plotly.graph_objects.Figure
        Vertical-bar timeline figure.
    """
    cluster_df = timeline_result.cluster_sentence_df
    row = cluster_df.loc[cluster_df["cluster_id"] == cluster_id]

    if row.empty:
        raise ValueError(f"No cluster with cluster_id={cluster_id} in cluster_sentence_df.")

    row = row.iloc[0]

    if time_unit == "min":
        x_values = row["sentence_start_min_list"]
        x_label = "Time (minutes)"
    elif time_unit == "ms":
        x_values = row["sentence_start_ms_list"]
        x_label = "Time (ms)"
    else:
        raise ValueError("time_unit must be 'min' or 'ms'.")

    sentences = row["sentence_text_list"]
    sentence_count = int(row["num_sentences"])

    # Prepare hover texts
    hover_texts = []
    for t, s in zip(x_values, sentences):
        # Wrap sentence text a bit in HTML <br> to keep hover readable
        wrapped = "<br>".join(str(s).splitlines())
        hover_texts.append(f"{t:.2f} → {wrapped}")

    fig = go.Figure()

    for t, hover in zip(x_values, hover_texts):
        fig.add_trace(
            go.Scatter(
                x=[t, t],
                y=[0, 1],
                mode="lines",
                line=dict(width=1.5, color="darkblue"),
                hovertext=hover,
                hoverinfo="text",
                hoverlabel=dict(bgcolor="lightgreen"),
                showlegend=False,
            )
        )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=10, b=25, t=40, pad=5),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(
            title=x_label,
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        title=(
            f"<span style='font-size:14px; font-weight: normal; color:darkblue;'>"
            f"{title_prefix} (Cluster {cluster_id})<br>{sentence_count} key sentences</span>"
        ),
    )

    return fig


# ---------------------------------------------------------------------
# Phrase bubble map (2D embedding from TopicCoreResult)
# ---------------------------------------------------------------------


def plot_phrase_bubble_map(
    core_result: TopicCoreResult,
    *,
    max_phrases: Optional[int] = None,
    show_text: bool = False,
    width: int = 900,
    height: int = 700,
    title: Optional[str] = None,
) -> px.scatter:
    """
    Plot a 2D bubble map of phrases using the (x, y) coordinates stored
    in TopicCoreResult.phrases_df.

    Parameters
    ----------
    core_result:
        Output of TopicModeler.fit_core().
        `core_result.phrases_df` must contain columns:
            'phrase', 'count', 'cluster_id', 'x', 'y'.
    max_phrases:
        Optional limit on the number of phrases to plot (top by count).
    show_text:
        If True, annotate points with phrase text.
    width, height:
        Figure size in pixels.
    title:
        Optional title. If None, a default one is constructed.

    Returns
    -------
    plotly.express.scatter
        Bubble chart Figure.
    """
    df = core_result.phrases_df.copy()

    # Only keep required columns and sort by frequency
    df = df[["phrase", "count", "cluster_id", "x", "y"]].copy()
    df.sort_values("count", ascending=False, inplace=True)

    if max_phrases is not None and max_phrases > 0:
        df = df.head(max_phrases).copy()

    # Treat cluster_id as string for discrete color mapping
    df["cluster_label"] = df["cluster_id"].astype(str)

    text_arg = df["phrase"] if show_text else None

    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="count",
        color="cluster_label",
        text=text_arg,
        hover_name="phrase",
        hover_data={
            "phrase": True,
            "count": True,
            "cluster_label": True,
            "x": False,
            "y": False,
        },
        labels={
            "cluster_label": "Cluster",
            "count": "Frequency",
        },
        width=width,
        height=height,
        size_max=40,
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(
            opacity=0.8,
            line=dict(width=0.7, color="DarkSlateGrey"),
        ),
        selector=dict(mode="markers"),
    )

    fig.update_layout(
        title=title
        or f"Inter-phrase Distance Map: Semantic Representation of Top {len(df)} Phrases",
        xaxis=dict(
            showline=True,
            showgrid=False,
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            showticklabels=False,
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor="rgb(204, 204, 204)",
            linecolor="rgb(204, 204, 204)",
            linewidth=2,
            showticklabels=False,
        ),
        plot_bgcolor="white",
        hoverlabel=dict(font_size=13),
    )

    return fig


# ---------------------------------------------------------------------
# Phrase treemap (cluster → phrases sized by frequency)
# ---------------------------------------------------------------------


def plot_phrase_treemap(
    core_result: TopicCoreResult,
    *,
    color_continuous_scale: str = "Viridis",
    width: int = 900,
    height: int = 500,
) -> px.treemap:
    """
    Plot a treemap of phrases grouped by cluster.

    The area encodes phrase frequency; color encodes the same or can
    be interpreted as intensity.

    Parameters
    ----------
    core_result:
        Output of TopicModeler.fit_core(); uses phrases_df.
    color_continuous_scale:
        Name of a Plotly continuous colormap (e.g. 'Viridis', 'Blues').
    width, height:
        Figure size in pixels.

    Returns
    -------
    plotly.express.treemap
        Treemap Figure.
    """
    df = core_result.phrases_df[["phrase", "count", "cluster_id"]].copy()
    df["cluster_label"] = "Cluster " + df["cluster_id"].astype(str)

    fig = px.treemap(
        df,
        path=[px.Constant("All Phrase Clusters"), "cluster_label", "phrase"],
        values="count",
        color="count",
        color_continuous_scale=color_continuous_scale,
        hover_data={"phrase": True, "count": True},
        labels={"count": "Frequency"},
        width=width,
        height=height,
    )

    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


