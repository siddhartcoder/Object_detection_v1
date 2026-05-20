"""
Interactive Plotly Dash Dashboard for BDD100K Dataset Analysis.

This module builds a multi-tab interactive dashboard that allows users to
explore dataset statistics, class distributions, bounding box properties,
and frame-level attribute breakdowns.

Usage::

    from src.data_analysis.bdd_parser import BDDDataset
    from src.data_analysis.dashboard import build_dashboard

    dataset = BDDDataset(
        train_json="data/labels/det_20/det_train.json",
        val_json="data/labels/det_20/det_val.json",
    )
    dataset.load()
    app = build_dashboard(dataset)
    app.run(debug=False, port=8050)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from src.data_analysis.bdd_parser import BDD_DETECTION_CLASSES, BDDDataset

# Colour map for classes
CLASS_COLOR_MAP = {
    "pedestrian": "#e6194b", "rider": "#f58231", "car": "#3cb44b",
    "truck": "#4363d8", "bus": "#911eb4", "train": "#42d4f4",
    "motorcycle": "#f032e6", "bicycle": "#bfef45",
    "traffic light": "#fabed4", "traffic sign": "#aaffc3",
}


def _build_class_df(dataset: BDDDataset) -> pd.DataFrame:
    """Build a long-form DataFrame of class instance counts for train and val.

    Args:
        dataset: Loaded BDDDataset object.

    Returns:
        DataFrame with columns ['class', 'split', 'count'].
    """
    train_counts = dataset.get_class_counts("train")
    val_counts = dataset.get_class_counts("val")
    rows = []
    for cls in BDD_DETECTION_CLASSES:
        rows.append({"class": cls, "split": "train", "count": train_counts[cls]})
        rows.append({"class": cls, "split": "val", "count": val_counts[cls]})
    return pd.DataFrame(rows)


def _build_bbox_df(dataset: BDDDataset, split: str = "train") -> pd.DataFrame:
    """Build a DataFrame of bounding box properties for the given split.

    Args:
        dataset: Loaded BDDDataset object.
        split: 'train' or 'val'.

    Returns:
        DataFrame with columns ['class', 'width', 'height', 'area', 'aspect_ratio'].
    """
    bbox_stats = dataset.get_bbox_stats(split)
    rows = []
    for cls in BDD_DETECTION_CLASSES:
        data = bbox_stats[cls]
        if len(data) == 0:
            continue
        # Sample up to 5000 points per class to keep the dashboard responsive
        idx = np.random.choice(len(data), size=min(5000, len(data)), replace=False)
        for i in idx:
            rows.append({
                "class": cls,
                "width": data[i, 0],
                "height": data[i, 1],
                "area": data[i, 2],
                "aspect_ratio": data[i, 3],
            })
    return pd.DataFrame(rows)


def _build_attr_df(dataset: BDDDataset) -> pd.DataFrame:
    """Build a DataFrame of frame-level attribute counts for train split.

    Args:
        dataset: Loaded BDDDataset object.

    Returns:
        DataFrame with columns ['attribute', 'value', 'count'].
    """
    rows = []
    for attr in ["weather", "scene", "timeofday"]:
        dist = dataset.get_attribute_distribution(attr, "train")
        for value, count in dist.items():
            rows.append({"attribute": attr, "value": value, "count": count})
    return pd.DataFrame(rows)


def build_dashboard(dataset: BDDDataset, debug: bool = False) -> dash.Dash:
    """Build and return the Plotly Dash dashboard application.

    The dashboard has four tabs:
    1. Overview — summary statistics card grid
    2. Class Distribution — bar and pie charts
    3. Bounding Box Analysis — scatter and violin plots
    4. Frame Attributes — weather / scene / time-of-day breakdowns

    Args:
        dataset: Loaded BDDDataset object.
        debug: If True, enables Dash debug mode.

    Returns:
        Configured Dash application object (call app.run() to start).
    """
    # Pre-compute DataFrames
    class_df = _build_class_df(dataset)
    bbox_train_df = _build_bbox_df(dataset, "train")
    bbox_val_df = _build_bbox_df(dataset, "val")
    attr_df = _build_attr_df(dataset)
    summary = dataset.summary()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="BDD100K EDA Dashboard",
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                dbc.Col(
                    html.H2(
                        "BDD100K Object Detection — EDA Dashboard",
                        className="text-center my-3",
                        style={"color": "#17a2b8"},
                    )
                )
            ),
            dbc.Tabs(
                [
                    # ---- Tab 1: Overview ----
                    dbc.Tab(
                        label="Overview",
                        tab_id="tab-overview",
                        children=_build_overview_tab(summary),
                    ),
                    # ---- Tab 2: Class Distribution ----
                    dbc.Tab(
                        label="Class Distribution",
                        tab_id="tab-class",
                        children=_build_class_tab(class_df),
                    ),
                    # ---- Tab 3: Bounding Box Analysis ----
                    dbc.Tab(
                        label="Bounding Box Analysis",
                        tab_id="tab-bbox",
                        children=_build_bbox_tab(bbox_train_df, bbox_val_df),
                    ),
                    # ---- Tab 4: Frame Attributes ----
                    dbc.Tab(
                        label="Frame Attributes",
                        tab_id="tab-attrs",
                        children=_build_attr_tab(attr_df),
                    ),
                ],
                id="main-tabs",
                active_tab="tab-overview",
            ),
        ],
    )

    return app


# ------------------------------------------------------------------
# Tab Builders
# ------------------------------------------------------------------


def _build_overview_tab(summary: dict) -> html.Div:
    """Build the Overview tab with summary statistic cards."""
    cards = [
        _stat_card("Training Images", f"{summary['train_frames']:,}", "primary"),
        _stat_card("Validation Images", f"{summary['val_frames']:,}", "info"),
        _stat_card("Train Instances", f"{summary['train_instances']:,}", "success"),
        _stat_card("Val Instances", f"{summary['val_instances']:,}", "warning"),
        _stat_card("Train Empty Frames", f"{summary['train_empty_frames']:,}", "danger"),
        _stat_card("Val Empty Frames", f"{summary['val_empty_frames']:,}", "secondary"),
    ]
    return html.Div(
        [
            dbc.Row([dbc.Col(card, md=2) for card in cards], className="my-4"),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(figure=_overview_bar_chart(summary)),
                    md=12,
                )
            ),
        ]
    )


def _stat_card(title: str, value: str, color: str) -> dbc.Card:
    """Return a Bootstrap card with a bold numeric value."""
    return dbc.Card(
        dbc.CardBody([
            html.H5(value, className="card-title text-center", style={"fontSize": "1.8rem"}),
            html.P(title, className="card-text text-center", style={"fontSize": "0.8rem"}),
        ]),
        color=color,
        inverse=True,
        className="m-1",
    )


def _overview_bar_chart(summary: dict) -> go.Figure:
    """Build a grouped bar chart of train vs val class instance counts."""
    tc = summary["train_class_counts"]
    vc = summary["val_class_counts"]
    fig = go.Figure(data=[
        go.Bar(name="Train", x=BDD_DETECTION_CLASSES, y=[tc[c] for c in BDD_DETECTION_CLASSES],
               marker_color="#4363d8"),
        go.Bar(name="Val", x=BDD_DETECTION_CLASSES, y=[vc[c] for c in BDD_DETECTION_CLASSES],
               marker_color="#e6194b"),
    ])
    fig.update_layout(
        barmode="group",
        title="Train vs Val — Instance Counts per Class",
        xaxis_title="Class",
        yaxis_title="Instance Count",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_class_tab(class_df: pd.DataFrame) -> html.Div:
    """Build the Class Distribution tab."""
    train_df = class_df[class_df["split"] == "train"].sort_values("count", ascending=False)

    bar_fig = px.bar(
        class_df,
        x="class",
        y="count",
        color="split",
        barmode="group",
        title="Instance Counts per Class (Train vs Val)",
        color_discrete_map={"train": "#4363d8", "val": "#e6194b"},
        template="plotly_dark",
    )

    pie_fig = px.pie(
        train_df,
        names="class",
        values="count",
        title="Training Split — Class Distribution (%)",
        color="class",
        color_discrete_map=CLASS_COLOR_MAP,
        template="plotly_dark",
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=bar_fig), md=6),
            dbc.Col(dcc.Graph(figure=pie_fig), md=6),
        ], className="mt-3"),
    ])


def _build_bbox_tab(train_df: pd.DataFrame, val_df: pd.DataFrame) -> html.Div:
    """Build the Bounding Box Analysis tab with a split selector."""
    violin_fig = px.violin(
        train_df,
        x="class",
        y="area",
        color="class",
        box=True,
        title="Bounding Box Area Distribution per Class (Train)",
        color_discrete_map=CLASS_COLOR_MAP,
        template="plotly_dark",
    )
    violin_fig.update_traces(meanline_visible=True)
    violin_fig.update_layout(showlegend=False)

    scatter_fig = px.scatter(
        train_df.sample(min(5000, len(train_df))),
        x="width",
        y="height",
        color="class",
        title="Bounding Box Width vs Height (Train — sampled 5k pts)",
        color_discrete_map=CLASS_COLOR_MAP,
        template="plotly_dark",
        opacity=0.5,
    )
    scatter_fig.update_layout(legend=dict(orientation="h"))

    ar_fig = px.histogram(
        train_df,
        x="aspect_ratio",
        color="class",
        nbins=100,
        title="Bounding Box Aspect Ratio (W/H) Distribution",
        range_x=[0, 5],
        color_discrete_map=CLASS_COLOR_MAP,
        template="plotly_dark",
        barmode="overlay",
        opacity=0.6,
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=violin_fig), md=12),
        ], className="mt-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=scatter_fig), md=6),
            dbc.Col(dcc.Graph(figure=ar_fig), md=6),
        ]),
    ])


def _build_attr_tab(attr_df: pd.DataFrame) -> html.Div:
    """Build the Frame Attributes tab with pie charts for each attribute."""
    figs = []
    for attr in ["weather", "scene", "timeofday"]:
        sub = attr_df[attr_df["attribute"] == attr]
        fig = px.pie(
            sub,
            names="value",
            values="count",
            title=attr.replace("timeofday", "Time of Day").title(),
            template="plotly_dark",
        )
        figs.append(dcc.Graph(figure=fig))

    return html.Div([
        dbc.Row([dbc.Col(f, md=4) for f in figs], className="mt-3"),
    ])
