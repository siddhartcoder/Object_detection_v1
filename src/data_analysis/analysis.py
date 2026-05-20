"""
BDD100K Dataset Exploratory Data Analysis.

This module provides comprehensive analysis functions for the BDD100K
object detection dataset, including class distribution analysis,
train/val split comparison, anomaly detection, and sample identification.

Functions are designed to be called independently or chained via run_full_analysis().
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.data_analysis.bdd_parser import (
    BDD_DETECTION_CLASSES,
    BDDDataset,
    BDDFrame,
    BDDLabel,
)

# Consistent colour palette for the 10 classes
CLASS_COLORS = {
    "pedestrian": "#e6194b",
    "rider": "#f58231",
    "car": "#3cb44b",
    "truck": "#4363d8",
    "bus": "#911eb4",
    "train": "#42d4f4",
    "motorcycle": "#f032e6",
    "bicycle": "#bfef45",
    "traffic light": "#fabed4",
    "traffic sign": "#aaffc3",
}


# ---------------------------------------------------------------------------
# 1. Class Distribution
# ---------------------------------------------------------------------------


def plot_class_distribution(
    dataset: BDDDataset,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Plot class instance distribution for train and val splits side-by-side.

    Args:
        dataset: Loaded BDDDataset object.
        output_dir: Directory to save the plot. If None, plot is shown inline.

    Returns:
        DataFrame with columns ['class', 'train_count', 'val_count', 'ratio'].
    """
    train_counts = dataset.get_class_counts("train")
    val_counts = dataset.get_class_counts("val")

    df = pd.DataFrame(
        {
            "class": BDD_DETECTION_CLASSES,
            "train_count": [train_counts[c] for c in BDD_DETECTION_CLASSES],
            "val_count": [val_counts[c] for c in BDD_DETECTION_CLASSES],
        }
    )
    df["ratio"] = (df["val_count"] / df["train_count"].replace(0, np.nan)).round(3)
    df = df.sort_values("train_count", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("BDD100K — Class Instance Distribution", fontsize=16, fontweight="bold")

    # Bar chart
    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width / 2, df["train_count"], width, label="Train", color="#4363d8", alpha=0.8)
    axes[0].bar(x + width / 2, df["val_count"], width, label="Val", color="#e6194b", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["class"], rotation=30, ha="right")
    axes[0].set_ylabel("Instance Count")
    axes[0].set_title("Instance Counts per Class")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Pie chart (train only)
    colors = [CLASS_COLORS[c] for c in df["class"]]
    axes[1].pie(
        df["train_count"],
        labels=df["class"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
    )
    axes[1].set_title("Train Split — Class Distribution (%)")

    plt.tight_layout()
    _save_or_show(fig, output_dir, "class_distribution.png")
    return df


# ---------------------------------------------------------------------------
# 2. Train / Val Split Analysis
# ---------------------------------------------------------------------------


def analyze_train_val_split(
    dataset: BDDDataset,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Analyse and compare train vs val splits at image and instance level.

    Args:
        dataset: Loaded BDDDataset object.
        output_dir: Directory to save plots.

    Returns:
        Dict with detailed split statistics.
    """
    summary = dataset.summary()

    print("\n" + "=" * 60)
    print("  BDD100K — Train / Val Split Analysis")
    print("=" * 60)
    print(f"  Training images   : {summary['train_frames']:>8,}")
    print(f"  Validation images : {summary['val_frames']:>8,}")
    print(f"  Train instances   : {summary['train_instances']:>8,}")
    print(f"  Val instances     : {summary['val_instances']:>8,}")
    print(f"  Train empty imgs  : {summary['train_empty_frames']:>8,}")
    print(f"  Val empty imgs    : {summary['val_empty_frames']:>8,}")

    # Per-class ratio table
    tc = summary["train_class_counts"]
    vc = summary["val_class_counts"]
    print("\n  Per-Class Instance Counts:")
    print(f"  {'Class':<16} {'Train':>10} {'Val':>8} {'Val/Train':>10}")
    print("  " + "-" * 46)
    for cls in BDD_DETECTION_CLASSES:
        ratio = vc[cls] / tc[cls] if tc[cls] > 0 else 0
        print(f"  {cls:<16} {tc[cls]:>10,} {vc[cls]:>8,} {ratio:>10.3f}")
    print("=" * 60)

    # Plot per-class val/train ratio
    ratios = [vc[c] / tc[c] if tc[c] > 0 else 0 for c in BDD_DETECTION_CLASSES]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(BDD_DETECTION_CLASSES, ratios, color=[CLASS_COLORS[c] for c in BDD_DETECTION_CLASSES])
    ax.axhline(y=np.mean(ratios), color="red", linestyle="--", label=f"Mean ratio = {np.mean(ratios):.3f}")
    ax.set_title("Val/Train Instance Ratio per Class", fontsize=14)
    ax.set_ylabel("Ratio (Val / Train)")
    ax.set_xticklabels(BDD_DETECTION_CLASSES, rotation=30, ha="right")
    ax.legend()
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{ratio:.3f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, output_dir, "val_train_ratio.png")

    return summary


# ---------------------------------------------------------------------------
# 3. Bounding Box Statistics
# ---------------------------------------------------------------------------


def plot_bbox_statistics(
    dataset: BDDDataset,
    split: str = "train",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Plot bounding box size, area, and aspect ratio distributions per class.

    Args:
        dataset: Loaded BDDDataset object.
        split: 'train' or 'val'.
        output_dir: Directory to save plots.

    Returns:
        DataFrame with per-class bbox statistics (mean/std of W, H, Area, AR).
    """
    bbox_stats = dataset.get_bbox_stats(split)
    rows = []

    for cls in BDD_DETECTION_CLASSES:
        data = bbox_stats[cls]
        if len(data) == 0:
            continue
        rows.append({
            "class": cls,
            "count": len(data),
            "mean_width": data[:, 0].mean(),
            "mean_height": data[:, 1].mean(),
            "mean_area": data[:, 2].mean(),
            "mean_aspect_ratio": data[:, 3].mean(),
            "std_width": data[:, 0].std(),
            "std_height": data[:, 1].std(),
        })

    df = pd.DataFrame(rows)

    # 4-subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"BDD100K Bounding Box Statistics — {split.capitalize()} Split", fontsize=15)

    # Mean Width
    axes[0, 0].barh(df["class"], df["mean_width"], color=[CLASS_COLORS[c] for c in df["class"]])
    axes[0, 0].set_title("Mean Bounding Box Width (px)")
    axes[0, 0].set_xlabel("Pixels")

    # Mean Height
    axes[0, 1].barh(df["class"], df["mean_height"], color=[CLASS_COLORS[c] for c in df["class"]])
    axes[0, 1].set_title("Mean Bounding Box Height (px)")
    axes[0, 1].set_xlabel("Pixels")

    # Mean Area (log scale)
    axes[1, 0].barh(df["class"], df["mean_area"], color=[CLASS_COLORS[c] for c in df["class"]])
    axes[1, 0].set_title("Mean Bounding Box Area (px²)")
    axes[1, 0].set_xlabel("Pixels²")
    axes[1, 0].set_xscale("log")

    # Mean Aspect Ratio
    axes[1, 1].barh(df["class"], df["mean_aspect_ratio"], color=[CLASS_COLORS[c] for c in df["class"]])
    axes[1, 1].axvline(x=1.0, color="red", linestyle="--", label="AR = 1 (square)")
    axes[1, 1].set_title("Mean Aspect Ratio (W/H)")
    axes[1, 1].set_xlabel("Width / Height")
    axes[1, 1].legend()

    plt.tight_layout()
    _save_or_show(fig, output_dir, f"bbox_statistics_{split}.png")
    return df


def plot_bbox_size_distributions(
    dataset: BDDDataset,
    split: str = "train",
    output_dir: Optional[str] = None,
) -> None:
    """Plot violin plots of bounding box area distribution per class.

    Args:
        dataset: Loaded BDDDataset object.
        split: 'train' or 'val'.
        output_dir: Directory to save plots.
    """
    bbox_stats = dataset.get_bbox_stats(split)

    all_areas, all_classes = [], []
    for cls in BDD_DETECTION_CLASSES:
        data = bbox_stats[cls]
        if len(data) > 0:
            # Cap to top 99th percentile to remove extreme outliers for visualisation
            areas = data[:, 2]
            cap = np.percentile(areas, 99)
            areas = np.minimum(areas, cap)
            all_areas.extend(areas.tolist())
            all_classes.extend([cls] * len(areas))

    df_area = pd.DataFrame({"class": all_classes, "area": all_areas})

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.violinplot(
        data=df_area,
        x="class",
        y="area",
        palette=CLASS_COLORS,
        ax=ax,
        inner="quartile",
    )
    ax.set_title(f"BDD100K Bounding Box Area Distribution — {split.capitalize()} Split", fontsize=14)
    ax.set_xlabel("Object Class")
    ax.set_ylabel("Bounding Box Area (px²)")
    ax.set_xticklabels(BDD_DETECTION_CLASSES, rotation=30, ha="right")
    plt.tight_layout()
    _save_or_show(fig, output_dir, f"bbox_area_violin_{split}.png")


# ---------------------------------------------------------------------------
# 4. Anomaly Detection
# ---------------------------------------------------------------------------


def detect_anomalies(
    dataset: BDDDataset,
    split: str = "train",
) -> Dict[str, List[BDDFrame]]:
    """Detect anomalous frames and labels in the dataset.

    Anomalies detected:
    - Empty frames (no labels)
    - Frames with extremely high object count (crowded scenes)
    - Labels with extreme aspect ratios (< 0.1 or > 20)
    - Labels with very small area (< 100 px²)
    - Truncated and occluded labels

    Args:
        dataset: Loaded BDDDataset object.
        split: 'train' or 'val'.

    Returns:
        Dict mapping anomaly type to list of affected BDDFrame objects.
    """
    frames = dataset.train_frames if split == "train" else dataset.val_frames
    anomalies: Dict[str, List[BDDFrame]] = {
        "empty_frames": [],
        "crowded_frames": [],
        "extreme_aspect_ratio": [],
        "tiny_objects": [],
        "truncated_objects": [],
        "occluded_objects": [],
    }

    for frame in tqdm(frames, desc=f"Scanning {split} for anomalies"):
        if frame.is_empty:
            anomalies["empty_frames"].append(frame)
            continue

        if frame.num_objects > 50:
            anomalies["crowded_frames"].append(frame)

        for lbl in frame.labels:
            ar = lbl.aspect_ratio
            if ar < 0.1 or ar > 20:
                anomalies["extreme_aspect_ratio"].append(frame)
                break

        for lbl in frame.labels:
            if lbl.area < 100:
                anomalies["tiny_objects"].append(frame)
                break

        for lbl in frame.labels:
            if lbl.is_truncated:
                anomalies["truncated_objects"].append(frame)
                break

        for lbl in frame.labels:
            if lbl.is_occluded:
                anomalies["occluded_objects"].append(frame)
                break

    print(f"\n[Anomaly Detection — {split} split]")
    for atype, flist in anomalies.items():
        print(f"  {atype:<28}: {len(flist):>6,} frames")

    return anomalies


# ---------------------------------------------------------------------------
# 5. Scene / Weather / Time-of-Day Analysis
# ---------------------------------------------------------------------------


def plot_attribute_distributions(
    dataset: BDDDataset,
    output_dir: Optional[str] = None,
) -> None:
    """Plot frame-level attribute distributions (weather, scene, time of day).

    Args:
        dataset: Loaded BDDDataset object.
        output_dir: Directory to save plots.
    """
    attributes = ["weather", "scene", "timeofday"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("BDD100K — Frame Attribute Distributions (Train Split)", fontsize=14)

    for ax, attr in zip(axes, attributes):
        dist = dataset.get_attribute_distribution(attr, "train")
        labels_ = list(dist.keys())
        values = list(dist.values())
        ax.pie(values, labels=labels_, autopct="%1.1f%%", startangle=90)
        ax.set_title(attr.replace("timeofday", "Time of Day").title())

    plt.tight_layout()
    _save_or_show(fig, output_dir, "attribute_distributions.png")


# ---------------------------------------------------------------------------
# 6. Interesting / Unique Sample Identification
# ---------------------------------------------------------------------------


def find_interesting_samples(
    dataset: BDDDataset,
    split: str = "val",
    top_n: int = 5,
) -> Dict[str, List[BDDFrame]]:
    """Identify visually interesting or unique samples in the dataset.

    Categories:
    - Most crowded: frames with the highest object count
    - Rare class frames: frames containing train / motorcycle / rider
    - Night scenes: time-of-day == 'night'
    - Rainy scenes: weather == 'rainy'

    Args:
        dataset: Loaded BDDDataset object.
        split: 'train' or 'val'.
        top_n: Number of samples to return per category.

    Returns:
        Dict mapping category name to list of BDDFrame objects.
    """
    frames = dataset.train_frames if split == "train" else dataset.val_frames

    # Most crowded
    sorted_by_crowd = sorted(frames, key=lambda f: f.num_objects, reverse=True)

    # Rare class frames (train vehicle)
    rare_train = [f for f in frames if any(l.category == "train" for l in f.labels)]

    # Night scenes
    night_frames = [f for f in frames if f.time_of_day == "night"]

    # Rainy scenes
    rainy_frames = [f for f in frames if f.weather == "rainy"]

    interesting = {
        "most_crowded": sorted_by_crowd[:top_n],
        "rare_train_class": rare_train[:top_n],
        "night_scenes": night_frames[:top_n],
        "rainy_scenes": rainy_frames[:top_n],
    }

    print(f"\n[Interesting Samples — {split} split]")
    for cat, sample_list in interesting.items():
        print(f"  {cat:<22}: {len(sample_list)} samples")

    return interesting


# ---------------------------------------------------------------------------
# 7. Full Analysis Pipeline
# ---------------------------------------------------------------------------


def run_full_analysis(
    dataset: BDDDataset,
    output_dir: str = "results/analysis",
) -> None:
    """Run the complete EDA pipeline and save all plots to output_dir.

    Args:
        dataset: Loaded BDDDataset object.
        output_dir: Directory where all plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n>>> Step 1/5: Class Distribution Analysis")
    plot_class_distribution(dataset, output_dir)

    print("\n>>> Step 2/5: Train/Val Split Analysis")
    analyze_train_val_split(dataset, output_dir)

    print("\n>>> Step 3/5: Bounding Box Statistics")
    plot_bbox_statistics(dataset, "train", output_dir)
    plot_bbox_size_distributions(dataset, "train", output_dir)

    print("\n>>> Step 4/5: Frame Attribute Distributions")
    plot_attribute_distributions(dataset, output_dir)

    print("\n>>> Step 5/5: Anomaly Detection")
    detect_anomalies(dataset, "train")
    detect_anomalies(dataset, "val")

    print(f"\n[Done] All analysis plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save_or_show(fig: plt.Figure, output_dir: Optional[str], filename: str) -> None:
    """Save figure to file or display it if no output_dir is given.

    Args:
        fig: Matplotlib figure to save or show.
        output_dir: Directory to save to; if None the figure is shown.
        filename: Output filename (e.g., 'class_distribution.png').
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()
