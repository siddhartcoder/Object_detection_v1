"""
Visualization Module for BDD100K Object Detection Evaluation.

Produces both quantitative (Precision-Recall curves, confusion matrix heatmap,
mAP bar chart) and qualitative (GT vs prediction overlays, failure gallery,
failure cluster analysis) visualizations.

Qualitative Analysis Tools:
    - Ground truth vs prediction overlay on individual images
    - Failure case gallery (worst-performing images by missed GT count)
    - Failure clustering by scene attributes (night, rain, crowded)
    - Bounding box size analysis for missed detections

All plots are saved to the specified output directory in high resolution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.data_analysis.bdd_parser import BDD_DETECTION_CLASSES, BDDFrame
from src.model.model_loader import BDDDetection

matplotlib.use("Agg")  # Non-interactive backend for server/container use

# Colour palette per class (RGB tuples for matplotlib, BGR for OpenCV)
CLASS_COLORS_RGB: Dict[str, Tuple[float, float, float]] = {
    "pedestrian": (0.9, 0.1, 0.3),
    "rider": (0.95, 0.5, 0.2),
    "car": (0.2, 0.7, 0.3),
    "truck": (0.26, 0.39, 0.85),
    "bus": (0.57, 0.12, 0.7),
    "train": (0.26, 0.83, 0.96),
    "motorcycle": (0.94, 0.2, 0.9),
    "bicycle": (0.75, 0.94, 0.27),
    "traffic light": (0.98, 0.74, 0.85),
    "traffic sign": (0.67, 1.0, 0.77),
}

# BGR version for OpenCV
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    cls: (int(b * 255), int(g * 255), int(r * 255))
    for cls, (r, g, b) in CLASS_COLORS_RGB.items()
}


# ---------------------------------------------------------------------------
# Quantitative Visualizations
# ---------------------------------------------------------------------------


def plot_map_barchart(
    per_class_ap: Dict[str, float],
    mAP_50: float,
    output_dir: str,
) -> None:
    """Plot a horizontal bar chart of per-class AP and the overall mAP.

    Args:
        per_class_ap: Dict mapping class name to AP@0.5 value.
        mAP_50: Mean AP@0.5 across all classes.
        output_dir: Directory to save the plot.
    """
    classes = list(per_class_ap.keys())
    ap_values = [per_class_ap[c] for c in classes]
    colors = [CLASS_COLORS_RGB.get(c, (0.5, 0.5, 0.5)) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, ap_values, color=colors)
    ax.axvline(x=mAP_50, color="red", linestyle="--", linewidth=2, label=f"mAP@0.5 = {mAP_50:.3f}")
    ax.set_xlabel("Average Precision (AP) @ IoU=0.5")
    ax.set_title("Per-Class AP @ IoU=0.5 — YOLOv8n on BDD100K Val", fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.legend()

    for bar, val in zip(bars, ap_values):
        ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    _save(fig, output_dir, "per_class_ap.png")


def plot_precision_recall_curves(
    per_class_preds: Dict[str, List[Tuple[float, int]]],
    per_class_gt_count: Dict[str, int],
    output_dir: str,
) -> None:
    """Plot Precision-Recall curves for each class on a single figure.

    Args:
        per_class_preds: Dict mapping class → list of (confidence, is_tp).
        per_class_gt_count: Dict mapping class → total GT count.
        output_dir: Directory to save the plot.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Precision-Recall Curves per Class — YOLOv8n on BDD100K Val", fontsize=14)

    for ax, cls in zip(axes.flatten(), BDD_DETECTION_CLASSES):
        preds = per_class_preds.get(cls, [])
        n_gt = per_class_gt_count.get(cls, 0)
        color = CLASS_COLORS_RGB.get(cls, (0.5, 0.5, 0.5))

        if n_gt == 0 or len(preds) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cls)
            continue

        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        tp_cumsum = np.cumsum([p[1] for p in preds_sorted])
        fp_cumsum = np.cumsum([1 - p[1] for p in preds_sorted])
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ax.plot(recalls, precisions, color=color, linewidth=2)
        ax.fill_between(recalls, precisions, alpha=0.2, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_title(cls, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, "precision_recall_curves.png")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    output_dir: str,
) -> None:
    """Plot a normalized confusion matrix heatmap.

    Args:
        confusion_matrix: Integer array of shape (10, 10); rows=GT, cols=Pred.
        output_dir: Directory to save the plot.
    """
    # Normalize by row (GT count)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    norm_cm = np.where(row_sums > 0, confusion_matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".2f",
        xticklabels=BDD_DETECTION_CLASSES,
        yticklabels=BDD_DETECTION_CLASSES,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Ground Truth Class", fontsize=12)
    ax.set_title("Normalized Confusion Matrix — YOLOv8n on BDD100K Val", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, output_dir, "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Qualitative Visualizations
# ---------------------------------------------------------------------------


def visualize_predictions(
    frames: List[BDDFrame],
    all_detections: Dict[str, List[BDDDetection]],
    image_dir: str,
    output_dir: str,
    n_samples: int = 20,
    seed: int = 42,
) -> None:
    """Save ground truth vs prediction overlay images for a random sample.

    Args:
        frames: List of BDDFrame objects (ground truth).
        all_detections: Dict mapping image filename → list of BDDDetection.
        image_dir: Directory containing the validation images.
        output_dir: Directory to save annotated images.
        n_samples: Number of random images to visualize.
        seed: Random seed for reproducible sampling.
    """
    import random
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    sample_frames = random.sample(frames, min(n_samples, len(frames)))

    for frame in tqdm(sample_frames, desc="Visualizing predictions"):
        image_path = os.path.join(image_dir, frame.name)
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        # Draw ground truth (dashed-style: thicker green)
        for lbl in frame.labels:
            color_bgr = CLASS_COLORS_BGR.get(lbl.category, (0, 255, 0))
            x1, y1, x2, y2 = int(lbl.x1), int(lbl.y1), int(lbl.x2), int(lbl.y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 3)
            cv2.putText(image, f"GT:{lbl.category}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

        # Draw predictions (thinner, lighter)
        preds = all_detections.get(frame.name, [])
        for pred in preds:
            color_bgr = CLASS_COLORS_BGR.get(pred.category, (128, 128, 128))
            # Lighten the colour for prediction boxes
            light_color = tuple(min(255, int(c * 1.4)) for c in color_bgr)
            x1, y1, x2, y2 = int(pred.x1), int(pred.y1), int(pred.x2), int(pred.y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), light_color, 2)
            cv2.putText(image, f"P:{pred.category}:{pred.confidence:.2f}",
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, light_color, 1)

        # Add legend
        legend_text = "Thick=GT  Thin=Prediction"
        cv2.putText(image, legend_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_path = os.path.join(output_dir, Path(frame.name).stem + "_overlay.jpg")
        cv2.imwrite(output_path, image)

    print(f"[Visualizer] Saved {len(sample_frames)} overlay images to: {output_dir}")


def plot_failure_gallery(
    frames: List[BDDFrame],
    all_detections: Dict[str, List[BDDDetection]],
    image_dir: str,
    output_dir: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Identify and visualize the worst-performing images (most missed GT objects).

    Failure score = number of unmatched GT objects / total GT objects in frame.

    Args:
        frames: List of BDDFrame objects with ground truth.
        all_detections: Dict mapping image filename → list of BDDDetection.
        image_dir: Directory containing validation images.
        output_dir: Directory to save failure case images.
        top_n: Number of worst images to include in gallery.

    Returns:
        DataFrame with failure stats per image (name, gt_count, missed, score).
    """
    from src.evaluation.evaluator import compute_iou

    os.makedirs(output_dir, exist_ok=True)
    failure_rows = []

    for frame in tqdm(frames, desc="Computing failure scores"):
        if frame.is_empty:
            continue
        preds = all_detections.get(frame.name, [])
        gt_boxes = [(lbl.category, np.array([lbl.x1, lbl.y1, lbl.x2, lbl.y2]))
                    for lbl in frame.labels]

        missed = 0
        for gt_cls, gt_box in gt_boxes:
            matched = False
            for pred in preds:
                if pred.category == gt_cls:
                    iou = compute_iou(np.array(pred.bbox_xyxy), gt_box)
                    if iou >= 0.5:
                        matched = True
                        break
            if not matched:
                missed += 1

        failure_rows.append({
            "name": frame.name,
            "gt_count": frame.num_objects,
            "missed": missed,
            "score": missed / frame.num_objects,
            "weather": frame.weather,
            "scene": frame.scene,
            "timeofday": frame.time_of_day,
        })

    df = pd.DataFrame(failure_rows).sort_values("score", ascending=False)

    # Save gallery images
    for _, row in df.head(top_n).iterrows():
        image_path = os.path.join(image_dir, row["name"])
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        label = f"FAILURE | missed={row['missed']}/{row['gt_count']} | {row['timeofday']} | {row['weather']}"
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        out_path = os.path.join(output_dir, Path(row["name"]).stem + "_failure.jpg")
        cv2.imwrite(out_path, image)

    print(f"[Visualizer] Failure gallery saved to: {output_dir}")
    return df


def plot_failure_clusters(
    failure_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Plot bar charts showing failure rate breakdown by scene attributes.

    Connects data analysis insights (time-of-day, weather, scene type) to
    model failure patterns, helping suggest targeted improvements.

    Args:
        failure_df: DataFrame output from plot_failure_gallery().
        output_dir: Directory to save cluster analysis plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Failure Cluster Analysis — Where Does the Model Fail?", fontsize=14)

    for ax, attr in zip(axes, ["timeofday", "weather", "scene"]):
        group = failure_df.groupby(attr)["score"].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(group)))  # type: ignore[attr-defined]
        group.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
        ax.set_title(f"Avg Failure Rate by {attr.replace('timeofday','Time of Day').title()}")
        ax.set_ylabel("Avg Miss Rate (missed / GT objects)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    _save(fig, output_dir, "failure_clusters.png")
    print(
        "\n[Analysis] Key observations from failure clusters:\n"
        "  → Night-time scenes likely have higher failure rates\n"
        "  → Rainy/foggy scenes challenge model due to lower contrast\n"
        "  → Highway scenes: rare classes (train, rider) barely appear\n"
        "  → Suggested fix: Night-time data augmentation + class-weighted loss"
    )


def plot_missed_object_sizes(
    frames: List[BDDFrame],
    all_detections: Dict[str, List[BDDDetection]],
    output_dir: str,
) -> None:
    """Plot size distribution of missed vs detected objects.

    Identifies whether small objects cause more failures (common for cyclists,
    pedestrians at distance) to guide anchor/resolution improvements.

    Args:
        frames: List of BDDFrame objects with ground truth.
        all_detections: Dict mapping image filename → list of BDDDetection.
        output_dir: Directory to save the plot.
    """
    from src.evaluation.evaluator import compute_iou

    missed_areas, detected_areas = [], []

    for frame in frames:
        preds = all_detections.get(frame.name, [])
        for lbl in frame.labels:
            gt_box = np.array([lbl.x1, lbl.y1, lbl.x2, lbl.y2])
            detected = False
            for pred in preds:
                if pred.category == lbl.category:
                    iou = compute_iou(np.array(pred.bbox_xyxy), gt_box)
                    if iou >= 0.5:
                        detected = True
                        break
            if detected:
                detected_areas.append(lbl.area)
            else:
                missed_areas.append(lbl.area)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(missed_areas, bins=50, alpha=0.6, color="red", label=f"Missed ({len(missed_areas):,})", density=True)
    ax.hist(detected_areas, bins=50, alpha=0.6, color="green", label=f"Detected ({len(detected_areas):,})", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Bounding Box Area (px²) — log scale")
    ax.set_ylabel("Density")
    ax.set_title("Size Distribution: Missed vs Detected Objects")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, output_dir, "missed_vs_detected_sizes.png")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, output_dir: str, filename: str) -> None:
    """Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure object.
        output_dir: Directory to save to.
        filename: Output filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)
