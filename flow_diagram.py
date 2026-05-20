"""
Flow Diagram Generator — BDD100K Object Detection Pipeline.

Generates a visually clean end-to-end flow diagram for the
Bosch Applied CV assignment that can be used in interviews.

Run:
    python flow_diagram.py
    → Saves: Sid_Work/flow_diagram.png  (high-resolution, print-ready)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",   # dark background
    "header":   "#1f6feb",   # blue  — section headers
    "task1":    "#388bfd",   # blue  — data analysis
    "task2":    "#3fb950",   # green — model
    "task3":    "#d29922",   # amber — evaluation
    "output":   "#bc4c00",   # orange — outputs
    "arrow":    "#8b949e",   # gray arrows
    "text_hdr": "#ffffff",   # white text on dark boxes
    "text_sub": "#c9d1d9",   # light gray sub-text
    "border":   "#30363d",   # box borders
    "docker":   "#2ea043",   # docker green
}

FIG_W, FIG_H = 24, 16


# ── Helper: draw a rounded box ───────────────────────────────────────────────
def box(ax, x, y, w, h, text, subtext="", color="#1f6feb", text_color="#ffffff",
        fontsize=11, subfontsize=8.5, radius=0.3, alpha=0.95, bold=True):
    """Draw a rounded rectangle with main label and optional sub-label."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor="#ffffff", linewidth=0.8, alpha=alpha,
        zorder=3,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.12 if subtext else 0), text,
            ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=weight, zorder=4)
    if subtext:
        ax.text(x, y - 0.22, subtext,
                ha="center", va="center", color="#e0e0e0",
                fontsize=subfontsize, fontstyle="italic", zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#8b949e", lw=1.8):
    """Draw a clean arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    mutation_scale=16,
                ),
                zorder=5)


def section_label(ax, x, y, text, color):
    """Draw a large section number / label."""
    ax.text(x, y, text, ha="center", va="center",
            color=color, fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                      edgecolor="none", alpha=0.15),
            zorder=6)


# ── Main diagram ─────────────────────────────────────────────────────────────
def draw_flow():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ───────────────────────────────────────────────────────────────
    ax.text(12, 15.3, "BDD100K Object Detection — End-to-End Pipeline",
            ha="center", va="center", color="#ffffff",
            fontsize=18, fontweight="bold")
    ax.text(12, 14.85, "Bosch Applied CV Assignment  |  YOLOv8n + Custom DataLoader + Plotly Dashboard",
            ha="center", va="center", color=C["text_sub"], fontsize=10)

    # divider line
    ax.plot([0.5, 23.5], [14.55, 14.55], color=C["border"], lw=1)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0 — Dataset
    # ══════════════════════════════════════════════════════════════════════════
    box(ax, 12, 13.7, 7, 0.9,
        "BDD100K Dataset",
        "100k Images (5.3 GB) + Labels (107 MB)  |  70k Train / 10k Val",
        color="#21262d", text_color="#79c0ff", fontsize=12)

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 1 — Data Analysis  (left column)
    # ══════════════════════════════════════════════════════════════════════════
    section_label(ax, 3.5, 13.0, "TASK 1 — Data Analysis  (10 pts)", C["task1"])

    # Parse box
    box(ax, 3.5, 12.1, 5.8, 0.8,
        "JSON Parser  (bdd_parser.py)",
        "BDDDataset  ·  BDDFrame  ·  BDDLabel  dataclasses",
        color="#1c2a3a", text_color="#79c0ff", fontsize=10)

    # Analysis boxes — 3 side by side
    for i, (lbl, sub) in enumerate([
        ("Class Distribution", "Bar + Pie charts\nTrain vs Val"),
        ("BBox Statistics", "Width / Height / Area\nAspect ratio violin"),
        ("Anomaly Detection", "Empty frames\nOccluded / Truncated"),
    ]):
        bx = 1.6 + i * 2.0
        box(ax, bx, 10.85, 1.75, 0.95, lbl, sub,
            color="#162032", text_color=C["task1"],
            fontsize=8.5, subfontsize=7, bold=False)
        arrow(ax, bx, 11.7, bx, 11.32, color=C["task1"])

    # Attribute box
    box(ax, 3.5, 9.65, 5.8, 0.8,
        "Frame Attributes",
        "Weather  ·  Scene  ·  Time-of-Day  distributions",
        color="#162032", text_color=C["task1"], fontsize=10, subfontsize=8)

    # Interesting samples
    box(ax, 3.5, 8.65, 5.8, 0.8,
        "Interesting Sample Identification",
        "Crowded scenes  ·  Rare classes  ·  Night  ·  Rainy",
        color="#162032", text_color=C["task1"], fontsize=10, subfontsize=8)

    # Dashboard
    box(ax, 3.5, 7.45, 5.8, 0.9,
        "Interactive Dashboard  (dashboard.py)",
        "Plotly Dash  ·  4 Tabs  ·  localhost:8050",
        color="#0d419d", text_color="#ffffff", fontsize=10)

    # Docker callout
    box(ax, 3.5, 6.35, 4.5, 0.7,
        "Docker Container  (data-only)",
        "Self-contained  ·  No host installs needed",
        color="#1a3a1a", text_color=C["docker"], fontsize=9, subfontsize=7.5)

    # Task 1 arrows (vertical chain)
    for y1, y2 in [(13.25, 12.5), (11.7, 11.32), (11.38, 10.05), (10.25, 9.05), (9.25, 8.05), (8.25, 7.9), (7.0, 6.7)]:
        arrow(ax, 3.5, y1, 3.5, y2, color=C["task1"])

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 2 — Model  (centre column)
    # ══════════════════════════════════════════════════════════════════════════
    section_label(ax, 12, 13.0, "TASK 2 — Model  (5 + 5 pts)", C["task2"])

    # Model choice box
    box(ax, 12, 12.1, 5.8, 0.85,
        "Model: YOLOv8n  (Ultralytics)",
        "Pre-trained COCO  →  BDD100K class mapping",
        color="#1a2e1a", text_color="#56d364", fontsize=11, subfontsize=8.5)

    # Architecture breakdown
    arch_items = [
        ("Backbone", "CSPDarknet\n+ C2f modules"),
        ("Neck", "PANet\nMulti-scale FPN"),
        ("Head", "Decoupled\nAnchor-free"),
        ("Loss", "BCE + CIoU\n+ DFL"),
    ]
    for i, (lbl, sub) in enumerate(arch_items):
        bx = 9.5 + i * 1.7
        box(ax, bx, 10.85, 1.5, 0.95, lbl, sub,
            color="#0d2818", text_color=C["task2"],
            fontsize=8.5, subfontsize=7, bold=False)
        arrow(ax, bx, 11.7, bx, 11.32, color=C["task2"])

    # Custom DataLoader
    box(ax, 12, 9.65, 5.8, 0.85,
        "Custom DataLoader  (dataloader.py)",
        "BDD100KDetectionDataset  ·  YOLO-format labels  ·  Augmentation",
        color="#0d2818", text_color=C["task2"], fontsize=10, subfontsize=8)

    # Data pipeline steps
    for i, lbl in enumerate(["Parse JSON", "Resize 640×640", "Normalise [0,1]", "collate_fn"]):
        bx = 9.5 + i * 1.7
        box(ax, bx, 8.65, 1.5, 0.7, lbl, "",
            color="#0a1e0a", text_color=C["task2"],
            fontsize=8, bold=False)
        arrow(ax, 9.5 + i * 1.7, 9.22, 9.5 + i * 1.7, 9.0, color=C["task2"])

    # Training pipeline
    box(ax, 12, 7.55, 5.8, 0.85,
        "Training Pipeline  (trainer.py)",
        "BDD JSON → YOLO format → yolo.train()  ·  1 epoch / 1k images demo",
        color="#0d2818", text_color=C["task2"], fontsize=10, subfontsize=8)

    # Saved model
    box(ax, 12, 6.45, 4.0, 0.7,
        "best.pt  checkpoint",
        "Fine-tuned weights saved",
        color="#1a3a1a", text_color=C["docker"], fontsize=9, subfontsize=7.5)

    # Task 2 vertical arrows
    for y1, y2 in [(13.25, 12.52), (11.7, 11.32), (11.38, 10.05), (10.22, 9.05), (9.25, 7.97), (7.12, 6.8)]:
        arrow(ax, 12, y1, 12, y2, color=C["task2"])

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 3 — Evaluation  (right column)
    # ══════════════════════════════════════════════════════════════════════════
    section_label(ax, 20.5, 13.0, "TASK 3 — Evaluation & Viz  (10 pts)", C["task3"])

    # Evaluator
    box(ax, 20.5, 12.1, 5.8, 0.85,
        "Evaluator  (evaluator.py)",
        "IoU matching  ·  TP / FP / FN  per class",
        color="#2d1f00", text_color="#e3b341", fontsize=11, subfontsize=8.5)

    # Quantitative metrics grid
    metrics = [
        ("mAP@0.5", "Primary\nbenchmark"),
        ("Precision", "False alarm\nrate"),
        ("Recall", "Safety-critical\nmiss rate"),
        ("Per-class AP", "Weak class\nidentification"),
    ]
    for i, (lbl, sub) in enumerate(metrics):
        bx = 18.2 + i * 1.6
        box(ax, bx, 10.85, 1.45, 0.95, lbl, sub,
            color="#1e1500", text_color=C["task3"],
            fontsize=8.5, subfontsize=7, bold=False)
        arrow(ax, bx, 11.7, bx, 11.32, color=C["task3"])

    # Confusion matrix
    box(ax, 20.5, 9.65, 5.8, 0.85,
        "Confusion Matrix  (10 × 10)",
        "Class confusion heatmap  ·  rider ↔ pedestrian  ·  car ↔ truck",
        color="#1e1500", text_color=C["task3"], fontsize=10, subfontsize=8)

    # Qualitative section
    box(ax, 20.5, 8.65, 5.8, 0.75,
        "Qualitative Visualization  (visualizer.py)",
        "",
        color="#2d1f00", text_color="#e3b341", fontsize=10)

    qual_items = [
        ("GT vs Pred\nOverlays", "#2d1f00"),
        ("Failure\nGallery", "#2d1f00"),
        ("Failure\nClusters", "#2d1f00"),
        ("Missed Size\nAnalysis", "#2d1f00"),
    ]
    for i, (lbl, col) in enumerate(qual_items):
        bx = 18.2 + i * 1.6
        box(ax, bx, 7.65, 1.45, 0.85, lbl, "",
            color=col, text_color=C["task3"],
            fontsize=7.5, bold=False)
        arrow(ax, bx, 8.27, bx, 8.07, color=C["task3"])

    # Cluster labels
    cluster_labels = ["Night", "Rain", "Crowd", "Small\nObjects"]
    for i, lbl in enumerate(cluster_labels):
        bx = 18.2 + i * 1.6
        box(ax, bx, 6.65, 1.45, 0.75, lbl, "",
            color="#110d00", text_color="#f0a500",
            fontsize=8, bold=False)
        arrow(ax, bx, 7.22, bx, 7.02, color=C["task3"])

    # Suggestions box
    box(ax, 20.5, 5.7, 5.8, 0.85,
        "Improvement Suggestions",
        "Night aug  ·  Focal loss  ·  SAHI  ·  Class-weighted sampling",
        color="#1e1500", text_color=C["task3"], fontsize=9.5, subfontsize=8)

    # Task 3 vertical arrows
    for y1, y2 in [(13.25, 12.52), (11.7, 11.32), (11.38, 10.05), (10.22, 9.05), (9.22, 9.02), (8.27, 8.05), (6.27, 6.12)]:
        arrow(ax, 20.5, y1, 20.5, y2, color=C["task3"])

    # ══════════════════════════════════════════════════════════════════════════
    # Cross-task arrows (dataset → 3 tasks)
    # ══════════════════════════════════════════════════════════════════════════
    # BDD100K → Task1
    ax.annotate("", xy=(3.5, 13.25), xytext=(8.5, 13.7),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5,
                                connectionstyle="arc3,rad=0.1"), zorder=5)
    # BDD100K → Task2
    ax.annotate("", xy=(12, 13.25), xytext=(12, 13.25),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5), zorder=5)
    arrow(ax, 12, 14.25, 12, 13.52, color="#555555")
    # BDD100K → Task3
    ax.annotate("", xy=(20.5, 13.25), xytext=(15.5, 13.7),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5,
                                connectionstyle="arc3,rad=-0.1"), zorder=5)

    # Task1 EDA → Task3 (connect insights)
    ax.annotate("", xy=(17.6, 9.65), xytext=(6.4, 9.65),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2,
                                linestyle="dashed",
                                connectionstyle="arc3,rad=-0.15"), zorder=4)
    ax.text(12, 9.72, "Data insights inform evaluation focus",
            ha="center", va="bottom", color="#555555", fontsize=7.5, style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # Bottom legend
    # ══════════════════════════════════════════════════════════════════════════
    ax.plot([0.5, 23.5], [5.1, 5.1], color=C["border"], lw=1)

    # Legend boxes
    legend_items = [
        (2.5, "Data Analysis", C["task1"]),
        (7.5, "Model & Training", C["task2"]),
        (12.5, "Evaluation & Viz", C["task3"]),
        (17.5, "Docker Container", C["docker"]),
        (21.5, "Output / Deliverable", C["output"]),
    ]
    for lx, lbl, col in legend_items:
        box(ax, lx, 4.65, 3.5, 0.6, lbl, "",
            color="#161b22", text_color=col, fontsize=9, bold=False)
        rect = FancyBboxPatch((lx - 1.75, 4.35), 3.5, 0.6,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor="none", edgecolor=col,
                              linewidth=1.5, zorder=6)
        ax.add_patch(rect)

    # Bottom tech stack
    tech = ("Tech Stack:  Python 3.10  ·  YOLOv8 (Ultralytics)  ·  "
            "PyTorch  ·  OpenCV  ·  Plotly Dash  ·  Seaborn  ·  Docker  ·  PEP8 / black / pylint")
    ax.text(12, 3.9, tech,
            ha="center", va="center", color=C["text_sub"], fontsize=8.5)

    # File map
    file_map = (
        "Key Files:  "
        "bdd_parser.py → data structures  |  "
        "analysis.py → EDA  |  "
        "dashboard.py → Plotly Dash  |  "
        "model_loader.py → YOLOv8  |  "
        "dataloader.py → PyTorch Dataset  |  "
        "trainer.py → fine-tuning  |  "
        "evaluator.py → mAP  |  "
        "visualizer.py → plots  |  "
        "main.py → CLI"
    )
    ax.text(12, 3.5, file_map,
            ha="center", va="center", color="#555555", fontsize=7.2)

    ax.text(12, 3.1,
            "Notebooks:  01_data_analysis.ipynb  ·  02_model.ipynb  ·  03_evaluation.ipynb  "
            "|  Docs: README.md  ·  docs/analysis_report.md",
            ha="center", va="center", color="#555555", fontsize=7.2)

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = "flow_diagram.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    print(f"[flow_diagram] Saved: {out_path}  ({FIG_W*180}×{FIG_H*180} px)")
    plt.show()


if __name__ == "__main__":
    draw_flow()
