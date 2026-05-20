# Bosch Applied CV Coding Assignment — BDD100K Object Detection

**Version:** 1.1.2 | **Author:** Siddharth | **Dataset:** BDD100K

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Dataset Setup](#dataset-setup)
4. [Task 1 — Data Analysis](#task-1--data-analysis)
5. [Task 2 — Model](#task-2--model)
6. [Task 3 — Evaluation & Visualization](#task-3--evaluation--visualization)
7. [Docker (Data Analysis Container)](#docker-data-analysis-container)
8. [Model & Evaluation — How to Run](#model--evaluation--how-to-run)
9. [Results Summary](#results-summary)

---

## Project Overview

This project provides an end-to-end solution for object detection on the **BDD100K** dataset as part of the Bosch Applied Computer Vision interview assignment. It covers:

- Exploratory Data Analysis (EDA) with class distribution, train/val split analysis, anomaly detection
- Interactive dashboard for dataset statistics
- Object detection model using **YOLOv8** (pre-trained, fine-tuned on BDD100K)
- Custom PyTorch `DataLoader` for BDD100K with training pipeline
- Quantitative (mAP, Precision, Recall) and qualitative (ground truth vs prediction overlays, failure clustering) evaluation

---

## Project Structure
# Object Detection with BDD100K — YOLOv8

```
Assignment/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── main.py
│
├── data/
│   ├── images/
│   │   ├── 100k/
│   │   │   ├── train/               ← 70k training images
│   │   │   └── val/                 ← 10k validation images
│   └── labels/
│       └── det_20/
│           ├── det_train.json       ← Training annotations
│           └── det_val.json         ← Validation annotations
│
├── src/
│   ├── data_analysis/
│   │   ├── __init__.py
│   │   ├── bdd_parser.py
│   │   ├── analysis.py
│   │   └── dashboard.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── dataloader.py
│   │   └── trainer.py
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py
│       └── visualizer.py
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model.ipynb
│   └── 03_evaluation.ipynb
│
└── docs/
    └── analysis_report.md
```

---

## Dataset Setup

1. Download the **BDD100K** dataset from [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/):
   - **100k Images** (~5.3 GB)
   - **Labels** (~107 MB)

2. Extract and place files so the structure matches:
```
   data/
   ├── images/100k/train/   ← 70,000 training images
   ├── images/100k/val/     ← 10,000 validation images
   └── labels/det_20/
       ├── det_train.json
       └── det_val.json
```

---

## Task 1 — Data Analysis

**10 detection classes:** `pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`

### Run via Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run full EDA
python main.py --task analysis \
  --train_json data/labels/det_20/det_train.json \
  --val_json data/labels/det_20/det_val.json \
  --image_dir data/images/100k

# Launch interactive dashboard
python main.py --task dashboard \
  --train_json data/labels/det_20/det_train.json \
  --val_json data/labels/det_20/det_val.json
```

### What it covers

| Analysis | Description |
|----------|-------------|
| Class Distribution | Bar/pie charts of instance counts per class (train & val) |
| Train/Val Split | Image count, label count, class-wise split ratio |
| Bounding Box Stats | Width/height/area distributions per class |
| Aspect Ratio | Histogram of W/H ratios, identifying thin vs wide objects |
| Anomaly Detection | Empty images, truncated boxes, extreme aspect ratios |
| Interesting Samples | Crowded scenes, rare classes, occluded objects |
| Dashboard | Interactive Plotly Dash app (runs at `localhost:8050`) |

### Run via Jupyter Notebook

```bash
jupyter notebook notebooks/01_data_analysis.ipynb
```

---

## Task 2 — Model

### Model Choice: **YOLOv8n (Ultralytics)**

**Why YOLOv8?**
- State-of-the-art single-stage object detector (2023)
- Excellent speed-accuracy tradeoff — real-time capable (30+ FPS on GPU)
- Pre-trained on COCO; can be fine-tuned on BDD100K classes which heavily overlap
- BDD100K's 10 classes all exist in COCO or are closely related
- Native support for mAP evaluation, inference visualization, and export (ONNX, TensorRT)
- Active community, well-documented architecture

**Architecture Summary:**
- Backbone: CSPDarknet (Cross-Stage Partial network) with C2f modules
- Neck: PANet (Path Aggregation Network) for multi-scale feature fusion
- Head: Decoupled head (separate classification + regression branches)
- Anchor-free detection with DFL (Distribution Focal Loss) for regression
- Input: 640×640 (configurable), Output: multi-scale bounding boxes + class probabilities

**Alternatives considered:**
- DETR / RT-DETR: Better accuracy, but much slower; overkill for this task
- Faster R-CNN: Two-stage detector, slower inference; not suited for real-time autonomous driving
- YOLOv5: Predecessor to YOLOv8, inferior performance

### Run Inference

```bash
python main.py --task model \
  --image_dir data/images/100k/val \
  --output_dir results/predictions/
```

### Run Training Pipeline (1 epoch on subset)

```bash
python main.py --task train \
  --train_json data/labels/det_20/det_train.json \
  --image_dir data/images/100k/train \
  --subset_size 1000 --epochs 1
```

### Run via Jupyter Notebook

```bash
jupyter notebook notebooks/02_model.ipynb
```

---

## Task 3 — Evaluation & Visualization

### Metrics Chosen

| Metric | Why Chosen |
|--------|-----------|
| **mAP@0.5** | Standard object detection benchmark metric; IoU=0.5 threshold |
| **mAP@0.5:0.95** | COCO-style stricter metric; tests across multiple IoU thresholds |
| **Precision** | Critical for autonomous driving — false positives cause unnecessary braking |
| **Recall** | Critical safety metric — missing pedestrians/cyclists is dangerous |
| **Per-class AP** | Identifies which object types the model struggles with |
| **Confusion Matrix** | Shows class confusion (e.g., rider vs pedestrian) |

### Run Evaluation

```bash
python main.py --task evaluate \
  --val_json data/labels/det_20/det_val.json \
  --image_dir data/images/100k/val \
  --output_dir results/evaluation/
```

### What it produces
- Per-class AP table
- Precision-Recall curves (per class)
- Confusion matrix heatmap
- Ground truth vs prediction overlays (sample images)
- Failure case gallery (worst-performing images)
- Failure cluster analysis (time-of-day, weather, scene type)

### Run via Jupyter Notebook

```bash
jupyter notebook notebooks/03_evaluation.ipynb
```

---

## Docker (Data Analysis Container)

The data analysis code is fully containerized. The Docker image includes all dependencies — **no additional installations needed**.

### Build the Container

```bash
docker build -t bdd-analysis .
```

### Run the Container

```bash
# Run analysis
docker run --rm \
  -v ./data/labels:/app/data/labels \
  -v ./data/images:/app/data/images \
  -v ./results:/app/results \
  -p 8080:8080 \
  bdd-analysis python main.py --task analysis \
    --train_json data/labels/det_20/det_train.json \
    --val_json data/labels/det_20/det_val.json \
    --image_dir data/images/100k

# Run interactive dashboard (accessible at http://localhost:8050)
docker run --rm \
  -v ./data/labels:/app/data/labels \
  -p 8050:8050 \
  bdd-analysis python main.py --task dashboard \
    --train_json data/labels/det_20/det_train.json \
    --val_json data/labels/det_20/det_val.json
```

### Using docker-compose

```bash
# Edit docker-compose.yml to set BDD100K_DATA_PATH to your data directory
docker-compose up bdd-analysis
# For dashboard:
docker-compose up bdd-dashboard
```

---

## Full Run — All Tasks

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Run data analysis
python main.py --task analysis \
  --train_json data/labels/det_20/det_train.json \
  --val_json data/labels/det_20/det_val.json \
  --image_dir data/images/100k

# 3. Run model inference on validation set
python main.py --task model \
  --val_json data/labels/det_20/det_val.json \
  --image_dir data/images/100k/val \
  --output_dir results/predictions/

# 4. Run training pipeline (subset, 1 epoch)
python main.py --task train \
  --train_json data/labels/det_20/det_train.json \
  --image_dir data/images/100k/train \
  --subset_size 1000 --epochs 1

# 5. Run evaluation & visualization
python main.py --task evaluate \
  --val_json data/labels/det_20/det_val.json \
  --image_dir data/images/100k/val \
  --output_dir results/evaluation/

# 6. Launch interactive dashboard
python main.py --task dashboard \
  --train_json data/labels/det_20/det_train.json \
  --val_json data/labels/det_20/det_val.json
```

**Key Observations (to be filled after running):**
- Class imbalance: `car` dominates; `train` is very rare
- Model strength: Large objects (car, truck, bus) detected reliably
- Model weakness: Small objects (rider, bicycle) at distance; night-time scenes
- Suggested improvements: Data augmentation for rare classes, night-time fine-tuning

---