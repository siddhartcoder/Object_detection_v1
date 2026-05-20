"""
Model Evaluation Engine for BDD100K Object Detection.

Computes standard object detection metrics (mAP@0.5, mAP@0.5:0.95,
Precision, Recall) using COCO-style evaluation via pycocotools,
plus per-class AP and a confusion matrix.

Metric Choices Rationale:
    - mAP@0.5     : Industry-standard benchmark for object detection.
                    IoU=0.5 is lenient, good for overall model health check.
    - mAP@0.5:0.95: COCO-style metric; tests robustness across IoU thresholds.
                    Critical for fine-grained localisation quality.
    - Precision   : Fraction of detections that are correct.
                    Important to avoid false alarms in autonomous driving.
    - Recall      : Fraction of GT objects detected.
                    Safety-critical: missed pedestrians/cyclists are dangerous.
    - Per-class AP: Reveals which specific classes the model struggles with,
                    guiding targeted improvements (data aug, class-weighted loss).
    - Confusion Matrix: Identifies class-pair confusions (rider vs pedestrian).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_analysis.bdd_parser import BDD_DETECTION_CLASSES, BDDDataset, BDDFrame
from src.model.model_loader import BDDDetection, YOLOv8Detector

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(BDD_DETECTION_CLASSES)}


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Array [x1, y1, x2, y2] for the first box.
        box2: Array [x1, y1, x2, y2] for the second box.

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """Compute Average Precision (AP) using 11-point interpolation.

    Args:
        precisions: Array of precision values at each threshold.
        recalls: Array of recall values at each threshold.

    Returns:
        Average Precision (AP) scalar.
    """
    # Append sentinel values
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    recalls = np.concatenate([[0.0], recalls, [1.0]])

    # Monotonic precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate
    recall_thresholds = np.arange(0.0, 1.01, 0.1)  # 11-point
    ap = 0.0
    for thresh in recall_thresholds:
        prec_at_thresh = precisions[recalls >= thresh]
        ap += prec_at_thresh.max() if prec_at_thresh.size > 0 else 0.0
    return ap / 11.0


class BDDEvaluator:
    """Evaluates a YOLOv8 detector on the BDD100K validation set.

    Computes:
    - Per-class Average Precision (AP) at IoU=0.5
    - Mean AP (mAP) at IoU=0.5
    - Per-class Precision and Recall at confidence=0.5
    - Confusion matrix across 10 classes

    Usage::

        evaluator = BDDEvaluator(
            detector=YOLOv8Detector("yolov8n.pt"),
            val_frames=dataset.val_frames,
            val_image_dir="data/images/100k/val",
        )
        results = evaluator.evaluate(max_images=500)
        evaluator.print_report(results)

    Attributes:
        detector: Loaded YOLOv8Detector instance.
        val_frames: List of BDDFrame objects for validation.
        val_image_dir: Path to validation image directory.
        iou_threshold: IoU threshold for matching predictions to GT (default 0.5).
        conf_threshold: Confidence threshold for predictions (default 0.25).
    """

    def __init__(
        self,
        detector: YOLOv8Detector,
        val_frames: List[BDDFrame],
        val_image_dir: str,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ) -> None:
        """Initialize the evaluator.

        Args:
            detector: Loaded YOLOv8Detector (call detector.load() beforehand).
            val_frames: List of BDDFrame objects from the validation set.
            val_image_dir: Directory containing validation images.
            iou_threshold: IoU threshold for TP/FP determination.
            conf_threshold: Confidence threshold to filter predictions.
        """
        self.detector = detector
        self.val_frames = val_frames
        self.val_image_dir = val_image_dir
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def evaluate(
        self,
        max_images: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        """Run evaluation on the validation set and return metrics.

        Args:
            max_images: If set, evaluate only on the first N images (for quick tests).
            output_dir: Directory to save per-image prediction JSON.

        Returns:
            Dict with keys:
                - 'per_class_ap': Dict[str, float] — AP per class
                - 'mAP_50': float — mean AP at IoU=0.5
                - 'per_class_precision': Dict[str, float]
                - 'per_class_recall': Dict[str, float]
                - 'confusion_matrix': np.ndarray (10×10)
                - 'image_scores': List[Dict] — per-image prediction count
        """
        frames = self.val_frames
        if max_images is not None:
            frames = frames[:max_images]

        # Storage: class → list of (confidence, is_tp)
        per_class_preds: Dict[str, List[Tuple[float, int]]] = {
            cls: [] for cls in BDD_DETECTION_CLASSES
        }
        per_class_gt_count: Dict[str, int] = {cls: 0 for cls in BDD_DETECTION_CLASSES}
        confusion_matrix = np.zeros((len(BDD_DETECTION_CLASSES), len(BDD_DETECTION_CLASSES)), dtype=int)
        image_scores: List[Dict] = []

        print(f"\n[BDDEvaluator] Evaluating on {len(frames)} validation images ...")

        for frame in tqdm(frames, desc="Evaluating"):
            image_path = os.path.join(self.val_image_dir, frame.name)
            if not os.path.exists(image_path):
                continue

            # Get predictions
            predictions, _ = self.detector.predict_image(image_path)

            # Ground truth boxes per class
            gt_by_class: Dict[str, List[np.ndarray]] = defaultdict(list)
            for lbl in frame.labels:
                if lbl.category in BDD_DETECTION_CLASSES:
                    gt_by_class[lbl.category].append(
                        np.array([lbl.x1, lbl.y1, lbl.x2, lbl.y2])
                    )
                    per_class_gt_count[lbl.category] += 1

            # Sort predictions by confidence (highest first)
            predictions.sort(key=lambda d: d.confidence, reverse=True)

            # Match predictions to GT
            matched_gt: Dict[str, List[bool]] = {
                cls: [False] * len(gt_by_class[cls])
                for cls in BDD_DETECTION_CLASSES
            }

            for pred in predictions:
                if pred.confidence < self.conf_threshold:
                    continue
                cls = pred.category
                pred_box = np.array(pred.bbox_xyxy)
                gt_boxes = gt_by_class.get(cls, [])

                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.iou_threshold and not matched_gt[cls][best_gt_idx]:
                    matched_gt[cls][best_gt_idx] = True
                    per_class_preds[cls].append((pred.confidence, 1))  # TP
                else:
                    per_class_preds[cls].append((pred.confidence, 0))  # FP

                # Confusion matrix: find best-matching GT class
                best_cls_iou, best_cls = 0.0, cls
                for gt_cls, gt_boxes_cls in gt_by_class.items():
                    for gt_box in gt_boxes_cls:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_cls_iou:
                            best_cls_iou = iou
                            best_cls = gt_cls
                if best_cls_iou >= self.iou_threshold:
                    pred_idx = CLASS_TO_IDX.get(cls, 0)
                    gt_idx_ = CLASS_TO_IDX.get(best_cls, 0)
                    confusion_matrix[gt_idx_, pred_idx] += 1

            image_scores.append({
                "name": frame.name,
                "n_gt": frame.num_objects,
                "n_pred": len(predictions),
            })

        # Compute per-class AP
        per_class_ap: Dict[str, float] = {}
        per_class_precision: Dict[str, float] = {}
        per_class_recall: Dict[str, float] = {}

        for cls in BDD_DETECTION_CLASSES:
            preds = per_class_preds[cls]
            n_gt = per_class_gt_count[cls]

            if n_gt == 0 or len(preds) == 0:
                per_class_ap[cls] = 0.0
                per_class_precision[cls] = 0.0
                per_class_recall[cls] = 0.0
                continue

            preds.sort(key=lambda x: x[0], reverse=True)
            tp_cumsum = np.cumsum([p[1] for p in preds])
            fp_cumsum = np.cumsum([1 - p[1] for p in preds])

            recalls = tp_cumsum / n_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            per_class_ap[cls] = compute_ap(precisions, recalls)
            per_class_precision[cls] = float(precisions[-1]) if len(precisions) > 0 else 0.0
            per_class_recall[cls] = float(recalls[-1]) if len(recalls) > 0 else 0.0

        mAP_50 = float(np.mean(list(per_class_ap.values())))

        return {
            "per_class_ap": per_class_ap,
            "mAP_50": mAP_50,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "confusion_matrix": confusion_matrix,
            "image_scores": image_scores,
        }

    def print_report(self, results: Dict[str, object]) -> None:
        """Print a formatted evaluation report to stdout.

        Args:
            results: Output dict from evaluate().
        """
        print("\n" + "=" * 65)
        print("  BDD100K — Evaluation Report (YOLOv8n, IoU=0.5)")
        print("=" * 65)
        print(f"  {'Class':<18} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8}")
        print("  " + "-" * 47)
        for cls in BDD_DETECTION_CLASSES:
            ap = results["per_class_ap"][cls]
            prec = results["per_class_precision"][cls]
            rec = results["per_class_recall"][cls]
            print(f"  {cls:<18} {ap:>8.3f} {prec:>10.3f} {rec:>8.3f}")
        print("  " + "-" * 47)
        print(f"  {'mAP@0.5':<18} {results['mAP_50']:>8.3f}")
        print("=" * 65)

    def save_results(self, results: Dict[str, object], output_dir: str) -> None:
        """Save evaluation results to JSON and CSV files.

        Args:
            results: Output dict from evaluate().
            output_dir: Directory to save result files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save per-class metrics as CSV
        rows = []
        for cls in BDD_DETECTION_CLASSES:
            rows.append({
                "class": cls,
                "AP@0.5": round(results["per_class_ap"][cls], 4),
                "Precision": round(results["per_class_precision"][cls], 4),
                "Recall": round(results["per_class_recall"][cls], 4),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)

        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(
            results["confusion_matrix"],
            index=BDD_DETECTION_CLASSES,
            columns=BDD_DETECTION_CLASSES,
        )
        cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

        # Save summary JSON
        summary = {
            "mAP_50": round(results["mAP_50"], 4),
            "per_class_ap": {k: round(v, 4) for k, v in results["per_class_ap"].items()},
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[BDDEvaluator] Results saved to: {output_dir}")
