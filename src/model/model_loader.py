"""
YOLOv8 Model Loader and Inference Engine for BDD100K Object Detection.

This module wraps the Ultralytics YOLOv8 model for inference on BDD100K
validation images, with BDD100K class remapping and result serialization.

Model Choice Rationale:
    YOLOv8n (nano) is selected as the inference backbone because:
    - State-of-the-art single-stage detector (2023, Ultralytics)
    - Real-time capable: >100 FPS on a modern GPU
    - BDD100K's 10 classes map cleanly onto COCO classes (pre-trained weights)
    - Decoupled head (separate cls/reg) improves accuracy over YOLOv5
    - Anchor-free with DFL loss for robust small-object detection
    - Native mAP evaluation, export (ONNX, TensorRT) out of the box

Architecture Summary:
    Backbone : CSPDarknet with C2f (Cross-Stage Partial + bottleneck fusion)
    Neck     : PANet (Path Aggregation Network) — multi-scale feature pyramid
    Head     : Decoupled detection head at 3 scales (P3/8, P4/16, P5/32)
    Loss     : BCE (classification) + CIoU (regression) + DFL (distribution)
    Input    : 640×640 default; any size supported with auto-padding
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# BDD100K classes and their mapping to COCO indices
BDD_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# COCO class indices that correspond to BDD100K classes
# Used to filter YOLOv8 COCO predictions to BDD-relevant classes
COCO_TO_BDD_MAP: Dict[int, str] = {
    0: "pedestrian",    # COCO: person
    1: "bicycle",       # COCO: bicycle
    2: "car",           # COCO: car
    3: "motorcycle",    # COCO: motorcycle
    5: "bus",           # COCO: bus
    6: "train",         # COCO: train
    7: "truck",         # COCO: truck
    9: "traffic light", # COCO: traffic light
    11: "stop sign",    # COCO: stop sign → traffic sign (partial)
}

BDD_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(BDD_CLASSES)}


class BDDDetection:
    """Represents a single detected object from model inference.

    Attributes:
        image_name: Filename of the source image.
        category: Predicted class name.
        class_id: Integer class index in BDD_CLASSES.
        confidence: Detection confidence score [0, 1].
        x1: Left bounding box coordinate.
        y1: Top bounding box coordinate.
        x2: Right bounding box coordinate.
        y2: Bottom bounding box coordinate.
    """

    def __init__(
        self,
        image_name: str,
        category: str,
        class_id: int,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        """Initialize a BDDDetection instance."""
        self.image_name = image_name
        self.category = category
        self.class_id = class_id
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        """Bounding box area in pixels²."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, object]:
        """Serialize detection to a plain dictionary."""
        return {
            "image_name": self.image_name,
            "category": self.category,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


class YOLOv8Detector:
    """YOLOv8-based object detector for BDD100K images.

    Wraps the Ultralytics YOLOv8 model and maps COCO predictions to
    BDD100K class space. Supports batch inference with configurable
    confidence and IoU NMS thresholds.

    Usage::

        detector = YOLOv8Detector(model_name="yolov8n.pt")
        detector.load()
        detections = detector.predict_image("path/to/image.jpg")

    Attributes:
        model_name: Ultralytics model identifier or path to .pt weights.
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        device: Torch device string ('cpu', 'cuda', 'mps').
        model: Loaded Ultralytics YOLO model object.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """Initialize the YOLOv8 detector.

        Args:
            model_name: YOLOv8 model variant ('yolov8n.pt', 'yolov8s.pt',
                        'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt') or
                        path to fine-tuned .pt checkpoint.
            conf_threshold: Confidence threshold for detections [0, 1].
            iou_threshold: NMS IoU threshold [0, 1].
            device: Inference device ('cpu', 'cuda:0', 'mps').
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None

    def load(self) -> None:
        """Download (if needed) and load the YOLOv8 model weights.

        Raises:
            ImportError: If ultralytics is not installed.
            FileNotFoundError: If a local checkpoint path does not exist.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            ) from exc

        print(f"[YOLOv8Detector] Loading model: {self.model_name} on {self.device}")
        self.model = YOLO(self.model_name)
        print("[YOLOv8Detector] Model loaded successfully.")
        print(f"[YOLOv8Detector] Model type: {type(self.model.model).__name__}")

    def predict_image(
        self,
        image_path: str,
        return_annotated: bool = False,
    ) -> Tuple[List[BDDDetection], Optional[np.ndarray]]:
        """Run inference on a single image and return BDD-mapped detections.

        Args:
            image_path: Absolute or relative path to the input image.
            return_annotated: If True, also return the annotated BGR image array.

        Returns:
            Tuple of (list of BDDDetection, annotated image or None).

        Raises:
            RuntimeError: If model has not been loaded via load().
            FileNotFoundError: If image_path does not exist.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        image_name = Path(image_path).name
        detections = self._parse_results(results[0], image_name)
        annotated = results[0].plot() if return_annotated else None
        return detections, annotated

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16,
    ) -> Dict[str, List[BDDDetection]]:
        """Run inference on a batch of images.

        Args:
            image_paths: List of image file paths.
            batch_size: Number of images per inference batch.

        Returns:
            Dict mapping image filename to list of BDDDetection objects.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        all_detections: Dict[str, List[BDDDetection]] = {}

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i: i + batch_size]
            results = self.model.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=True,
            )
            for result in results:
                image_name = Path(result.path).name
                all_detections[image_name] = self._parse_results(result, image_name)

            print(
                f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images",
                end="\r",
            )

        print()
        return all_detections

    def _parse_results(self, result, image_name: str) -> List[BDDDetection]:
        """Convert Ultralytics Result object to list of BDDDetection objects.

        Filters out COCO classes that have no BDD100K equivalent and maps
        the remaining classes to BDD class names.

        Args:
            result: Ultralytics Result object from model.predict().
            image_name: Filename of the source image.

        Returns:
            List of BDDDetection objects.
        """
        detections: List[BDDDetection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            if cls_id not in COCO_TO_BDD_MAP:
                continue
            bdd_class = COCO_TO_BDD_MAP[cls_id]
            if bdd_class not in BDD_CLASS_TO_IDX:
                continue
            detections.append(
                BDDDetection(
                    image_name=image_name,
                    category=bdd_class,
                    class_id=BDD_CLASS_TO_IDX[bdd_class],
                    confidence=float(conf),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[BDDDetection],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on an image.

        Args:
            image: BGR image array (H, W, 3).
            detections: List of BDDDetection objects to draw.
            color_map: Optional dict mapping class name to BGR colour tuple.

        Returns:
            Annotated BGR image array.
        """
        if color_map is None:
            np.random.seed(42)
            color_map = {
                cls: tuple(int(c) for c in np.random.randint(50, 230, 3))
                for cls in BDD_CLASSES
            }

        annotated = image.copy()
        for det in detections:
            color = color_map.get(det.category, (0, 255, 0))
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.category} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        return annotated
