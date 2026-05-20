"""
BDD100K Dataset Parser and Data Structures.

This module provides classes and functions to parse and manage
the BDD100K dataset for object detection tasks.

Classes:
    BDDFrame: Represents a single annotated image frame.
    BDDLabel: Represents a single object annotation within a frame.
    BDDDataset: Loads and manages the full BDD100K detection dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# The 10 official BDD100K object detection classes
BDD_DETECTION_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(BDD_DETECTION_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


@dataclass
class BDDLabel:
    """Represents a single bounding-box annotation within a BDD100K frame.

    Attributes:
        label_id: Unique identifier of the label within the frame.
        category: Object class name (must be one of BDD_DETECTION_CLASSES).
        box2d: Bounding box coordinates {"x1", "y1", "x2", "y2"}.
        attributes: Optional dict of label attributes (occluded, truncated, etc.).
    """

    label_id: int
    category: str
    box2d: Dict[str, float]
    attributes: Dict[str, object] = field(default_factory=dict)

    @property
    def x1(self) -> float:
        """Left x-coordinate of the bounding box."""
        return self.box2d["x1"]

    @property
    def y1(self) -> float:
        """Top y-coordinate of the bounding box."""
        return self.box2d["y1"]

    @property
    def x2(self) -> float:
        """Right x-coordinate of the bounding box."""
        return self.box2d["x2"]

    @property
    def y2(self) -> float:
        """Bottom y-coordinate of the bounding box."""
        return self.box2d["y2"]

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height) of the bounding box."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def is_occluded(self) -> bool:
        """Whether the object is marked as occluded."""
        return bool(self.attributes.get("occluded", False))

    @property
    def is_truncated(self) -> bool:
        """Whether the object is marked as truncated (partially out of frame)."""
        return bool(self.attributes.get("truncated", False))

    def is_valid(self) -> bool:
        """Returns True if the bounding box has positive area and valid coords."""
        return (
            self.x1 < self.x2
            and self.y1 < self.y2
            and self.area > 0
            and self.category in BDD_DETECTION_CLASSES
        )

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return bounding box as (x_center, y_center, width, height) tuple."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy, self.width, self.height)


@dataclass
class BDDFrame:
    """Represents a single annotated image frame from BDD100K.

    Attributes:
        name: Filename of the image (e.g., "0000f77c-6257be58.jpg").
        labels: List of BDDLabel annotations; empty if no objects.
        attributes: Frame-level attributes (weather, scene, timeofday).
        timestamp: Frame timestamp in milliseconds.
    """

    name: str
    labels: List[BDDLabel] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    timestamp: int = 0

    @property
    def image_name(self) -> str:
        """Filename of the image."""
        return self.name

    @property
    def num_objects(self) -> int:
        """Number of annotated objects in this frame."""
        return len(self.labels)

    @property
    def is_empty(self) -> bool:
        """True if the frame has no object annotations."""
        return len(self.labels) == 0

    @property
    def weather(self) -> str:
        """Weather condition of the frame (e.g., 'clear', 'rainy')."""
        return self.attributes.get("weather", "unknown")

    @property
    def scene(self) -> str:
        """Scene type (e.g., 'city street', 'highway', 'residential')."""
        return self.attributes.get("scene", "unknown")

    @property
    def time_of_day(self) -> str:
        """Time of day (e.g., 'daytime', 'night', 'dawn/dusk')."""
        return self.attributes.get("timeofday", "unknown")

    def get_labels_by_class(self, category: str) -> List[BDDLabel]:
        """Return all labels matching the given category.

        Args:
            category: Class name to filter by.

        Returns:
            List of BDDLabel objects of the requested class.
        """
        return [lbl for lbl in self.labels if lbl.category == category]

    def class_counts(self) -> Dict[str, int]:
        """Return a dict mapping class name to instance count for this frame."""
        counts: Dict[str, int] = {cls: 0 for cls in BDD_DETECTION_CLASSES}
        for lbl in self.labels:
            if lbl.category in counts:
                counts[lbl.category] += 1
        return counts


class BDDDataset:
    """Loads and manages the BDD100K detection dataset from JSON annotation files.

    Usage::

        dataset = BDDDataset(
            train_json="data/labels/det_20/det_train.json",
            val_json="data/labels/det_20/det_val.json",
            image_dir="data/images/100k",
        )
        dataset.load()
        summary = dataset.summary()

    Attributes:
        train_json: Path to training annotation JSON file.
        val_json: Path to validation annotation JSON file.
        image_dir: Root directory containing 'train/' and 'val/' image folders.
        train_frames: Parsed list of BDDFrame objects for training split.
        val_frames: Parsed list of BDDFrame objects for validation split.
    """

    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720

    def __init__(
        self,
        train_json: str,
        val_json: str,
        image_dir: Optional[str] = None,
    ) -> None:
        """Initialize BDDDataset with paths to annotation files.

        Args:
            train_json: Path to det_train.json annotation file.
            val_json: Path to det_val.json annotation file.
            image_dir: Optional root path to image directories.
        """
        self.train_json = train_json
        self.val_json = val_json
        self.image_dir = image_dir

        self.train_frames: List[BDDFrame] = []
        self.val_frames: List[BDDFrame] = []
        self._loaded = False

    def load(self) -> None:
        """Parse both train and val annotation JSON files into BDDFrame objects."""
        print(f"[BDDDataset] Loading training annotations from: {self.train_json}")
        self.train_frames = self._parse_json(self.train_json)
        print(f"[BDDDataset] Loaded {len(self.train_frames)} training frames.")

        print(f"[BDDDataset] Loading validation annotations from: {self.val_json}")
        self.val_frames = self._parse_json(self.val_json)
        print(f"[BDDDataset] Loaded {len(self.val_frames)} validation frames.")

        self._loaded = True

    def _parse_json(self, json_path: str) -> List[BDDFrame]:
        """Parse a BDD100K detection JSON file into a list of BDDFrame objects.

        Args:
            json_path: Absolute or relative path to the JSON annotation file.

        Returns:
            List of BDDFrame objects parsed from the JSON.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the JSON structure is invalid.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as fh:
            raw_data = json.load(fh)

        frames: List[BDDFrame] = []
        for entry in raw_data:
            frame_name = entry.get("name", "")
            frame_attrs = entry.get("attributes", {})
            timestamp = entry.get("timestamp", 0)
            raw_labels = entry.get("labels", []) or []

            parsed_labels: List[BDDLabel] = []
            for raw_lbl in raw_labels:
                # Only process labels with box2d (skip segmentation, lanes)
                box2d = raw_lbl.get("box2d")
                if box2d is None:
                    continue
                category = raw_lbl.get("category", "")
                if category not in BDD_DETECTION_CLASSES:
                    continue
                lbl = BDDLabel(
                    label_id=raw_lbl.get("id", -1),
                    category=category,
                    box2d=box2d,
                    attributes=raw_lbl.get("attributes", {}),
                )
                if lbl.is_valid():
                    parsed_labels.append(lbl)

            frames.append(
                BDDFrame(
                    name=frame_name,
                    labels=parsed_labels,
                    attributes=frame_attrs,
                    timestamp=timestamp,
                )
            )

        return frames

    def get_image_path(self, frame: BDDFrame, split: str) -> str:
        """Construct absolute image path for a given frame.

        Args:
            frame: BDDFrame object.
            split: Dataset split, either 'train' or 'val'.

        Returns:
            Absolute path to the image file.

        Raises:
            ValueError: If image_dir was not provided during initialization.
        """
        if self.image_dir is None:
            raise ValueError("image_dir was not provided during initialization.")
        return os.path.join(self.image_dir, split, frame.name)

    def get_class_counts(self, split: str = "train") -> Dict[str, int]:
        """Compute total instance count per class for a given split.

        Args:
            split: Either 'train' or 'val'.

        Returns:
            Dict mapping class name to total instance count.
        """
        frames = self.train_frames if split == "train" else self.val_frames
        counts: Dict[str, int] = {cls: 0 for cls in BDD_DETECTION_CLASSES}
        for frame in frames:
            for lbl in frame.labels:
                if lbl.category in counts:
                    counts[lbl.category] += 1
        return counts

    def get_all_labels(self, split: str = "train") -> List[BDDLabel]:
        """Return a flat list of all BDDLabel objects for the given split.

        Args:
            split: Either 'train' or 'val'.

        Returns:
            Flat list of all BDDLabel annotations.
        """
        frames = self.train_frames if split == "train" else self.val_frames
        return [lbl for frame in frames for lbl in frame.labels]

    def get_empty_frames(self, split: str = "train") -> List[BDDFrame]:
        """Return frames that have no object annotations.

        Args:
            split: Either 'train' or 'val'.

        Returns:
            List of BDDFrame objects with no labels.
        """
        frames = self.train_frames if split == "train" else self.val_frames
        return [f for f in frames if f.is_empty]

    def summary(self) -> Dict[str, object]:
        """Return a summary dictionary of dataset statistics.

        Returns:
            Dict with keys: train_frames, val_frames, train_instances,
            val_instances, train_class_counts, val_class_counts,
            train_empty_frames, val_empty_frames.
        """
        train_counts = self.get_class_counts("train")
        val_counts = self.get_class_counts("val")
        return {
            "train_frames": len(self.train_frames),
            "val_frames": len(self.val_frames),
            "train_instances": sum(train_counts.values()),
            "val_instances": sum(val_counts.values()),
            "train_class_counts": train_counts,
            "val_class_counts": val_counts,
            "train_empty_frames": len(self.get_empty_frames("train")),
            "val_empty_frames": len(self.get_empty_frames("val")),
        }

    def get_bbox_stats(self, split: str = "train") -> Dict[str, np.ndarray]:
        """Compute per-class bounding box statistics for the given split.

        Args:
            split: Either 'train' or 'val'.

        Returns:
            Dict mapping class name to array of shape (N, 4):
            columns are [width, height, area, aspect_ratio].
        """
        frames = self.train_frames if split == "train" else self.val_frames
        per_class: Dict[str, list] = {cls: [] for cls in BDD_DETECTION_CLASSES}

        for frame in frames:
            for lbl in frame.labels:
                if lbl.category in per_class:
                    per_class[lbl.category].append(
                        [lbl.width, lbl.height, lbl.area, lbl.aspect_ratio]
                    )

        return {
            cls: np.array(vals) if vals else np.empty((0, 4))
            for cls, vals in per_class.items()
        }

    def get_attribute_distribution(
        self, attribute: str, split: str = "train"
    ) -> Dict[str, int]:
        """Count frame-level attribute values (weather, scene, timeofday).

        Args:
            attribute: One of 'weather', 'scene', or 'timeofday'.
            split: Either 'train' or 'val'.

        Returns:
            Dict mapping attribute value to frame count.
        """
        frames = self.train_frames if split == "train" else self.val_frames
        distribution: Dict[str, int] = {}
        for frame in frames:
            value = frame.attributes.get(attribute, "unknown")
            distribution[value] = distribution.get(value, 0) + 1
        return distribution
