"""
Custom PyTorch DataLoader for BDD100K Object Detection.

This module implements a PyTorch Dataset class that loads BDD100K images
and annotations for model training and evaluation. Supports configurable
augmentations, subset sampling, and YOLO-format label conversion.

Classes:
    BDD100KDetectionDataset: PyTorch Dataset for BDD100K object detection.
    BDD100KDataModule: Convenience wrapper to create train/val DataLoaders.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_analysis.bdd_parser import BDD_DETECTION_CLASSES, BDDDataset, BDDFrame

# Class name → integer index mapping (for YOLO-style labels)
CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(BDD_DETECTION_CLASSES)}

# Default image dimensions for BDD100K
BDD_IMG_WIDTH = 1280
BDD_IMG_HEIGHT = 720


class BDD100KDetectionDataset(Dataset):
    """PyTorch Dataset for BDD100K object detection training and evaluation.

    Each item is a resized image tensor and a list of YOLO-format targets:
        [class_id, cx_norm, cy_norm, w_norm, h_norm]
    where coordinates are normalised to [0, 1] relative to image size.

    Usage::

        dataset = BDD100KDetectionDataset(
            frames=bdd_dataset.train_frames,
            image_dir="data/images/100k/train",
            img_size=640,
            subset_size=1000,
        )
        image, targets = dataset[0]

    Attributes:
        frames: List of BDDFrame objects to use.
        image_dir: Directory containing the JPEG images.
        img_size: Square size to resize images to.
        transforms: Optional callable applied to the PIL/numpy image.
    """

    def __init__(
        self,
        frames: List[BDDFrame],
        image_dir: str,
        img_size: int = 640,
        subset_size: Optional[int] = None,
        transforms: Optional[Callable] = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize the dataset.

        Args:
            frames: List of BDDFrame objects (from BDDDataset.train_frames etc.).
            image_dir: Directory containing image files.
            img_size: Target image size (square); images are resized to img_size×img_size.
            subset_size: If set, randomly subsample this many frames from the full list.
            transforms: Optional torchvision-compatible transform pipeline.
            augment: If True, apply basic data augmentation (flips, colour jitter).
            seed: Random seed for reproducible subset sampling.
        """
        self.image_dir = image_dir
        self.img_size = img_size
        self.transforms = transforms
        self.augment = augment

        # Filter to frames whose image file exists
        valid_frames = [
            f for f in frames
            if os.path.exists(os.path.join(image_dir, f.name))
        ]

        if len(valid_frames) < len(frames):
            print(
                f"[BDD100KDetectionDataset] Warning: {len(frames) - len(valid_frames)} "
                f"frames skipped (images not found in {image_dir})."
            )

        if subset_size is not None and subset_size < len(valid_frames):
            random.seed(seed)
            valid_frames = random.sample(valid_frames, subset_size)
            print(f"[BDD100KDetectionDataset] Using subset of {subset_size} frames.")

        self.frames: List[BDDFrame] = valid_frames
        print(f"[BDD100KDetectionDataset] Dataset size: {len(self.frames)} frames.")

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return a single (image, targets) pair.

        Args:
            idx: Index of the frame to retrieve.

        Returns:
            Tuple of:
                - image: Float tensor of shape (3, img_size, img_size), normalised to [0, 1].
                - targets: Float tensor of shape (N, 6) where each row is
                  [sample_idx=0, class_id, cx_norm, cy_norm, w_norm, h_norm].
                  Returns empty tensor of shape (0, 6) if frame has no labels.
        """
        frame = self.frames[idx]
        image_path = os.path.join(self.image_dir, frame.name)

        # Load image with OpenCV (BGR)
        image = cv2.imread(image_path)
        if image is None:
            # Return blank image if file is missing/corrupted
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            # Apply augmentation before resize
            if self.augment:
                image = self._augment(image)
            # Resize to target size
            image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert BGR → RGB, then to float tensor [0, 1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transforms is not None:
            image_tensor = self.transforms(image_tensor)

        # Build targets: [batch_idx=0, cls, cx_n, cy_n, w_n, h_n]
        targets = self._build_targets(frame)
        return image_tensor, targets

    def _build_targets(self, frame: BDDFrame) -> torch.Tensor:
        """Convert BDDFrame labels to normalised YOLO-format target tensor.

        Args:
            frame: BDDFrame with bounding box annotations.

        Returns:
            Float tensor of shape (N, 6): [0, class_id, cx_n, cy_n, w_n, h_n].
        """
        if frame.is_empty:
            return torch.zeros((0, 6), dtype=torch.float32)

        rows = []
        for lbl in frame.labels:
            cls_id = CLASS_TO_IDX.get(lbl.category, -1)
            if cls_id == -1:
                continue
            # Normalise coordinates by original image size
            cx_n = (lbl.x1 + lbl.x2) / 2.0 / BDD_IMG_WIDTH
            cy_n = (lbl.y1 + lbl.y2) / 2.0 / BDD_IMG_HEIGHT
            w_n = lbl.width / BDD_IMG_WIDTH
            h_n = lbl.height / BDD_IMG_HEIGHT
            # Clamp to [0, 1]
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            w_n = max(0.0, min(1.0, w_n))
            h_n = max(0.0, min(1.0, h_n))
            rows.append([0.0, float(cls_id), cx_n, cy_n, w_n, h_n])

        if not rows:
            return torch.zeros((0, 6), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        """Apply basic data augmentation to a BGR image.

        Augmentations applied (each with 50% probability):
        - Horizontal flip
        - Brightness/contrast jitter

        Args:
            image: BGR numpy array.

        Returns:
            Augmented BGR numpy array.
        """
        # Horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        # Brightness jitter
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function to handle variable-length target tensors.

        Stacks images into a single batch tensor and concatenates targets,
        updating the batch index column so each target row knows which
        image in the batch it belongs to.

        Args:
            batch: List of (image, targets) tuples from __getitem__.

        Returns:
            Tuple of (images_tensor [B, 3, H, W], targets_tensor [M, 6]).
        """
        images, targets_list = zip(*batch)
        images_tensor = torch.stack(images, dim=0)

        updated_targets = []
        for i, targets in enumerate(targets_list):
            if targets.shape[0] > 0:
                targets[:, 0] = i  # Set batch index
                updated_targets.append(targets)

        if updated_targets:
            targets_tensor = torch.cat(updated_targets, dim=0)
        else:
            targets_tensor = torch.zeros((0, 6), dtype=torch.float32)

        return images_tensor, targets_tensor


class BDD100KDataModule:
    """Convenience class to create BDD100K train and validation DataLoaders.

    Usage::

        data_module = BDD100KDataModule(
            train_json="data/labels/det_20/det_train.json",
            val_json="data/labels/det_20/det_val.json",
            train_image_dir="data/images/100k/train",
            val_image_dir="data/images/100k/val",
            batch_size=16,
            img_size=640,
            subset_size=1000,  # for quick experiments
        )
        train_loader, val_loader = data_module.build()

    Attributes:
        train_json: Path to training annotation JSON.
        val_json: Path to validation annotation JSON.
        train_image_dir: Path to training image directory.
        val_image_dir: Path to validation image directory.
        batch_size: Mini-batch size for DataLoader.
        img_size: Target image size for resizing.
        num_workers: Number of DataLoader worker processes.
        subset_size: Optional subset for quick training experiments.
    """

    def __init__(
        self,
        train_json: str,
        val_json: str,
        train_image_dir: str,
        val_image_dir: str,
        batch_size: int = 16,
        img_size: int = 640,
        num_workers: int = 4,
        subset_size: Optional[int] = None,
        augment_train: bool = True,
    ) -> None:
        """Initialize the data module.

        Args:
            train_json: Path to det_train.json.
            val_json: Path to det_val.json.
            train_image_dir: Path to training images.
            val_image_dir: Path to validation images.
            batch_size: Batch size for DataLoader.
            img_size: Square target size for image resize.
            num_workers: Number of workers for DataLoader.
            subset_size: If set, only use this many training images.
            augment_train: Whether to apply augmentation during training.
        """
        self.train_json = train_json
        self.val_json = val_json
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.subset_size = subset_size
        self.augment_train = augment_train

    def build(self) -> Tuple[DataLoader, DataLoader]:
        """Parse annotations and build train + val DataLoaders.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        bdd_dataset = BDDDataset(
            train_json=self.train_json,
            val_json=self.val_json,
        )
        bdd_dataset.load()

        train_ds = BDD100KDetectionDataset(
            frames=bdd_dataset.train_frames,
            image_dir=self.train_image_dir,
            img_size=self.img_size,
            subset_size=self.subset_size,
            augment=self.augment_train,
        )

        val_ds = BDD100KDetectionDataset(
            frames=bdd_dataset.val_frames,
            image_dir=self.val_image_dir,
            img_size=self.img_size,
            augment=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=BDD100KDetectionDataset.collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=BDD100KDetectionDataset.collate_fn,
            pin_memory=True,
        )

        print(
            f"[BDD100KDataModule] Train DataLoader: {len(train_ds)} images, "
            f"{len(train_loader)} batches"
        )
        print(
            f"[BDD100KDataModule] Val DataLoader  : {len(val_ds)} images, "
            f"{len(val_loader)} batches"
        )

        return train_loader, val_loader
