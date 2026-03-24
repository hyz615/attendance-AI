"""Cell classification: determine if a cell contains 'A', is blank, or unknown.

Critical design: Uses Otsu auto-thresholding and background normalization
to handle the grey/white alternating row pattern in Ontario CE sheets.
"""

import cv2
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CellLabel(str, Enum):
    A = "A"
    BLANK = "BLANK"
    UNKNOWN = "UNKNOWN"


def normalize_cell_background(cell: np.ndarray) -> np.ndarray:
    """Normalize cell background to white to handle grey/white alternating rows.

    Shifts the histogram so the median (background) becomes ~255,
    making grey-row and white-row cells comparable.
    """
    if cell.size == 0:
        return cell
    bg = float(np.median(cell))
    if bg < 20:
        return cell
    if bg > 240:
        return cell  # already white background
    shift = 255.0 - bg
    normalized = np.clip(cell.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    return normalized


class BaselineClassifier:
    """Rule-based cell classifier using Otsu thresholding and shape analysis.

    Handles grey/white alternating rows by normalizing cell backgrounds.
    Tuned to prefer false positives over false negatives (never miss an A).
    """

    def __init__(self, config: dict):
        cfg = config.get("cell_classification", {})
        self.min_dark_ratio = cfg.get("min_dark_ratio_for_A", 0.06)
        self.max_blank_ratio = cfg.get("max_dark_ratio_for_blank", 0.02)
        self.min_contour_area_ratio = cfg.get("min_contour_area_ratio", 0.015)
        self.center_tolerance = cfg.get("center_mass_tolerance", 0.40)
        self.do_normalize = cfg.get("normalize_background", True)

    def classify(self, cell_image: np.ndarray) -> tuple[CellLabel, float]:
        """Classify a single cell image.

        Returns:
            (label, confidence) where confidence is 0.0-1.0
        """
        if cell_image is None or cell_image.size == 0:
            return CellLabel.BLANK, 1.0

        h, w = cell_image.shape[:2]
        cell_area = h * w
        if cell_area < 16:
            return CellLabel.BLANK, 1.0

        # Guard: uniform cells are blank (prevents Otsu noise on flat images)
        cell_std = float(np.std(cell_image))
        if cell_std < 8.0:
            return CellLabel.BLANK, 1.0

        # Step 1: Normalize background (grey rows → white)
        if self.do_normalize:
            cell = normalize_cell_background(cell_image)
        else:
            cell = cell_image

        # Step 2: Otsu thresholding (auto-adapts to cell content)
        _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Step 3: Remove small noise with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Dark pixel ratio after Otsu + cleanup
        dark_pixels = np.sum(binary > 0)
        dark_ratio = dark_pixels / cell_area

        # If very few dark pixels → BLANK
        if dark_ratio < self.max_blank_ratio:
            return CellLabel.BLANK, 1.0 - dark_ratio / max(self.max_blank_ratio, 0.001)

        # Step 4: Shape analysis on remaining dark pixels
        if dark_ratio >= self.min_dark_ratio:
            is_a, shape_conf = self._check_a_shape(binary, h, w)
            if is_a:
                combined_conf = min(1.0, 0.5 + dark_ratio + shape_conf * 0.3)
                return CellLabel.A, combined_conf

        # In-between zone: lower threshold check (prefer FP over FN)
        if dark_ratio >= self.max_blank_ratio:
            is_a, shape_conf = self._check_a_shape(binary, h, w)
            if is_a and shape_conf > 0.25:
                return CellLabel.A, 0.4 + dark_ratio
            # If there's reasonable ink but shape doesn't match A,
            # still lean toward A (prefer false positive)
            if dark_ratio > self.min_dark_ratio * 0.7:
                return CellLabel.A, 0.35 + dark_ratio
            return CellLabel.UNKNOWN, 0.5

        return CellLabel.BLANK, 0.8

    def _check_a_shape(
        self, binary: np.ndarray, h: int, w: int
    ) -> tuple[bool, float]:
        """Analyze if the dark pixels form an A-like pattern.

        Checks:
        1. Content is roughly centered in the cell
        2. Has reasonable contour area
        3. Bounding box has letter-like aspect ratio
        4. Dark pixels are concentrated (not scattered noise)
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return False, 0.0

        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        cell_area = h * w

        if area < cell_area * self.min_contour_area_ratio:
            return False, 0.0

        # Bounding box of the mark
        bx, by, bw, bh = cv2.boundingRect(largest)

        # Check centeredness (how close the mark center is to cell center)
        cx = bx + bw / 2
        cy = by + bh / 2
        center_x_off = abs(cx - w / 2) / (w / 2) if w > 0 else 1
        center_y_off = abs(cy - h / 2) / (h / 2) if h > 0 else 1

        centered = (
            center_x_off < self.center_tolerance
            and center_y_off < self.center_tolerance
        )

        # Aspect ratio of the mark (A is roughly as tall as wide, or taller)
        mark_aspect = bh / max(bw, 1)
        reasonable_aspect = 0.3 < mark_aspect < 3.5

        # Size relative to cell
        size_ratio = (bw * bh) / cell_area
        reasonable_size = size_ratio > 0.04

        # Solidity check — how filled the convex hull is (A has moderate solidity)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        reasonable_solidity = solidity > 0.15

        score = 0.0
        if centered:
            score += 0.30
        if reasonable_aspect:
            score += 0.25
        if reasonable_size:
            score += 0.25
        if reasonable_solidity:
            score += 0.20

        is_a = score >= 0.35  # lenient threshold (prefer FP)
        return is_a, score


class CNNClassifier:
    """CNN-based cell classifier (requires a trained model).

    Falls back to BaselineClassifier if no model is available.
    """

    def __init__(self, config: dict, model_path: str | None = None):
        self.config = config
        self.model = None
        self.baseline = BaselineClassifier(config)
        self.input_size = (32, 32)

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load a trained PyTorch CNN model."""
        try:
            import torch
            from attendance_ai.models.cnn_model import AttendanceCNN

            self.model = AttendanceCNN()
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info(f"CNN model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load CNN model: {e}. Using baseline.")
            self.model = None

    def classify(self, cell_image: np.ndarray) -> tuple[CellLabel, float]:
        """Classify cell using CNN if available, else baseline."""
        if self.model is None:
            return self.baseline.classify(cell_image)

        import torch

        # Normalize background before CNN too
        cell = normalize_cell_background(cell_image)
        resized = cv2.resize(cell, self.input_size)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        label_map = {0: CellLabel.A, 1: CellLabel.BLANK, 2: CellLabel.UNKNOWN}
        return label_map.get(pred.item(), CellLabel.UNKNOWN), conf.item()


class CellClassifier:
    """Unified classifier interface — selects between baseline and CNN."""

    def __init__(self, config: dict, model_path: str | None = None):
        self.use_cnn = model_path is not None
        if self.use_cnn:
            self.classifier = CNNClassifier(config, model_path)
        else:
            self.classifier = BaselineClassifier(config)
        logger.info(f"Cell classifier: {'CNN' if self.use_cnn else 'Baseline'}")

    def classify(self, cell_image: np.ndarray) -> tuple[CellLabel, float]:
        return self.classifier.classify(cell_image)

    def classify_batch(
        self, cell_images: list[np.ndarray]
    ) -> list[tuple[CellLabel, float]]:
        """Classify a batch of cell images."""
        return [self.classify(img) for img in cell_images]
