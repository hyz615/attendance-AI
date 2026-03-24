"""Preprocessing module: grayscale, CLAHE, denoise, adaptive threshold, edge detection."""

import cv2
import numpy as np
import logging

from attendance_ai.utils.image_utils import to_grayscale, resize_to_width, auto_rotate

logger = logging.getLogger(__name__)


class Preprocessor:
    """Performs image preprocessing for attendance sheet recognition."""

    def __init__(self, config: dict):
        cfg = config.get("preprocessing", {})
        self.target_width = cfg.get("target_width", 2480)
        self.adaptive_block = cfg.get("adaptive_block_size", 21)
        self.adaptive_c = cfg.get("adaptive_c", 10)
        self.denoise_h = cfg.get("denoise_h", 10)
        self.denoise_tw = cfg.get("denoise_template_window", 7)
        self.denoise_sw = cfg.get("denoise_search_window", 21)
        self.use_clahe = cfg.get("use_clahe", True)
        self.clahe_clip = cfg.get("clahe_clip_limit", 2.0)
        self.clahe_tile = cfg.get("clahe_tile_size", 8)
        self.orientation = config.get("sheet", {}).get("orientation", "landscape")

    def run(self, image: np.ndarray) -> dict:
        """Run full preprocessing pipeline.

        Returns dict with intermediate results:
            - original: input image
            - rotated: auto-rotated image
            - resized: resized to target width
            - grayscale: grayscale version
            - enhanced: CLAHE-enhanced grayscale (if enabled)
            - denoised: denoised grayscale
            - binary: adaptive-thresholded binary
            - edges: Canny edge map
        """
        logger.info("=== PREPROCESSING ===")
        results = {"original": image}

        # Auto-rotate to match expected sheet orientation (landscape for Ontario CE)
        rotated = auto_rotate(image, self.orientation)
        results["rotated"] = rotated
        logger.info(f"After rotation: {rotated.shape[1]}x{rotated.shape[0]}")

        # Resize to standard width
        resized = resize_to_width(rotated, self.target_width)
        results["resized"] = resized
        logger.info(f"Resized to: {resized.shape[1]}x{resized.shape[0]}")

        # Grayscale
        gray = to_grayscale(resized)
        results["grayscale"] = gray

        # CLAHE contrast enhancement (critical for uneven lighting in phone photos)
        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_tile, self.clahe_tile),
            )
            enhanced = clahe.apply(gray)
            results["enhanced"] = enhanced
            logger.info("CLAHE contrast enhancement applied")
        else:
            enhanced = gray
            results["enhanced"] = enhanced

        # Denoise
        denoised = cv2.fastNlMeansDenoising(
            enhanced, None, self.denoise_h, self.denoise_tw, self.denoise_sw
        )
        results["denoised"] = denoised
        logger.info("Denoising applied")

        # Adaptive threshold (inverted: text=white, background=black)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block,
            self.adaptive_c,
        )
        results["binary"] = binary
        logger.info("Adaptive thresholding applied")

        # Edge detection
        edges = cv2.Canny(denoised, 50, 150)
        results["edges"] = edges
        logger.info("Edge detection applied")

        return results
