"""Document detection: find sheet contour and apply perspective correction."""

import cv2
import numpy as np
import logging

from attendance_ai.utils.image_utils import order_points, four_point_transform

logger = logging.getLogger(__name__)


class DocumentDetector:
    """Detects the attendance sheet boundary and applies perspective correction."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, preprocessed: dict) -> dict:
        """Detect document contour and deskew.

        Args:
            preprocessed: dict from Preprocessor.run()

        Returns:
            dict with:
                - warped: perspective-corrected colour image
                - warped_gray: grayscale of warped
                - contour: detected document contour (or None)
                - success: bool
        """
        logger.info("=== DOCUMENT DETECTION ===")
        resized = preprocessed["resized"]
        edges = preprocessed["edges"]

        result = {
            "warped": resized,
            "warped_gray": preprocessed.get("enhanced", preprocessed["grayscale"]),
            "contour": None,
            "success": False,
        }

        contour = self._find_document_contour(edges, resized.shape)
        if contour is not None:
            logger.info("Document contour found, applying perspective transform")
            result["contour"] = contour
            result["warped"] = four_point_transform(resized, contour)
            gray_w = cv2.cvtColor(result["warped"], cv2.COLOR_BGR2GRAY)
            # Apply CLAHE on warped gray if original was enhanced
            if "enhanced" in preprocessed:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_w = clahe.apply(gray_w)
            result["warped_gray"] = gray_w
            result["success"] = True
        else:
            logger.warning(
                "No document contour found — using full image (may already be cropped)"
            )
            result["success"] = False

        h, w = result["warped"].shape[:2]
        logger.info(f"Document size after deskew: {w}x{h}")
        return result

    def _find_document_contour(
        self, edges: np.ndarray, img_shape: tuple
    ) -> np.ndarray | None:
        """Find the largest 4-point contour that looks like a sheet of paper."""
        h, w = img_shape[:2]
        min_area = h * w * 0.2  # at least 20% of image

        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:10]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if self._is_reasonable_quad(pts, w, h):
                    logger.info(f"Found document quad with area {area:.0f}")
                    return pts

        # Fallback: try convex hull of largest contour
        if contours:
            hull = cv2.convexHull(contours[0])
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                if self._is_reasonable_quad(pts, w, h):
                    return pts

        return None

    @staticmethod
    def _is_reasonable_quad(pts: np.ndarray, img_w: int, img_h: int) -> bool:
        """Check if the quad has reasonable proportions for a document."""
        ordered = order_points(pts)
        tl, tr, br, bl = ordered

        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)

        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2

        if avg_w < img_w * 0.3 or avg_h < img_h * 0.3:
            return False

        aspect = avg_h / max(avg_w, 1)
        # Attendance sheets are roughly portrait A4 ~ 1.414
        if aspect < 0.5 or aspect > 3.0:
            return False

        return True
