"""Cell extraction: extract individual cells from the detected table grid.

Tuned for Ontario CE attendance sheets:
- Dynamic cell padding (proportional to cell size) to remove grid lines
- Latest column detection that ignores empty student rows
- Handles 35-row layout with ~25 filled students
"""

import cv2
import numpy as np
import logging

from attendance_ai.pipeline.classify_cell import normalize_cell_background

logger = logging.getLogger(__name__)


class CellExtractor:
    """Extracts individual cells from the attendance grid."""

    def __init__(self, config: dict):
        tbl = config.get("table", {})
        self.header_rows = tbl.get("header_rows", 3)
        self.max_student_rows = tbl.get("max_student_rows", 35)

    def run(
        self,
        warped_gray: np.ndarray,
        table_info: dict,
    ) -> dict:
        """Extract attendance cells and name cells.

        Args:
            warped_gray: grayscale deskewed document image
            table_info: dict from TableDetector.run()

        Returns dict with:
            - student_cells: list of dicts per student row, each with:
                - row_index: int
                - name_region: cropped name cell image
                - attendance_cells: list of dicts per column
                - is_empty: whether the row appears to be an empty slot
            - latest_column_index: int (rightmost non-empty column)
            - total_rows: int (non-empty students only)
            - all_rows: int (including empty rows)
        """
        logger.info("=== CELL EXTRACTION ===")
        rows = table_info["rows"]
        cols = table_info["cols"]
        name_col = table_info["name_col"]
        h_img, w_img = warped_gray.shape[:2]

        # Skip header rows
        data_rows = rows[self.header_rows:] if len(rows) > self.header_rows else rows
        logger.info(
            f"Total rows: {len(rows)}, data rows (after skipping {self.header_rows} headers): {len(data_rows)}"
        )

        if not cols:
            logger.error("No attendance columns detected!")
            return {
                "student_cells": [],
                "latest_column_index": -1,
                "total_rows": 0,
                "all_rows": 0,
            }

        student_cells = []
        for row_idx, (y1, y2) in enumerate(data_rows):
            # Dynamic padding: ~8% of cell dimension, min 2, max 8
            row_h = y2 - y1
            pad_y = max(2, min(8, int(row_h * 0.08)))

            # Crop name region
            nx1, nx2 = name_col
            name_img = self._safe_crop(warped_gray, y1, y2, nx1, nx2, pad_y, 2)

            # Check if this is an empty row (no student enrolled)
            is_empty = self._is_empty_row(name_img)

            # Crop each attendance cell
            att_cells = []
            for col_idx, (x1, x2) in enumerate(cols):
                col_w = x2 - x1
                pad_x = max(2, min(8, int(col_w * 0.08)))
                cell_img = self._safe_crop(warped_gray, y1, y2, x1, x2, pad_y, pad_x)
                att_cells.append({
                    "col_index": col_idx,
                    "image": cell_img,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                })

            student_cells.append({
                "row_index": row_idx,
                "name_region": name_img,
                "attendance_cells": att_cells,
                "is_empty": is_empty,
            })

        # Count non-empty students
        filled_students = [s for s in student_cells if not s["is_empty"]]
        empty_students = [s for s in student_cells if s["is_empty"]]
        logger.info(
            f"Students: {len(filled_students)} filled, {len(empty_students)} empty rows"
        )

        # Find latest column (only considering filled student rows)
        latest_col_idx = self._find_latest_column(
            warped_gray, data_rows, cols, [s["is_empty"] for s in student_cells]
        )
        logger.info(
            f"Attendance columns: {len(cols)}, latest column index: {latest_col_idx}"
        )

        return {
            "student_cells": student_cells,
            "latest_column_index": latest_col_idx,
            "total_rows": len(filled_students),
            "all_rows": len(student_cells),
        }

    def _safe_crop(
        self, img: np.ndarray, y1: int, y2: int, x1: int, x2: int,
        pad_y: int = 3, pad_x: int = 3,
    ) -> np.ndarray:
        """Crop cell with dynamic padding to remove grid lines."""
        h, w = img.shape[:2]
        cy1 = max(0, y1 + pad_y)
        cy2 = min(h, y2 - pad_y)
        cx1 = max(0, x1 + pad_x)
        cx2 = min(w, x2 - pad_x)
        if cy2 <= cy1 or cx2 <= cx1:
            return np.zeros((max(y2 - y1, 1), max(x2 - x1, 1)), dtype=np.uint8)
        return img[cy1:cy2, cx1:cx2].copy()

    def _is_empty_row(self, name_img: np.ndarray) -> bool:
        """Determine if a student row is empty (no name written).

        Uses dark pixel ratio on the name region. Empty rows
        have very few dark pixels (just grey/white background + grid lines).
        """
        if name_img is None or name_img.size == 0:
            return True

        # Normalize background, then threshold
        normalized = normalize_cell_background(name_img)
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        dark_ratio = np.sum(binary > 0) / max(binary.size, 1)
        # Name regions with actual text typically have > 5% dark pixels
        return dark_ratio < 0.03

    def _find_latest_column(
        self,
        gray: np.ndarray,
        data_rows: list[tuple[int, int]],
        cols: list[tuple[int, int]],
        is_empty_flags: list[bool],
    ) -> int:
        """Find rightmost column with significant content, ignoring empty student rows.

        Scans from right to left and returns the first column
        where enough *filled* student cells have dark pixel content.
        """
        if not cols or not data_rows:
            return max(0, len(cols) - 1)

        # Only consider filled student rows
        filled_rows = [
            (y1, y2)
            for (y1, y2), empty in zip(data_rows, is_empty_flags)
            if not empty
        ]

        if not filled_rows:
            return max(0, len(cols) - 1)

        # At least 1 cell in the column must have an A-like mark
        min_filled_cells = 1

        for col_idx in range(len(cols) - 1, -1, -1):
            x1, x2 = cols[col_idx]
            pad_x = max(2, min(8, int((x2 - x1) * 0.08)))
            filled = 0

            for y1, y2 in filled_rows:
                pad_y = max(2, min(8, int((y2 - y1) * 0.08)))
                cell = self._safe_crop(gray, y1, y2, x1, x2, pad_y, pad_x)
                if cell.size == 0:
                    continue

                # Skip uniform cells (no content)
                if np.std(cell) < 8.0:
                    continue

                # Normalize background and use Otsu
                cell_norm = normalize_cell_background(cell)
                _, binary = cv2.threshold(
                    cell_norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )
                dark_ratio = np.sum(binary > 0) / max(binary.size, 1)
                if dark_ratio > 0.04:
                    filled += 1

            if filled >= min_filled_cells:
                logger.info(
                    f"Column {col_idx}: {filled}/{len(filled_rows)} cells with content → selected as latest"
                )
                return col_idx

        # Fallback: rightmost column
        logger.warning("No columns with sufficient content found, using rightmost")
        return len(cols) - 1
