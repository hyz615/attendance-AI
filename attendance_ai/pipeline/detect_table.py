"""Table detection: find horizontal & vertical grid lines and cell boundaries.

Tuned for Ontario CE attendance sheets photographed by phone:
- Handles slight rotation / perspective artifacts
- Robust to grey/white alternating row shading
- Filters lines to attendance grid region
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TableDetector:
    """Detects table grid structure from the deskewed document image."""

    def __init__(self, config: dict):
        cfg = config.get("line_detection", {})
        self.h_kernel_ratio = cfg.get("horizontal_kernel_ratio", 0.03)
        self.v_kernel_ratio = cfg.get("vertical_kernel_ratio", 0.03)
        self.merge_distance = cfg.get("merge_distance_pixels", 12)

        tbl = config.get("table", {})
        self.header_rows = tbl.get("header_rows", 3)
        self.table_y_start = tbl.get("table_y_start_ratio", 0.13)
        self.table_y_end = tbl.get("table_y_end_ratio", 0.87)
        self.grid_x_start = tbl.get("attendance_grid", {}).get("x_start_ratio", 0.40)
        self.grid_x_end = tbl.get("attendance_grid", {}).get("x_end_ratio", 0.95)
        self.name_x_start = tbl.get("name_column", {}).get("x_start_ratio", 0.02)
        self.name_x_end = tbl.get("name_column", {}).get("x_end_ratio", 0.40)
        self.max_student_rows = tbl.get("max_student_rows", 35)

    def run(self, warped_gray: np.ndarray) -> dict:
        """Detect table grid lines and cell structure.

        Returns dict with:
            - h_lines: list of y-coordinates of horizontal lines
            - v_lines: list of x-coordinates of vertical lines (all)
            - v_lines_grid: vertical lines in attendance grid area
            - rows: list of (y_start, y_end) tuples
            - cols: list of (x_start, x_end) tuples for attendance grid
            - name_col: (x_start, x_end) for the name column
            - binary: binary image used for detection
        """
        logger.info("=== TABLE DETECTION ===")
        h, w = warped_gray.shape[:2]

        # Create binary for line detection using adaptive threshold
        binary = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10,
        )

        # Detect horizontal lines
        h_lines = self._detect_horizontal_lines(binary, h, w)
        logger.info(f"Detected {len(h_lines)} horizontal lines (raw)")

        # Detect vertical lines
        v_lines = self._detect_vertical_lines(binary, h, w)
        logger.info(f"Detected {len(v_lines)} vertical lines (raw)")

        # Filter horizontal lines to table region
        y_min = int(h * self.table_y_start)
        y_max = int(h * self.table_y_end)
        h_lines_table = [y for y in h_lines if y_min <= y <= y_max]

        # Filter vertical lines to attendance grid region
        x_min_grid = int(w * self.grid_x_start)
        x_max_grid = int(w * self.grid_x_end)
        v_lines_grid = [x for x in v_lines if x_min_grid <= x <= x_max_grid]

        # Also include boundary lines if not already detected
        if h_lines_table and h_lines_table[0] > y_min + 20:
            h_lines_table.insert(0, y_min)
        if h_lines_table and h_lines_table[-1] < y_max - 20:
            h_lines_table.append(y_max)
        if not h_lines_table:
            h_lines_table = [y_min, y_max]

        if v_lines_grid and v_lines_grid[0] > x_min_grid + 15:
            v_lines_grid.insert(0, x_min_grid)
        if v_lines_grid and v_lines_grid[-1] < x_max_grid - 15:
            v_lines_grid.append(x_max_grid)
        if not v_lines_grid:
            v_lines_grid = [x_min_grid, x_max_grid]

        logger.info(
            f"After filtering: {len(h_lines_table)} h-lines, {len(v_lines_grid)} v-lines in grid"
        )

        # Build rows from horizontal lines
        rows = self._lines_to_intervals(sorted(h_lines_table))

        # Filter out rows that are too narrow (likely noise) or too tall
        if rows:
            median_row_h = np.median([r[1] - r[0] for r in rows])
            rows = [
                r for r in rows
                if (r[1] - r[0]) > median_row_h * 0.4
                and (r[1] - r[0]) < median_row_h * 2.5
            ]

        # Cap to max student rows
        if len(rows) > self.max_student_rows + self.header_rows:
            rows = rows[: self.max_student_rows + self.header_rows]

        # Build attendance grid columns from vertical lines
        cols = self._lines_to_intervals(sorted(v_lines_grid))

        # Filter out columns that are too narrow or too wide
        if cols:
            median_col_w = np.median([c[1] - c[0] for c in cols])
            cols = [
                c for c in cols
                if (c[1] - c[0]) > median_col_w * 0.3
                and (c[1] - c[0]) < median_col_w * 3.0
            ]

        # Name column bounds
        name_col = (int(w * self.name_x_start), int(w * self.name_x_end))

        logger.info(f"Table structure: {len(rows)} rows, {len(cols)} attendance columns")

        return {
            "h_lines": sorted(h_lines_table),
            "v_lines": sorted(v_lines),
            "v_lines_grid": sorted(v_lines_grid),
            "rows": rows,
            "cols": cols,
            "name_col": name_col,
            "binary": binary,
        }

    def _detect_horizontal_lines(
        self, binary: np.ndarray, h: int, w: int
    ) -> list[int]:
        """Detect horizontal lines using morphological operations."""
        kernel_len = max(int(w * self.h_kernel_ratio), 30)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # Find line positions via horizontal projection
        projection = np.sum(h_mask, axis=1)
        threshold = w * 0.10  # lower threshold for phone photos
        line_ys = np.where(projection > threshold)[0]

        return self._merge_nearby(line_ys.tolist(), self.merge_distance)

    def _detect_vertical_lines(
        self, binary: np.ndarray, h: int, w: int
    ) -> list[int]:
        """Detect vertical lines using morphological operations."""
        kernel_len = max(int(h * self.v_kernel_ratio), 30)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        projection = np.sum(v_mask, axis=0)
        threshold = h * 0.10  # lower threshold for phone photos
        line_xs = np.where(projection > threshold)[0]

        return self._merge_nearby(line_xs.tolist(), self.merge_distance)

    @staticmethod
    def _merge_nearby(positions: list[int], distance: int) -> list[int]:
        """Merge nearby line positions into single representative positions."""
        if not positions:
            return []
        merged = []
        group = [positions[0]]
        for p in positions[1:]:
            if p - group[-1] <= distance:
                group.append(p)
            else:
                merged.append(int(np.mean(group)))
                group = [p]
        merged.append(int(np.mean(group)))
        return merged

    @staticmethod
    def _lines_to_intervals(lines: list[int]) -> list[tuple[int, int]]:
        """Convert sorted line positions into (start, end) intervals."""
        intervals = []
        for i in range(len(lines) - 1):
            gap = lines[i + 1] - lines[i]
            if gap > 5:  # skip zero-width intervals
                intervals.append((lines[i], lines[i + 1]))
        return intervals
