"""Table detection: find attendance grid via projection-based line detection.

Robust approach for phone photos of Ontario CE attendance sheets:
1. Adaptive threshold → binary image
2. Morphological opening to isolate horizontal / vertical line structures
3. Projection profiles → peak detection for grid lines
4. Auto-detect grid region from regular spacing patterns
5. No dependency on document perspective correction
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TableDetector:
    """Detects the attendance grid from the preprocessed (but not warped) image."""

    def __init__(self, config: dict):
        tbl = config.get("table", {})
        self.header_rows = tbl.get("header_rows", 2)
        self.max_student_rows = tbl.get("max_student_rows", 35)

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def run(self, gray: np.ndarray) -> dict:
        """Detect grid lines and cell structure.

        Args:
            gray: grayscale image (after rotation + resize + CLAHE)

        Returns dict with:
            h_lines      – sorted y-positions of horizontal grid lines
            v_lines      – sorted x-positions of ALL vertical lines
            v_lines_grid – x-positions of attendance date-column lines
            rows         – list of (y_start, y_end) tuples (ALL rows incl. header)
            cols         – list of (x_start, x_end) for attendance date columns
            name_col     – (x_start, x_end) for the student name area
            binary       – the binary image used
        """
        logger.info("=== TABLE DETECTION (projection-based) ===")
        h, w = gray.shape[:2]

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10,
        )

        # ── Phase 1: detect line structures ──
        h_lines_raw = self._detect_h_lines(binary, h, w)
        v_lines_raw = self._detect_v_lines(binary, h, w)
        logger.info(f"Raw detection: {len(h_lines_raw)} h-lines, {len(v_lines_raw)} v-lines")

        # ── Phase 2: find regularly-spaced grid lines ──
        h_grid, h_period = self._find_regular_lines(h_lines_raw, min_count=8)
        v_grid, v_period = self._find_regular_lines(v_lines_raw, min_count=5)
        logger.info(
            f"Regular grid: {len(h_grid)} h-lines (period≈{h_period:.1f}), "
            f"{len(v_grid)} v-lines (period≈{v_period:.1f})"
        )

        # ── Phase 3: merge close lines (remove double-detections) ──
        h_grid = self._merge_close_lines(h_grid, min_gap=max(8, h_period * 0.4))
        v_grid = self._merge_close_lines(v_grid, min_gap=max(8, v_period * 0.4))
        logger.info(
            f"After merge: {len(h_grid)} h-lines, {len(v_grid)} v-lines"
        )

        # ── Phase 4: build rows / cols ──
        rows = self._lines_to_intervals(h_grid)
        cols = self._lines_to_intervals(v_grid)

        # Infer name column: everything left of the first date-column vertical line
        if v_grid:
            name_x_end = v_grid[0]
            # Estimate name column start by looking for the first strong v-line
            # to the left of the grid
            left_v = [x for x in v_lines_raw if x < name_x_end - 20]
            name_x_start = left_v[0] if left_v else max(0, int(w * 0.01))
        else:
            name_x_start = 0
            name_x_end = int(w * 0.35)
        name_col = (name_x_start, name_x_end)

        # Find sub-columns within name area (internal v-lines)
        name_area_v = sorted([x for x in v_lines_raw if name_x_start < x < name_x_end - 10])
        # Build sub-column intervals
        name_sub_cols = []
        boundaries = [name_x_start] + name_area_v + [name_x_end]
        for i in range(len(boundaries) - 1):
            x1, x2 = boundaries[i], boundaries[i + 1]
            if x2 - x1 > 15:  # skip very narrow gaps
                name_sub_cols.append((x1, x2))

        logger.info(
            f"Table: {len(rows)} rows, {len(cols)} attendance cols, "
            f"name col x=[{name_col[0]}, {name_col[1]}], "
            f"{len(name_sub_cols)} name sub-cols"
        )

        return {
            "h_lines": h_grid,
            "v_lines": v_lines_raw,
            "v_lines_grid": v_grid,
            "rows": rows,
            "cols": cols,
            "name_col": name_col,
            "name_sub_cols": name_sub_cols,
            "binary": binary,
        }

    # ──────────────────────────────────────────────────
    # Line detection helpers
    # ──────────────────────────────────────────────────

    def _detect_h_lines(self, binary: np.ndarray, h: int, w: int) -> list[int]:
        """Detect horizontal lines via morphological opening + projection."""
        # Use 8% of width as minimum line length — catches grid lines
        kernel_len = max(int(w * 0.08), 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        projection = np.sum(mask, axis=1).astype(float)
        if projection.max() == 0:
            return []

        # Dynamic threshold: 20% of max projection value
        threshold = projection.max() * 0.20
        return self._peaks_from_projection(projection, threshold, min_distance=5)

    def _detect_v_lines(self, binary: np.ndarray, h: int, w: int) -> list[int]:
        """Detect vertical lines via morphological opening + projection."""
        kernel_len = max(int(h * 0.08), 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        projection = np.sum(mask, axis=0).astype(float)
        if projection.max() == 0:
            return []

        threshold = projection.max() * 0.20
        return self._peaks_from_projection(projection, threshold, min_distance=5)

    @staticmethod
    def _peaks_from_projection(
        proj: np.ndarray, threshold: float, min_distance: int
    ) -> list[int]:
        """Find peak positions in a projection profile (simple peak finding)."""
        peaks = []
        above = proj > threshold
        i = 0
        n = len(proj)
        while i < n:
            if above[i]:
                # Find extent of this peak region
                start = i
                while i < n and above[i]:
                    i += 1
                # Peak center = position of maximum within this region
                segment = proj[start:i]
                peak_pos = start + int(np.argmax(segment))
                if not peaks or (peak_pos - peaks[-1]) >= min_distance:
                    peaks.append(peak_pos)
            else:
                i += 1
        return peaks

    # ──────────────────────────────────────────────────
    # Regular spacing analysis
    # ──────────────────────────────────────────────────

    def _find_regular_lines(
        self, lines: list[int], min_count: int = 5
    ) -> tuple[list[int], float]:
        """Find the largest subset of lines with approximately regular spacing.

        Uses histogram of pairwise spacings to find the dominant period,
        then greedily selects lines that match that period.
        """
        if len(lines) < min_count:
            return lines, 0.0

        sorted_lines = sorted(lines)
        spacings = np.diff(sorted_lines)

        if len(spacings) == 0:
            return sorted_lines, 0.0

        # Build histogram of spacings to find dominant period
        # Use bins of 2 pixels
        max_spacing = int(np.percentile(spacings, 90)) + 1
        min_spacing = max(5, int(np.percentile(spacings, 10)))
        if max_spacing <= min_spacing:
            return sorted_lines, float(np.median(spacings))

        hist_bins = np.arange(min_spacing, max_spacing + 2, 2)
        if len(hist_bins) < 2:
            return sorted_lines, float(np.median(spacings))

        hist, bin_edges = np.histogram(spacings, bins=hist_bins)
        if hist.max() == 0:
            return sorted_lines, float(np.median(spacings))

        # Dominant period = bin center with highest count
        best_bin = np.argmax(hist)
        period = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2

        logger.debug(f"Dominant period: {period:.1f} px (count={hist[best_bin]})")

        # Greedily select lines matching this period (±35% tolerance)
        tolerance = period * 0.35
        selected = [sorted_lines[0]]
        for line in sorted_lines[1:]:
            gap = line - selected[-1]
            if gap < period - tolerance:
                # Too close — might be noise; keep the one with more "support"
                continue
            elif gap <= period + tolerance:
                selected.append(line)
            elif gap <= 2 * period + tolerance:
                # Missing one line in between — still add this one
                selected.append(line)
            else:
                # Too far — could be a section break; only continue if enough lines remain
                remaining = [l for l in sorted_lines if l > line]
                if len(remaining) + 1 >= min_count:
                    # restart from here
                    candidate = [line]
                    for ll in remaining:
                        g = ll - candidate[-1]
                        if period - tolerance <= g <= period + tolerance:
                            candidate.append(ll)
                        elif g <= 2 * period + tolerance:
                            candidate.append(ll)
                    if len(candidate) > len(selected):
                        selected = candidate

        # Refine period using the final selection
        if len(selected) > 1:
            final_spacings = np.diff(selected)
            # Only use spacings close to one period (exclude double-gaps)
            single_spacings = final_spacings[final_spacings < period * 1.5]
            if len(single_spacings) > 0:
                period = float(np.median(single_spacings))

        return selected, period

    # ──────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────

    @staticmethod
    def _merge_close_lines(lines: list[int], min_gap: float) -> list[int]:
        """Merge lines that are closer than min_gap, keeping the average position."""
        if not lines:
            return lines
        merged = []
        group = [lines[0]]
        for line in lines[1:]:
            if line - group[-1] <= min_gap:
                group.append(line)
            else:
                merged.append(int(np.mean(group)))
                group = [line]
        merged.append(int(np.mean(group)))
        return merged

    @staticmethod
    def _lines_to_intervals(lines: list[int]) -> list[tuple[int, int]]:
        """Convert sorted line positions into (start, end) intervals."""
        intervals = []
        for i in range(len(lines) - 1):
            gap = lines[i + 1] - lines[i]
            if gap > 3:
                intervals.append((lines[i], lines[i + 1]))
        return intervals
