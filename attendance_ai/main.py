"""Main attendance processing pipeline — orchestrates all stages."""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from attendance_ai.utils.image_utils import (
    load_image,
    save_debug_image,
    draw_grid_on_image,
    draw_cells_on_image,
)
from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_table_v2 import TableDetector
from attendance_ai.pipeline.extract_cells import CellExtractor
from attendance_ai.pipeline.classify_cell import CellClassifier, CellLabel
from attendance_ai.pipeline.ocr_names import ocr_all_fields_batch
from attendance_ai.pipeline.aggregate import AttendanceAggregator

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "template.json"


def _classify_column_relative(
    cell_images: list[np.ndarray | None],
    config: dict,
) -> list[tuple[CellLabel, float]]:
    """Classify cells using column-relative thresholding + OCR confirmation.

    Two-pass approach:
    1) Compute dark_ratio on center-cropped cells (avoids border artifacts).
    2) For borderline cells, run shape-validated OCR to detect 'A'.
    """
    from attendance_ai.pipeline.classify_cell import normalize_cell_background

    # Step 1: compute dark_ratio on CENTER-CROPPED cells to exclude borders.
    # At ~21px cells, even 1px of border bleed is ~5% area, causing false positives.
    CROP_FRAC = 0.18  # trim 18% from each side → inner 64%
    metrics = []
    for img in cell_images:
        if img is None or img.size == 0:
            metrics.append({"dark_ratio": 0.0, "center_dr": 0.0, "valid": False})
            continue

        cell_std = float(np.std(img))
        if cell_std < 5.0:
            metrics.append({"dark_ratio": 0.0, "center_dr": 0.0, "valid": True})
            continue

        h, w = img.shape[:2]
        norm = normalize_cell_background(img)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Full-cell dark ratio
        dark_ratio = float(np.sum(binary > 0) / max(binary.size, 1))

        # Center-crop dark ratio (excludes border bleed)
        cy1 = int(h * CROP_FRAC)
        cy2 = max(cy1 + 1, h - int(h * CROP_FRAC))
        cx1 = int(w * CROP_FRAC)
        cx2 = max(cx1 + 1, w - int(w * CROP_FRAC))
        center = binary[cy1:cy2, cx1:cx2]
        center_dr = float(np.sum(center > 0) / max(center.size, 1))

        metrics.append({"dark_ratio": dark_ratio, "center_dr": center_dr, "valid": True})

    # Step 2: column baseline using center dark ratio
    valid_ratios = [m["center_dr"] for m in metrics if m["valid"]]
    if not valid_ratios:
        return [(CellLabel.BLANK, 1.0)] * len(cell_images)

    baseline = float(np.percentile(valid_ratios, 25))

    # IQR from lower half only (blank cells)
    median_ratio = float(np.median(valid_ratios))
    lower_half = [r for r in valid_ratios if r <= median_ratio]
    if len(lower_half) >= 4:
        iqr = float(np.percentile(lower_half, 75) - np.percentile(lower_half, 25))
    else:
        iqr = 0.02
    capped_iqr = min(iqr, 0.05)

    # High-confidence threshold (definitely A by center_dr alone)
    high_threshold = max(
        baseline + 1.5 * max(capped_iqr, 0.02) + 0.03,
        baseline * 2.0 + 0.02,
        0.20,
    )

    # OCR candidate threshold
    ocr_candidate_threshold = max(baseline + 0.03, 0.12)

    logger.info(
        f"Column stats: baseline={baseline:.4f}, iqr={iqr:.4f}, "
        f"high_thr={high_threshold:.4f}, ocr_thr={ocr_candidate_threshold:.4f}"
    )

    # Step 3: first pass — pixel-ratio classification on center_dr
    results = []
    ocr_needed_indices = []
    for i, m in enumerate(metrics):
        if not m["valid"]:
            results.append((CellLabel.BLANK, 1.0))
            continue

        dr = m["center_dr"]
        if dr >= high_threshold:
            excess = dr - high_threshold
            conf = min(1.0, 0.5 + excess * 5)
            results.append((CellLabel.A, conf))
        elif dr < 0.04:
            results.append((CellLabel.BLANK, 1.0))
        elif dr >= ocr_candidate_threshold:
            results.append((CellLabel.BLANK, 0.5))  # placeholder
            ocr_needed_indices.append(i)
        else:
            results.append((CellLabel.BLANK, max(0.5, 1.0 - dr / max(high_threshold, 0.01))))

    # Step 4: OCR + shape validation on borderline cells
    if ocr_needed_indices:
        ocr_results = _ocr_detect_a_batch(
            [cell_images[i] for i in ocr_needed_indices]
        )
        for idx, is_a in zip(ocr_needed_indices, ocr_results):
            if is_a:
                cdr = metrics[idx]["center_dr"]
                fdr = metrics[idx]["dark_ratio"]
                # Require minimum full-cell dark ratio to avoid FPs from noise
                if fdr >= 0.10:
                    results[idx] = (CellLabel.A, min(1.0, 0.5 + cdr))

    n_absent = sum(1 for r in results if r[0] == CellLabel.A)
    logger.info(f"Results: {n_absent}A {len(results)-n_absent}P (OCR checked {len(ocr_needed_indices)})")
    return results


def _ocr_detect_a_batch(cell_images: list[np.ndarray | None]) -> list[bool]:
    """Detect 'A' in small attendance cells using shape analysis + OCR.

    Combines contour-shape validation with Tesseract OCR.
    Parallelized via threads.
    """
    try:
        import pytesseract
        import os
        for p in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"D:\Program Files\Tesseract-OCR\tesseract.exe",
        ]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    except ImportError:
        return [False] * len(cell_images)

    from attendance_ai.pipeline.classify_cell import normalize_cell_background
    from concurrent.futures import ThreadPoolExecutor

    def _is_line_like(stats_row, cell_h, cell_w):
        """Check if a connected component looks like a border line, not a character."""
        bw = stats_row[cv2.CC_STAT_WIDTH]
        bh = stats_row[cv2.CC_STAT_HEIGHT]
        bx = stats_row[cv2.CC_STAT_LEFT]
        by = stats_row[cv2.CC_STAT_TOP]
        ba = stats_row[cv2.CC_STAT_AREA]
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        fill = ba / max(bw * bh, 1)
        # Lines: very thin (aspect > 4) and low fill ratio
        if aspect > 4 and fill < 0.5:
            return True
        # Edge-hugging: blob touching edge and very thin
        if (bx <= 1 or bx + bw >= cell_w - 1 or by <= 1 or by + bh >= cell_h - 1):
            if aspect > 3:
                return True
        return False

    def _ocr_one(img):
        if img is None or img.size == 0:
            return False
        try:
            h, w = img.shape[:2]
            if h < 5 or w < 5:
                return False

            norm = normalize_cell_background(img)
            _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if np.sum(binary > 127) / binary.size < 0.5:
                binary = cv2.bitwise_not(binary)

            # Find connected components in the dark content
            inv = cv2.bitwise_not(binary)
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv)
            if n_labels <= 1:
                return False

            # Filter out line-like components (border artifacts)
            char_areas = []
            for lbl_idx in range(1, n_labels):
                if not _is_line_like(stats[lbl_idx], h, w):
                    char_areas.append(stats[lbl_idx][cv2.CC_STAT_AREA])

            # If no character-like blobs remain, it's just border noise
            if not char_areas:
                return False
            total_char_area = sum(char_areas)
            # Character should cover at least 3% of cell
            if total_char_area < binary.size * 0.03:
                return False

            # Upscale for OCR
            target_h = 80
            scale = max(4, target_h // max(h, 1))
            up = cv2.resize(binary, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            border = 10
            up = cv2.copyMakeBorder(up, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=255)

            # Try PSM 10 (single char) with whitelist — primary
            text = pytesseract.image_to_string(
                up,
                config="--psm 10 --oem 3 -c tessedit_char_whitelist=AaPp",
            ).strip().upper()
            if text == "A":
                return True

            # Try PSM 8 (single word) with whitelist
            text8 = pytesseract.image_to_string(
                up,
                config="--psm 8 --oem 3 -c tessedit_char_whitelist=AaPp",
            ).strip().upper()
            if text8 == "A":
                return True

            # Fallback: no whitelist — check if result starts with 'A'
            # (whitelist can sometimes prevent recognition of stylized A)
            text_free = pytesseract.image_to_string(
                up,
                config="--psm 10 --oem 3",
            ).strip().upper()
            return text_free == "A"
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_ocr_one, cell_images))

    return results


def load_config(config_path: str | None = None) -> dict:
    """Load pipeline configuration from JSON file."""
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        logger.warning(f"Config not found at {path}, using defaults")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def process_attendance_sheet(
    image_path: str,
    config_path: str | None = None,
    model_path: str | None = None,
    debug: bool = True,
    debug_dir: str | None = None,
    column_index: int | None = None,
) -> dict:
    """Process an attendance sheet photo and extract attendance data.

    Args:
        image_path: path to the input photo
        config_path: path to template.json config (optional)
        model_path: path to trained CNN model (optional, uses baseline if None)
        debug: whether to save debug visualizations
        debug_dir: directory for debug output (default: debug_output/)
        column_index: specific column to classify (None = auto-detect latest)

    Returns:
        dict with:
            - students: list of {name, status, confidence, raw_label}
            - summary: {should_attend, absent, present}
            - csv_string: CSV output
            - json_summary: JSON summary string
            - debug_images: list of saved debug image paths (if debug=True)
            - timing: dict of stage durations
            - total_columns: int (number of attendance columns)
            - selected_column: int (which column was classified)
    """
    config = load_config(config_path)
    if debug_dir is None:
        debug_dir = config.get("debug", {}).get("output_dir", "debug_output")

    debug_images = []
    timing = {}

    # ─── Stage 1: Preprocessing ───
    t0 = time.time()
    logger.info("▶ Stage 1: Preprocessing")
    image = load_image(image_path)
    preprocessor = Preprocessor(config)
    preprocessed = preprocessor.run(image)
    timing["preprocess"] = round(time.time() - t0, 3)

    if debug:
        debug_images.append(
            save_debug_image(preprocessed["binary"], debug_dir, "01_binary.png")
        )
        debug_images.append(
            save_debug_image(preprocessed["edges"], debug_dir, "02_edges.png")
        )
        if "enhanced" in preprocessed:
            debug_images.append(
                save_debug_image(preprocessed["enhanced"], debug_dir, "01b_enhanced.png")
            )

    # ─── Stage 2: Table Detection (directly on preprocessed image) ───
    t0 = time.time()
    logger.info("▶ Stage 2: Table Detection")
    # Use CLAHE-enhanced image for line detection (better contrast for structure)
    detect_gray = preprocessed.get("enhanced", preprocessed["grayscale"])
    # Use raw grayscale for cell extraction (CLAHE distorts cell content)
    cell_gray = preprocessed["grayscale"]
    work_color = preprocessed["resized"]
    table_detector = TableDetector(config)
    table_info = table_detector.run(detect_gray)
    timing["table_detection"] = round(time.time() - t0, 3)

    if debug:
        grid_vis = draw_grid_on_image(
            work_color, table_info["h_lines"], table_info["v_lines_grid"]
        )
        debug_images.append(
            save_debug_image(grid_vis, debug_dir, "03_grid.png")
        )

    # ─── Stage 3: Cell Extraction ───
    t0 = time.time()
    logger.info("▶ Stage 3: Cell Extraction")
    extractor = CellExtractor(config)
    cell_result = extractor.run(cell_gray, table_info)
    timing["cell_extraction"] = round(time.time() - t0, 3)

    student_cells = cell_result["student_cells"]
    total_columns = len(table_info["cols"])
    auto_latest_col = cell_result["latest_column_index"]
    # Use user-specified column if valid, otherwise auto-detected
    if column_index is not None and 0 <= column_index < total_columns:
        latest_col = column_index
    else:
        latest_col = auto_latest_col

    if not student_cells:
        logger.error("No student rows found!")
        return {
            "students": [],
            "summary": {"should_attend": 0, "absent": 0, "present": 0},
            "csv_string": "Student,Status\n",
            "json_summary": '{"should_attend": 0, "absent": 0, "present": 0}',
            "debug_images": debug_images,
            "timing": timing,
            "error": "No student rows detected in the image.",
        }

    # ─── Stage 4: Cell Classification (ALL columns) ───
    t0 = time.time()
    logger.info("▶ Stage 4: Cell Classification (all %d columns)", total_columns)

    filled_student_cells = [sc for sc in student_cells if not sc.get("is_empty", False)]

    # Build per-column classification: grid[col_idx] = list of (label, conf) per filled student
    # Parallelize across columns since each is independent
    from concurrent.futures import ThreadPoolExecutor

    def _classify_col(col_idx):
        col_images = []
        for sc in filled_student_cells:
            if col_idx < len(sc["attendance_cells"]):
                col_images.append(sc["attendance_cells"][col_idx]["image"])
            else:
                col_images.append(None)
        return _classify_column_relative(col_images, config)

    with ThreadPoolExecutor(max_workers=4) as pool:
        all_col_classifications = list(pool.map(_classify_col, range(total_columns)))

    timing["classification"] = round(time.time() - t0, 3)

    # For backward compat, extract single-column classifications for the selected column
    classifications = all_col_classifications[latest_col] if latest_col < total_columns else []

    if debug:
        # Visualize ALL columns
        cells_for_vis = []
        vis_labels = []
        for col_idx in range(total_columns):
            col_cls = all_col_classifications[col_idx]
            for sc, (lbl, _conf) in zip(filled_student_cells, col_cls):
                if col_idx < len(sc["attendance_cells"]):
                    cd = sc["attendance_cells"][col_idx]
                    cells_for_vis.append({"x": cd["x"], "y": cd["y"], "w": cd["w"], "h": cd["h"]})
                    vis_labels.append(lbl.value)
        cell_vis = draw_cells_on_image(work_color, cells_for_vis, vis_labels)
        debug_images.append(
            save_debug_image(cell_vis, debug_dir, "05_classifications.png")
        )

    # ─── Stage 5: OCR All Info Fields ───
    t0 = time.time()
    logger.info("▶ Stage 5: OCR All Fields")
    student_info = ocr_all_fields_batch(
        filled_student_cells,
        gray_image=cell_gray,
        table_info=table_info,
    )
    student_names = [info.get("name", f"Student {i+1}") for i, info in enumerate(student_info)]
    timing["ocr_names"] = round(time.time() - t0, 3)
    logger.info(f"  OCR extracted {len(student_names)} student records")

    # ─── Stage 6: Aggregation ───
    t0 = time.time()
    logger.info("▶ Stage 6: Aggregation")
    aggregator = AttendanceAggregator()
    result = aggregator.run(
        filled_student_cells, classifications, latest_col,
        student_names=student_names,
    )

    # Build full attendance grid: each student gets a list of statuses across all columns
    grid = []
    for row_i in range(len(filled_student_cells)):
        row_statuses = []
        for col_idx in range(total_columns):
            lbl, conf = all_col_classifications[col_idx][row_i]
            row_statuses.append("A" if lbl == CellLabel.A else "P")
        grid.append(row_statuses)

    result["grid"] = grid
    result["student_info"] = student_info
    timing["aggregation"] = round(time.time() - t0, 3)

    result["debug_images"] = debug_images
    result["timing"] = timing
    result["total_columns"] = total_columns
    result["selected_column"] = latest_col

    logger.info(f"✅ Pipeline complete. Timing: {timing}")
    return result
