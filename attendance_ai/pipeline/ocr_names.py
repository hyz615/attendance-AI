"""OCR module for extracting student names from name cells.

Uses Tesseract OCR to read printed text from the name column.
Falls back gracefully to 'Student N' if Tesseract is unavailable.
"""

import cv2
import numpy as np
import logging
import os
import re

logger = logging.getLogger(__name__)

# Auto-detect Tesseract on Windows
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\Program Files\Tesseract-OCR\tesseract.exe",
]

_tesseract_available = False
try:
    import pytesseract

    # Try to find tesseract binary
    for _p in _TESSERACT_PATHS:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
    pytesseract.get_tesseract_version()
    _tesseract_available = True
    logger.info("Tesseract OCR available")
except Exception:
    logger.warning("Tesseract OCR not available — will use fallback names")


def ocr_name_cell(name_image: np.ndarray, row_index: int) -> str:
    """Extract student name from a name cell image.

    Args:
        name_image: grayscale image of the name cell region
        row_index: 0-based row index (for fallback naming)

    Returns:
        Extracted name string, or 'Student N' if OCR fails
    """
    if not _tesseract_available:
        return f"Student {row_index + 1}"

    if name_image is None or name_image.size == 0:
        return f"Student {row_index + 1}"

    try:
        processed = _preprocess_for_ocr(name_image)
        text = pytesseract.image_to_string(
            processed,
            config="--psm 7 --oem 3",
        )
        text = _clean_ocr_text(text.strip())
        if text and len(text) >= 2:
            return text
        else:
            return f"Student {row_index + 1}"

    except Exception as e:
        logger.debug(f"OCR failed for row {row_index}: {e}")
        return f"Student {row_index + 1}"


def ocr_name_cells_batch(
    student_cells: list[dict],
    gray_image: np.ndarray | None = None,
    table_info: dict | None = None,
) -> list[str]:
    """Extract names for all student rows.

    If gray_image and table_info are provided, uses sub-column OCR
    for much better accuracy (reads last name and first name separately).

    Args:
        student_cells: list of student cell dicts from CellExtractor
        gray_image: full grayscale image (for sub-column extraction)
        table_info: dict from TableDetector (with name_sub_cols)

    Returns:
        List of student name strings
    """
    all_info = ocr_all_fields_batch(student_cells, gray_image, table_info)
    return [info.get("name", f"Student {i+1}") for i, info in enumerate(all_info)]


def ocr_all_fields_batch(
    student_cells: list[dict],
    gray_image: np.ndarray | None = None,
    table_info: dict | None = None,
) -> list[dict]:
    """Extract all info fields (name, grade, OEN, DOB, gender) for all students.

    Returns:
        List of dicts, each with keys: name, grade, oen, dob, gender
    """
    if gray_image is not None and table_info is not None:
        return _ocr_all_sub_columns(student_cells, gray_image, table_info)

    # Fallback: per-cell OCR (names only)
    results = []
    for sc in student_cells:
        idx = sc.get("row_index", len(results))
        name_img = sc.get("name_region")
        name = ocr_name_cell(name_img, idx)
        results.append({"name": name, "grade": "", "oen": "", "dob": "", "gender": ""})
    return results


def _ocr_all_sub_columns(
    student_cells: list[dict],
    gray: np.ndarray,
    table_info: dict,
) -> list[dict]:
    """OCR all info sub-columns: surname, given name, grade, OEN, DOB, gender.

    Ontario CE form sub-columns (typically 6):
      0: row number (narrow, skip)
      1: surname
      2: given name
      3: grade
      4: OEN (9-digit student number)
      5: DOB (DD/MM/YY) — may include gender if no separator detected
    If there are 7+ sub-cols, the last narrow one is gender.
    """
    sub_cols = table_info.get("name_sub_cols", [])
    rows = table_info.get("rows", [])
    header_rows = 0

    if not sub_cols or len(sub_cols) < 2:
        logger.warning("No name sub-columns detected, falling back to full-cell OCR")
        results = []
        for sc in student_cells:
            idx = sc.get("row_index", len(results))
            name = ocr_name_cell(sc.get("name_region"), idx)
            results.append({"name": name, "grade": "", "oen": "", "dob": "", "gender": ""})
        return results

    # Classify sub-columns by width
    col_info = [(x2 - x1, i, x1, x2) for i, (x1, x2) in enumerate(sub_cols)]

    # Skip the first narrow column (row numbers)
    start_idx = 0
    if col_info[0][0] < 40 and len(col_info) > 2:
        start_idx = 1

    # Map field indices: surname, given_name, grade, oen, dob, gender
    field_names = ["surname", "given_name", "grade", "oen", "dob", "gender"]
    remaining = col_info[start_idx:]

    # If only 5 remaining cols (no separate gender), split last wide col into DOB + gender
    if len(remaining) == 5:
        last_w, last_i, last_x1, last_x2 = remaining[-1]
        if last_w > 80:  # wide enough to contain both DOB and gender
            split_x = last_x1 + int(last_w * 0.65)  # DOB gets ~65%, gender ~35%
            remaining[-1] = (split_x - last_x1, last_i, last_x1, split_x)  # dob part
            remaining.append((last_x2 - split_x, last_i, split_x, last_x2))  # gender part
            logger.info(f"Split last sub-col into DOB x:{last_x1}-{split_x} + Gender x:{split_x}-{last_x2}")

    field_cols = {}  # field_name -> (w, i, x1, x2)
    for fi, f in enumerate(field_names):
        if fi < len(remaining):
            field_cols[f] = remaining[fi]

    logger.info(
        "Sub-column mapping: "
        + ", ".join(f"{f}=x:{c[2]}-{c[3]}(w={c[0]})" for f, c in field_cols.items())
    )

    data_rows = rows[header_rows:] if len(rows) > header_rows else rows
    results = []
    sc_idx = 0

    for sc in student_cells:
        row_idx = sc.get("row_index", sc_idx)
        empty_record = {"name": f"Student {row_idx + 1}", "grade": "", "oen": "", "dob": "", "gender": ""}

        if sc.get("is_empty", False) or row_idx >= len(data_rows):
            results.append(empty_record)
            sc_idx += 1
            continue

        y1, y2 = data_rows[row_idx]
        pad_y = max(2, int((y2 - y1) * 0.08))

        record = {"grade": "", "oen": "", "dob": "", "gender": ""}

        # OCR each field
        name_parts = []
        for field, (_, _, x1, x2) in field_cols.items():
            pad_x = max(2, int((x2 - x1) * 0.04))
            cell = gray[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
            if cell.size == 0:
                continue

            if field in ("surname", "given_name"):
                text = _ocr_single_subcol(cell)
                if text:
                    name_parts.append(text)
            elif field == "grade":
                text = _ocr_single_subcol(cell)
                record["grade"] = _clean_grade(text) if text else ""
            elif field == "oen":
                text = _ocr_digits(cell)
                record["oen"] = text
            elif field == "dob":
                text = _ocr_date(cell)
                record["dob"] = text
            elif field == "gender":
                text = _ocr_gender(cell)
                record["gender"] = text

        name = " ".join(name_parts).strip()
        record["name"] = name if len(name) >= 2 else f"Student {row_idx + 1}"
        results.append(record)
        sc_idx += 1

    return results


def _ocr_digits(cell: np.ndarray) -> str:
    """OCR a cell containing digits (OEN)."""
    if not _tesseract_available:
        return ""
    h, w = cell.shape[:2]
    if h < 3 or w < 5:
        return ""
    target_h = 80
    scale = max(4, target_h // max(h, 1))
    up = cv2.resize(cell, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.sum(binary > 127) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)
    try:
        text = pytesseract.image_to_string(
            binary, config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
        )
        return re.sub(r"[^0-9]", "", text.strip())
    except Exception:
        return ""


def _ocr_date(cell: np.ndarray) -> str:
    """OCR a cell containing a date (DD/MM/YY)."""
    if not _tesseract_available:
        return ""
    h, w = cell.shape[:2]
    if h < 3 or w < 5:
        return ""
    target_h = 80
    scale = max(4, target_h // max(h, 1))
    up = cv2.resize(cell, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.sum(binary > 127) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)
    try:
        text = pytesseract.image_to_string(
            binary, config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/MFmf"
        )
        text = text.strip()
        # Keep digits and slashes
        text = re.sub(r"[^0-9/MFmf]", "", text)
        return text
    except Exception:
        return ""


def _clean_grade(text: str) -> str:
    """Clean grade OCR result."""
    text = re.sub(r"[^A-Za-z0-9 ]", "", text).strip()
    return text


def _clean_gender(text: str) -> str:
    """Extract M or F from OCR result."""
    text = text.strip().upper()
    if "M" in text:
        return "M"
    if "F" in text:
        return "F"
    return text[:1] if text else ""


def _ocr_gender(cell: np.ndarray) -> str:
    """OCR a single-character gender cell (M or F)."""
    if not _tesseract_available:
        return ""
    h, w = cell.shape[:2]
    if h < 3 or w < 3:
        return ""
    # Very aggressive upscale for tiny cells
    target_h = 120
    scale = max(6, target_h // max(h, 1))
    up = cv2.resize(cell, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.sum(binary > 127) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)
    try:
        # Try PSM 10 (single character) first
        text = pytesseract.image_to_string(
            binary, config="--psm 10 --oem 3 -c tessedit_char_whitelist=MFmf"
        )
        text = text.strip().upper()
        if text in ("M", "F"):
            return text
        # Fallback: PSM 8 (single word)
        text = pytesseract.image_to_string(
            binary, config="--psm 8 --oem 3 -c tessedit_char_whitelist=MFmf"
        )
        text = text.strip().upper()
        if text in ("M", "F"):
            return text
        return ""
    except Exception:
        return ""


def _ocr_single_subcol(cell: np.ndarray) -> str:
    """OCR a single sub-column cell (e.g., last name or first name)."""
    if not _tesseract_available:
        return ""

    h, w = cell.shape[:2]
    if h < 3 or w < 5:
        return ""

    # Aggressive upscale for tiny cells
    target_h = 80
    scale = max(4, target_h // max(h, 1))
    up = cv2.resize(cell, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Otsu binarization
    blurred = cv2.GaussianBlur(up, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Ensure dark text on white background
    if np.sum(binary > 127) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)

    try:
        text = pytesseract.image_to_string(
            binary,
            config="--psm 7 --oem 3",
        )
        return _clean_ocr_text(text.strip())
    except Exception:
        return ""


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess a name cell image for better OCR accuracy."""
    h, w = image.shape[:2]

    # Aggressive upscale for tiny cells
    target_h = 80
    if h < target_h:
        scale = max(2, target_h // max(h, 1))
        image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Ensure text is dark on white background
    if np.sum(binary > 127) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)

    return binary


def _clean_ocr_text(text: str) -> str:
    """Clean up raw OCR output to extract a reasonable name."""
    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove common OCR garbage characters
    text = re.sub(r"[|_~`@#$%^&*()+=\[\]{}\\/<>0-9]", "", text)

    # Remove single-character fragments
    parts = text.split()
    parts = [p for p in parts if len(p) >= 2 or p in (",", ".")]
    text = " ".join(parts)

    return text.strip()
