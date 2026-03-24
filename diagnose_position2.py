"""Detailed position diagnostic: OCR header, check col0, visualize latest column."""
import cv2
import numpy as np
import json
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_table_v2 import TableDetector
from attendance_ai.pipeline.extract_cells import CellExtractor

img = cv2.imread("1774321122561.jpg")
config = json.load(open("attendance_ai/config/template.json"))
pp = Preprocessor(config)
res = pp.run(img)
gray = res["grayscale"]
detect_gray = res.get("enhanced", gray)

td = TableDetector(config)
ti = td.run(detect_gray)
rows = ti["rows"]
cols = ti["cols"]
v_grid = ti["v_lines_grid"]

# 1. OCR the header rows for each column to understand column labels
print("=== HEADER OCR (identifying what each column means) ===")
for header_row in range(3):
    y1, y2 = rows[header_row]
    print(f"\nHeader Row {header_row} (y={y1}-{y2}):")
    for c_idx in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 29, 30]:
        if c_idx >= len(cols):
            continue
        x1, x2 = cols[c_idx]
        cell = gray[y1:y2, x1:x2]
        h, w = cell.shape
        sc = max(6, 100 // max(h, 1))
        up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
        _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        t = pytesseract.image_to_string(bn, config="--psm 10 --oem 3").strip()
        print(f"  Col {c_idx:2d} (x={x1}-{x2}, w={x2-x1}): '{t}'")

# 2. Check if Col 0 is a date column or something else
print("\n=== COL 0 ANALYSIS (is it a date column?) ===")
x1, x2 = cols[0]
print(f"Col 0: x={x1} to {x2}, width={x2-x1} (others ~{cols[1][1]-cols[1][0]})")

# Save col 0 content for first 5 data rows
for r in range(5):
    y1, y2 = rows[3 + r]
    cell = gray[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2]
    h, w = cell.shape
    sc = max(6, 100 // max(h, 1))
    up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
    _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3").strip()
    print(f"  Row {r} Col 0: '{t}' (mean={cell.mean():.1f}, std={np.std(cell):.1f})")

# 3. Save zoomed header row showing date labels
print("\n=== Date header zoom ===")
# Header row with date numbers (likely row 2)
for hr in range(3):
    y1, y2 = rows[hr]
    # Full header row across all date columns
    x_start = cols[0][0]
    x_end = cols[-1][1]
    header_cells = gray[y1:y2, x_start:x_end]
    h, w = header_cells.shape
    sc = max(4, 80 // max(h, 1))
    up = cv2.resize(header_cells, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"debug_output/header_row{hr}_dates.png", up)
    t = pytesseract.image_to_string(up, config="--psm 7 --oem 3").strip()
    print(f"  Header row {hr}: '{t}'")

# 4. Check the rightmost columns to find where actual data ends
print("\n=== RIGHT-SIDE COLUMN ANALYSIS (checking where data ends) ===")
ext = CellExtractor(config)
cr = ext.run(gray, ti)

# For each of the last 10 columns, count how many non-empty student rows have content
data_rows = rows[3:]
filled_mask = [not sc["is_empty"] for sc in cr["student_cells"]]
filled_rows_data = [(y1, y2) for (y1, y2), f in zip(data_rows, filled_mask) if f]

from attendance_ai.pipeline.classify_cell import normalize_cell_background

for c_idx in range(max(0, len(cols) - 10), len(cols)):
    x1, x2 = cols[c_idx]
    pad_x = max(2, int((x2 - x1) * 0.08))
    filled_count = 0
    ratios = []
    for y1, y2 in filled_rows_data:
        pad_y = max(2, int((y2 - y1) * 0.08))
        cell = gray[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
        if cell.size == 0 or np.std(cell) < 5:
            ratios.append(0.0)
            continue
        norm = normalize_cell_background(cell)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        dr = float(np.sum(binary > 0) / binary.size)
        ratios.append(dr)
        if dr > 0.15:
            filled_count += 1
    mean_dr = np.mean(ratios) if ratios else 0
    print(f"  Col {c_idx:2d} (x={x1}-{x2}): {filled_count}/{len(filled_rows_data)} filled (>0.15), mean_dr={mean_dr:.3f}")

print(f"\nCurrent latest_column_index = {cr['latest_column_index']}")
print(f"Total filled students = {cr['total_rows']}")
