"""Classify ALL columns to find the real latest date column."""
import cv2
import numpy as np
import json
import logging
logging.basicConfig(level=logging.INFO)

from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_table_v2 import TableDetector
from attendance_ai.pipeline.extract_cells import CellExtractor
from attendance_ai.pipeline.classify_cell import normalize_cell_background

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

ext = CellExtractor(config)
cr = ext.run(gray, ti)

# Get filled student rows
data_rows = rows[3:]
filled_mask = [not sc["is_empty"] for sc in cr["student_cells"]]
filled_indices = [i for i, f in enumerate(filled_mask) if f]
print(f"Total data rows: {len(data_rows)}, filled: {len(filled_indices)}")
print()

# For each column, run column-relative classification
print("=== Column-relative classification for ALL columns ===")
print(f"{'Col':>4s} {'X range':>12s} {'nA':>4s} {'nP':>4s} {'baseline':>8s} {'threshold':>9s} {'Pattern'}")

for c_idx in range(len(cols)):
    x1, x2 = cols[c_idx]
    pad_x = max(2, int((x2 - x1) * 0.08))
    
    # Extract cells for this column
    cell_images = []
    for ri in filled_indices:
        y1, y2 = data_rows[ri]
        pad_y = max(2, int((y2 - y1) * 0.08))
        cell = gray[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
        cell_images.append(cell)
    
    # Compute dark ratios
    ratios = []
    for cell in cell_images:
        if cell.size == 0 or np.std(cell) < 5:
            ratios.append(0.0)
            continue
        norm = normalize_cell_background(cell)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        dr = float(np.sum(binary > 0) / max(binary.size, 1))
        ratios.append(dr)
    
    if not ratios:
        continue
    
    baseline = float(np.median(ratios))
    iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
    threshold = max(baseline + 1.5 * max(iqr, 0.02), baseline * 1.8 + 0.03)
    
    n_absent = sum(1 for dr in ratios if dr > threshold)
    n_present = len(ratios) - n_absent
    
    # Simple visual pattern: A for absent cells, . for present
    pattern = ""
    for dr in ratios[:20]:
        if dr > threshold:
            pattern += "A"
        else:
            pattern += "."
    
    print(f"C{c_idx:02d}  x={x1:4d}-{x2:4d}  {n_absent:3d}  {n_present:3d}  {baseline:7.4f}  {threshold:8.4f}  {pattern}")

print()
print("=== Summary: columns with highest absent count ===")
# Also check the right region beyond the grid
print("\n=== Content beyond grid (x > 1742) ===")
# Check raw v-lines beyond grid
all_v = sorted(ti['v_lines'])
right_v = [x for x in all_v if x > 1742]
print(f"Raw v-lines beyond grid: {right_v}")
# Try to OCR a few cells from the right region
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if right_v:
    for x in right_v:
        # Header cell
        for hr in range(3):
            y1, y2 = rows[hr]
            cell = gray[y1:y2, max(x-30, 0):min(x+30, gray.shape[1])]
            if cell.size == 0:
                continue
            h, w = cell.shape
            sc = max(4, 80 // max(h, 1))
            up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
            _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3").strip()
            print(f"  Header {hr} at x~{x}: '{t}'")
    # Data cell at the right boundary columns
    y1, y2 = data_rows[0]
    for x in right_v:
        width_guess = 56  # similar to col0
        cell = gray[y1 + 2:y2 - 2, x + 2:min(x + width_guess - 2, gray.shape[1])]
        if cell.size > 0:
            print(f"  Data row 0, x~{x}: mean={cell.mean():.1f}, std={np.std(cell):.1f}")
