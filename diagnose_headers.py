"""Identify column date labels from header + verify cell alignment."""
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
color = res["resized"]

td = TableDetector(config)
ti = td.run(detect_gray)
rows = ti["rows"]
cols = ti["cols"]

# OCR each header cell for ALL header rows
# Use the header row that likely has date numbers
print("=== Header cell OCR (all rows, all cols) ===")
for hr in range(3):
    y1, y2 = rows[hr]
    results = []
    for ci in range(len(cols)):
        x1, x2 = cols[ci]
        cell = gray[y1:y2, x1:x2]
        h, w = cell.shape
        sc = max(8, 120 // max(h, 1))
        up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
        _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Also try with single char mode
        t = pytesseract.image_to_string(bn, config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789").strip()
        results.append(t)
    print(f"HR{hr}: {' '.join(f'{r:>3s}' for r in results)}")

# Save each header cell as separate image for first 10 cols
print("\n=== Saving header cells ===")
for hr in range(3):
    y1, y2 = rows[hr]
    for ci in range(min(10, len(cols))):
        x1, x2 = cols[ci]
        cell = gray[y1:y2, x1:x2]
        sc = max(8, 120 // max(cell.shape[0], 1))
        up = cv2.resize(cell, (cell.shape[1] * sc, cell.shape[0] * sc), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"debug_output/hdr_r{hr}_c{ci}.png", up)

# Also try: read the ENTIRE header row as a single strip for date identification
print("\n=== Full header row strips ===")
for hr in range(3):
    y1, y2 = rows[hr]
    strip = gray[y1:y2, cols[0][0]:cols[-1][1]]
    h, w = strip.shape
    sc = max(8, 120 // max(h, 1))
    up = cv2.resize(strip, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
    _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"debug_output/hdr_strip_{hr}.png", bn)
    t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/ ").strip()
    print(f"  HR{hr}: '{t}'")

# Check vertical alignment: do name cells from name_sub_cols align with
# actual text content? Save aligned visualization
print("\n=== Name cell alignment check ===")
data_rows = rows[3:]
name_sub = ti.get("name_sub_cols", [])

# Draw name sub-column boundaries on a zoomed crop
crop_y1 = rows[2][0]
crop_y2 = data_rows[min(7, len(data_rows)-1)][1]
name_x1 = ti["name_col"][0]
name_x2 = ti["name_col"][1]

name_region = color[crop_y1:crop_y2, name_x1-5:name_x2+5].copy()
for x1, x2 in name_sub:
    rx1 = x1 - name_x1 + 5
    rx2 = x2 - name_x1 + 5
    cv2.line(name_region, (rx1, 0), (rx1, name_region.shape[0]), (0, 0, 255), 1)
    cv2.line(name_region, (rx2, 0), (rx2, name_region.shape[0]), (255, 0, 0), 1)

# Scale up for visibility
h, w = name_region.shape[:2]
sc = max(3, 500 // max(h, 1))
name_vis = cv2.resize(name_region, (w * sc, h * sc), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("debug_output/name_alignment.png", name_vis)
print(f"Saved name_alignment.png (scale {sc}x)")

# Also save attendance cell alignment for latest column
latest = cr["latest_column_index"]
print(f"\n=== Attendance cell alignment (latest col {latest}) ===")
if latest < len(cols):
    att_x1 = cols[max(0, latest-3)][0]
    att_x2 = cols[min(len(cols)-1, latest+1)][1]
    att_region = color[crop_y1:crop_y2, att_x1-5:att_x2+5].copy()
    
    # Mark the latest column
    lx1 = cols[latest][0] - att_x1 + 5
    lx2 = cols[latest][1] - att_x1 + 5
    cv2.rectangle(att_region, (lx1, 0), (lx2, att_region.shape[0]), (0, 0, 255), 1)
    
    h, w = att_region.shape[:2]
    sc = max(3, 500 // max(h, 1))
    att_vis = cv2.resize(att_region, (w * sc, h * sc), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("debug_output/att_alignment.png", att_vis)
    print(f"Saved att_alignment.png (scale {sc}x)")
