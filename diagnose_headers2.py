"""OCR header cells to identify column labels."""
import cv2
import numpy as np
import json
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_table_v2 import TableDetector

img = cv2.imread("1774321122561.jpg")
config = json.load(open("attendance_ai/config/template.json"))
pp = Preprocessor(config)
res = pp.run(img)
gray = res["grayscale"]
detect_gray = res.get("enhanced", gray)

td = TableDetector(config)
ti = td.run(detect_gray)
rows = ti["rows"]
name_sub = ti.get("name_sub_cols", [])
cols = ti["cols"]

# OCR header for each name sub-column
print("=== Name sub-column headers ===")
for hr in range(3):
    y1, y2 = rows[hr]
    for i, (x1, x2) in enumerate(name_sub):
        cell = gray[y1:y2, x1:x2]
        h, w = cell.shape
        if h < 3 or w < 5:
            continue
        sc = max(8, 120 // max(h, 1))
        up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
        _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3").strip()
        print(f'  HR{hr} SubCol{i} (x={x1}-{x2}, w={x2-x1}): "{t}"')

# OCR the entire header area as one block
print("\n=== Full header block OCR ===")
header_y1 = rows[0][0]
header_y2 = rows[2][1]
# Name area headers
name_hdr = gray[header_y1:header_y2, name_sub[0][0]:name_sub[-1][1]]
h, w = name_hdr.shape
sc = max(6, 200 // max(h, 1))
up = cv2.resize(name_hdr, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
_, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("debug_output/name_header_block.png", bn)
t = pytesseract.image_to_string(bn, config="--psm 6 --oem 3").strip()
print(f'Name header: "{t}"')

# Date area headers - try row by row with higher scale
print("\n=== Date header rows (10x scale, adaptive) ===")
for hr in range(3):
    y1, y2 = rows[hr]
    strip = gray[y1:y2, cols[0][0]:cols[-1][1]]
    h, w = strip.shape
    sc = max(10, 150 // max(h, 1))
    up = cv2.resize(strip, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
    adaptive = cv2.adaptiveThreshold(
        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    cv2.imwrite(f"debug_output/hdr_strip_adaptive_{hr}.png", adaptive)
    t = pytesseract.image_to_string(adaptive, config="--psm 7 --oem 3").strip()
    print(f'  HR{hr}: "{t}"')

# Check what the summary columns to the right contain
print("\n=== Summary columns (right of date grid) ===")
# From earlier: v-lines at absolute x=1748, 1802, 1856
summary_cols = [(1742, 1802), (1802, 1856), (1856, 2107)]
for sc_i, (x1, x2) in enumerate(summary_cols):
    x2_clip = min(x2, gray.shape[1])
    # Header
    for hr in range(3):
        y1, y2 = rows[hr]
        cell = gray[y1:y2, x1:x2_clip]
        h, w = cell.shape
        if h < 3 or w < 3:
            continue
        sc = max(8, 120 // max(h, 1))
        up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
        _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3").strip()
        print(f'  Summary col {sc_i} (x={x1}-{x2_clip}) HR{hr}: "{t}"')
    # First data row
    dy1, dy2 = rows[3]
    cell = gray[dy1:dy2, x1:x2_clip]
    h, w = cell.shape
    if h >= 3 and w >= 3:
        scs = max(8, 120 // max(h, 1))
        up = cv2.resize(cell, (w * scs, h * scs), interpolation=cv2.INTER_CUBIC)
        _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3").strip()
        print(f'  Summary col {sc_i} (x={x1}-{x2_clip}) Data0: "{t}"')
