"""Test different OCR approaches on name cells."""
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
nc = ti["name_col"]

name_v_lines = [355, 385, 501, 627, 681, 794, 946]

# Approach 1: Full strip OCR
y_start = rows[3][0]
y_end = rows[-1][1]
strip = gray[y_start:y_end, nc[0] + 4 : nc[1] - 4]
h, w = strip.shape
row_h = h // len(rows[3:])
scale = max(4, 100 // max(row_h, 1))
strip_up = cv2.resize(strip, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
_, strip_bin = cv2.threshold(strip_up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
text = pytesseract.image_to_string(strip_bin, config="--psm 6 --oem 3")
lines = [l.strip() for l in text.split("\n") if l.strip()]
print("== Full strip OCR (PSM 6) ==")
for i, l in enumerate(lines[:20]):
    print(f"  {i}: {l}")
print(f"  Total lines: {len(lines)}")
print()

# Approach 2: Per-row, sub-columns separately
print("== Per-row sub-columns ==")
for i in range(min(8, len(rows) - 3)):
    ri = rows[3 + i]
    parts = []
    for x1, x2, label in [
        (385, 501, "A"),
        (501, 627, "B"),
        (681, 794, "C"),
        (794, 946, "D"),
    ]:
        cell = gray[ri[0] + 2 : ri[1] - 2, x1 + 2 : x2 - 2]
        ch, cw = cell.shape
        if ch < 4 or cw < 4:
            parts.append(("", label))
            continue
        sc = max(4, 80 // max(ch, 1))
        up = cv2.resize(cell, (cw * sc, ch * sc), interpolation=cv2.INTER_CUBIC)
        _, bn = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        t = pytesseract.image_to_string(bn, config="--psm 7 --oem 3").strip()
        parts.append((t, label))
    print(f"  Row {i}: {' | '.join(f'{lb}={t}' for t, lb in parts)}")
print()

# Approach 3: Per-row, combined name sub-cols (x=385 to x=946) with aggressive upscale
print("== Per-row full name area (x=385-946, scale 6x) ==")
for i in range(min(10, len(rows) - 3)):
    ri = rows[3 + i]
    cell = gray[ri[0] + 2 : ri[1] - 2, 385 + 2 : 946 - 2]
    ch, cw = cell.shape
    sc = max(6, 120 // max(ch, 1))
    up = cv2.resize(cell, (cw * sc, ch * sc), interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold for uneven backgrounds
    adaptive = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    t = pytesseract.image_to_string(adaptive, config="--psm 7 --oem 3").strip()
    print(f"  Row {i}: \"{t}\"")
