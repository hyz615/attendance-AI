"""Diagnose which column/row positions are being recognized."""
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
color = res["resized"]

td = TableDetector(config)
ti = td.run(detect_gray)

rows = ti["rows"]
cols = ti["cols"]
h_lines = ti["h_lines"]
v_grid = ti["v_lines_grid"]
name_col = ti["name_col"]

header_rows = 3
data_rows = rows[header_rows:]

print(f"=== Grid: {len(rows)} rows, {len(cols)} cols ===")
print(f"Header rows (0-2): y={rows[0][0]}-{rows[2][1]}")
print(f"Data rows (3-{len(rows)-1}): y={data_rows[0][0]}-{data_rows[-1][1]}")
print(f"Col 0: x={cols[0][0]}-{cols[0][1]} (w={cols[0][1]-cols[0][0]})")
print(f"Col 1: x={cols[1][0]}-{cols[1][1]} (w={cols[1][1]-cols[1][0]})")
print(f"Col 30: x={cols[30][0]}-{cols[30][1]} (w={cols[30][1]-cols[30][0]})")
print()

# Analyze dark_ratio for ALL columns across a few filled student rows
# to see which columns actually have content
print("=== Dark ratio per column for first 5 data rows ===")
print(f"{'Row':>4s}", end="")
for c in range(len(cols)):
    print(f" C{c:02d}", end="")
print()

for r in range(min(10, len(data_rows))):
    y1, y2 = data_rows[r]
    pad_y = max(2, int((y2 - y1) * 0.08))
    print(f"R{r:02d}:", end="")
    for c in range(len(cols)):
        x1, x2 = cols[c]
        pad_x = max(2, int((x2 - x1) * 0.08))
        cell = gray[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
        if cell.size == 0:
            print("  -- ", end="")
            continue
        std = np.std(cell)
        if std < 5:
            print("  .  ", end="")
            continue
        norm = normalize_cell_background(cell)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        dr = np.sum(binary > 0) / binary.size
        mark = " *" if dr > 0.15 else "  "
        print(f"{dr:4.2f}{mark[1]}", end="")
    print()

print()
print("Legend: . = uniform (blank), * = likely 'A' mark")
print()

# Draw the grid on the color image with col numbers
vis = color.copy()
for i, (y1, y2) in enumerate(rows):
    cv2.line(vis, (0, y1), (vis.shape[1], y1), (0, 255, 0), 1)
for i, x in enumerate(v_grid):
    cv2.line(vis, (x, rows[0][0]), (x, rows[-1][1]), (255, 0, 0), 1)
    # Label columns
    if i < len(cols):
        cx = (cols[i][0] + cols[i][1]) // 2
        cv2.putText(vis, str(i), (cx - 5, rows[0][0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

# Mark the "latest column" with a thick red box
ext = CellExtractor(config)
cr = ext.run(gray, ti)
latest = cr["latest_column_index"]
if latest < len(cols):
    x1, x2 = cols[latest]
    y_top = data_rows[0][0]
    y_bot = data_rows[-1][1]
    cv2.rectangle(vis, (x1, y_top), (x2, y_bot), (0, 0, 255), 2)
    cv2.putText(vis, f"LATEST={latest}", (x1, y_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Also mark non-empty rows
for i, sc in enumerate(cr["student_cells"]):
    if not sc["is_empty"]:
        y1, y2 = data_rows[i]
        cv2.putText(vis, f"R{i}", (5, (y1 + y2) // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

cv2.imwrite("debug_output/grid_positions.png", vis)
print("Saved: debug_output/grid_positions.png")

# Also save zoomed view of the header rows to see what column labels are
header_region = color[rows[0][0]:rows[2][1], name_col[0]:cols[-1][1]]
h, w = header_region.shape[:2]
header_up = cv2.resize(header_region, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("debug_output/header_zoomed.png", header_up)
print("Saved: debug_output/header_zoomed.png")
