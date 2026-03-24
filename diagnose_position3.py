"""Save annotated image showing exactly which cells are being recognized."""
import cv2
import numpy as np
import json

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
name_col = ti["name_col"]
name_sub = ti.get("name_sub_cols", [])
v_grid = ti["v_lines_grid"]

ext = CellExtractor(config)
cr = ext.run(gray, ti)
latest = cr["latest_column_index"]

# Create annotated image
vis = color.copy()

# Draw all h-lines in green
for y in ti["h_lines"]:
    cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0), 1)

# Draw all v-grid lines in blue
for x in v_grid:
    cv2.line(vis, (x, rows[0][0]), (x, rows[-1][1]), (255, 0, 0), 1)

# Draw name sub-columns in cyan
for x1, x2 in name_sub:
    cv2.line(vis, (x1, rows[0][0]), (x1, rows[-1][1]), (255, 255, 0), 1)

# Mark header rows in yellow
for hr in range(3):
    y1, y2 = rows[hr]
    cv2.rectangle(vis, (0, y1), (vis.shape[1], y2), (0, 255, 255), 1)
    cv2.putText(vis, f"HDR{hr}", (5, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Mark data rows
data_rows = rows[3:]
for i, (y1, y2) in enumerate(data_rows):
    is_empty = cr["student_cells"][i]["is_empty"] if i < len(cr["student_cells"]) else True
    label = f"D{i}" if not is_empty else f"D{i}(E)"
    c = (0, 200, 0) if not is_empty else (128, 128, 128)
    cv2.putText(vis, label, (5, (y1 + y2) // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)

# Highlight the latest column with thick red border
if latest < len(cols):
    x1, x2 = cols[latest]
    y_top = data_rows[0][0]
    y_bot = data_rows[-1][1]
    cv2.rectangle(vis, (x1, y_top), (x2, y_bot), (0, 0, 255), 3)
    cv2.putText(vis, f"LATEST=C{latest}", (x1 - 20, y_top - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Highlight name OCR sub-columns (the ones we use) in magenta
# Our OCR uses the first two widest sub-cols after skipping narrow row# col
used_name_cols = []
if name_sub:
    widths = [(x2 - x1, i, x1, x2) for i, (x1, x2) in enumerate(name_sub)]
    if widths[0][0] < 40 and len(widths) > 2:
        widths = widths[1:]
    used_name_cols = sorted(widths[:2], key=lambda c: c[2])

for w, i, x1, x2 in used_name_cols:
    y_top = data_rows[0][0]
    y_bot = data_rows[-1][1]
    cv2.rectangle(vis, (x1, y_top), (x2, y_bot), (255, 0, 255), 2)
    cv2.putText(vis, f"NAME_{i}", (x1, y_top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

# Add column numbers at top
for i in range(len(cols)):
    x1, x2 = cols[i]
    cx = (x1 + x2) // 2
    cv2.putText(vis, str(i), (cx - 5, rows[0][0] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imwrite("debug_output/annotated_grid.png", vis)
print("Saved: debug_output/annotated_grid.png")

# Also save zoomed sections
# 1. Top-left corner (headers + first 3 data rows, name area + first 5 cols)
zoom1 = vis[rows[0][0] - 10 : data_rows[5][1] + 10, max(0, name_col[0] - 10) : cols[5][1] + 10]
cv2.imwrite("debug_output/zoom_topleft.png", zoom1)
print(f"Saved: debug_output/zoom_topleft.png ({zoom1.shape})")

# 2. Top-right corner (latest column area)
if latest < len(cols):
    x1 = max(0, cols[max(0, latest - 3)][0] - 10)
    x2 = min(vis.shape[1], cols[min(len(cols) - 1, latest + 1)][1] + 10)
    zoom2 = vis[rows[0][0] - 10 : data_rows[5][1] + 10, x1:x2]
    cv2.imwrite("debug_output/zoom_latest_cols.png", zoom2)
    print(f"Saved: debug_output/zoom_latest_cols.png ({zoom2.shape})")

# 3. Full width, first 5 rows only
zoom3 = vis[rows[0][0] - 10 : data_rows[5][1] + 10, :]
h, w = zoom3.shape[:2]
sc = max(1, min(3, 4000 // max(w, 1)))
zoom3_up = cv2.resize(zoom3, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("debug_output/zoom_full_width.png", zoom3_up)
print(f"Saved: debug_output/zoom_full_width.png ({zoom3_up.shape})")

# 4. Save individual name cells that are OCR'd
for i in range(min(5, len(data_rows))):
    y1, y2 = data_rows[i]
    for j, (_, _, nx1, nx2) in enumerate(used_name_cols):
        pad_y = max(2, int((y2 - y1) * 0.08))
        pad_x = max(2, int((nx2 - nx1) * 0.04))
        cell = gray[y1 + pad_y:y2 - pad_y, nx1 + pad_x:nx2 - pad_x]
        # Upscale
        h, w = cell.shape
        sc = max(4, 80 // max(h, 1))
        up = cv2.resize(cell, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"debug_output/name_r{i}_sub{j}.png", up)

# Print what the image looks like at the name sub-col boundaries for row 3 (first data row)
y1, y2 = data_rows[0]
print(f"\nFirst data row (y={y1}-{y2}):")
print(f"  Full name area: x={name_col[0]}-{name_col[1]}")
print(f"  Sub-columns: {name_sub}")
print(f"  Used for OCR: {[(x1, x2) for _, _, x1, x2 in used_name_cols]}")
for _, _, x1, x2 in used_name_cols:
    cell = gray[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2]
    print(f"    x={x1}-{x2}: shape={cell.shape}, mean={cell.mean():.1f}, std={np.std(cell):.1f}")

# Print the entire resized image area that the grid covers
print(f"\nGrid coverage:")
print(f"  X: {name_col[0]} to {cols[-1][1]} ({cols[-1][1] - name_col[0]}px) out of {gray.shape[1]}px ({(cols[-1][1] - name_col[0])/gray.shape[1]*100:.1f}%)")
print(f"  Y: {rows[0][0]} to {rows[-1][1]} ({rows[-1][1] - rows[0][0]}px) out of {gray.shape[0]}px ({(rows[-1][1] - rows[0][0])/gray.shape[0]*100:.1f}%)")
print(f"  Left margin: {name_col[0]}px")
print(f"  Right margin: {gray.shape[1] - cols[-1][1]}px")
print(f"  Top margin: {rows[0][0]}px")
print(f"  Bottom margin: {gray.shape[0] - rows[-1][1]}px")
