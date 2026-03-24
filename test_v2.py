"""Test the new projection-based table detector on the example image."""
import cv2
import numpy as np
import json
import logging
import os

logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

os.makedirs("debug_output/test_v2", exist_ok=True)

config = json.load(open("attendance_ai/config/template.json"))
orig = cv2.imread("1774321122561.jpg")
print(f"Original: {orig.shape}")

# Preprocess (rotate + resize + CLAHE) but skip document detection
from attendance_ai.pipeline.preprocess import Preprocessor
prep = Preprocessor(config)
preprocessed = prep.run(orig)
enhanced = preprocessed.get("enhanced", preprocessed["grayscale"])
print(f"Enhanced: {enhanced.shape}")
cv2.imwrite("debug_output/test_v2/00_enhanced.png", enhanced)

# New table detector
from attendance_ai.pipeline.detect_table_v2 import TableDetector
td = TableDetector(config)
table_info = td.run(enhanced)

h_lines = table_info["h_lines"]
v_lines_grid = table_info["v_lines_grid"]
rows = table_info["rows"]
cols = table_info["cols"]
name_col = table_info["name_col"]

print(f"\n=== RESULTS ===")
print(f"H-lines: {len(h_lines)}")
print(f"V-lines (grid): {len(v_lines_grid)}")
print(f"Rows: {len(rows)}")
print(f"Cols: {len(cols)}")
print(f"Name col: {name_col}")

if rows:
    heights = [r[1]-r[0] for r in rows]
    print(f"Row heights: min={min(heights)}, max={max(heights)}, median={np.median(heights):.1f}")
    print(f"  First 5 rows: {rows[:5]}")
    print(f"  Last 5 rows: {rows[-5:]}")

if cols:
    widths = [c[1]-c[0] for c in cols]
    print(f"Col widths: min={min(widths)}, max={max(widths)}, median={np.median(widths):.1f}")
    print(f"  First 5 cols: {cols[:5]}")
    print(f"  Last 5 cols: {cols[-5:]}")

# Draw on enhanced image (convert to color for annotation)
vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Draw horizontal grid lines
for y in h_lines:
    cv2.line(vis, (0, y), (enhanced.shape[1], y), (0, 255, 0), 1)

# Draw vertical grid lines    
for x in v_lines_grid:
    cv2.line(vis, (x, 0), (x, enhanced.shape[0]), (0, 0, 255), 1)

# Draw name column
cv2.rectangle(vis, (name_col[0], h_lines[0] if h_lines else 0), 
              (name_col[1], h_lines[-1] if h_lines else enhanced.shape[0]), (255, 0, 0), 2)

cv2.imwrite("debug_output/test_v2/01_grid_overlay.png", vis)

# Extract and save a few cells
if rows and cols:
    header_rows = config.get("table", {}).get("header_rows", 2)
    data_rows = rows[header_rows:] if len(rows) > header_rows else rows
    print(f"\nData rows: {len(data_rows)}")
    
    # Save first 5 name cell crops
    for i, (y1, y2) in enumerate(data_rows[:5]):
        nx1, nx2 = name_col
        name_crop = enhanced[max(0,y1+2):min(enhanced.shape[0],y2-2), 
                             max(0,nx1+2):min(enhanced.shape[1],nx2-2)]
        if name_crop.size > 0:
            cv2.imwrite(f"debug_output/test_v2/name_row{i}.png", name_crop)
            print(f"  Name row {i}: shape={name_crop.shape}, mean={np.mean(name_crop):.1f}, std={np.std(name_crop):.1f}")
    
    # Save first 5 attendance cells from last column
    if cols:
        last_col_idx = len(cols) - 1
        cx1, cx2 = cols[last_col_idx]
        for i, (y1, y2) in enumerate(data_rows[:5]):
            cell = enhanced[max(0,y1+2):min(enhanced.shape[0],y2-2),
                           max(0,cx1+2):min(enhanced.shape[1],cx2-2)]
            if cell.size > 0:
                cv2.imwrite(f"debug_output/test_v2/cell_row{i}_lastcol.png", cell)
                print(f"  Cell [{i}, last]: shape={cell.shape}, mean={np.mean(cell):.1f}, std={np.std(cell):.1f}")

print("\nDone. Images in debug_output/test_v2/")
