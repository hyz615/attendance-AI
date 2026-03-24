"""Analyze the cell extraction and classification from the new pipeline."""
import cv2
import numpy as np
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

config = json.load(open("attendance_ai/config/template.json"))
orig = cv2.imread("1774321122561.jpg")

from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_table_v2 import TableDetector
from attendance_ai.pipeline.extract_cells import CellExtractor

prep = Preprocessor(config)
preprocessed = prep.run(orig)
gray = preprocessed.get("enhanced", preprocessed["grayscale"])

td = TableDetector(config)
table_info = td.run(gray)

ext = CellExtractor(config)
cell_result = ext.run(gray, table_info)

latest = cell_result["latest_column_index"]
total_cols = len(table_info["cols"])
print(f"Cols: {total_cols}, latest: {latest}")
print(f"Students: {cell_result['total_rows']} filled, {cell_result['all_rows']} total")

# Save cells from the latest column and a few other columns for comparison
os.makedirs("debug_output/cell_analysis", exist_ok=True)
filled = [s for s in cell_result["student_cells"] if not s.get("is_empty", False)]

print(f"\n--- Analyzing latest column (col {latest}) ---")
for i, sc in enumerate(filled[:15]):
    if latest < len(sc["attendance_cells"]):
        cell = sc["attendance_cells"][latest]
        img = cell["image"]
        mean_v = np.mean(img)
        std_v = np.std(img)
        min_v = np.min(img)
        max_v = np.max(img)
        cv2.imwrite(f"debug_output/cell_analysis/row{i}_col{latest}.png", img)
        print(f"  Row {i}: shape={img.shape}, mean={mean_v:.1f}, std={std_v:.1f}, min={min_v}, max={max_v}")

# Also look at a few other columns to see what blank vs marked cells look like
print(f"\n--- Column sample comparison (row 0) ---")
if filled:
    sc = filled[0]
    for col_idx, cell in enumerate(sc["attendance_cells"]):
        img = cell["image"]
        mean_v = np.mean(img)
        std_v = np.std(img)
        cv2.imwrite(f"debug_output/cell_analysis/r0_c{col_idx}.png", img)
        print(f"  Col {col_idx}: shape={img.shape}, mean={mean_v:.1f}, std={std_v:.1f}")

# Check a few name regions
print(f"\n--- Name regions (first 5) ---")
for i, sc in enumerate(filled[:5]):
    name_img = sc.get("name_region")
    if name_img is not None:
        cv2.imwrite(f"debug_output/cell_analysis/name_row{i}.png", name_img)
        print(f"  Row {i}: shape={name_img.shape}, mean={np.mean(name_img):.1f}, std={np.std(name_img):.1f}")

# Verify: is the selected latest column correct? 
# Check what fraction of cells have content in each column
print(f"\n--- Column content analysis ---")
for col_idx in range(total_cols):
    n_with_content = 0
    total_std = 0
    for sc in filled:
        if col_idx < len(sc["attendance_cells"]):
            img = sc["attendance_cells"][col_idx]["image"]
            if img.size > 0 and np.std(img) > 8.0:
                n_with_content += 1
            total_std += np.std(img)
    avg_std = total_std / max(len(filled), 1)
    print(f"  Col {col_idx}: {n_with_content}/{len(filled)} with content, avg_std={avg_std:.1f}")
