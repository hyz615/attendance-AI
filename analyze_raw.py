"""Analyze cells from raw grayscale (not CLAHE) to understand classification."""
import cv2
import numpy as np
import json
import logging

logging.basicConfig(level=logging.WARNING)

config = json.load(open("attendance_ai/config/template.json"))
orig = cv2.imread("1774321122561.jpg")

from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_table_v2 import TableDetector
from attendance_ai.pipeline.extract_cells import CellExtractor
from attendance_ai.pipeline.classify_cell import CellClassifier, normalize_cell_background

prep = Preprocessor(config)
preprocessed = prep.run(orig)

# Use CLAHE for detection, raw gray for cells
detect_gray = preprocessed.get("enhanced", preprocessed["grayscale"])
cell_gray = preprocessed["grayscale"]

td = TableDetector(config)
table_info = td.run(detect_gray)
ext = CellExtractor(config)

# Run extraction on raw grayscale
cell_result = ext.run(cell_gray, table_info)
latest = cell_result["latest_column_index"]
filled = [s for s in cell_result["student_cells"] if not s.get("is_empty", False)]

print(f"Total cols: {len(table_info['cols'])}, latest col: {latest}")
print(f"Students: {len(filled)}")

# Analyze the latest column cells on RAW grayscale
print(f"\n--- Latest column (col {latest}) cell analysis ---")
for i, sc in enumerate(filled[:20]):
    if latest < len(sc["attendance_cells"]):
        cell = sc["attendance_cells"][latest]
        img = cell["image"]
        mean_v = np.mean(img)
        std_v = np.std(img)
        
        # What does the classifier see after normalization?
        norm = normalize_cell_background(img)
        norm_std = np.std(norm)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        dark_ratio = np.sum(binary > 0) / binary.size
        
        print(f"  Row {i:2d}: mean={mean_v:5.1f} std={std_v:4.1f} | norm_std={norm_std:4.1f} dark_ratio={dark_ratio:.4f}")

# Also show a sample of column patterns for the first few rows
print(f"\n--- All columns for row 0 (raw grayscale) ---")
sc = filled[0]
for col_idx, cell in enumerate(sc["attendance_cells"]):
    img = cell["image"]
    mean_v = np.mean(img)
    std_v = np.std(img)
    tag = "BLANK" if std_v < 8 else ""
    print(f"  Col {col_idx:2d}: mean={mean_v:5.1f} std={std_v:4.1f} {tag}")

# Show column 10 (which was dark on CLAHE) vs column 11 (which was light)
print(f"\n--- Col 10 vs Col 11 for all rows ---")
for i, sc in enumerate(filled[:20]):
    imgs = []
    for cidx in [10, 11]:
        if cidx < len(sc["attendance_cells"]):
            img = sc["attendance_cells"][cidx]["image"]
            imgs.append(f"mean={np.mean(img):5.1f} std={np.std(img):4.1f}")
        else:
            imgs.append("N/A")
    print(f"  Row {i:2d}: Col10=[{imgs[0]}]  Col11=[{imgs[1]}]")
