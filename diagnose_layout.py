"""Comprehensive visual diagnosis of the attendance sheet."""
import cv2
import numpy as np
import json
import os

os.makedirs("debug_output/diagnosis", exist_ok=True)

# Load original
orig = cv2.imread("1774321122561.jpg")
print(f"Original image: {orig.shape}")  # H, W, C
gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# Run preprocessing to get the actual warped image
from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_document import DocumentDetector

config = json.load(open("attendance_ai/config/template.json"))
prep = Preprocessor(config)
preprocessed = prep.run(orig)
doc = DocumentDetector(config)
doc_result = doc.run(preprocessed)
warped = doc_result["warped"]
warped_gray = doc_result["warped_gray"]
print(f"Warped image: {warped.shape}")

# Save warped with a grid overlay at template ratio positions
h, w = warped_gray.shape
annotated = warped.copy()
# Template region boundaries
y_start = int(h * 0.14)
y_end = int(h * 0.88)
x_name_start = int(w * 0.01)
x_name_end = int(w * 0.36)
x_grid_start = int(w * 0.36)
x_grid_end = int(w * 0.93)

# Draw template regions
cv2.rectangle(annotated, (x_name_start, y_start), (x_name_end, y_end), (0, 255, 0), 2)
cv2.rectangle(annotated, (x_grid_start, y_start), (x_grid_end, y_end), (0, 0, 255), 2)
cv2.putText(annotated, "NAME AREA", (x_name_start+10, y_start+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv2.putText(annotated, "GRID AREA", (x_grid_start+10, y_start+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imwrite("debug_output/diagnosis/00_template_regions.png", annotated)

# Crop the table region
table_region = warped_gray[y_start:y_end, :]
cv2.imwrite("debug_output/diagnosis/01_table_region.png", table_region)
print(f"Table region: {table_region.shape}")

# Crop just the grid area
grid_region = warped_gray[y_start:y_end, x_grid_start:x_grid_end]
cv2.imwrite("debug_output/diagnosis/02_grid_region.png", grid_region)
print(f"Grid region: {grid_region.shape}")

# Crop just the name area
name_region = warped_gray[y_start:y_end, x_name_start:x_name_end]
cv2.imwrite("debug_output/diagnosis/03_name_region.png", name_region)
print(f"Name region: {name_region.shape}")

# Horizontal projection of the full table region
h_proj = np.mean(table_region, axis=1)
print(f"\nHorizontal projection: min={h_proj.min():.1f}, max={h_proj.max():.1f}, mean={h_proj.mean():.1f}")

# Find the darkest rows (grid lines appear as dark horizontal bands)
# Invert: grid lines are dark, so low values = lines
h_proj_inv = 255 - h_proj  # now high = dark = grid line
threshold_h = np.mean(h_proj_inv) + 0.5 * np.std(h_proj_inv)
print(f"H projection threshold: {threshold_h:.1f}")

# Find peaks in h_proj_inv
from scipy.signal import find_peaks
try:
    peaks_h, props_h = find_peaks(h_proj_inv, height=threshold_h, distance=10)
    print(f"Horizontal line candidates: {len(peaks_h)}")
    for i, p in enumerate(peaks_h):
        print(f"  Line {i}: y_offset={p} -> y_abs={p + y_start}, intensity={h_proj_inv[p]:.1f}")
except ImportError:
    # Manual peak finding
    peaks_h = []
    for i in range(1, len(h_proj_inv) - 1):
        if h_proj_inv[i] > threshold_h and h_proj_inv[i] >= h_proj_inv[i-1] and h_proj_inv[i] >= h_proj_inv[i+1]:
            if not peaks_h or (i - peaks_h[-1]) > 10:
                peaks_h.append(i)
    print(f"Horizontal line candidates (manual): {len(peaks_h)}")
    for i, p in enumerate(peaks_h):
        print(f"  Line {i}: y_offset={p} -> y_abs={p + y_start}, intensity={h_proj_inv[p]:.1f}")

# Vertical projection of the grid area
v_proj = np.mean(grid_region, axis=0)
v_proj_inv = 255 - v_proj
threshold_v = np.mean(v_proj_inv) + 0.5 * np.std(v_proj_inv)

try:
    peaks_v, props_v = find_peaks(v_proj_inv, height=threshold_v, distance=5)
    print(f"\nVertical line candidates: {len(peaks_v)}")
    for i, p in enumerate(peaks_v[:30]):  # show first 30
        print(f"  Line {i}: x_offset={p} -> x_abs={p + x_grid_start}, intensity={v_proj_inv[p]:.1f}")
except ImportError:
    peaks_v = []
    for i in range(1, len(v_proj_inv) - 1):
        if v_proj_inv[i] > threshold_v and v_proj_inv[i] >= v_proj_inv[i-1] and v_proj_inv[i] >= v_proj_inv[i+1]:
            if not peaks_v or (i - peaks_v[-1]) > 5:
                peaks_v.append(i)
    print(f"\nVertical line candidates (manual): {len(peaks_v)}")
    for i, p in enumerate(peaks_v[:30]):
        print(f"  Line {i}: x_offset={p} -> x_abs={p + x_grid_start}, intensity={v_proj_inv[p]:.1f}")

# Save projection plots as simple images
proj_h_img = np.zeros((len(h_proj), 300), dtype=np.uint8)
for i, v in enumerate(h_proj):
    proj_h_img[i, :int(v * 300 / 255)] = 200
cv2.imwrite("debug_output/diagnosis/04_h_projection.png", proj_h_img)

proj_v_img = np.zeros((300, len(v_proj)), dtype=np.uint8)
for i, v in enumerate(v_proj):
    proj_v_img[:int(v * 300 / 255), i] = 200
cv2.imwrite("debug_output/diagnosis/05_v_projection.png", proj_v_img)

# Also: look at binary of the grid region with different methods
_, binary_otsu = cv2.threshold(grid_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite("debug_output/diagnosis/06_grid_binary_otsu.png", binary_otsu)

binary_adapt = cv2.adaptiveThreshold(grid_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imwrite("debug_output/diagnosis/07_grid_binary_adaptive.png", binary_adapt)

print("\nDiagnostic images saved to debug_output/diagnosis/")
