"""Deeper analysis: projection on binary grid region to find actual grid structure."""
import cv2
import numpy as np
import json
import os
from scipy.signal import find_peaks

os.makedirs("debug_output/diagnosis2", exist_ok=True)

config = json.load(open("attendance_ai/config/template.json"))
orig = cv2.imread("1774321122561.jpg")

from attendance_ai.pipeline.preprocess import Preprocessor
from attendance_ai.pipeline.detect_document import DocumentDetector

prep = Preprocessor(config)
preprocessed = prep.run(orig)
doc = DocumentDetector(config)
doc_result = doc.run(preprocessed)
warped_gray = doc_result["warped_gray"]
h, w = warped_gray.shape

# Template boundaries
y_start = int(h * 0.14)
y_end = int(h * 0.88)
x_grid_start = int(w * 0.36)
x_grid_end = int(w * 0.93)

# Crop grid region
grid = warped_gray[y_start:y_end, x_grid_start:x_grid_end]
gh, gw = grid.shape
print(f"Grid region: {gh}x{gw}")

# Binary of grid region (adaptive threshold)
grid_binary = cv2.adaptiveThreshold(
    grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 8
)
cv2.imwrite("debug_output/diagnosis2/grid_binary.png", grid_binary)

# Horizontal projection on binary grid
h_proj_bin = np.sum(grid_binary, axis=1).astype(float) / (gw * 255)
# h_proj_bin: fraction of black pixels per row (0=all white, 1=all black)

# A grid line row should have many dark pixels (high h_proj_bin)
print(f"\nH-projection on binary: min={h_proj_bin.min():.4f}, max={h_proj_bin.max():.4f}, mean={h_proj_bin.mean():.4f}")

# Find peaks (grid lines)
peaks_h, _ = find_peaks(h_proj_bin, height=0.15, distance=10, prominence=0.02)
print(f"\nHorizontal grid lines from binary projection: {len(peaks_h)}")
for i, p in enumerate(peaks_h):
    print(f"  y_offset={p}, y_abs={p + y_start}, proj={h_proj_bin[p]:.4f}")

if len(peaks_h) > 1:
    spacings = np.diff(peaks_h)
    print(f"\nRow spacings: min={spacings.min()}, max={spacings.max()}, median={np.median(spacings):.1f}, mean={spacings.mean():.1f}")
    print(f"All spacings: {spacings.tolist()}")

# Vertical projection on binary grid
v_proj_bin = np.sum(grid_binary, axis=0).astype(float) / (gh * 255)
peaks_v, _ = find_peaks(v_proj_bin, height=0.15, distance=8, prominence=0.02)
print(f"\nVertical grid lines from binary projection: {len(peaks_v)}")
for i, p in enumerate(peaks_v[:40]):
    print(f"  x_offset={p}, x_abs={p + x_grid_start}, proj={v_proj_bin[p]:.4f}")

if len(peaks_v) > 1:
    spacings_v = np.diff(peaks_v)
    print(f"\nColumn spacings: min={spacings_v.min()}, max={spacings_v.max()}, median={np.median(spacings_v):.1f}, mean={spacings_v.mean():.1f}")

# Also try morphological line detection with SMALL kernels on grid region
# Horizontal lines
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
h_mask = cv2.morphologyEx(grid_binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
cv2.imwrite("debug_output/diagnosis2/h_lines_mask.png", h_mask)
h_proj_morph = np.sum(h_mask, axis=1)
peaks_hm, _ = find_peaks(h_proj_morph, height=gw*0.1, distance=8)
print(f"\nMorphological h-lines (kernel=30): {len(peaks_hm)} lines")
if len(peaks_hm) > 1:
    sp = np.diff(peaks_hm)
    print(f"  Row spacings: min={sp.min()}, max={sp.max()}, median={np.median(sp):.1f}")
    # Show first 10 and last 10
    for i, p in enumerate(peaks_hm[:10]):
        print(f"  Line {i}: y={p + y_start}, proj={h_proj_morph[p]}")
    if len(peaks_hm) > 20:
        print(f"  ... ({len(peaks_hm) - 20} more) ...")
        for i, p in enumerate(peaks_hm[-10:]):
            print(f"  Line {len(peaks_hm)-10+i}: y={p + y_start}, proj={h_proj_morph[p]}")

# Vertical lines  
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
v_mask = cv2.morphologyEx(grid_binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
cv2.imwrite("debug_output/diagnosis2/v_lines_mask.png", v_mask)
v_proj_morph = np.sum(v_mask, axis=0)
peaks_vm, _ = find_peaks(v_proj_morph, height=gh*0.1, distance=8)
print(f"\nMorphological v-lines (kernel=30): {len(peaks_vm)} lines")
if len(peaks_vm) > 1:
    sp = np.diff(peaks_vm)
    print(f"  Col spacings: min={sp.min()}, max={sp.max()}, median={np.median(sp):.1f}")
    for i, p in enumerate(peaks_vm[:30]):
        print(f"  Line {i}: x={p + x_grid_start}, proj={v_proj_morph[p]}")

print("\nDone. Diagnostic images in debug_output/diagnosis2/")
