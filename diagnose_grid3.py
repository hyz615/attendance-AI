"""Test robust grid detection via large-kernel morphology + intersection analysis."""
import cv2
import numpy as np
import json
import os
from scipy.signal import find_peaks

os.makedirs("debug_output/diagnosis3", exist_ok=True)

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
print(f"Image: {h}x{w}")

# Template boundaries
y_start = int(h * 0.14)
y_end = int(h * 0.88)
x_name_start = int(w * 0.01)
x_name_end = int(w * 0.36)
x_grid_start = int(w * 0.36)
x_grid_end = int(w * 0.93)

# Work on entire image (not cropped) for line detection
binary = cv2.adaptiveThreshold(
    warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 25, 12
)

# ─── Horizontal lines with LARGE kernel ───
# Use 15% of image width as minimum line length
h_kernel_len = max(int(w * 0.15), 100)
print(f"\nH-kernel length: {h_kernel_len}")
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
cv2.imwrite("debug_output/diagnosis3/h_mask_large.png", h_mask)

# Project and find lines
h_proj = np.sum(h_mask, axis=1)
h_threshold = np.max(h_proj) * 0.15
peaks_h, _ = find_peaks(h_proj, height=h_threshold, distance=8)
print(f"H-lines (large kernel): {len(peaks_h)}")
for i, p in enumerate(peaks_h):
    print(f"  y={p}, proj={h_proj[p]}")

# ─── Vertical lines with LARGE kernel ───
v_kernel_len = max(int(h * 0.15), 100)
print(f"\nV-kernel length: {v_kernel_len}")
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
cv2.imwrite("debug_output/diagnosis3/v_mask_large.png", v_mask)

v_proj = np.sum(v_mask, axis=0)
v_threshold = np.max(v_proj) * 0.15
peaks_v, _ = find_peaks(v_proj, height=v_threshold, distance=8)
print(f"V-lines (large kernel): {len(peaks_v)}")
for i, p in enumerate(peaks_v):
    print(f"  x={p}, proj={v_proj[p]}")

# ─── Intersections ───
intersections = cv2.bitwise_and(h_mask, v_mask)
# Dilate to make intersections more visible
inter_dilated = cv2.dilate(intersections, np.ones((5, 5)), iterations=2)
cv2.imwrite("debug_output/diagnosis3/intersections.png", inter_dilated)

# Find intersection points
contours, _ = cv2.findContours(inter_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centers = []
for c in contours:
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

print(f"\nIntersection points: {len(centers)}")
if centers:
    centers.sort(key=lambda p: (p[1], p[0]))
    # Cluster by Y to find rows
    ys = sorted(set([c[1] for c in centers]))
    print(f"Unique Y values: {len(ys)}")
    # Merge close Y values
    merged_ys = []
    group = [ys[0]]
    for y in ys[1:]:
        if y - group[-1] <= 10:
            group.append(y)
        else:
            merged_ys.append(int(np.mean(group)))
            group = [y]
    merged_ys.append(int(np.mean(group)))
    print(f"Merged Y rows: {len(merged_ys)}")
    for y in merged_ys:
        print(f"  y={y}")
    
    if len(merged_ys) > 1:
        spacings = np.diff(merged_ys)
        print(f"\nRow spacings: {spacings.tolist()}")
        print(f"  median={np.median(spacings):.1f}")

    # Cluster by X to find columns
    xs = sorted(set([c[0] for c in centers]))
    merged_xs = []
    group = [xs[0]]
    for x in xs[1:]:
        if x - group[-1] <= 10:
            group.append(x)
        else:
            merged_xs.append(int(np.mean(group)))
            group = [x]
    merged_xs.append(int(np.mean(group)))
    print(f"\nMerged X columns: {len(merged_xs)}")
    for x in merged_xs:
        print(f"  x={x}")

# ─── Also try: detect grid in just the right half (attendance area) ───
print("\n\n=== Grid-area only detection ===")
grid_gray = warped_gray[y_start:y_end, x_grid_start:x_grid_end]
gh, gw = grid_gray.shape
grid_bin = cv2.adaptiveThreshold(grid_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)

# Use 60% of grid width for horizontal kernel (very strict)
gh_kernel_len = max(int(gw * 0.6), 100)
gh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gh_kernel_len, 1))
gh_mask = cv2.morphologyEx(grid_bin, cv2.MORPH_OPEN, gh_kernel, iterations=2)
cv2.imwrite("debug_output/diagnosis3/grid_h_mask_strict.png", gh_mask)

gh_proj = np.sum(gh_mask, axis=1)
gh_threshold = np.max(gh_proj) * 0.15
gpeaks_h, _ = find_peaks(gh_proj, height=gh_threshold, distance=10)
print(f"Grid H-lines (strict, kernel={gh_kernel_len}): {len(gpeaks_h)}")
for i, p in enumerate(gpeaks_h):
    print(f"  y_offset={p}, y_abs={p + y_start}")

if len(gpeaks_h) > 1:
    sp = np.diff(gpeaks_h)
    print(f"Row spacings: {sp.tolist()}")
    print(f"  median={np.median(sp):.1f}")

# Use 60% of grid height for vertical kernel (very strict)
gv_kernel_len = max(int(gh * 0.6), 100)
gv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gv_kernel_len))
gv_mask = cv2.morphologyEx(grid_bin, cv2.MORPH_OPEN, gv_kernel, iterations=2)
cv2.imwrite("debug_output/diagnosis3/grid_v_mask_strict.png", gv_mask)

gv_proj = np.sum(gv_mask, axis=0)
gv_threshold = np.max(gv_proj) * 0.15
gpeaks_v, _ = find_peaks(gv_proj, height=gv_threshold, distance=10)
print(f"Grid V-lines (strict, kernel={gv_kernel_len}): {len(gpeaks_v)}")
for i, p in enumerate(gpeaks_v):
    print(f"  x_offset={p}, x_abs={p + x_grid_start}")

if len(gpeaks_v) > 1:
    sp = np.diff(gpeaks_v)
    print(f"Col spacings: {sp.tolist()}")
    print(f"  median={np.median(sp):.1f}")

# Save annotated image with detected grid
annotated = warped.copy()
for p in gpeaks_h:
    y_abs = p + y_start
    cv2.line(annotated, (x_grid_start, y_abs), (x_grid_end, y_abs), (0, 255, 0), 1)
for p in gpeaks_v:
    x_abs = p + x_grid_start
    cv2.line(annotated, (x_abs, y_start), (x_abs, y_end), (0, 0, 255), 1)
cv2.imwrite("debug_output/diagnosis3/detected_grid.png", annotated)

print("\nDone.")
