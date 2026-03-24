"""Test HoughLinesP for grid detection, and test skipping document warping."""
import cv2
import numpy as np
import json
import os
from scipy.signal import find_peaks

os.makedirs("debug_output/diagnosis4", exist_ok=True)

config = json.load(open("attendance_ai/config/template.json"))
orig = cv2.imread("1774321122561.jpg")
print(f"Original: {orig.shape}")  # H, W, C

# Work on original image directly (no document detection)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
oh, ow = gray.shape

# Apply CLAHE for better contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
cv2.imwrite("debug_output/diagnosis4/00_enhanced.png", enhanced)

# Estimate the attendance grid area in the original image
# The form should be taking up most of the photo
# Let's do adaptive threshold and look for lines
binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 21, 10)

# HoughLinesP on the original
edges = cv2.Canny(enhanced, 30, 100)
cv2.imwrite("debug_output/diagnosis4/01_edges.png", edges)

# Detect lines with HoughLinesP
min_line_length = min(oh, ow) * 0.15  # at least 15% of smaller dimension
max_line_gap = 10
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, 
                         minLineLength=min_line_length, maxLineGap=max_line_gap)

if lines is None:
    print("No lines found!")
else:
    print(f"HoughLinesP found {len(lines)} line segments")
    
    h_lines = []  # (y, x1, x2) for horizontal lines
    v_lines = []  # (x, y1, y2) for vertical lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        
        if angle < 5:  # Nearly horizontal
            y_avg = (y1 + y2) / 2
            h_lines.append((y_avg, min(x1, x2), max(x1, x2), length))
        elif angle > 85:  # Nearly vertical  
            x_avg = (x1 + x2) / 2
            v_lines.append((x_avg, min(y1, y2), max(y1, y2), length))
    
    print(f"Horizontal segments: {len(h_lines)}")
    print(f"Vertical segments: {len(v_lines)}")
    
    # Cluster horizontal lines by y position
    if h_lines:
        h_ys = sorted([l[0] for l in h_lines])
        merged_h = []
        group = [h_ys[0]]
        for y in h_ys[1:]:
            if y - group[-1] <= 5:
                group.append(y)
            else:
                merged_h.append(int(np.mean(group)))
                group = [y]
        merged_h.append(int(np.mean(group)))
        print(f"\nClustered horizontal lines: {len(merged_h)}")
        for i, y in enumerate(merged_h):
            count = sum(1 for l in h_lines if abs(l[0] - y) <= 5)
            total_len = sum(l[3] for l in h_lines if abs(l[0] - y) <= 5)
            print(f"  y={y}, segments={count}, total_length={total_len:.0f}")
    
    if v_lines:
        v_xs = sorted([l[0] for l in v_lines])
        merged_v = []
        group = [v_xs[0]]
        for x in v_xs[1:]:
            if x - group[-1] <= 5:
                group.append(x)
            else:
                merged_v.append(int(np.mean(group)))
                group = [x]
        merged_v.append(int(np.mean(group)))
        print(f"\nClustered vertical lines: {len(merged_v)}")
        for i, x in enumerate(merged_v):
            count = sum(1 for l in v_lines if abs(l[0] - x) <= 5)
            total_len = sum(l[3] for l in v_lines if abs(l[0] - y) <= 5)
            print(f"  x={x}, segments={count}")
    
    # Draw all detected lines on the image
    vis = orig.copy()
    for y_avg, x1, x2, length in h_lines:
        cv2.line(vis, (int(x1), int(y_avg)), (int(x2), int(y_avg)), (0, 255, 0), 1)
    for x_avg, y1, y2, length in v_lines:
        cv2.line(vis, (int(x_avg), int(y1)), (int(x_avg), int(y2)), (0, 0, 255), 1)
    cv2.imwrite("debug_output/diagnosis4/02_hough_lines.png", vis)

# Also: try direct projection on the original enhanced image
# Focus on the right portion (attendance grid)
print("\n\n=== Direct projection on original (right half) ===")
# Estimate grid area on original: roughly right 60% of the image
x_start = int(ow * 0.35)
x_end = int(ow * 0.95)
y_top = int(oh * 0.10)
y_bot = int(oh * 0.92)

grid_crop = enhanced[y_top:y_bot, x_start:x_end]
cv2.imwrite("debug_output/diagnosis4/03_grid_crop.png", grid_crop)
print(f"Grid crop: {grid_crop.shape}")

# Binary of grid crop
grid_bin = cv2.adaptiveThreshold(grid_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 8)

# Morphological horizontal lines with moderate kernel
gch, gcw = grid_crop.shape
h_kern = max(int(gcw * 0.3), 50)
print(f"Grid h-kernel: {h_kern}")
h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kern, 1))
h_mask = cv2.morphologyEx(grid_bin, cv2.MORPH_OPEN, h_ker, iterations=1)
cv2.imwrite("debug_output/diagnosis4/04_grid_h_mask.png", h_mask)

h_proj = np.sum(h_mask, axis=1)
if h_proj.max() > 0:
    h_thresh = h_proj.max() * 0.15
    peaks_h, _ = find_peaks(h_proj, height=h_thresh, distance=5)
    print(f"Grid H-lines: {len(peaks_h)}")
    if len(peaks_h) > 1:
        sp = np.diff(peaks_h)
        print(f"  Row spacings: min={sp.min()}, max={sp.max()}, median={np.median(sp):.1f}")
        print(f"  First 10: {[p + y_top for p in peaks_h[:10]]}")
        print(f"  Last 10: {[p + y_top for p in peaks_h[-10:]]}")

# Vertical lines  
v_kern = max(int(gch * 0.3), 50)
print(f"Grid v-kernel: {v_kern}")
v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kern))
v_mask = cv2.morphologyEx(grid_bin, cv2.MORPH_OPEN, v_ker, iterations=1)
cv2.imwrite("debug_output/diagnosis4/05_grid_v_mask.png", v_mask)

v_proj = np.sum(v_mask, axis=0)
if v_proj.max() > 0:
    v_thresh = v_proj.max() * 0.15
    peaks_v, _ = find_peaks(v_proj, height=v_thresh, distance=5)
    print(f"Grid V-lines: {len(peaks_v)}")
    if len(peaks_v) > 1:
        sp = np.diff(peaks_v)
        print(f"  Col spacings: min={sp.min()}, max={sp.max()}, median={np.median(sp):.1f}")
        for p in peaks_v:
            print(f"    x_abs={p + x_start}")

# Draw the detected grid on original
vis2 = orig.copy()
if h_proj.max() > 0:
    for p in peaks_h:
        y_abs = p + y_top
        cv2.line(vis2, (x_start, y_abs), (x_end, y_abs), (0, 255, 0), 1)
if v_proj.max() > 0:
    for p in peaks_v:
        x_abs = p + x_start
        cv2.line(vis2, (x_abs, y_top), (x_abs, y_bot), (0, 0, 255), 1)
cv2.imwrite("debug_output/diagnosis4/06_detected_grid_on_orig.png", vis2)

print("\nDone.")
