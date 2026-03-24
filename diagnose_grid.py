"""Deep diagnosis: dump line positions, cell coordinates, and pixel analysis."""
import cv2
import numpy as np
import json

# Load config and image
config = json.load(open("attendance_ai/config/template.json"))
img = cv2.imread("debug_output/test_run/03_warped.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Warped image: {img.shape}")  # H, W, C

# Run table detection to get line positions
from attendance_ai.pipeline.detect_table import TableDetector
td = TableDetector(config)
table_info = td.run(gray)

h_lines = sorted(table_info["h_lines"])
v_lines = sorted(table_info["v_lines"])
print(f"\nH-lines ({len(h_lines)}): {h_lines}")
print(f"V-lines ({len(v_lines)}): {v_lines}")

# Show row heights
print("\nRow heights:")
for i in range(len(h_lines) - 1):
    h = h_lines[i + 1] - h_lines[i]
    print(f"  Row {i}: y={h_lines[i]}-{h_lines[i+1]}, height={h}")

# Show column widths
print("\nColumn widths:")
for i in range(len(v_lines) - 1):
    w = v_lines[i + 1] - v_lines[i]
    print(f"  Col {i}: x={v_lines[i]}-{v_lines[i+1]}, width={w}")

# Run cell extraction
from attendance_ai.pipeline.extract_cells import CellExtractor
ce = CellExtractor(config)
cell_result = ce.run(gray, table_info)
print(f"\nExtracted: {len(cell_result['student_cells'])} students, latest col={cell_result['latest_column_index']}")

# Show first few cell coordinates
for i, sc in enumerate(cell_result["student_cells"][:5]):
    name_img = sc.get("name_image")
    print(f"\nStudent {i}:")
    if name_img is not None:
        print(f"  Name image: shape={name_img.shape}")
    for j, ac in enumerate(sc["attendance_cells"]):
        print(f"  Att col {j}: x={ac['x']}, y={ac['y']}, w={ac['w']}, h={ac['h']}, img_shape={ac['image'].shape}")

# Now test: what does a REAL attendance cell look like?
# Pick a cell that should be checkmark/blank and examine pixel distribution
latest_col = cell_result["latest_column_index"]
print(f"\n--- Pixel analysis of latest column cells (col {latest_col}) ---")
for i, sc in enumerate(cell_result["student_cells"]):
    if latest_col < len(sc["attendance_cells"]):
        cell = sc["attendance_cells"][latest_col]
        cimg = cell["image"]
        if cimg.size > 0:
            mean_v = np.mean(cimg)
            std_v = np.std(cimg)
            min_v = np.min(cimg)
            max_v = np.max(cimg)
            # histogram bins
            hist = cv2.calcHist([cimg], [0], None, [256], [0, 256]).flatten()
            peak_bin = np.argmax(hist)
            print(f"  Student {i}: mean={mean_v:.1f}, std={std_v:.1f}, min={min_v}, max={max_v}, peak_bin={peak_bin}, shape={cimg.shape}")
