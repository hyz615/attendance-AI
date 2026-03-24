"""Analyze debug cell images to understand classification."""
import cv2
import numpy as np
import os

# Check warped image dimensions
warped = cv2.imread("debug_output/test_run/03_warped.png")
if warped is not None:
    print(f"Warped image: {warped.shape}")

# Check cell images
cells_dir = "debug_output/test_run/cells"
if os.path.exists(cells_dir):
    for f in sorted(os.listdir(cells_dir)):
        img = cv2.imread(os.path.join(cells_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            mean_val = np.mean(img)
            std_val = np.std(img)
            # Otsu threshold
            _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ink_ratio = 1.0 - np.count_nonzero(otsu) / otsu.size
            print(f"  {f}: shape={img.shape}, mean={mean_val:.1f}, std={std_val:.1f}, ink_ratio={ink_ratio:.4f}")
else:
    print("No cells directory found!")

# Also analyze what the classifier does
print("\n--- Re-running classifier on cells ---")
from attendance_ai.pipeline.classify_cell import CellClassifier, normalize_cell_background
import json
config = json.load(open("attendance_ai/config/template.json"))
classifier = CellClassifier(config)

for f in sorted(os.listdir(cells_dir)):
    img = cv2.imread(os.path.join(cells_dir, f), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Step through what the classifier does
        norm = normalize_cell_background(img)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        ink_pixels = cv2.countNonZero(binary)
        total_pixels = binary.shape[0] * binary.shape[1]
        ink_ratio = ink_pixels / total_pixels
        
        label, conf = classifier.classify(img)
        print(f"  {f}: label={label.value}, conf={conf:.3f}, ink_ratio={ink_ratio:.4f}, shape={img.shape}")
        
        # Save the normalized + binary versions for inspection
        cv2.imwrite(f"debug_output/test_run/cells/norm_{f}", norm)
        cv2.imwrite(f"debug_output/test_run/cells/bin_{f}", binary)
