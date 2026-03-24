"""Build diagnostic montages of classified cells."""
import cv2
import numpy as np
import os
import glob

def build_montage(folder, title, output_path, max_cells=100, cell_display_size=60):
    """Build a montage image from cell PNGs in a folder."""
    files = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not files:
        print(f"No files in {folder}")
        return
    
    # Sample evenly if too many
    if len(files) > max_cells:
        step = len(files) / max_cells
        files = [files[int(i * step)] for i in range(max_cells)]
    
    cols_per_row = 20
    rows_needed = (len(files) + cols_per_row - 1) // cols_per_row
    
    pad = 2
    label_h = 14
    tile_w = cell_display_size + pad * 2
    tile_h = cell_display_size + pad * 2 + label_h
    
    canvas_w = cols_per_row * tile_w
    canvas_h = rows_needed * tile_h + 30  # 30 for title
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(canvas, f"{title} ({len(files)} shown)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    for i, fpath in enumerate(files):
        row = i // cols_per_row
        col = i % cols_per_row
        x = col * tile_w + pad
        y = row * tile_h + 30 + pad
        
        img = cv2.imread(fpath)
        if img is None:
            continue
        
        # Resize to display size
        resized = cv2.resize(img, (cell_display_size, cell_display_size))
        canvas[y:y+cell_display_size, x:x+cell_display_size] = resized
        
        # Label with filename info (dark_ratio)
        fname = os.path.basename(fpath)
        # Extract dr from filename like c00_r00_dr0.100.png
        parts = fname.replace('.png', '').split('_')
        label = parts[0] + parts[1]  # c00r00
        dr_part = [p for p in parts if p.startswith('dr')]
        if dr_part:
            label += ' ' + dr_part[0]
        
        cv2.putText(canvas, label, (x, y + cell_display_size + 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 200), 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"Saved {output_path} ({canvas_w}x{canvas_h})")


if __name__ == '__main__':
    build_montage('debug_output/cells_A', 'Detected as A', 
                  'debug_output/montage_A.png', max_cells=200)
    build_montage('debug_output/cells_borderline', 'Borderline (classified P)', 
                  'debug_output/montage_borderline.png', max_cells=200)
    
    # Also build montages sorted by dark_ratio for the borderline ones
    # to see which ones are closest to being classified as A
    files = sorted(glob.glob('debug_output/cells_borderline/*.png'))
    # Sort by dark_ratio (extracted from filename)
    def get_dr(f):
        try:
            return float(os.path.basename(f).split('dr')[1].replace('.png', ''))
        except:
            return 0.0
    files_by_dr = sorted(files, key=get_dr, reverse=True)
    
    # Save top-50 highest dark_ratio borderline cells
    top_dir = 'debug_output/cells_borderline_top'
    os.makedirs(top_dir, exist_ok=True)
    for f in files_by_dr[:50]:
        import shutil
        shutil.copy2(f, top_dir)
    
    build_montage(top_dir, 'Top 50 Borderline (highest dark_ratio)', 
                  'debug_output/montage_borderline_top50.png', max_cells=50)
    
    print("\nDone. Check debug_output/montage_*.png")
