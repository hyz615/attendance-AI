"""Image utility functions for the attendance AI pipeline."""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk. Handles unicode paths on Windows."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Use numpy to handle unicode paths on Windows
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    logger.info(f"Loaded image: {path.name} ({img.shape[1]}x{img.shape[0]})")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image maintaining aspect ratio to target width."""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)
    logger.debug(f"Resized from {w}x{h} to {target_width}x{new_h}")
    return resized


def auto_rotate(image: np.ndarray, expected_orientation: str = "landscape") -> np.ndarray:
    """Rotate image to match the expected sheet orientation.

    Args:
        image: input image
        expected_orientation: 'landscape' or 'portrait'
    """
    h, w = image.shape[:2]
    is_landscape = w > h

    if expected_orientation == "landscape":
        if not is_landscape:
            logger.info("Image is portrait but sheet is landscape, rotating 90° CW")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif expected_orientation == "portrait":
        if is_landscape:
            logger.info("Image is landscape but sheet is portrait, rotating 90° CW")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    logger.info(f"Image orientation matches expected ({expected_orientation}), no rotation")
    return image


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform given 4 corner points."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def save_debug_image(image: np.ndarray, output_dir: str, filename: str) -> str:
    """Save a debug/visualization image to output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / filename
    cv2.imwrite(str(filepath), image)
    logger.debug(f"Debug image saved: {filepath}")
    return str(filepath)


def draw_grid_on_image(
    image: np.ndarray,
    h_lines: list,
    v_lines: list,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw detected grid lines on image for debugging."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    for y in h_lines:
        cv2.line(vis, (0, y), (w, y), color, thickness)
    for x in v_lines:
        cv2.line(vis, (x, 0), (x, h), color, thickness)
    return vis


def draw_cells_on_image(
    image: np.ndarray,
    cells: list,
    labels: list = None,
    color_map: dict = None,
) -> np.ndarray:
    """Draw cell bounding boxes with optional labels for debugging."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if color_map is None:
        color_map = {"A": (0, 0, 255), "BLANK": (0, 255, 0), "UNKNOWN": (0, 255, 255)}

    default_color = (255, 255, 0)

    for i, cell in enumerate(cells):
        x, y, cw, ch = cell["x"], cell["y"], cell["w"], cell["h"]
        label = labels[i] if labels and i < len(labels) else None
        c = color_map.get(label, default_color) if label else default_color
        cv2.rectangle(vis, (x, y), (x + cw, y + ch), c, 2)
        if label:
            cv2.putText(vis, label, (x + 2, y + ch - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    return vis
