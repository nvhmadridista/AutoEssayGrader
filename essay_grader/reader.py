from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def _resize_max(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Resize keeping aspect ratio so the longest side <= max_side."""
    h, w = img.shape[:2]
    scale = min(max_side / float(max(h, w)), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _deskew(img_gray: np.ndarray) -> np.ndarray:
    """Attempt to deskew an image using the text angle estimated via Hough lines.

    This is a heuristic suitable for simple scanned pages.
    """
    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return img_gray

    # Collect angles near horizontal
    angles = []
    for rho_theta in lines[:100]:
        rho, theta = rho_theta[0]
        # Convert to degrees
        angle = (theta * 180.0 / np.pi) - 90.0
        # Focus on small tilt angles
        if -20 <= angle <= 20:
            angles.append(angle)

    if not angles:
        return img_gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img_gray

    # Rotate to correct skew
    h, w = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def load_and_preprocess(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image, convert to grayscale, deskew, resize.

    Returns a tuple of (original_bgr, processed_gray).
    Raises FileNotFoundError if image cannot be loaded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img = _resize_max(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize contrast and reduce noise
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    gray = cv2.equalizeHist(gray)

    # Deskew
    gray = _deskew(gray)

    return img, gray
