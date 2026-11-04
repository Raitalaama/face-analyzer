"""Image input/output utilities for the face analyzer application."""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from a file.

    Args:
        image_path: Path to the image file.

    Returns:
        The loaded image as a NumPy array in uint8 BGR format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not decode image at path: {image_path}")
    return image


def save_image(image: np.ndarray, image_path: str | Path) -> Path:
    """Save an image to a file, creating parent directories if needed.

    The format is chosen from the file extension, e.g. .jpg or .png.

    Args:
        image: Image in BGR format.
        image_path: Destination file path.

    Returns:
        The destination path.

    Raises:
        ValueError: If OpenCV cannot write the image, e.g. because of an
            unsupported file extension.
    """
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(image_path), image)
    except cv2.error as e:
        raise ValueError(f"Could not write image to {image_path}: {e}") from e
    if not written:
        raise ValueError(f"Could not write image to {image_path}.")
    return image_path
