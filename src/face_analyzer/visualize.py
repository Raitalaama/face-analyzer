"""Visualization utilities for the face analyzer application."""

from pathlib import Path

import cv2
import numpy as np

from face_analyzer.detection import Face
from face_analyzer.image_io import save_image

BOX_COLOR = (0, 255, 0)
LANDMARK_COLOR = (0, 0, 255)


def draw_faces(image: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Draw detected faces on a copy of an image.

    Each face gets its bounding box, its five landmarks and its score.

    Args:
        image: Image in BGR format. It is not modified.
        faces: Faces to draw.

    Returns:
        A copy of the image with the faces drawn on it.
    """
    image = image.copy()
    for face in faces:
        x, y, w, h = round(face.x), round(face.y), round(face.width), round(face.height)
        cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOR, 2)
        for px, py in (face.right_eye, face.left_eye, face.nose_tip, face.right_mouth_corner, face.left_mouth_corner):
            cv2.circle(image, (round(px), round(py)), 2, LANDMARK_COLOR, -1)
        cv2.putText(image, f"{face.score:.2f}", (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
    return image


def save_faces(image: np.ndarray, faces: list[Face], output_path: str | Path) -> Path:
    """Draw detected faces on an image and save the result.

    Args:
        image: Image in BGR format. It is not modified.
        faces: Faces to draw.
        output_path: Destination file path. The extension sets the format.

    Returns:
        The destination path.

    Raises:
        ValueError: If the image cannot be written.
    """
    return save_image(draw_faces(image, faces), output_path)
