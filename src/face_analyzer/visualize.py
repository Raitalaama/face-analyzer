"""Visualization utilities for the face analyzer application."""

from pathlib import Path

import cv2
import numpy as np

from face_analyzer.detection import Face
from face_analyzer.image_io import save_image

# Colors are BGR. The person's right side is drawn in warm colors and the left
# side in cool colors, so swapped or mirrored landmarks are easy to spot.
BOX_COLOR = (0, 255, 0)  # green
RIGHT_EYE_COLOR = (0, 0, 255)  # red
LEFT_EYE_COLOR = (255, 0, 0)  # blue
NOSE_TIP_COLOR = (0, 255, 255)  # yellow
RIGHT_MOUTH_CORNER_COLOR = (255, 0, 255)  # magenta
LEFT_MOUTH_CORNER_COLOR = (255, 255, 0)  # cyan


def draw_faces(image: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Draw detected faces on a copy of an image.

    Each face gets its bounding box, its score and its five landmarks, each
    landmark in its own color: right eye red, left eye blue, nose tip yellow,
    right mouth corner magenta and left mouth corner cyan.

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
        landmarks = (
            (face.right_eye, RIGHT_EYE_COLOR),
            (face.left_eye, LEFT_EYE_COLOR),
            (face.nose_tip, NOSE_TIP_COLOR),
            (face.right_mouth_corner, RIGHT_MOUTH_CORNER_COLOR),
            (face.left_mouth_corner, LEFT_MOUTH_CORNER_COLOR),
        )
        for (px, py), color in landmarks:
            cv2.circle(image, (round(px), round(py)), 3, color, -1)
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
