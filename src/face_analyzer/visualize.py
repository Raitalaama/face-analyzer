"""Visualization utilities for the face analyzer application."""

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from face_analyzer.detection import Face
from face_analyzer.image_io import save_image
from face_analyzer.labels import Emotion

if TYPE_CHECKING:
    # Only for type hints: importing classification at runtime loads torch and transformers.
    from face_analyzer.classification import EmotionPrediction

# Colors are BGR. The person's right side is drawn in warm colors and the left
# side in cool colors, so swapped or mirrored landmarks are easy to spot.
BOX_COLOR = (0, 255, 0)  # green
RIGHT_EYE_COLOR = (0, 0, 255)  # red
LEFT_EYE_COLOR = (255, 0, 0)  # blue
NOSE_TIP_COLOR = (0, 255, 255)  # yellow
RIGHT_MOUTH_CORNER_COLOR = (255, 0, 255)  # magenta
LEFT_MOUTH_CORNER_COLOR = (255, 255, 0)  # cyan
EMOTION_TEXT_COLOR = (0, 0, 0)  # black, on a label in BOX_COLOR


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


def draw_emotions(
    image: np.ndarray, faces: list[Face], predictions: list[EmotionPrediction], top_k: int = 1
) -> np.ndarray:
    """Draw each face's most probable emotions on a copy of an image.

    The top_k emotions and their probabilities, e.g. "happy 0.87", are drawn
    one per line, most probable first, on a label just below the face's
    bounding box, or inside the bottom of the box if there is no room below it.
    Labels near the left or right edge are shifted to stay inside the image.

    Args:
        image: Image in BGR format. It is not modified.
        faces: Faces the predictions belong to.
        predictions: One prediction per face, in the same order as faces.
        top_k: How many of the most probable emotions to list per face.
            Ties keep canonical Emotion order.

    Returns:
        A copy of the image with the emotions drawn on it.

    Raises:
        ValueError: If faces and predictions have different lengths, or top_k
            is not between 1 and the number of emotions.
    """
    if len(faces) != len(predictions):
        raise ValueError(f"Expected one prediction per face, got {len(predictions)} for {len(faces)} faces.")
    if not 1 <= top_k <= len(Emotion):
        raise ValueError(f"top_k must be between 1 and {len(Emotion)}, got {top_k}.")
    image = image.copy()
    font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 3
    for face, prediction in zip(faces, predictions, strict=True):
        ranked = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
        lines = [f"{emotion} {probability:.2f}" for emotion, probability in ranked[:top_k]]
        sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
        text_w = max(w for (w, _), _ in sizes)
        line_h = max(h + baseline for (_, h), baseline in sizes)
        text_h = max(h for (_, h), _ in sizes)
        label_h = len(lines) * line_h + 2 * pad
        label_w = text_w + 2 * pad
        img_h, img_w = image.shape[:2]
        # cv2.rectangle includes its end corner, so the label's last pixel is at x + label_w.
        x = max(0, min(round(face.x), img_w - 1 - label_w))
        top = round(face.y + face.height)
        if top + label_h > img_h - 1:
            top -= label_h
        top = max(0, top)
        cv2.rectangle(image, (x, top), (x + label_w, top + label_h), BOX_COLOR, -1)
        for i, line in enumerate(lines):
            origin = (x + pad, top + pad + i * line_h + text_h)
            cv2.putText(image, line, origin, font, scale, EMOTION_TEXT_COLOR, thickness)
    return image


def save_faces(
    image: np.ndarray,
    faces: list[Face],
    output_path: str | Path,
    emotion_predictions: list[EmotionPrediction] | None = None,
    emotion_top_k: int = 1,
) -> Path:
    """Draw detected faces on an image and save the result.

    Args:
        image: Image in BGR format. It is not modified.
        faces: Faces to draw.
        output_path: Destination file path. The extension sets the format.
        emotion_predictions: Optional predictions, one per face in the same
            order. If given, each face's emotions are drawn too.
        emotion_top_k: How many of the most probable emotions to draw per
            face when emotion_predictions is given.

    Returns:
        The destination path.

    Raises:
        ValueError: If the image cannot be written, emotion_predictions
            does not have one prediction per face, or emotion_top_k is out
            of range.
    """
    drawn = draw_faces(image, faces)
    if emotion_predictions is not None:
        drawn = draw_emotions(drawn, faces, emotion_predictions, top_k=emotion_top_k)
    return save_image(drawn, output_path)
