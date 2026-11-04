"""Face detection module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np

type Point = tuple[float, float]


@dataclass(frozen=True)
class Face:
    """A face found by the detector.

    Coordinates are in pixels of the input image. Right and left are from the
    person's point of view, so the right eye appears on the image's left.

    Attributes:
        x: Left edge of the bounding box.
        y: Top edge of the bounding box.
        width: Width of the bounding box.
        height: Height of the bounding box.
        right_eye: Right eye (x, y).
        left_eye: Left eye (x, y).
        nose_tip: Nose tip (x, y).
        right_mouth_corner: Right mouth corner (x, y).
        left_mouth_corner: Left mouth corner (x, y).
        score: Detection confidence, between 0 and 1.
    """

    x: float
    y: float
    width: float
    height: float
    right_eye: Point
    left_eye: Point
    nose_tip: Point
    right_mouth_corner: Point
    left_mouth_corner: Point
    score: float

    @classmethod
    def from_row(cls, row: np.ndarray) -> Self:
        """Create a face from one row of the detector output.

        Args:
            row: Array of 15 values: box x, y, w, h; right eye, left eye,
                nose tip, right mouth corner and left mouth corner as x, y
                pairs; and the confidence score.

        Returns:
            The face described by the row.

        Raises:
            ValueError: If the row does not have 15 values.
        """
        if len(row) != 15:
            raise ValueError(f"Expected a detector row of 15 values, got {len(row)}.")
        v = [float(value) for value in row]
        return cls(
            x=v[0],
            y=v[1],
            width=v[2],
            height=v[3],
            right_eye=(v[4], v[5]),
            left_eye=(v[6], v[7]),
            nose_tip=(v[8], v[9]),
            right_mouth_corner=(v[10], v[11]),
            left_mouth_corner=(v[12], v[13]),
            score=v[14],
        )


class FaceDetector:
    """Face detector using a pre-trained YuNet model."""

    def __init__(self, model_path: Path, confidence_threshold: float = 0.9):
        """Load the YuNet model.

        Args:
            model_path: Path to the YuNet ONNX model file.
            confidence_threshold: Minimum score, between 0 and 1, for a
                detection to be returned.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(0, 0),
            score_threshold=self.confidence_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, image: np.ndarray) -> list[Face]:
        """Detect faces in an image.

        Args:
            image: BGR image as returned by cv2.imread.

        Returns:
            The detected faces, empty if none were found.

        Raises:
            ValueError: If the image is not a 3-channel image. YuNet returns no
                faces for grayscale or BGRA input instead of failing.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected a 3-channel BGR image, got shape {image.shape}.")
        height, width = image.shape[:2]
        self.model.setInputSize((width, height))
        _, rows = self.model.detect(image)
        if rows is None:
            return []
        return [Face.from_row(row) for row in rows]
