from pathlib import Path

import numpy as np
import pytest

from face_analyzer.detection import Face, FaceDetector
from face_analyzer.image_io import load_image

ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "models" / "face_detection_yunet_2026may.onnx"
GROUP_IMAGE = ROOT / "tests" / "data" / "group.jpg"

needs_model = pytest.mark.skipif(not MODEL_PATH.is_file(), reason="model not downloaded, run `make models`")


def test_face_from_row_maps_columns():
    row = np.arange(15, dtype=np.float32)
    face = Face.from_row(row)
    assert (face.x, face.y, face.width, face.height) == (0, 1, 2, 3)
    assert face.right_eye == (4, 5)
    assert face.left_eye == (6, 7)
    assert face.nose_tip == (8, 9)
    assert face.right_mouth_corner == (10, 11)
    assert face.left_mouth_corner == (12, 13)
    assert face.score == 14


def test_face_from_row_returns_python_floats():
    face = Face.from_row(np.full(15, 0.5, dtype=np.float32))
    assert type(face.score) is float
    assert type(face.nose_tip[0]) is float


def test_face_from_row_rejects_wrong_length():
    with pytest.raises(ValueError, match="15 values"):
        Face.from_row(np.zeros(14))


@needs_model
def test_detect_returns_empty_list_for_blank_image():
    detector = FaceDetector(MODEL_PATH)
    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []


@needs_model
def test_detect_finds_faces_in_group_photo():
    faces = FaceDetector(MODEL_PATH).detect(load_image(GROUP_IMAGE))
    assert len(faces) == 5
    for face in faces:
        assert face.score >= 0.9
        landmarks = (face.right_eye, face.left_eye, face.nose_tip, face.right_mouth_corner, face.left_mouth_corner)
        for px, py in landmarks:
            assert face.x <= px <= face.x + face.width
            assert face.y <= py <= face.y + face.height
        # Right/left are from the person's point of view.
        assert face.right_eye[0] < face.left_eye[0]


@needs_model
@pytest.mark.parametrize("shape", [(50, 50), (50, 50, 1), (50, 50, 4)])
def test_detect_rejects_non_bgr_images(shape):
    with pytest.raises(ValueError, match="3-channel"):
        FaceDetector(MODEL_PATH).detect(np.zeros(shape, dtype=np.uint8))
