import numpy as np
import pytest
from conftest import GROUP_IMAGE

from face_analyzer.detection import Face, FaceDetector
from face_analyzer.image_io import load_image


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


def test_detect_returns_empty_list_for_blank_image(face_detector: FaceDetector):
    assert face_detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []


def test_detect_finds_faces_in_group_photo(face_detector: FaceDetector):
    faces = face_detector.detect(load_image(GROUP_IMAGE))
    assert len(faces) == 5
    for face in faces:
        assert face.score >= 0.9
        landmarks = (face.right_eye, face.left_eye, face.nose_tip, face.right_mouth_corner, face.left_mouth_corner)
        for px, py in landmarks:
            assert face.x <= px <= face.x + face.width
            assert face.y <= py <= face.y + face.height
        # Right/left are from the person's point of view.
        assert face.right_eye[0] < face.left_eye[0]


@pytest.mark.parametrize("shape", [(50, 50), (50, 50, 1), (50, 50, 4)])
def test_detect_rejects_non_bgr_images(face_detector: FaceDetector, shape):
    with pytest.raises(ValueError, match="3-channel"):
        face_detector.detect(np.zeros(shape, dtype=np.uint8))
