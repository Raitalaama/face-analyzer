import numpy as np
import pytest
from conftest import GROUP_IMAGE

from face_analyzer.detection import Face, FaceDetector, crop_face
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


def _box_face(x: float, y: float, width: float, height: float) -> Face:
    return Face.from_row(np.array([x, y, width, height, *([0.0] * 10), 0.9]))


def _gradient_image(height: int = 100, width: int = 120) -> np.ndarray:
    """Image where every pixel is distinct, so crops can be compared exactly."""
    ys, xs = np.mgrid[:height, :width]
    return np.dstack([ys, xs, np.full_like(ys, 255)]).astype(np.uint8)


def test_crop_face_is_square_and_centered():
    image = _gradient_image()
    crop = crop_face(image, _box_face(40, 30, 20, 30), margin_pct=0.0)
    # Side is the longer box side (30); the 20-wide box is widened by 5 on each side.
    assert crop.shape == (30, 30, 3)
    assert np.array_equal(crop, image[30:60, 35:65])


def test_crop_face_adds_margin_on_each_side():
    crop = crop_face(_gradient_image(), _box_face(40, 30, 20, 20), margin_pct=0.25)
    assert crop.shape == (30, 30, 3)


def test_crop_face_pads_outside_image_with_black():
    image = _gradient_image()
    crop = crop_face(image, _box_face(-10, -5, 30, 30), margin_pct=0.0)
    assert crop.shape == (30, 30, 3)
    assert not crop[:5].any()  # rows above the image
    assert not crop[:, :10].any()  # columns left of the image
    assert np.array_equal(crop[5:, 10:], image[:25, :20])


def test_crop_face_pads_past_bottom_right_corner():
    image = _gradient_image()
    crop = crop_face(image, _box_face(110, 90, 20, 20), margin_pct=0.0)
    assert crop.shape == (20, 20, 3)
    assert np.array_equal(crop[:10, :10], image[90:, 110:])
    assert not crop[10:].any()
    assert not crop[:, 10:].any()


def test_crop_face_does_not_share_memory_with_image():
    image = _gradient_image()
    crop = crop_face(image, _box_face(40, 30, 20, 20))
    assert not np.shares_memory(crop, image)


def test_crop_face_rejects_negative_margin():
    with pytest.raises(ValueError, match="non-negative"):
        crop_face(_gradient_image(), _box_face(40, 30, 20, 20), margin_pct=-0.1)


def test_crop_face_on_detected_faces_is_square(face_detector: FaceDetector):
    image = load_image(GROUP_IMAGE)
    for face in face_detector.detect(image):
        crop = crop_face(image, face)
        assert crop.shape[0] == crop.shape[1]
        assert crop.shape[2] == 3
