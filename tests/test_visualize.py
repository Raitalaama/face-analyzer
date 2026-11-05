import numpy as np
import pytest

from face_analyzer.classification import EmotionPrediction
from face_analyzer.detection import Face
from face_analyzer.image_io import load_image, save_image
from face_analyzer.labels import Emotion
from face_analyzer.visualize import BOX_COLOR, draw_emotions, draw_faces, save_faces


def _face() -> Face:
    return Face.from_row(np.array([10, 20, 30, 40, 15, 30, 30, 30, 22, 40, 15, 50, 30, 50, 0.95]))


def test_draw_faces_returns_annotated_copy():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    drawn = draw_faces(image, [_face()])
    assert not image.any()
    assert drawn.any()


def test_save_faces_writes_readable_image(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = save_faces(image, [_face()], tmp_path / "sub" / "faces.png")
    assert load_image(out).shape == image.shape


def test_save_image_round_trips_png(tmp_path):
    image = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = save_image(image, tmp_path / "img.png")
    assert np.array_equal(load_image(out), image)


def test_save_image_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="Could not write image"):
        save_image(np.zeros((8, 8, 3), dtype=np.uint8), tmp_path / "img.notanimage")


def test_load_image_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(tmp_path / "missing.jpg")


def test_load_image_undecodable_file_raises_value_error(tmp_path):
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image")
    with pytest.raises(ValueError, match="Could not decode image"):
        load_image(fake)


def test_draw_faces_colors_each_landmark():
    face = _face()
    drawn = draw_faces(np.zeros((100, 100, 3), dtype=np.uint8), [face])
    colors = {
        tuple(int(c) for c in drawn[round(py), round(px)])
        for px, py in (face.right_eye, face.left_eye, face.nose_tip, face.right_mouth_corner, face.left_mouth_corner)
    }
    assert len(colors) == 5


def _happy() -> EmotionPrediction:
    return EmotionPrediction({e: 0.7 if e is Emotion.HAPPY else 0.05 for e in Emotion})


def test_draw_emotions_draws_label_below_box_on_copy():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    drawn = draw_emotions(image, [_face()], [_happy()])
    assert not image.any()
    # The face box ends at y=60; the label starts there in BOX_COLOR.
    assert tuple(int(c) for c in drawn[61, 11]) == BOX_COLOR
    assert not drawn[:60].any()


def test_draw_emotions_moves_label_inside_box_at_bottom_edge():
    image = np.zeros((65, 100, 3), dtype=np.uint8)
    drawn = draw_emotions(image, [_face()], [_happy()])
    assert tuple(int(c) for c in drawn[58, 11]) == BOX_COLOR


def test_draw_emotions_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one prediction per face"):
        draw_emotions(np.zeros((100, 100, 3), dtype=np.uint8), [_face(), _face()], [_happy()])


def test_save_faces_draws_emotions_only_when_given(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    plain = load_image(save_faces(image, [_face()], tmp_path / "plain.png"))
    with_emotions = load_image(save_faces(image, [_face()], tmp_path / "emotions.png", [_happy()]))
    assert not plain[63:80].any()
    assert with_emotions[63:80].any()


def _label_height(drawn: np.ndarray) -> int:
    """Number of rows below the face box (which ends at y=61) that contain the label."""
    return int((drawn[62:, 11] == BOX_COLOR).all(axis=1).sum())


def test_draw_emotions_top_k_adds_one_line_per_emotion():
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    one = _label_height(draw_emotions(image, [_face()], [_happy()]))
    three = _label_height(draw_emotions(image, [_face()], [_happy()], top_k=3))
    assert three > 2 * one


@pytest.mark.parametrize("top_k", [0, len(Emotion) + 1])
def test_draw_emotions_rejects_out_of_range_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        draw_emotions(np.zeros((100, 100, 3), dtype=np.uint8), [_face()], [_happy()], top_k=top_k)


def _label_columns(drawn: np.ndarray, row: int) -> np.ndarray:
    return np.flatnonzero((drawn[row] == BOX_COLOR).all(axis=1))


def test_draw_emotions_keeps_label_inside_right_edge():
    # Box from x=80 to x=110 sticks out of the 100 px wide image.
    face = Face.from_row(np.array([80, 20, 30, 40, *[0] * 10, 0.95]))
    image = np.zeros((200, 100, 3), dtype=np.uint8)
    drawn = draw_emotions(image, [face], [_happy()], top_k=3)
    columns = _label_columns(drawn, 62)
    assert columns.max() == 99
    # The whole label is drawn: wider than the 20 px of box still in the image.
    assert len(columns) > 40


def test_draw_emotions_keeps_label_inside_left_edge():
    face = Face.from_row(np.array([-15, 20, 30, 40, *[0] * 10, 0.95]))
    drawn = draw_emotions(np.zeros((100, 100, 3), dtype=np.uint8), [face], [_happy()])
    assert _label_columns(drawn, 62).min() == 0
