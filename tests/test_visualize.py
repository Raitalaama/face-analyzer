import numpy as np
import pytest

from face_analyzer.detection import Face
from face_analyzer.image_io import load_image, save_image
from face_analyzer.visualize import draw_faces, save_faces


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
