from pathlib import Path

import pytest

from face_analyzer.cli import main
from face_analyzer.image_io import load_image

ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "models" / "face_detection_yunet_2026may.onnx"
GROUP_IMAGE = ROOT / "tests" / "data" / "group.jpg"
BASE_ARGS = ["--manifest-path", str(ROOT / "configs" / "models.yaml"), "--artifact-dir", str(ROOT / "models")]


def test_missing_image_exits_with_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "detect", str(tmp_path / "missing.jpg")])
    assert exc.value.code == 1
    assert "face-analyzer: error: Image not found" in capsys.readouterr().err


def test_unknown_model_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "detect", str(GROUP_IMAGE), "--model", "nope"])
    assert exc.value.code == 2
    assert "unknown model 'nope'" in capsys.readouterr().err


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="model not downloaded, run `make models`")
def test_detect_saves_annotated_image(tmp_path, capsys):
    out = tmp_path / "faces.jpg"
    main([*BASE_ARGS, "detect", str(GROUP_IMAGE), "-o", str(out)])
    assert "Found 5 face(s)" in capsys.readouterr().out
    assert load_image(out).shape == load_image(GROUP_IMAGE).shape
