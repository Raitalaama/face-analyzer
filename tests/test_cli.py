from pathlib import Path

import pytest
from conftest import ARTIFACT_DIR, GROUP_IMAGE, MANIFEST_PATH

from face_analyzer.cli import main
from face_analyzer.image_io import load_image

BASE_ARGS = ["--manifest-path", str(MANIFEST_PATH), "--artifact-dir", str(ARTIFACT_DIR)]


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


def test_undecodable_image_exits_with_error(tmp_path, capsys):
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image")
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "detect", str(fake)])
    assert exc.value.code == 1
    assert "Could not decode image" in capsys.readouterr().err


def test_failed_model_download_exits_with_error(tmp_path, capsys):
    manifest = tmp_path / "models.yaml"
    # Port 9 (discard) on localhost refuses the connection, like being offline.
    manifest.write_text(f"yunet:\n  url: http://127.0.0.1:9/model.onnx\n  sha256: {'0' * 64}\n  filename: model.onnx\n")
    args = ["--manifest-path", str(manifest), "--artifact-dir", str(tmp_path / "models")]
    with pytest.raises(SystemExit) as exc:
        main([*args, "detect", str(GROUP_IMAGE)])
    assert exc.value.code == 1
    assert "face-analyzer: error:" in capsys.readouterr().err


def test_detect_saves_annotated_image(yunet_path: Path, tmp_path, capsys):
    out = tmp_path / "faces.jpg"
    main([*BASE_ARGS, "detect", str(GROUP_IMAGE), "-o", str(out)])
    assert "Found 5 face(s)" in capsys.readouterr().out
    assert load_image(out).shape == load_image(GROUP_IMAGE).shape
