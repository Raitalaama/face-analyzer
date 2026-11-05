import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import ARTIFACT_DIR, GROUP_IMAGE, MANIFEST_PATH

import face_analyzer.cli as cli
from face_analyzer.classification import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, EmotionPrediction
from face_analyzer.cli import main
from face_analyzer.detection import crop_face
from face_analyzer.image_io import load_image, save_image
from face_analyzer.labels import Emotion

BASE_ARGS = ["--manifest-path", str(MANIFEST_PATH), "--artifact-dir", str(ARTIFACT_DIR)]


def test_missing_image_exits_with_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "detect", str(tmp_path / "missing.jpg")])
    assert exc.value.code == 1
    assert "face-analyzer: error: Image not found" in capsys.readouterr().err


def test_unknown_model_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "detect", str(GROUP_IMAGE), "--detection-model", "nope"])
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


@pytest.fixture
def classifier_calls(monkeypatch):
    """Record the arguments the CLI creates EmotionClassifier with.

    Returns a function that installs the classifier the CLI should get.
    """
    calls: list[dict] = []

    def install(classifier) -> list[dict]:
        def create(**kwargs):
            calls.append(kwargs)
            return classifier

        monkeypatch.setattr(cli, "EmotionClassifier", create)
        return calls

    return install


class _AlwaysSurprised:
    """Fake classifier that predicts "surprise" for every face."""

    def __init__(self):
        self.images: list[np.ndarray] = []

    def predict(self, images):
        self.images.extend(images)
        probabilities = {e: 0.9 if e is Emotion.SURPRISE else 0.1 / (len(Emotion) - 1) for e in Emotion}
        return [EmotionPrediction(probabilities) for _ in images]


def test_analyze_missing_image_exits_with_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "analyze", str(tmp_path / "missing.jpg")])
    assert exc.value.code == 1
    assert "Image not found" in capsys.readouterr().err


def test_analyze_with_real_models(yunet_path, emotion_classifier, classifier_calls, tmp_path, capsys):
    calls = classifier_calls(emotion_classifier)
    out = tmp_path / "analyzed.jpg"
    main([*BASE_ARGS, "analyze", str(GROUP_IMAGE), "-o", str(out)])

    assert calls == [{"model_name": DEFAULT_MODEL_NAME, "model_revision": DEFAULT_MODEL_REVISION, "device": None}]
    stdout = capsys.readouterr().out
    emotions = re.findall(r"^Face \d+: (\w+) \(", stdout, flags=re.MULTILINE)
    assert len(emotions) == 5
    assert emotions.count("happy") >= 3, stdout
    assert load_image(out).shape == load_image(GROUP_IMAGE).shape


def test_analyze_passes_model_options_to_classifier(yunet_path, classifier_calls, capsys):
    calls = classifier_calls(_AlwaysSurprised())
    main(
        [
            *BASE_ARGS,
            "analyze",
            str(GROUP_IMAGE),
            "--analyze-model",
            "org/other-model",
            "--analyze-model-revision",
            "abc123",
            "--device",
            "cpu",
        ]
    )
    assert calls == [{"model_name": "org/other-model", "model_revision": "abc123", "device": "cpu"}]


def test_analyze_classifies_rgb_square_crops(face_detector, classifier_calls, capsys):
    classifier = _AlwaysSurprised()
    classifier_calls(classifier)
    main([*BASE_ARGS, "analyze", str(GROUP_IMAGE)])

    image = load_image(GROUP_IMAGE)
    expected = [cv2.cvtColor(crop_face(image, face), cv2.COLOR_BGR2RGB) for face in face_detector.detect(image)]
    assert len(classifier.images) == len(expected) == 5
    for got, want in zip(classifier.images, expected, strict=True):
        assert got.shape[0] == got.shape[1]
        assert np.array_equal(got, want)
    assert capsys.readouterr().out.count("surprise (0.90)") == 5


def test_analyze_image_without_faces(yunet_path, classifier_calls, tmp_path, capsys):
    blank = save_image(np.zeros((240, 320, 3), dtype=np.uint8), tmp_path / "blank.png")
    classifier = _AlwaysSurprised()
    classifier_calls(classifier)
    out = tmp_path / "out.png"
    main([*BASE_ARGS, "analyze", str(blank), "-o", str(out)])

    assert classifier.images == []
    assert "Face " not in capsys.readouterr().out
    assert out.is_file()


def test_analyze_invalid_device_exits_before_loading_model(yunet_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "analyze", str(GROUP_IMAGE), "--device", "gpu"])
    assert exc.value.code == 1
    assert "face-analyzer: error: Invalid device 'gpu'" in capsys.readouterr().err
