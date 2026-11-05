from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from conftest import GROUP_IMAGE

import face_analyzer.classification as classification
from face_analyzer.classification import EmotionClassifier, EmotionPrediction, resolve_device
from face_analyzer.detection import FaceDetector, crop_face
from face_analyzer.image_io import load_image
from face_analyzer.labels import Emotion

UNIFORM = [1 / len(Emotion)] * len(Emotion)


def _prediction(**probabilities: float) -> EmotionPrediction:
    """Prediction with the given probabilities and the rest spread evenly."""
    rest = (1 - sum(probabilities.values())) / (len(Emotion) - len(probabilities))
    return EmotionPrediction({e: probabilities.get(e.value, rest) for e in Emotion})


def test_from_probabilities_maps_values_in_canonical_order():
    values = [0.0, 0.1, 0.0, 0.6, 0.1, 0.1, 0.1]
    prediction = EmotionPrediction.from_probabilities(values)
    assert list(prediction.probabilities) == list(Emotion)
    assert list(prediction.probabilities.values()) == values


def test_from_probabilities_converts_to_python_floats():
    prediction = EmotionPrediction.from_probabilities(np.array(UNIFORM, dtype=np.float32))
    assert all(type(p) is float for p in prediction.probabilities.values())


def test_from_probabilities_rejects_wrong_count():
    with pytest.raises(ValueError, match="Expected 7 probabilities"):
        EmotionPrediction.from_probabilities([0.5, 0.5])


def test_prediction_rejects_wrong_emotion_order():
    reversed_probs = dict(zip(reversed(Emotion), UNIFORM, strict=True))
    with pytest.raises(ValueError, match="in that order"):
        EmotionPrediction(reversed_probs)


def test_prediction_rejects_missing_emotion():
    with pytest.raises(ValueError, match="in that order"):
        EmotionPrediction({e: 1 / 6 for e in list(Emotion)[:-1]})


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_prediction_rejects_values_outside_unit_interval(bad):
    values = [bad, 1 - bad] + [0.0] * (len(Emotion) - 2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        EmotionPrediction.from_probabilities(values)


def test_prediction_rejects_values_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        EmotionPrediction.from_probabilities([0.1] * len(Emotion))


def test_prediction_tolerates_float32_rounding():
    values = [0.1428571] * 6 + [1 - 6 * 0.1428571 + 5e-5]
    EmotionPrediction.from_probabilities(values)


def test_emotion_and_confidence_are_the_most_probable():
    prediction = _prediction(happy=0.6, sad=0.2)
    assert prediction.emotion is Emotion.HAPPY
    assert prediction.confidence == pytest.approx(0.6)


def test_emotion_ties_go_to_earlier_canonical_emotion():
    prediction = _prediction(sad=0.4, fear=0.4)
    assert prediction.emotion is Emotion.FEAR


def test_str_lists_every_emotion():
    text = str(_prediction(happy=0.6))
    assert text.startswith("EmotionPrediction(")
    assert "happy: 0.600" in text
    assert all(e.value in text for e in Emotion)


# Fakes for the Hugging Face processor and model, so the tests need no download.
# The fake model's outputs are in a different order from the canonical one.
FAKE_ID2LABEL = {0: "neutral", 1: "happy", 2: "angry", 3: "sad", 4: "surprise", 5: "fear", 6: "disgust"}


class _FakeBatch(dict):
    device = None

    def to(self, device):
        self.device = device
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, images, return_tensors):
        self.calls += 1
        assert return_tensors == "pt"
        # One fake feature per image: the image's mean value picks the "winning" model output.
        return _FakeBatch(pixel_values=torch.tensor([float(np.mean(image)) for image in images]))


class _FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(id2label=FAKE_ID2LABEL)
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, pixel_values):
        # Strongly favour model output index int(value) for each image.
        logits = torch.zeros(len(pixel_values), len(FAKE_ID2LABEL))
        logits[torch.arange(len(pixel_values)), pixel_values.long()] = 10.0
        return SimpleNamespace(logits=logits)


@pytest.fixture
def fake_hf(monkeypatch):
    """Replace the Hugging Face loaders with fakes and record what they were asked for."""
    loaded = {"processor": _FakeProcessor(), "model": _FakeModel(), "calls": []}

    def processor_from_pretrained(name, revision=None, local_files_only=False):
        loaded["calls"].append(("processor", name, revision, local_files_only))
        return loaded["processor"]

    def model_from_pretrained(name, revision=None, local_files_only=False):
        loaded["calls"].append(("model", name, revision, local_files_only))
        return loaded["model"]

    monkeypatch.setattr(
        classification, "AutoImageProcessor", SimpleNamespace(from_pretrained=processor_from_pretrained)
    )
    monkeypatch.setattr(
        classification, "AutoModelForImageClassification", SimpleNamespace(from_pretrained=model_from_pretrained)
    )
    return loaded


def _image(model_index: int) -> np.ndarray:
    """RGB image whose mean makes the fake model favour the given output index."""
    return np.full((4, 4, 3), model_index, dtype=np.uint8)


def test_classifier_loads_model_and_processor_at_revision(fake_hf):
    EmotionClassifier("org/model", model_revision="abc123", device=torch.device("cpu"))
    assert fake_hf["calls"] == [
        ("processor", "org/model", "abc123", False),
        ("model", "org/model", "abc123", False),
    ]
    assert fake_hf["model"].device == torch.device("cpu")
    assert fake_hf["model"].eval_called


def test_classifier_passes_local_files_only(fake_hf):
    EmotionClassifier("org/model", device=torch.device("cpu"), local_files_only=True)
    assert all(call[3] is True for call in fake_hf["calls"])


def test_classifier_maps_model_outputs_to_canonical_emotions(fake_hf):
    classifier = EmotionClassifier("org/model", device=torch.device("cpu"))
    # Model output 1 is "happy", 0 is "neutral" and 6 is "disgust".
    predictions = classifier.predict([_image(1), _image(0), _image(6)])
    assert [p.emotion for p in predictions] == [Emotion.HAPPY, Emotion.NEUTRAL, Emotion.DISGUST]
    for prediction in predictions:
        assert list(prediction.probabilities) == list(Emotion)
        assert prediction.confidence > 0.99


def test_classifier_returns_empty_list_without_calling_model(fake_hf):
    classifier = EmotionClassifier("org/model", device=torch.device("cpu"))
    assert classifier.predict([]) == []
    assert fake_hf["processor"].calls == 0


def test_classifier_rejects_model_with_unknown_labels(fake_hf):
    fake_hf["model"].config.id2label = {**FAKE_ID2LABEL, 6: "contempt"}
    with pytest.raises(ValueError, match="contempt"):
        EmotionClassifier("org/model", device=torch.device("cpu"))


def test_real_model_finds_happy_faces_in_group_photo(face_detector: FaceDetector, emotion_classifier):
    image = load_image(GROUP_IMAGE)
    faces = face_detector.detect(image)
    crops = [cv2.cvtColor(crop_face(image, face), cv2.COLOR_BGR2RGB) for face in faces]
    predictions = emotion_classifier.predict(crops)
    emotions = [p.emotion for p in predictions]
    assert len(predictions) == len(faces) == 5
    assert emotions.count(Emotion.HAPPY) >= 3, emotions


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend the machine has the given number of CUDA devices (default 2)."""

    def set_count(count: int = 2) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)

    set_count()
    return set_count


@pytest.mark.parametrize(("count", "expected"), [(2, "cuda"), (0, "cpu")])
def test_resolve_device_defaults_to_cuda_if_available(fake_cuda, count, expected):
    fake_cuda(count)
    assert resolve_device(None) == torch.device(expected)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", torch.device("cuda:0")])
def test_resolve_device_accepts_valid_devices(fake_cuda, device):
    assert resolve_device(device) == torch.device(device)


@pytest.mark.parametrize(
    ("device", "count", "message"),
    [
        ("gpu", 2, "Invalid device 'gpu'"),
        ("", 2, "Invalid device ''"),
        ("cuda", 0, "CUDA is not available"),
        ("cuda:2", 2, "only 2 CUDA device"),
    ],
)
def test_resolve_device_rejects_unusable_devices(fake_cuda, device, count, message):
    fake_cuda(count)
    with pytest.raises(ValueError, match=message):
        resolve_device(device)


def test_classifier_checks_device_before_loading_model(fake_hf, fake_cuda):
    fake_cuda(0)
    with pytest.raises(ValueError, match="CUDA is not available"):
        EmotionClassifier("org/model", device="cuda")
    assert fake_hf["calls"] == []
