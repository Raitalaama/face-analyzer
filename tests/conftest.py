"""Shared test paths and fixtures."""

import os
from pathlib import Path

import pytest

from face_analyzer.artifacts import ArtifactStore, artifact_exists
from face_analyzer.classification import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, EmotionClassifier
from face_analyzer.detection import FaceDetector

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "configs" / "models.yaml"
ARTIFACT_DIR = ROOT / "models"
GROUP_IMAGE = ROOT / "tests" / "data" / "group.jpg"

REQUIRE_MODELS_ENV = "FACE_ANALYZER_REQUIRE_MODELS"


def _model_unavailable(reason: str) -> None:
    """Skip the test, or fail it if FACE_ANALYZER_REQUIRE_MODELS=1."""
    if os.environ.get(REQUIRE_MODELS_ENV) == "1":
        pytest.fail(f"{reason} ({REQUIRE_MODELS_ENV}=1)")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def yunet_path() -> Path:
    """Path to the verified YuNet model from the manifest.

    Skips the test if the model is missing or its hash does not match. With
    FACE_ANALYZER_REQUIRE_MODELS=1 the test fails instead, so CI cannot pass
    without running the model tests.
    """
    artifact = ArtifactStore.from_manifest(MANIFEST_PATH, ARTIFACT_DIR).get("yunet")
    if not artifact_exists(artifact):
        _model_unavailable(f"model {artifact.name!r} missing or invalid at {artifact.file_path}, run `make models`")
    return artifact.file_path


@pytest.fixture
def face_detector(yunet_path: Path) -> FaceDetector:
    """Face detector using the manifest's YuNet model."""
    return FaceDetector(yunet_path)


@pytest.fixture(scope="session")
def emotion_classifier() -> EmotionClassifier:
    """Emotion classifier loaded from the local Hugging Face cache.

    Never downloads. Skips the test if the model is not cached, or fails it
    with FACE_ANALYZER_REQUIRE_MODELS=1.
    """
    try:
        return EmotionClassifier(DEFAULT_MODEL_NAME, model_revision=DEFAULT_MODEL_REVISION, local_files_only=True)
    except OSError as e:
        _model_unavailable(
            f"emotion model {DEFAULT_MODEL_NAME}@{DEFAULT_MODEL_REVISION[:7]} not in the Hugging Face cache: {e}"
        )
