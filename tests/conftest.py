"""Shared test paths and fixtures."""

import os
from pathlib import Path

import pytest

from face_analyzer.artifacts import ArtifactStore, artifact_exists
from face_analyzer.detection import FaceDetector

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "configs" / "models.yaml"
ARTIFACT_DIR = ROOT / "models"
GROUP_IMAGE = ROOT / "tests" / "data" / "group.jpg"

REQUIRE_MODELS_ENV = "FACE_ANALYZER_REQUIRE_MODELS"


@pytest.fixture(scope="session")
def yunet_path() -> Path:
    """Path to the verified YuNet model from the manifest.

    Skips the test if the model is missing or its hash does not match. With
    FACE_ANALYZER_REQUIRE_MODELS=1 the test fails instead, so CI cannot pass
    without running the model tests.
    """
    artifact = ArtifactStore.from_manifest(MANIFEST_PATH, ARTIFACT_DIR).get("yunet")
    if not artifact_exists(artifact):
        reason = f"model {artifact.name!r} missing or invalid at {artifact.file_path}, run `make models`"
        if os.environ.get(REQUIRE_MODELS_ENV) == "1":
            pytest.fail(f"{reason} ({REQUIRE_MODELS_ENV}=1)")
        pytest.skip(reason)
    return artifact.file_path


@pytest.fixture
def face_detector(yunet_path: Path) -> FaceDetector:
    """Face detector using the manifest's YuNet model."""
    return FaceDetector(yunet_path)
