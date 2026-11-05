"""Classify facial expressions using a pretrained model."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from face_analyzer.labels import Emotion, canonical_order, labels_from_id2label

DEFAULT_MODEL_NAME = "trpakov/vit-face-expression"
DEFAULT_MODEL_REVISION = "ef0bc6fc34241b6587e7e009e7711357be28c024"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Turn a device name into a torch device and check that it can be used.

    Args:
        device: Device such as "cpu", "cuda" or "cuda:1". None picks CUDA if
            available, otherwise CPU.

    Returns:
        The resolved device.

    Raises:
        ValueError: If the device name is invalid, CUDA is requested but not
            available, or the CUDA device index does not exist.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except RuntimeError as e:
        raise ValueError(f"Invalid device {device!r}, use e.g. 'cpu', 'cuda' or 'cuda:0'.") from e
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Device {device!r} requested, but CUDA is not available.")
        count = torch.cuda.device_count()
        if resolved.index is not None and resolved.index >= count:
            raise ValueError(f"Device {device!r} requested, but only {count} CUDA device(s) are available.")
    return resolved


@dataclass(frozen=True)
class EmotionPrediction:
    """Predicted emotion probabilities for one face.

    Attributes:
        probabilities: Probability of each emotion, in canonical Emotion order.
            The values are between 0 and 1 and sum to 1.
    """

    probabilities: dict[Emotion, float]

    def __post_init__(self) -> None:
        """Check that the probabilities form a valid distribution over all emotions.

        Raises:
            ValueError: If an emotion is missing, a value is outside [0, 1], or
                the values do not sum to 1.
        """
        if list(self.probabilities) != list(Emotion):
            raise ValueError(
                f"Expected probabilities for {list(Emotion)} in that order, got {list(self.probabilities)}."
            )
        if any(not 0.0 <= p <= 1.0 for p in self.probabilities.values()):
            raise ValueError(f"Probabilities must be between 0 and 1, got {self.probabilities}.")
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"Probabilities must sum to 1, got {total}.")

    @classmethod
    def from_probabilities(cls, values: Sequence[float]) -> Self:
        """Create a prediction from probabilities in canonical Emotion order.

        Args:
            values: One probability per emotion, in the order of the Emotion enum.

        Returns:
            The prediction.

        Raises:
            ValueError: If there is not exactly one value per emotion, or the
                values are not a valid probability distribution.
        """
        if len(values) != len(Emotion):
            raise ValueError(f"Expected {len(Emotion)} probabilities, got {len(values)}.")
        return cls({emotion: float(p) for emotion, p in zip(Emotion, values, strict=True)})

    @property
    def emotion(self) -> Emotion:
        """The most probable emotion. Ties go to the earlier emotion in canonical order."""
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        """Probability of the most probable emotion."""
        return self.probabilities[self.emotion]

    def __str__(self) -> str:
        """Return a string representation of the prediction."""
        probs_str = ", ".join(f"{emotion.value}: {p:.3f}" for emotion, p in self.probabilities.items())
        return f"EmotionPrediction({probs_str})"


class EmotionClassifier:
    """Classify facial expressions using a pretrained model."""

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Load a pretrained image classification model from the Hugging Face Hub.

        Args:
            model_name: Hub model id, e.g. "trpakov/vit-face-expression".
            model_revision: Commit hash, branch or tag to load. Defaults to the
                latest version on the Hub.
            device: Device to run the model on, e.g. "cpu" or "cuda:0".
                Defaults to CUDA if available, otherwise CPU. It is checked
                before the model is loaded.
            local_files_only: Load only from the local Hugging Face cache,
                without downloading.

        Raises:
            OSError: If the model cannot be loaded, e.g. it is not cached and
                local_files_only is True.
            ValueError: If the device cannot be used, or the model's labels do
                not cover each emotion exactly once.
        """
        # Check the device first, so a bad device fails before the slow model load.
        self.model_device = resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=model_revision, local_files_only=local_files_only
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name, revision=model_revision, local_files_only=local_files_only
        )
        self.model.to(self.model_device)

        self.model.eval()  # Set the model to evaluation mode
        self.labels = labels_from_id2label(self.model.config.id2label)
        self.order = canonical_order(self.labels)

    def predict(self, images: list[np.ndarray]) -> list[EmotionPrediction]:
        """Classify the facial expression in the given images.

        Args:
            images: A list of NumPy arrays representing the images in RGB format.

        Returns:
            A list of emotion predictions.
        """
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt").to(self.model_device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[:, self.order].cpu().tolist()
        return [EmotionPrediction.from_probabilities(row) for row in probs]
