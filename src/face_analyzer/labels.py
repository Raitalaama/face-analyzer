"""Canonical emotion labels and mapping from model-specific label names."""

from collections.abc import Mapping
from enum import StrEnum, auto


class Emotion(StrEnum):
    """The seven basic emotions, in the project's canonical order."""

    ANGRY = auto()
    DISGUST = auto()
    FEAR = auto()
    HAPPY = auto()
    SAD = auto()
    SURPRISE = auto()
    NEUTRAL = auto()


# Other names models use for the same emotions. Keys are normalized (lowercase,
# no surrounding whitespace). Canonical names like "happy" need no entry here.
_ALIASES: dict[str, Emotion] = {
    "anger": Emotion.ANGRY,
    "disgusted": Emotion.DISGUST,
    "fearful": Emotion.FEAR,
    "happiness": Emotion.HAPPY,
    "sadness": Emotion.SAD,
    "surprised": Emotion.SURPRISE,
}


def parse_label(raw: str) -> Emotion:
    """Convert a model's label name to an Emotion.

    Matching ignores case and surrounding whitespace, and accepts common
    alternative names such as "happiness" or "fearful".

    Args:
        raw: Label name as a model reports it, e.g. "Happy".

    Returns:
        The matching emotion.

    Raises:
        ValueError: If the label is not a known emotion name.
    """
    name = raw.strip().lower()
    if name in Emotion:
        return Emotion(name)
    if name in _ALIASES:
        return _ALIASES[name]
    raise ValueError(f"Unknown emotion label {raw!r}.")


def labels_from_id2label(id2label: Mapping[int | str, str]) -> list[Emotion]:
    """Map a model's output indices to emotions.

    Args:
        id2label: A model's ``config.id2label``: output index to label name.
            Indices may be ints or numeric strings.

    Returns:
        A list where position i holds the emotion of the model's output i.

    Raises:
        ValueError: If the indices are not exactly 0..n-1, a label is unknown,
            or the labels do not cover every emotion exactly once.
    """
    indexed = {int(index): label for index, label in id2label.items()}
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError(f"Label indices must be 0..{len(indexed) - 1}, got {sorted(indexed)}.")
    emotions = [parse_label(indexed[i]) for i in range(len(indexed))]

    duplicates = sorted({e.value for e in emotions if emotions.count(e) > 1})
    missing = [e.value for e in Emotion if e not in emotions]
    if duplicates or missing:
        raise ValueError(
            f"Model labels must cover each emotion exactly once; duplicates: {duplicates}, missing: {missing}."
        )
    return emotions


def canonical_order(model_labels: list[Emotion]) -> list[int]:
    """Compute the column order that rearranges model outputs into canonical order.

    Indexing a model's output columns with the result puts them in the order
    of the Emotion enum: ``probs[:, canonical_order(labels)]``.

    Args:
        model_labels: Emotion of each model output index, as returned by
            labels_from_id2label.

    Returns:
        For each Emotion in canonical order, the model's output index for it.
    """
    return [model_labels.index(emotion) for emotion in Emotion]
