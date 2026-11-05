import pytest

from face_analyzer.labels import Emotion, canonical_order, labels_from_id2label, parse_label


def test_emotion_canonical_order():
    assert [e.value for e in Emotion] == ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


@pytest.mark.parametrize("emotion", list(Emotion))
def test_parse_label_accepts_canonical_names(emotion):
    assert parse_label(emotion.value) is emotion


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Happy", Emotion.HAPPY),
        ("  NEUTRAL ", Emotion.NEUTRAL),
        ("anger", Emotion.ANGRY),
        ("Disgusted", Emotion.DISGUST),
        ("fearful", Emotion.FEAR),
        ("happiness", Emotion.HAPPY),
        ("sadness", Emotion.SAD),
        ("Surprised", Emotion.SURPRISE),
    ],
)
def test_parse_label_normalizes_case_whitespace_and_aliases(raw, expected):
    assert parse_label(raw) is expected


@pytest.mark.parametrize("raw", ["contempt", "", "happy!"])
def test_parse_label_rejects_unknown_labels(raw):
    with pytest.raises(ValueError, match="Unknown emotion label"):
        parse_label(raw)


# id2label of a model whose outputs are in a different order and use alias names.
SCRAMBLED_ID2LABEL = {
    0: "Neutral",
    1: "happiness",
    2: "anger",
    3: "Sad",
    4: "surprised",
    5: "fear",
    6: "disgust",
}
SCRAMBLED_LABELS = [
    Emotion.NEUTRAL,
    Emotion.HAPPY,
    Emotion.ANGRY,
    Emotion.SAD,
    Emotion.SURPRISE,
    Emotion.FEAR,
    Emotion.DISGUST,
]


def test_labels_from_id2label_maps_each_index():
    assert labels_from_id2label(SCRAMBLED_ID2LABEL) == SCRAMBLED_LABELS


def test_labels_from_id2label_accepts_string_indices_in_any_order():
    id2label = {str(i): label for i, label in reversed(SCRAMBLED_ID2LABEL.items())}
    assert labels_from_id2label(id2label) == SCRAMBLED_LABELS


def test_labels_from_id2label_rejects_gaps_in_indices():
    id2label = {i if i < 3 else i + 1: label for i, label in SCRAMBLED_ID2LABEL.items()}
    with pytest.raises(ValueError, match="indices"):
        labels_from_id2label(id2label)


def test_labels_from_id2label_rejects_duplicates_and_missing():
    id2label = {**SCRAMBLED_ID2LABEL, 6: "happy"}
    with pytest.raises(ValueError, match=r"duplicates: \['happy'\], missing: \['disgust'\]"):
        labels_from_id2label(id2label)


def test_labels_from_id2label_rejects_missing_emotion():
    id2label = {i: label for i, label in SCRAMBLED_ID2LABEL.items() if i < 6}
    with pytest.raises(ValueError, match=r"missing: \['disgust'\]"):
        labels_from_id2label(id2label)


def test_labels_from_id2label_rejects_unknown_label():
    with pytest.raises(ValueError, match="contempt"):
        labels_from_id2label({**SCRAMBLED_ID2LABEL, 7: "contempt"})


def test_canonical_order_is_identity_for_canonical_labels():
    assert canonical_order(list(Emotion)) == list(range(len(Emotion)))


def test_canonical_order_reorders_model_outputs():
    order = canonical_order(SCRAMBLED_LABELS)
    # Taking the model's outputs in this order gives canonical Emotion order.
    assert [SCRAMBLED_LABELS[i] for i in order] == list(Emotion)
