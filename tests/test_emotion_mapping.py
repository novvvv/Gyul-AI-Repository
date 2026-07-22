from app.core.emotion_mapping import (
    CONTRACT_EMOTIONS,
    aggregate_emotion,
    map_ser_label,
    raw_emotion_counts,
)


def _turn(probs: dict, label: str | None = None) -> dict:
    dominant = label or max(probs, key=probs.get)
    return {
        "user_text": "테스트",
        "voice_emotion": {"label": dominant, "confidence": 0.9, "probs": probs},
        "bot_reply": "응답",
    }


class TestMapSerLabel:
    def test_seven_labels_map_into_contract(self):
        for ser in ("happiness", "angry", "disgust", "fear", "neutral", "sadness", "surprise"):
            assert map_ser_label(ser) in CONTRACT_EMOTIONS

    def test_specific_mappings(self):
        assert map_ser_label("fear") == "TENSION"
        assert map_ser_label("surprise") == "CONFUSION"
        assert map_ser_label("happiness") == "CONFIDENCE"
        assert map_ser_label("neutral") == "NEUTRAL"

    def test_unknown_label_falls_back_to_neutral(self):
        assert map_ser_label("boredom") == "NEUTRAL"


class TestAggregateEmotion:
    def test_empty_session_is_neutral(self):
        assert aggregate_emotion([]) == {
            "dominant": "NEUTRAL",
            "scores": {"neutral": 1.0},
        }

    def test_dominant_and_scores(self):
        turns = [
            _turn({"fear": 0.8, "neutral": 0.2}),
            _turn({"fear": 0.6, "surprise": 0.4}),
        ]
        result = aggregate_emotion(turns)

        assert result["dominant"] == "TENSION"
        assert set(result["scores"]) <= {e.lower() for e in CONTRACT_EMOTIONS}
        assert abs(sum(result["scores"].values()) - 1.0) < 0.01
        assert result["scores"]["tension"] == 0.7

    def test_downgraded_labels_fold_into_neutral(self):
        turns = [_turn({"sadness": 0.5, "angry": 0.3, "disgust": 0.2})]
        result = aggregate_emotion(turns)
        assert result["dominant"] == "NEUTRAL"
        assert result["scores"] == {"neutral": 1.0}

    def test_scores_only_contain_present_emotions(self):
        turns = [_turn({"fear": 1.0})]
        result = aggregate_emotion(turns)
        assert result["scores"] == {"tension": 1.0}
        assert "calm" not in result["scores"]

    def test_turn_without_probs_ignored(self):
        turns = [
            {"user_text": "x", "voice_emotion": None, "bot_reply": "y"},
            _turn({"happiness": 1.0}),
        ]
        assert aggregate_emotion(turns)["dominant"] == "CONFIDENCE"


def test_raw_emotion_counts_preserves_original_labels():
    turns = [
        _turn({"sadness": 0.9}, label="sadness"),
        _turn({"sadness": 0.8}, label="sadness"),
        _turn({"fear": 0.7}, label="fear"),
    ]
    assert raw_emotion_counts(turns) == {"sadness": 2, "fear": 1}
