import re

from app.ws.interview_session import InterviewSessionState, new_session_id


def test_session_id_format_and_uniqueness():
    session_id = new_session_id()
    assert re.fullmatch(r"intv-\d{8}T\d{6}Z-[0-9a-f]{8}", session_id)
    assert new_session_id() != session_id


def test_record_turn_accumulates_log_and_speech():
    state = InterviewSessionState(email="ai-test@example.com", member_id=42)
    state.record_turn(
        {
            "user_text": "안녕하세요",
            "voice_emotion": {"label": "fear", "confidence": 0.8, "probs": {}},
            "bot_reply": "반가워요",
            "at": "2026-07-22T10:00:00Z",
            "speech_samples": 16000 * 3,
        }
    )

    assert state.turn_count == 1
    assert state.speech_duration_sec == 3
    # 리포트 스키마(SessionTurn 호환) 키만 남고 speech_samples는 분리 집계
    assert "speech_samples" not in state.turns[0]
    assert state.turns[0]["user_text"] == "안녕하세요"


def test_defaults():
    state = InterviewSessionState(email="a@b.c", member_id=None)
    assert state.persona_id == "gyul"
    assert state.turns == []
    assert state.started_at.tzinfo is not None
