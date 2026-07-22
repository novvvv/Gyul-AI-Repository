import re

import pytest

from app.services import report_builder
from app.ws.interview_session import InterviewSessionState

ISO_UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def state() -> InterviewSessionState:
    state = InterviewSessionState(email="ai-test@example.com", member_id=42)
    state.record_turn(
        {
            "user_text": "요즘 면접이 걱정돼요",
            "voice_emotion": {
                "label": "fear",
                "confidence": 0.8,
                "probs": {"fear": 0.8, "neutral": 0.2},
            },
            "bot_reply": "그 마음 이해해요",
            "at": "2026-07-22T10:00:00Z",
            "speech_samples": 16000 * 5,
        }
    )
    return state


@pytest.fixture
def stub_summary(monkeypatch):
    monkeypatch.setattr(
        report_builder,
        "_generate_summary",
        lambda state, ended_at: ("사용자는 전반적으로 긴장 상태를 보였습니다.", "openai"),
    )


async def test_schema_v1_required_fields(state, stub_summary):
    report = await report_builder.build_analysis_report(state)

    assert report["schemaVersion"] == 1
    assert report["sessionId"] == state.session_id
    assert report["email"] == "ai-test@example.com"
    assert report["phase"] == "PHASE_1"
    assert ISO_UTC_Z.fullmatch(report["startedAt"])
    assert ISO_UTC_Z.fullmatch(report["endedAt"])
    assert report["emotion"]["dominant"] == "TENSION"
    assert report["summary"] == "사용자는 전반적으로 긴장 상태를 보였습니다."
    assert report["metrics"]["turnCount"] == 1
    assert report["metrics"]["speechDurationSec"] == 5
    assert report["metrics"]["rawEmotionCounts"] == {"fear": 1}
    assert report["metrics"]["llmBackend"] == "openai"


async def test_summary_failure_uses_fallback(state, monkeypatch):
    def boom(state, ended_at):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(report_builder, "_generate_summary", boom)

    report = await report_builder.build_analysis_report(state)
    assert report["summary"] == report_builder.FALLBACK_SUMMARY
    assert "llmBackend" not in report["metrics"]


async def test_summary_is_truncated(state, monkeypatch):
    monkeypatch.setattr(
        report_builder,
        "_generate_summary",
        lambda state, ended_at: ("가" * 100_000, "openai"),
    )

    report = await report_builder.build_analysis_report(state)
    assert len(report["summary"]) == report_builder.SUMMARY_MAX_CHARS
