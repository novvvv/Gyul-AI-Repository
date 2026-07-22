"""분석 리포트 스키마 v1 조립 (연동 가이드 §3.2).

summary는 기존 scripts/session_report 파이프라인(LLM 자기성찰 리포트)을 재사용하고,
감정은 app/core/emotion_mapping으로 계약 enum에 맞춘다.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.core.emotion_mapping import aggregate_emotion, raw_emotion_counts
from app.ws.interview_session import InterviewSessionState

logger = logging.getLogger(__name__)

# summary는 TEXT 저장 — 64KB 이내 권장(가이드 §3.2)이라 여유를 두고 절단
SUMMARY_MAX_CHARS = 60_000

FALLBACK_SUMMARY = "대화 내용이 충분하지 않아 상세 분석을 생성하지 못했습니다."


def _iso_utc(moment: datetime) -> str:
    # 계약: ISO-8601 UTC + 'Z' 접미 필수 (+09:00 오프셋 금지)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_summary(state: InterviewSessionState, ended_at: datetime) -> tuple[str, str | None]:
    from scripts.session_report.generate import generate_session_report

    snapshot = {
        "session": {
            "user_id": state.email,
            "session_id": state.session_id,
            "persona_id": state.persona_id,
            "started_at": _iso_utc(state.started_at),
            "ended_at": _iso_utc(ended_at),
        },
        "turns": state.turns,
    }
    report = generate_session_report(snapshot)
    summary = (report.get("comprehensive_report") or "").strip()
    llm_backend = (report.get("meta") or {}).get("llm_backend")
    return summary, llm_backend


async def build_analysis_report(state: InterviewSessionState) -> dict:
    """InterviewSessionState → Kafka 발행용 스키마 v1 dict."""
    ended_at = datetime.now(timezone.utc)

    summary = ""
    llm_backend = None
    try:
        summary, llm_backend = await asyncio.to_thread(
            _generate_summary, state, ended_at
        )
    except Exception as e:
        logger.warning("summary 생성 실패 — 폴백 사용 (sessionId=%s): %s", state.session_id, e)
    if not summary:
        summary = FALLBACK_SUMMARY
    summary = summary[:SUMMARY_MAX_CHARS]

    metrics: dict = {
        "turnCount": state.turn_count,
        "speechDurationSec": state.speech_duration_sec,
        "rawEmotionCounts": raw_emotion_counts(state.turns),
    }
    if llm_backend:
        metrics["llmBackend"] = llm_backend

    return {
        "schemaVersion": 1,
        "sessionId": state.session_id,
        "email": state.email,  # JWT sub 그대로 — 가공 금지 (§3.2)
        "phase": "PHASE_1",
        "startedAt": _iso_utc(state.started_at),
        "endedAt": _iso_utc(ended_at),
        "emotion": aggregate_emotion(state.turns),
        "summary": summary,
        "metrics": metrics,
    }
