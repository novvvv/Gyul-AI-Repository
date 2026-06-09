from __future__ import annotations

import json
import re
from typing import Any

from app.llm.text_generator import TextGenerationError, generate_chat_completion, resolve_text_backend
from scripts.session_report.aggregate import build_aggregates
from scripts.session_report.format import format_report_markdown
from scripts.session_report.prompt import (
    REPORT_MAX_TOKENS,
    build_narrative_messages,
    build_report_messages,
)
from scripts.session_report.schema import SessionSnapshot

DEFAULT_DISCLAIMER = (
    "본 리포트는 AI 기반 자기성찰 도구이며, 전문 심리상담·의학적 진단을 대체하지 않습니다."
)

MIN_REPORT_SENTENCES = 6


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("LLM response is not valid JSON")


def _sentence_count(text: str) -> int:
    parts = re.split(r"[.!?。]\s*|\n+", text.strip())
    return len([p for p in parts if len(p.strip()) > 4])


def _generate_narrative(
    payload: dict[str, Any], aggregates: dict[str, Any]
) -> str:
    messages = build_narrative_messages(payload, aggregates)
    return generate_chat_completion(
        messages,
        temperature=0.5,
        max_tokens=REPORT_MAX_TOKENS,
    ).strip()


def _build_template_narrative(
    snapshot: SessionSnapshot, aggregates: dict[str, Any]
) -> str:
    if not snapshot.turns or not any(t.user_text for t in snapshot.turns):
        return "오늘은 아직 나눈 이야기가 없어요. 다음에 편하게 말씀해 주시면 함께 돌아볼게요."

    quote = next((t.user_text for t in snapshot.turns if t.user_text), "")
    voice = aggregates.get("voice_dominant") or []
    face = aggregates.get("face_dominant") or []

    lines: list[str] = []
    lines.append(
        "오늘 대화를 돌아보면, 당신은 자신의 마음을 꺼내려는 용기를 보여주셨어요. "
        "말로 감정을 표현하는 일은 쉽지 않은데, 그럼에도 이야기를 이어가신 점이 인상적이에요."
    )
    if quote:
        lines.append(
            f"특히 「{quote}」라고 하셨을 때, 그 안에 담긴 마음이 느껴졌어요. "
            "겉으로 드러낸 말 너머에도 더 깊은 생각이 있었을 수 있어요."
        )
    if voice and face and voice[0] != face[0]:
        lines.append(
            "목소리와 표정이 조금 다르게 느껴진 순간도 있었어요. "
            "그건 괜찮아요. 마음은 한 가지 색으로만 보이지 않을 때가 많거든요."
        )
    elif voice:
        lines.append(
            "목소리에는 여러 감정의 결이 스며 있었어요. "
            "한순간의 감정이 당신 전체를 말해주지는 않아요."
        )
    lines.append(
        "오늘 나눈 이야기 속에서, 당신이 무엇을 소중히 여기고 있는지 조금 더 선명해진 것 같아요. "
        "스스로를 돌아보는 이 시간 자체가 이미 의미 있는 걸음이에요."
    )
    lines.append(
        "다음에 마음이 편할 때, 오늘 가장 마음에 남는 순간을 조금 더 천천히 들여다보는 것도 좋겠어요. "
        "결은 언제든 옆에서 함께 듣고 있을게요."
    )
    return " ".join(lines)


def _ensure_comprehensive_report(
    llm_section: dict[str, Any],
    snapshot: SessionSnapshot,
    aggregates: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    report = (llm_section.get("comprehensive_report") or "").strip()
    if report and _sentence_count(report) >= MIN_REPORT_SENTENCES:
        return report

    summary = (llm_section.get("summary") or "").strip()
    if summary and _sentence_count(summary) >= MIN_REPORT_SENTENCES:
        return summary

    try:
        narrative = _generate_narrative(payload, aggregates)
        if narrative and _sentence_count(narrative) >= 3:
            return narrative
    except TextGenerationError:
        pass

    return _build_template_narrative(snapshot, aggregates)


def snapshot_to_payload(snapshot: SessionSnapshot) -> dict[str, Any]:
    return {
        "session": {
            "user_id": snapshot.user_id,
            "session_id": snapshot.session_id,
            "persona_id": snapshot.persona_id,
            "started_at": snapshot.started_at,
            "ended_at": snapshot.ended_at,
        },
        "turns": [
            {
                "user_text": t.user_text,
                "voice_emotion": (
                    {
                        "label": t.voice_emotion.label,
                        "confidence": t.voice_emotion.confidence,
                        "probs": t.voice_emotion.probs,
                    }
                    if t.voice_emotion
                    else None
                ),
                "face_emotion": (
                    {
                        "label": t.face_emotion.label,
                        "confidence": t.face_emotion.confidence,
                    }
                    if t.face_emotion
                    else None
                ),
                "bot_reply": t.bot_reply,
                "at": t.at,
            }
            for t in snapshot.turns
        ],
    }


def _empty_structured_section(
    snapshot: SessionSnapshot,
    aggregates: dict[str, Any],
) -> dict[str, Any]:
    quotes = [t.user_text for t in snapshot.turns if t.user_text][:3]
    return {
        "topics": [],
        "quotes": quotes,
        "patterns": [],
        "strengths": [],
        "reflection_questions": [],
        "next_topics": [],
        "disclaimer": DEFAULT_DISCLAIMER,
    }


def generate_session_report(snapshot_data: dict[str, Any]) -> dict[str, Any]:
    snapshot = SessionSnapshot.from_dict(snapshot_data)
    aggregates = build_aggregates(snapshot)
    payload = snapshot_to_payload(snapshot)
    backend = resolve_text_backend()

    llm_section: dict[str, Any] = _empty_structured_section(snapshot, aggregates)
    json_error: str | None = None

    try:
        messages = build_report_messages(payload, aggregates)
        raw = generate_chat_completion(
            messages,
            temperature=0.45,
            max_tokens=REPORT_MAX_TOKENS,
        )
        parsed = _extract_json(raw)
        llm_section.update(parsed)
    except (TextGenerationError, ValueError, json.JSONDecodeError) as e:
        json_error = str(e)

    comprehensive_report = _ensure_comprehensive_report(
        llm_section, snapshot, aggregates, payload
    )
    llm_section["comprehensive_report"] = comprehensive_report
    llm_section["summary"] = comprehensive_report[:150] + (
        "…" if len(comprehensive_report) > 150 else ""
    )

    if not llm_section.get("disclaimer"):
        llm_section["disclaimer"] = DEFAULT_DISCLAIMER

    meta = {
        "session_id": snapshot.session_id,
        "persona_id": snapshot.persona_id,
        "started_at": snapshot.started_at,
        "ended_at": snapshot.ended_at,
        "llm_backend": backend,
        **aggregates,
    }

    report_json = {
        "meta": meta,
        "comprehensive_report": comprehensive_report,
        "summary": llm_section.get("summary", ""),
        "topics": llm_section.get("topics") or [],
        "quotes": llm_section.get("quotes") or [],
        "patterns": llm_section.get("patterns") or [],
        "strengths": llm_section.get("strengths") or [],
        "reflection_questions": llm_section.get("reflection_questions") or [],
        "next_topics": llm_section.get("next_topics") or [],
        "disclaimer": llm_section.get("disclaimer", DEFAULT_DISCLAIMER),
        "turns": payload["turns"],
        "aggregates": aggregates,
    }
    if json_error:
        report_json["generation_error"] = json_error

    report_md = format_report_markdown(llm_section, meta)
    return {
        "report_json": report_json,
        "report_md": report_md,
        "llm_backend": backend,
    }
