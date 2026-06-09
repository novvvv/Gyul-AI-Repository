from __future__ import annotations

from typing import Any


def _lines(items: list[str], bullet: str = "-") -> str:
    if not items:
        return f"{bullet} (없음)"
    return "\n".join(f"{bullet} {item}" for item in items)


def format_report_markdown(report_json: dict[str, Any], meta: dict[str, Any]) -> str:
    session_id = meta.get("session_id", "")
    started = meta.get("started_at") or "-"
    ended = meta.get("ended_at") or "-"
    backend = meta.get("llm_backend", "-")

    return f"""# 오늘의 대화 리포트

- 세션: `{session_id}`
- 시간: {started} ~ {ended}
- 생성 모델: {backend}

## 종합 레포트
{report_json.get("comprehensive_report", report_json.get("summary", ""))}

## 한 줄 요약
{report_json.get("summary", "")}

## 핵심 주제
{_lines(report_json.get("topics") or [])}

## 핵심 발화
{_lines(report_json.get("quotes") or [])}

## 관찰된 패턴
{_lines(report_json.get("patterns") or [])}

## 잘 하고 있는 점
{_lines(report_json.get("strengths") or [])}

## 성찰 질문
{_lines(report_json.get("reflection_questions") or [], bullet="1.")}

## 다음에 이야기해볼 수 있는 것
{_lines(report_json.get("next_topics") or [])}

## 감정 집계
- 음성 지배 감정: {", ".join(meta.get("voice_dominant") or []) or "-"}
- 표정 지배 감정: {", ".join(meta.get("face_dominant") or []) or "-"}
- 감정 변화: {", ".join(meta.get("emotion_shifts") or []) or "-"}

---
{report_json.get("disclaimer") or "본 리포트는 AI 기반 자기성찰 도구이며 전문 진단을 대체하지 않습니다."}
"""
