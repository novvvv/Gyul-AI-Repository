from app.memory.session_memory import ConversationTurn, SessionContext


def build_summary_messages(
    context: SessionContext,
    previous_summary: str,
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    transcript = _format_history(history)
    previous = previous_summary or "아직 저장된 요약 없음"
    return [
        {
            "role": "system",
            "content": (
                "너는 한국어 대화 세션을 요약하는 분석가다. "
                "다음 세션에서 문맥을 이어갈 수 있도록 핵심만 간결하게 정리해라. "
                "진단을 단정하지 말고 관찰된 내용만 적어라."
            ),
        },
        {
            "role": "user",
            "content": (
                f"사용자 ID: {context.user_id}\n"
                f"세션 ID: {context.session_id}\n"
                f"페르소나: {context.persona_id}\n\n"
                f"기존 요약:\n{previous}\n\n"
                f"최근 대화:\n{transcript}\n\n"
                "아래 항목을 포함해 5문장 이내로 요약해줘.\n"
                "- 현재 대화 주제\n"
                "- 사용자가 반복해서 드러낸 감정 또는 걱정\n"
                "- 다음 대화에서 이어받아야 할 핵심 맥락\n"
                "- 대화 상대가 유지해야 할 톤"
            ),
        },
    ]


def fallback_summary(
    previous_summary: str,
    history: list[ConversationTurn],
    max_chars: int = 900,
) -> str:
    parts = []
    if previous_summary:
        parts.append(f"이전 요약: {previous_summary.strip()}")
    if history:
        parts.append("최근 대화: " + _format_history(history))
    summary = "\n".join(parts).strip()
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:]


def _format_history(history: list[ConversationTurn]) -> str:
    lines = []
    for turn in history:
        emotion = f" / 감정: {turn.emotion}" if turn.emotion else ""
        speaker = "사용자" if turn.role == "user" else "AI"
        lines.append(f"{speaker}{emotion}: {turn.content}")
    return "\n".join(lines) if lines else "최근 대화 없음"
