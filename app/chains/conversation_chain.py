from app.memory.session_memory import ConversationTurn, SessionContext


PERSONA_SYSTEM_PROMPTS = {
    "default": (
        "너는 한국어 대화 어시스턴트다. 사용자의 감정과 이전 대화 흐름을 반영해 "
        "짧고 자연스럽게 답한다."
    ),
    "gyul": (
        "너는 결이라는 이름의 자기분석 대화 상대다. 사용자가 스스로를 더 잘 "
        "이해하도록 부드럽게 되묻고, 단정적인 진단은 피한다."
        "최대한 사용자가 한 말에 공감을 해주고, 위로를 건네면서 대화를 진행한다."
        "계속해서 사용자의 반응을 이끌어내면서 대답한다."
    ),
    "interviewer": (
        "너는 차분한 AI 면접관이다. 답변을 경청하되 면접 흐름을 유지하고, "
        "필요하면 짧은 꼬리 질문을 한다."
    ),
}


def build_system_prompt(context: SessionContext) -> str:
    return PERSONA_SYSTEM_PROMPTS.get(
        context.persona_id,
        PERSONA_SYSTEM_PROMPTS["default"],
    )


def build_messages(
    context: SessionContext,
    history: list[ConversationTurn],
    user_text: str,
    emotion: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(context)}]

    for turn in history:
        content = turn.content
        if turn.role == "user" and turn.emotion:
            content = f"{content}\n[감정 신호: {turn.emotion}]"
        messages.append({"role": turn.role, "content": content})

    messages.append(
        {
            "role": "user",
            "content": (
                f"{user_text}\n"
                f"[현재 감정 신호: {emotion}]\n"
                "위 감정 신호는 답변의 톤 조절에만 참고해줘."
            ),
        }
    )
    return messages
