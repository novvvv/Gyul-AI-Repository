import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.memory.session_memory import SessionContext, create_session_memory
from app.services.llm_service import llm_reply_service


def main() -> None:
    context = SessionContext("local-user", "local-session", "gyul")
    memory = create_session_memory()
    memory.clear(context)

    memory.append_user_message(context, "나는 면접에서 첫 질문을 받을 때 긴장해.", "fear")
    memory.append_assistant_message(
        context,
        "첫 질문에서 긴장이 커지는군요. 그 순간 어떤 생각이 가장 먼저 떠오르나요?",
    )

    reply = llm_reply_service.generate_reply(
        text="대답을 잘 못하면 평가가 나빠질 것 같다는 생각이 들어요.",
        emotion="fear",
        context=context,
        history=memory.recent_messages(context),
    )
    memory.append_user_message(
        context,
        "대답을 잘 못하면 평가가 나빠질 것 같다는 생각이 들어요.",
        "fear",
    )
    memory.append_assistant_message(context, reply)

    print("Reply:")
    print(reply)
    print("History count:", len(memory.recent_messages(context)))


if __name__ == "__main__":
    main()
