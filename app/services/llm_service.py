from app.chains.conversation_chain import build_messages
from app.llm.text_generator import TextGenerationError, generate_chat_completion
from app.memory.session_memory import ConversationTurn, SessionContext


class LLMReplyService:
    def generate_reply(
        self,
        text: str,
        emotion: str,
        context: SessionContext | None = None,
        history: list[ConversationTurn] | None = None,
    ) -> str:
        if context is None:
            context = SessionContext(
                user_id="anonymous",
                session_id="default",
                persona_id="default",
            )
        messages = build_messages(context, history or [], text, emotion)
        try:
            return generate_chat_completion(messages, temperature=0.7)
        except TextGenerationError as e:
            return f"AI 답변 생성 실패: {e}"


llm_reply_service = LLMReplyService()
