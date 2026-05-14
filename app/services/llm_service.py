import json
import os
from urllib import error, request

from app.chains.conversation_chain import build_messages
from app.core.config import OPENAI_MODEL
from app.memory.session_memory import ConversationTurn, SessionContext


class LLMReplyService:
    def generate_reply(
        self,
        text: str,
        emotion: str,
        context: SessionContext | None = None,
        history: list[ConversationTurn] | None = None,
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return f"OPENAI_API_KEY가 없어서 감정({emotion})만 분석했어요."

        messages = self._build_messages(text, emotion, context, history or [])
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.7,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            return f"AI 답변 호출 실패(HTTP {e.code}): {detail[:180]}"
        except Exception as e:
            return f"AI 답변 호출 실패: {e}"

    def _build_messages(
        self,
        text: str,
        emotion: str,
        context: SessionContext | None,
        history: list[ConversationTurn],
    ) -> list[dict[str, str]]:
        if context is None:
            context = SessionContext(
                user_id="anonymous",
                session_id="default",
                persona_id="default",
            )
        return build_messages(context, history, text, emotion)


llm_reply_service = LLMReplyService()
