import json
import os
from urllib import error, request

from app.chains.langchain_conversation import (
    LangChainConversation,
    LangChainUnavailableError,
    UnsupportedLLMProviderError,
)
from app.chains.conversation_chain import build_messages
from app.core.config import LLM_PROVIDER, OPENAI_MODEL
from app.memory.session_memory import ConversationTurn, SessionContext


class LLMReplyService:
    def __init__(self) -> None:
        self.langchain_conversation = LangChainConversation()

    def generate_reply(
        self,
        text: str,
        emotion: str,
        context: SessionContext | None = None,
        history: list[ConversationTurn] | None = None,
    ) -> str:
        api_key = self._get_api_key()
        if not api_key:
            return f"{self._api_key_name()}가 없어서 감정({emotion})만 분석했어요."

        messages = self._build_messages(text, emotion, context, history or [])
        try:
            return self.langchain_conversation.invoke(messages)
        except LangChainUnavailableError:
            if LLM_PROVIDER != "openai":
                return (
                    "Gemini 호출에 필요한 langchain-google-genai 패키지가 없어요. "
                    "pip install -r requirements.txt를 실행해주세요."
                )
            return self._generate_reply_with_openai_rest(api_key, messages)
        except UnsupportedLLMProviderError as e:
            return f"AI 답변 호출 실패: {e}"
        except Exception as e:
            return f"AI 답변 호출 실패: {e}"

    def _generate_reply_with_openai_rest(
        self,
        api_key: str,
        messages: list[dict[str, str]],
    ) -> str:
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

    def _get_api_key(self) -> str | None:
        if LLM_PROVIDER == "gemini":
            return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    def _api_key_name(self) -> str:
        if LLM_PROVIDER == "gemini":
            return "GOOGLE_API_KEY 또는 GEMINI_API_KEY"
        return "OPENAI_API_KEY"


llm_reply_service = LLMReplyService()
