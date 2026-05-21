from app.chains.langchain_conversation import (
    LangChainConversation,
    LangChainUnavailableError,
    UnsupportedLLMProviderError,
)
from app.chains.summary_chain import build_summary_messages, fallback_summary
from app.core.config import LLM_PROVIDER, LOCAL_LLM_PROVIDERS
from app.memory.session_memory import ConversationTurn, SessionContext
from app.services.local_llm_service import local_llm_service


class SummaryService:
    def __init__(self) -> None:
        self.langchain_conversation = LangChainConversation()

    def summarize(
        self,
        context: SessionContext,
        previous_summary: str,
        history: list[ConversationTurn],
    ) -> str:
        if not history and not previous_summary:
            return ""

        messages = build_summary_messages(context, previous_summary, history)
        if LLM_PROVIDER in LOCAL_LLM_PROVIDERS:
            return self._summarize_with_local_llm(previous_summary, history, messages)

        try:
            return self.langchain_conversation.invoke(messages)
        except (
            LangChainUnavailableError,
            UnsupportedLLMProviderError,
            Exception,
        ):
            return fallback_summary(previous_summary, history)

    def _summarize_with_local_llm(
        self,
        previous_summary: str,
        history: list[ConversationTurn],
        messages: list[dict[str, str]],
    ) -> str:
        if not local_llm_service.is_loaded:
            return fallback_summary(previous_summary, history)
        try:
            return local_llm_service.generate(messages)
        except Exception:
            return fallback_summary(previous_summary, history)


summary_service = SummaryService()
