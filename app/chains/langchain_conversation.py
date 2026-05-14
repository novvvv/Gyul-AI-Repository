import os

from app.core.config import GEMINI_MODEL, LLM_PROVIDER, OPENAI_MODEL


class LangChainUnavailableError(RuntimeError):
    pass


class UnsupportedLLMProviderError(RuntimeError):
    pass


class LangChainConversation:
    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model_name: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        self.provider = provider
        self.model_name = model_name or self._default_model_name(provider)
        self.temperature = temperature

    def invoke(self, messages: list[dict[str, str]]) -> str:
        if self.provider == "gemini":
            return self._invoke_gemini(messages)
        if self.provider == "openai":
            return self._invoke_openai(messages)
        raise UnsupportedLLMProviderError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def _invoke_openai(self, messages: list[dict[str, str]]) -> str:
        try:
            from langchain.messages import AIMessage, HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise LangChainUnavailableError(str(e)) from e

        chain_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                chain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                chain_messages.append(AIMessage(content=content))
            else:
                chain_messages.append(HumanMessage(content=content))

        model = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        response = model.invoke(chain_messages)
        return str(response.content).strip()

    def _invoke_gemini(self, messages: list[dict[str, str]]) -> str:
        try:
            from langchain.messages import AIMessage, HumanMessage, SystemMessage
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise LangChainUnavailableError(str(e)) from e

        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        chain_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                chain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                chain_messages.append(AIMessage(content=content))
            else:
                chain_messages.append(HumanMessage(content=content))

        model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=google_api_key,
        )
        response = model.invoke(chain_messages)
        return str(response.content).strip()

    def _default_model_name(self, provider: str) -> str:
        if provider == "gemini":
            return GEMINI_MODEL
        return OPENAI_MODEL
