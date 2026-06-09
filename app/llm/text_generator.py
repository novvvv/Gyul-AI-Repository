"""대화·보고서 공통 텍스트 LLM 라우터.

OPENAI_API_KEY가 있으면 gpt-4o-mini, 없으면 Kanana(로컬 HF)를 사용한다.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from app.core.config import KANANA_MODEL_ID, OPENAI_MODEL
from app.services.local_llm_service import local_llm_service


class TextGenerationError(RuntimeError):
    pass


def resolve_text_backend() -> str:
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "kanana"


def generate_chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    backend = resolve_text_backend()
    if backend == "openai":
        return _generate_with_openai(messages, temperature=temperature, max_tokens=max_tokens)
    return _generate_with_kanana(
        messages, temperature=temperature, max_tokens=max_tokens
    )


def ensure_kanana_loaded() -> None:
    from app.llm.kanana_support import patch_kanana_llama_config

    patch_kanana_llama_config()
    if local_llm_service.is_loaded and local_llm_service.model_id == KANANA_MODEL_ID:
        return
    if local_llm_service.is_loaded:
        local_llm_service.tokenizer = None
        local_llm_service.model = None
    local_llm_service.model_id = KANANA_MODEL_ID
    local_llm_service.load()


def _generate_with_openai(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int | None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise TextGenerationError("OPENAI_API_KEY is missing")

    try:
        from app.chains.langchain_conversation import (
            LangChainConversation,
            LangChainUnavailableError,
        )

        model = LangChainConversation(
            provider="openai",
            model_name=OPENAI_MODEL,
            temperature=temperature,
        )
        return model.invoke(messages)
    except LangChainUnavailableError:
        return _generate_with_openai_rest(api_key, messages, temperature, max_tokens)
    except Exception as e:
        raise TextGenerationError(f"OpenAI generation failed: {e}") from e


def _generate_with_openai_rest(
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> str:
    payload: dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

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
        with request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise TextGenerationError(f"OpenAI HTTP {e.code}: {detail[:240]}") from e
    except Exception as e:
        raise TextGenerationError(f"OpenAI REST failed: {e}") from e


def _generate_with_kanana(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int | None = None,
) -> str:
    try:
        ensure_kanana_loaded()
        prev_temp = local_llm_service.temperature
        prev_max = local_llm_service.max_new_tokens
        local_llm_service.temperature = temperature
        if max_tokens is not None:
            local_llm_service.max_new_tokens = max_tokens
        try:
            return local_llm_service.generate(messages)
        finally:
            local_llm_service.temperature = prev_temp
            local_llm_service.max_new_tokens = prev_max
    except Exception as e:
        raise TextGenerationError(f"Kanana generation failed: {e}") from e
