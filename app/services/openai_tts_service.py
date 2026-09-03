"""OpenAI TTS — 사용자 SER 감정에 맞춰 낭독 톤을 바꾼다.

Fish Audio 서비스와 **같은 인터페이스**(`is_enabled` / `synthesize_b64`)를 갖는다.
app/ws/pipeline.py 가 둘 중 켜져 있는 쪽을 골라 쓴다.

참고: Whisper(`audio/transcriptions`)는 STT다. 낭독은 `audio/speech` 엔드포인트를 쓴다.
"""

from __future__ import annotations

import base64
import json
import logging
from urllib import error, request

from app.core.config import (
    ENABLE_OPENAI_TTS,
    OPENAI_API_KEY,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_VOICE,
)

logger = logging.getLogger(__name__)

OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

# 사용자 SER → 낭독 지시문.
# gpt-4o-mini-tts 는 `instructions` 로 톤을 바꿀 수 있다. 봇이 공감 톤으로 읽는다.
SER_TONE: dict[str, str] = {
    "sadness": "차분하고 낮은 톤으로, 조금 느리게. 위로하듯이 읽어라.",
    "fear": "부드럽고 안심시키는 톤으로, 천천히 읽어라.",
    "angry": "감정을 누그러뜨리는 차분한 톤으로 읽어라.",
    "disgust": "담담하고 중립적인 톤으로 읽어라.",
    "happiness": "밝고 가벼운 톤으로, 미소 띤 목소리로 읽어라.",
    "surprise": "약간 올라간 톤으로 반가움을 담아 읽어라.",
    "neutral": "차분하고 담담한 톤으로 읽어라.",
}

DEFAULT_TONE = "차분하고 담담한 톤으로 읽어라."


class OpenAiTtsError(RuntimeError):
    pass


class OpenAiTtsService:
    def __init__(
        self,
        *,
        api_key: str = OPENAI_API_KEY,
        voice: str = OPENAI_TTS_VOICE,
        model: str = OPENAI_TTS_MODEL,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.voice = (voice or "").strip()
        self.model = (model or "").strip()

    @property
    def is_enabled(self) -> bool:
        return ENABLE_OPENAI_TTS and bool(self.api_key) and bool(self.voice)

    def synthesize(self, reply: str, user_emotion: str) -> bytes | None:
        if not self.is_enabled:
            return None

        text = (reply or "").strip()
        if not text:
            return None

        payload: dict[str, object] = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "mp3",
        }
        # instructions 는 gpt-4o-mini-tts 계열에서만 지원된다.
        if "gpt-4o" in self.model:
            payload["instructions"] = SER_TONE.get(user_emotion, DEFAULT_TONE)

        req = request.Request(
            OPENAI_TTS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                audio = resp.read()
                if not audio:
                    raise OpenAiTtsError("empty audio response")
                return audio
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:400]
            logger.warning("OpenAI TTS HTTP %s: %s", e.code, detail)
            raise OpenAiTtsError(f"OpenAI TTS HTTP {e.code}: {detail}") from e
        except Exception as e:  # noqa: BLE001
            logger.warning("OpenAI TTS failed: %s", e)
            raise OpenAiTtsError(str(e)) from e

    def synthesize_b64(self, reply: str, user_emotion: str) -> str | None:
        try:
            audio = self.synthesize(reply, user_emotion)
        except OpenAiTtsError:
            return None
        if not audio:
            return None
        return base64.b64encode(audio).decode("ascii")


openai_tts_service = OpenAiTtsService()
