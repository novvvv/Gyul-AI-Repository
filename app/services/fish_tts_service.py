"""Fish Audio TTS — SER 감정 태그 + 고정 voice(reference_id)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any
from urllib import error, request

from app.core.config import (
    ENABLE_FISH_TTS,
    FISH_AUDIO_API_KEY,
    FISH_AUDIO_MODEL,
    FISH_AUDIO_VOICE_ID,
)

logger = logging.getLogger(__name__)

# 사용자 SER → Fish Audio 감정 태그 (봇이 공감 톤으로 읽기)
SER_EMOTION_TAGS: dict[str, str] = {
    "sadness": "[sad][soft]",
    "angry": "[soft]",
    "fear": "[whispering][soft]",
    "happiness": "",
    "surprise": "[soft]",
    "disgust": "[soft]",
    "neutral": "",
}

FISH_TTS_URL = "https://api.fish.audio/v1/tts"


class FishTtsError(RuntimeError):
    pass


class FishTtsService:
    def __init__(
        self,
        *,
        api_key: str = FISH_AUDIO_API_KEY,
        voice_id: str = FISH_AUDIO_VOICE_ID,
        model: str = FISH_AUDIO_MODEL,
    ) -> None:
        self.api_key = api_key.strip()
        self.voice_id = voice_id.strip()
        self.model = model.strip()

    @property
    def is_enabled(self) -> bool:
        return ENABLE_FISH_TTS and bool(self.api_key) and bool(self.voice_id)

    def _build_tts_text(self, reply: str, user_emotion: str) -> str:
        tags = SER_EMOTION_TAGS.get(user_emotion, "")
        body = reply.strip()
        if not body:
            return ""
        return f"{tags}{body}" if tags else body

    def synthesize(self, reply: str, user_emotion: str) -> bytes | None:
        if not self.is_enabled:
            return None

        text = self._build_tts_text(reply, user_emotion)
        if not text:
            return None

        payload: dict[str, Any] = {
            "text": text,
            "reference_id": self.voice_id,
            "format": "mp3",
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            FISH_TTS_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "model": self.model,
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                audio = resp.read()
                if not audio:
                    raise FishTtsError("empty audio response")
                return audio
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:400]
            logger.warning("Fish TTS HTTP %s: %s", e.code, detail)
            raise FishTtsError(f"Fish TTS HTTP {e.code}: {detail}") from e
        except Exception as e:
            logger.warning("Fish TTS failed: %s", e)
            raise FishTtsError(str(e)) from e

    def synthesize_b64(self, reply: str, user_emotion: str) -> str | None:
        try:
            audio = self.synthesize(reply, user_emotion)
        except FishTtsError:
            return None
        if not audio:
            return None
        return base64.b64encode(audio).decode("ascii")


fish_tts_service = FishTtsService()
