"""WebSocket 대화 파이프라인 공유 로직.

/ws/predict(데모)와 /ws/interview(계약)가 같은 VAD→SER→LLM→TTS 흐름을 쓰도록
제어 메시지 파싱, 발화 조립(VAD), 최종 턴 처리를 이 모듈로 추출했다.
"""

import asyncio
import json
from collections import deque
from datetime import datetime, timezone

import numpy as np
from fastapi import WebSocket

from app.core.config import (
    MIN_CHUNK_SECONDS,
    TARGET_SR,
    VAD_MIN_SPEECH_SECONDS,
    VAD_RMS_THRESHOLD,
    VAD_SILENCE_SECONDS,
)
from app.memory.session_memory import SessionContext, SessionMemory
from app.services.fish_tts_service import fish_tts_service
from app.services.llm_service import LLMReplyService
from app.services.ser_service import SERService


def parse_utterance_text(text_msg: str) -> str | None:
    try:
        text_payload = json.loads(text_msg)
    except json.JSONDecodeError:
        return None

    if (
        isinstance(text_payload, dict)
        and text_payload.get("type") == "utterance_text"
        and isinstance(text_payload.get("text"), str)
    ):
        utterance_text = text_payload["text"].strip()
        return utterance_text or None
    return None


def parse_session_context(
    text_msg: str,
    current_context: SessionContext,
) -> SessionContext | None:
    try:
        text_payload = json.loads(text_msg)
    except json.JSONDecodeError:
        return None

    if not isinstance(text_payload, dict) or text_payload.get("type") != "session_start":
        return None

    return SessionContext(
        user_id=str(text_payload.get("user_id") or current_context.user_id),
        session_id=str(text_payload.get("session_id") or current_context.session_id),
        persona_id=str(text_payload.get("persona_id") or current_context.persona_id),
    )


class UtteranceAssembler:
    """PCM 스트림에서 RMS 기반 VAD로 발화 단위를 조립한다."""

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        min_chunk_seconds: float = MIN_CHUNK_SECONDS,
        rms_threshold: float = VAD_RMS_THRESHOLD,
        min_speech_seconds: float = VAD_MIN_SPEECH_SECONDS,
        silence_seconds: float = VAD_SILENCE_SECONDS,
    ) -> None:
        self.min_samples = int(target_sr * min_chunk_seconds)
        self.min_speech_samples = int(target_sr * min_speech_seconds)
        self.silence_limit_samples = int(target_sr * silence_seconds)
        self.rms_threshold = rms_threshold
        self.stream_buffer = np.array([], dtype=np.float32)
        self.utterance_buffer = np.array([], dtype=np.float32)
        self.speech_active = False
        self.silence_samples = 0

    def feed(self, chunk: bytes) -> None:
        """16-bit PCM mono bytes 반영."""
        pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        # partial 윈도우는 마지막 min_samples만 쓰므로 그 이상은 유지하지 않는다.
        self.stream_buffer = np.concatenate([self.stream_buffer, pcm])[
            -self.min_samples :
        ]

        rms = float(np.sqrt(np.mean(np.square(pcm)) + 1e-12))
        if rms >= self.rms_threshold:
            self.speech_active = True
            self.silence_samples = 0
            self.utterance_buffer = np.concatenate([self.utterance_buffer, pcm])
        elif self.speech_active:
            self.silence_samples += len(pcm)
            self.utterance_buffer = np.concatenate([self.utterance_buffer, pcm])

    def partial_window(self) -> np.ndarray | None:
        if len(self.stream_buffer) >= self.min_samples:
            return self.stream_buffer[-self.min_samples :]
        return None

    def pop_completed_utterance(self) -> np.ndarray | None:
        """침묵이 기준을 넘으면 완결된 발화를 반환하고 상태를 리셋한다."""
        if (
            self.speech_active
            and self.silence_samples >= self.silence_limit_samples
            and len(self.utterance_buffer) >= self.min_speech_samples
        ):
            audio = self.utterance_buffer
            self._reset_utterance()
            return audio
        return None

    def flush(self) -> np.ndarray | None:
        """flush 제어 메시지: 최소 발화 길이 충족 시 발화 반환, 항상 리셋."""
        audio = (
            self.utterance_buffer
            if len(self.utterance_buffer) >= self.min_speech_samples
            else None
        )
        self._reset_utterance()
        return audio

    def _reset_utterance(self) -> None:
        self.utterance_buffer = np.array([], dtype=np.float32)
        self.speech_active = False
        self.silence_samples = 0


async def run_final_turn(
    websocket: WebSocket,
    ser: SERService,
    llm: LLMReplyService,
    memory: SessionMemory,
    context: SessionContext,
    audio: np.ndarray,
    pending_texts: deque,
) -> dict:
    """발화 하나를 SER→LLM→(TTS)로 처리해 final 응답을 보내고 턴 기록을 반환한다."""
    final_result = ser.predict_from_audio(audio)
    text_for_reply = pending_texts.popleft() if pending_texts else "방금 발화"
    history = memory.recent_messages(context)
    ai_reply = await asyncio.to_thread(
        llm.generate_reply,
        text_for_reply,
        final_result["label"],
        context,
        history,
    )
    memory.append_user_message(context, text_for_reply, final_result["label"])
    memory.append_assistant_message(context, ai_reply)

    payload: dict = {
        "type": "final",
        "session_id": context.session_id,
        "persona_id": context.persona_id,
        "text": text_for_reply,
        "reply": ai_reply,
        **final_result,
    }

    if fish_tts_service.is_enabled:
        audio_b64 = await asyncio.to_thread(
            fish_tts_service.synthesize_b64,
            ai_reply,
            final_result["label"],
        )
        if audio_b64:
            payload["reply_audio_b64"] = audio_b64
            payload["reply_audio_format"] = "mp3"

    await websocket.send_json(payload)

    return {
        "user_text": text_for_reply,
        "voice_emotion": {
            "label": final_result["label"],
            "confidence": final_result["confidence"],
            "probs": final_result["probs"],
        },
        "bot_reply": ai_reply,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "speech_samples": int(len(audio)),
    }
