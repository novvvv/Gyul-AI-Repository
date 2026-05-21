import asyncio
import json
from collections import deque

import numpy as np
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.core.config import (
    MIN_CHUNK_SECONDS,
    TARGET_SR,
    VAD_MIN_SPEECH_SECONDS,
    VAD_RMS_THRESHOLD,
    VAD_SILENCE_SECONDS,
)
from app.memory.session_memory import SessionContext, SessionMemory
from app.memory.summary_memory import SummaryMemory
from app.services.llm_service import LLMReplyService
from app.services.ser_service import SERService
from app.services.summary_service import SummaryService


def _context_from_query(websocket: WebSocket) -> SessionContext:
    return SessionContext(
        user_id=websocket.query_params.get("user_id", "anonymous"),
        session_id=websocket.query_params.get("session_id", "default"),
        persona_id=websocket.query_params.get("persona_id", "default"),
    )


async def _send_final_result(
    websocket: WebSocket,
    ser: SERService,
    llm: LLMReplyService,
    memory: SessionMemory,
    summaries: SummaryMemory,
    context: SessionContext,
    audio: np.ndarray,
    pending_texts: deque,
) -> None:
    final_result = ser.predict_from_audio(audio)
    text_for_reply = pending_texts.popleft() if pending_texts else "방금 발화"
    history = memory.recent_messages(context)
    session_summary = summaries.get_summary(context)
    ai_reply = await asyncio.to_thread(
        llm.generate_reply,
        text_for_reply,
        final_result["label"],
        context,
        history,
        session_summary,
    )
    memory.append_user_message(context, text_for_reply, final_result["label"])
    memory.append_assistant_message(context, ai_reply)
    await websocket.send_json(
        {
            "type": "final",
            "session_id": context.session_id,
            "persona_id": context.persona_id,
            "text": text_for_reply,
            "reply": ai_reply,
            **final_result,
        }
    )


def _parse_utterance_text(text_msg: str) -> str | None:
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


def _parse_session_context(
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


async def predict_websocket(
    websocket: WebSocket,
    ser: SERService,
    llm: LLMReplyService,
    memory: SessionMemory,
    summaries: SummaryMemory,
    summary: SummaryService,
) -> None:
    await websocket.accept()

    if not ser.is_loaded:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close(code=1011)
        return

    min_samples = int(TARGET_SR * MIN_CHUNK_SECONDS)
    vad_min_speech_samples = int(TARGET_SR * VAD_MIN_SPEECH_SECONDS)
    vad_silence_samples = int(TARGET_SR * VAD_SILENCE_SECONDS)
    stream_buffer = np.array([], dtype=np.float32)
    utterance_buffer = np.array([], dtype=np.float32)
    pending_texts = deque()
    speech_active = False
    silence_samples = 0
    context = _context_from_query(websocket)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                text_msg = message["text"]
                if text_msg == "flush":
                    if len(utterance_buffer) >= vad_min_speech_samples:
                        await _send_final_result(
                            websocket,
                            ser,
                            llm,
                            memory,
                            summaries,
                            context,
                            utterance_buffer,
                            pending_texts,
                        )
                    utterance_buffer = np.array([], dtype=np.float32)
                    speech_active = False
                    silence_samples = 0
                else:
                    next_context = _parse_session_context(text_msg, context)
                    if next_context:
                        context = next_context
                        await websocket.send_json(
                            {
                                "type": "session_ready",
                                "session_id": context.session_id,
                                "persona_id": context.persona_id,
                            }
                        )
                        continue

                    if text_msg == "session_end":
                        history = memory.recent_messages(context)
                        previous_summary = summaries.get_summary(context)
                        session_summary = await asyncio.to_thread(
                            summary.summarize,
                            context,
                            previous_summary,
                            history,
                        )
                        if session_summary:
                            summaries.save_summary(context, session_summary)
                        await websocket.send_json(
                            {
                                "type": "session_closed",
                                "session_id": context.session_id,
                                "turn_count": len(memory.recent_messages(context)),
                                "summary": session_summary,
                            }
                        )
                        continue

                    utterance_text = _parse_utterance_text(text_msg)
                    if utterance_text:
                        pending_texts.append(utterance_text)
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue

            # Expect 16-bit PCM mono audio bytes from client.
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            stream_buffer = np.concatenate([stream_buffer, pcm])

            rms = float(np.sqrt(np.mean(np.square(pcm)) + 1e-12))
            is_speech = rms >= VAD_RMS_THRESHOLD

            if is_speech:
                speech_active = True
                silence_samples = 0
                utterance_buffer = np.concatenate([utterance_buffer, pcm])
            elif speech_active:
                silence_samples += len(pcm)
                utterance_buffer = np.concatenate([utterance_buffer, pcm])

            if len(stream_buffer) >= min_samples:
                audio = stream_buffer[-min_samples:]
                partial_result = ser.predict_from_audio(audio)
                await websocket.send_json({"type": "partial", **partial_result})

            if (
                speech_active
                and silence_samples >= vad_silence_samples
                and len(utterance_buffer) >= vad_min_speech_samples
            ):
                await _send_final_result(
                    websocket,
                    ser,
                    llm,
                    memory,
                    summaries,
                    context,
                    utterance_buffer,
                    pending_texts,
                )
                utterance_buffer = np.array([], dtype=np.float32)
                speech_active = False
                silence_samples = 0

    except WebSocketDisconnect:
        return
