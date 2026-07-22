"""데모용 WebSocket (/ws/predict) — 인증 없는 로컬 데모/테스트 전용.

공유 파이프라인 로직은 app/ws/pipeline.py 참고. 계약 구현은 /ws/interview.
"""

from collections import deque

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.memory.session_memory import SessionContext, SessionMemory
from app.services.llm_service import LLMReplyService
from app.services.ser_service import SERService
from app.ws.pipeline import (
    UtteranceAssembler,
    parse_session_context,
    parse_utterance_text,
    run_final_turn,
)


def _context_from_query(websocket: WebSocket) -> SessionContext:
    return SessionContext(
        user_id=websocket.query_params.get("user_id", "anonymous"),
        session_id=websocket.query_params.get("session_id", "default"),
        persona_id=websocket.query_params.get("persona_id", "default"),
    )


async def predict_websocket(
    websocket: WebSocket,
    ser: SERService,
    llm: LLMReplyService,
    memory: SessionMemory,
) -> None:
    await websocket.accept()

    if not ser.is_loaded:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close(code=1011)
        return

    assembler = UtteranceAssembler()
    pending_texts = deque()
    context = _context_from_query(websocket)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                text_msg = message["text"]
                if text_msg == "flush":
                    audio = assembler.flush()
                    if audio is not None:
                        await run_final_turn(
                            websocket, ser, llm, memory, context, audio, pending_texts
                        )
                else:
                    next_context = parse_session_context(text_msg, context)
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
                        await websocket.send_json(
                            {
                                "type": "session_closed",
                                "session_id": context.session_id,
                                "turn_count": len(memory.recent_messages(context)),
                            }
                        )
                        continue

                    utterance_text = parse_utterance_text(text_msg)
                    if utterance_text:
                        pending_texts.append(utterance_text)
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue

            assembler.feed(chunk)

            partial_audio = assembler.partial_window()
            if partial_audio is not None:
                partial_result = ser.predict_from_audio(partial_audio)
                await websocket.send_json({"type": "partial", **partial_result})

            utterance = assembler.pop_completed_utterance()
            if utterance is not None:
                await run_final_turn(
                    websocket, ser, llm, memory, context, utterance, pending_texts
                )

    except WebSocketDisconnect:
        return
