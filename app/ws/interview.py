"""인증 필수 인터뷰 WebSocket (연동 가이드 §2).

/ws/interview?token=<ACCESS_TOKEN> 핸드셰이크에서 Spring 발급 JWT를 검증하고
sub(email)를 세션에 바인딩한 뒤, 데모와 동일한 VAD→SER→LLM→TTS 파이프라인을
수행하며 리포트 발행용 턴 로그를 축적한다. 세션 중 토큰 만료 재검증은 하지
않는다(§2.4 기본안).

close code: 4401(토큰 없음/무효/만료), 4403(권한 없음), 1000(정상 종료).
"""

import json
import logging
from collections import deque

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.core.security import TokenError, verify_access_token
from app.memory.session_memory import SessionContext, SessionMemory
from app.services.llm_service import LLMReplyService
from app.services.ser_service import SERService
from app.ws.interview_session import InterviewSessionState
from app.ws.pipeline import (
    UtteranceAssembler,
    parse_utterance_text,
    run_final_turn,
)

logger = logging.getLogger(__name__)

CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN = 4403
ALLOWED_ROLES = frozenset({"MEMBER", "ADMIN"})


async def authenticate(websocket: WebSocket) -> dict | None:
    """핸드셰이크 인증. 성공 시 accept 후 payload 반환, 실패 시 close 후 None.

    브라우저가 close code를 받을 수 있도록 accept 후 즉시 close한다
    (accept 없이 close하면 403 핸드셰이크 거부로 바뀌어 코드가 전달되지 않음).
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.accept()
        await websocket.close(code=CLOSE_UNAUTHORIZED)
        return None

    try:
        payload = verify_access_token(token)
    except TokenError as e:
        logger.info("interview WS 인증 실패: %s", e.reason)
        await websocket.accept()
        await websocket.close(code=CLOSE_UNAUTHORIZED)
        return None

    if payload.get("role") not in ALLOWED_ROLES:
        await websocket.accept()
        await websocket.close(code=CLOSE_FORBIDDEN)
        return None

    await websocket.accept()
    return payload


def _parse_persona(text_msg: str, default: str) -> str | None:
    """session_start 메시지에서 persona_id만 취한다.

    user_id/session_id는 서버 권한(JWT email·서버 생성 sessionId)이므로 무시.
    """
    try:
        text_payload = json.loads(text_msg)
    except json.JSONDecodeError:
        return None
    if not isinstance(text_payload, dict) or text_payload.get("type") != "session_start":
        return None
    return str(text_payload.get("persona_id") or default)


async def interview_websocket(
    websocket: WebSocket,
    ser: SERService,
    llm: LLMReplyService,
    memory: SessionMemory,
) -> None:
    payload = await authenticate(websocket)
    if payload is None:
        return

    state = InterviewSessionState(
        email=payload["sub"],
        member_id=payload.get("memberId"),
    )
    context = SessionContext(
        user_id=state.email,
        session_id=state.session_id,
        persona_id=state.persona_id,
    )

    await websocket.send_json(
        {
            "type": "session_ready",
            "session_id": state.session_id,
            "persona_id": state.persona_id,
            "email": state.email,
            "member_id": state.member_id,
        }
    )

    assembler = UtteranceAssembler()
    pending_texts = deque()

    async def finalize_turn(audio) -> None:
        turn = await run_final_turn(
            websocket, ser, llm, memory, context, audio, pending_texts
        )
        state.record_turn(turn)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                text_msg = message["text"]
                if text_msg == "flush":
                    audio = assembler.flush()
                    if audio is not None and ser.is_loaded:
                        await finalize_turn(audio)
                    continue

                if text_msg == "session_end":
                    await websocket.send_json(
                        {
                            "type": "session_closed",
                            "session_id": state.session_id,
                            "turn_count": state.turn_count,
                        }
                    )
                    await websocket.close(code=1000)
                    break

                persona_id = _parse_persona(text_msg, state.persona_id)
                if persona_id is not None:
                    state.persona_id = persona_id
                    context = SessionContext(
                        user_id=state.email,
                        session_id=state.session_id,
                        persona_id=persona_id,
                    )
                    await websocket.send_json(
                        {
                            "type": "session_ready",
                            "session_id": state.session_id,
                            "persona_id": persona_id,
                            "email": state.email,
                            "member_id": state.member_id,
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

            if not ser.is_loaded:
                await websocket.send_json({"type": "error", "error": "Model not loaded"})
                continue

            assembler.feed(chunk)

            partial_audio = assembler.partial_window()
            if partial_audio is not None:
                partial_result = ser.predict_from_audio(partial_audio)
                await websocket.send_json({"type": "partial", **partial_result})

            utterance = assembler.pop_completed_utterance()
            if utterance is not None:
                await finalize_turn(utterance)

    except WebSocketDisconnect:
        pass
