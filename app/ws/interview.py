"""인증 필수 인터뷰 WebSocket (연동 가이드 §2).

/ws/interview?token=<ACCESS_TOKEN> 핸드셰이크에서 Spring 발급 JWT를 검증하고
sub(email)를 세션에 바인딩한다. 세션 중 토큰 만료 재검증은 하지 않는다(§2.4 기본안).

close code: 4401(토큰 없음/무효/만료), 4403(권한 없음), 1000(정상 종료).
"""

import logging

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from app.core.security import TokenError, verify_access_token

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


async def interview_websocket(websocket: WebSocket) -> None:
    payload = await authenticate(websocket)
    if payload is None:
        return

    email = payload["sub"]
    member_id = payload.get("memberId")

    await websocket.send_json(
        {
            "type": "session_ready",
            "email": email,
            "member_id": member_id,
        }
    )

    # 인증 골격 단계: 대화 파이프라인 연결 전까지는 session_end만 처리한다.
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") == "session_end":
                await websocket.send_json({"type": "session_closed"})
                await websocket.close(code=1000)
                break
    except WebSocketDisconnect:
        pass
