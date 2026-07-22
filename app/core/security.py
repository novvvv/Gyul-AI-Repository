"""Spring 발급 JWT 교차 검증 (연동 가이드 §1).

Spring이 HS256 + 공유 시크릿(JWT_SECRET)으로 서명한 Access Token을 검증한다.
Access Token claims: sub(email), memberId(회원 PK), role, iat/exp.
"""

import logging

import jwt

from app.core import config

logger = logging.getLogger(__name__)

# 시계 오차 허용 (가이드 §1.3 — 만료 오판 방지용 최소값)
LEEWAY_SECONDS = 10


class TokenError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def verify_access_token(token: str) -> dict:
    """서명+만료 검증 후 payload 반환. 실패 시 TokenError.

    Refresh Token에는 memberId claim이 없으므로(가이드 §1.2) sub와 함께
    memberId 존재도 요구해 Refresh Token 유입을 차단한다.
    """
    if not config.JWT_SECRET:
        logger.warning("JWT_SECRET 미설정 — /ws/interview 인증 불가 (.env 확인)")
        raise TokenError("invalid")

    try:
        # algorithms 명시는 alg 혼동 공격 방지를 위해 필수 (가이드 §1.3)
        payload = jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=["HS256"],
            leeway=LEEWAY_SECONDS,
        )
    except jwt.ExpiredSignatureError:
        raise TokenError("expired")
    except jwt.InvalidTokenError:
        raise TokenError("invalid")

    if not payload.get("sub") or payload.get("memberId") is None:
        raise TokenError("invalid")
    return payload
