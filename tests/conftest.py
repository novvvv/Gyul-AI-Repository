import sys
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Spring과 공유하는 시크릿의 테스트 대역 (32바이트 이상 규칙 준수)
TEST_JWT_SECRET = "test-jwt-secret-shared-with-spring-32bytes"


def make_access_token(
    secret: str = TEST_JWT_SECRET,
    *,
    expires_in: int = 1800,
    algorithm: str = "HS256",
    **overrides,
) -> str:
    """Spring JwtProvider가 발급하는 Access Token 형태의 테스트 토큰 생성.

    overrides에 claim=None을 주면 해당 claim을 제거한다 (Refresh Token 흉내 등).
    """
    now = int(time.time())
    payload = {
        "sub": "ai-test@example.com",
        "memberId": 42,
        "role": "MEMBER",
        "iat": now,
        "exp": now + expires_in,
    }
    for key, value in overrides.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return jwt.encode(payload, secret, algorithm=algorithm)


@pytest.fixture
def jwt_secret(monkeypatch) -> str:
    from app.core import config

    monkeypatch.setattr(config, "JWT_SECRET", TEST_JWT_SECRET)
    return TEST_JWT_SECRET
