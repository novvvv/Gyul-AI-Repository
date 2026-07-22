import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.conftest import make_access_token


@pytest.fixture
def client(jwt_secret):
    from app.main import app

    # 컨텍스트 매니저 없이 생성해 startup(SER 모델 로드)을 건너뛴다 —
    # 인증 계층은 모델 로드와 무관하게 동작해야 한다.
    return TestClient(app)


def _expect_close(ws, code: int) -> None:
    with pytest.raises(WebSocketDisconnect) as exc:
        ws.receive_json()
    assert exc.value.code == code


def test_connect_without_token_closes_4401(client):
    with client.websocket_connect("/ws/interview") as ws:
        _expect_close(ws, 4401)


def test_connect_with_forged_token_closes_4401(client):
    token = make_access_token()
    forged = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    with client.websocket_connect(f"/ws/interview?token={forged}") as ws:
        _expect_close(ws, 4401)


def test_connect_with_expired_token_closes_4401(client):
    token = make_access_token(expires_in=-60)
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        _expect_close(ws, 4401)


def test_connect_with_unknown_role_closes_4403(client):
    token = make_access_token(role="GUEST")
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        _expect_close(ws, 4403)


def test_connect_with_valid_token_binds_session(client):
    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session_ready"
        assert ready["email"] == "ai-test@example.com"
        assert ready["member_id"] == 42


def test_session_end_closes_normally(client):
    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ws.receive_json()  # session_ready
        ws.send_text("session_end")
        closed = ws.receive_json()
        assert closed["type"] == "session_closed"
        _expect_close(ws, 1000)
