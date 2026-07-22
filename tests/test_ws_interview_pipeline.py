import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_access_token

SR = 16000


@pytest.fixture
def client(jwt_secret, monkeypatch):
    import app.ws.interview as interview_module
    from app.main import app
    from app.services.kafka_producer import report_kafka_producer
    from app.services.llm_service import llm_reply_service
    from app.services.ser_service import ser_service

    # 리포트 빌더(LLM 호출)와 Kafka 발행은 기본 스텁 — 발행 검증 테스트에서 재패치
    async def noop_build(state):
        return {"schemaVersion": 1, "sessionId": state.session_id, "email": state.email}

    async def noop_publish(report):
        return True

    monkeypatch.setattr(interview_module, "build_analysis_report", noop_build)
    monkeypatch.setattr(report_kafka_producer, "publish_report", noop_publish)

    # SER/LLM을 가짜로 대체해 모델 없이 파이프라인 흐름만 검증
    monkeypatch.setattr(ser_service, "feature_extractor", object())
    monkeypatch.setattr(ser_service, "model", object())
    monkeypatch.setattr(
        ser_service,
        "predict_from_audio",
        lambda audio: {
            "label": "fear",
            "confidence": 0.8,
            "probs": {"fear": 0.8, "neutral": 0.2},
        },
    )
    monkeypatch.setattr(
        llm_reply_service,
        "generate_reply",
        lambda text, emotion, context, history: "테스트 응답",
    )
    return TestClient(app)


def _pcm_bytes(amplitude: int, seconds: float) -> bytes:
    return np.full(int(SR * seconds), amplitude, dtype=np.int16).tobytes()


def test_full_turn_flow_records_final(client):
    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session_ready"
        assert ready["session_id"].startswith("intv-")
        assert ready["persona_id"] == "gyul"

        ws.send_json({"type": "utterance_text", "text": "요즘 면접이 걱정돼요"})
        ws.send_bytes(_pcm_bytes(3000, 0.5))  # 발화
        ws.send_bytes(_pcm_bytes(0, 0.8))  # 침묵 → 발화 종료

        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()

        assert message["type"] == "final"
        assert message["text"] == "요즘 면접이 걱정돼요"
        assert message["reply"] == "테스트 응답"
        assert message["label"] == "fear"
        assert message["session_id"] == ready["session_id"]

        ws.send_text("session_end")
        closed = ws.receive_json()
        assert closed["type"] == "session_closed"
        assert closed["turn_count"] == 1


def test_flush_finalizes_pending_speech(client):
    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ws.receive_json()  # session_ready

        ws.send_json({"type": "utterance_text", "text": "잘 부탁드립니다"})
        ws.send_bytes(_pcm_bytes(3000, 0.5))
        ws.send_text("flush")

        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()
        assert message["type"] == "final"
        assert message["text"] == "잘 부탁드립니다"


def test_session_end_publishes_analysis_report(client, monkeypatch):
    import app.ws.interview as interview_module
    from app.services.kafka_producer import report_kafka_producer

    published = []

    async def fake_build(state):
        return {"schemaVersion": 1, "sessionId": state.session_id, "email": state.email}

    async def fake_publish(report):
        published.append(report)
        return True

    monkeypatch.setattr(interview_module, "build_analysis_report", fake_build)
    monkeypatch.setattr(report_kafka_producer, "publish_report", fake_publish)

    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ready = ws.receive_json()

        ws.send_json({"type": "utterance_text", "text": "안녕하세요"})
        ws.send_bytes(_pcm_bytes(3000, 0.5))
        ws.send_text("flush")
        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()
        assert message["type"] == "final"

        ws.send_text("session_end")
        closed = ws.receive_json()
        assert closed["type"] == "session_closed"

    # session_end 경로 1회만 발행 (finally 중복 발행 없음)
    assert len(published) == 1
    assert published[0]["sessionId"] == ready["session_id"]
    assert published[0]["email"] == "ai-test@example.com"


def test_disconnect_without_session_end_still_publishes(client, monkeypatch):
    import app.ws.interview as interview_module
    from app.services.kafka_producer import report_kafka_producer

    published = []

    async def fake_build(state):
        return {"schemaVersion": 1, "sessionId": state.session_id, "email": state.email}

    async def fake_publish(report):
        published.append(report)
        return True

    monkeypatch.setattr(interview_module, "build_analysis_report", fake_build)
    monkeypatch.setattr(report_kafka_producer, "publish_report", fake_publish)

    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ws.receive_json()  # session_ready
        ws.send_json({"type": "utterance_text", "text": "안녕하세요"})
        ws.send_bytes(_pcm_bytes(3000, 0.5))
        ws.send_text("flush")
        message = ws.receive_json()
        while message["type"] == "partial":
            message = ws.receive_json()
        assert message["type"] == "final"
        # session_end 없이 그냥 연결 종료

    assert len(published) == 1


def test_empty_session_publishes_nothing(client, monkeypatch):
    from app.services.kafka_producer import report_kafka_producer

    published = []

    async def fake_publish(report):
        published.append(report)
        return True

    monkeypatch.setattr(report_kafka_producer, "publish_report", fake_publish)

    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ws.receive_json()  # session_ready
        ws.send_text("session_end")
        ws.receive_json()  # session_closed

    assert published == []


def test_session_start_changes_persona_only(client):
    token = make_access_token()
    with client.websocket_connect(f"/ws/interview?token={token}") as ws:
        ready = ws.receive_json()
        server_session_id = ready["session_id"]

        # 클라이언트가 보낸 user_id/session_id는 무시, persona만 반영
        ws.send_json(
            {
                "type": "session_start",
                "user_id": "attacker@example.com",
                "session_id": "client-fake",
                "persona_id": "interviewer",
            }
        )
        updated = ws.receive_json()
        assert updated["type"] == "session_ready"
        assert updated["persona_id"] == "interviewer"
        assert updated["session_id"] == server_session_id
        assert updated["email"] == "ai-test@example.com"
