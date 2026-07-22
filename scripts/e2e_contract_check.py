"""Spring 상대 E2E 계약 검증 (연동 가이드 §7 체크리스트).

사전 조건은 docs/e2e-checklist.md 참고:
- Spring 레포에서 docker compose up -d (MySQL/Redis/Kafka) + ./gradlew bootRun (:8080)
- FastAPI 서버 기동 (:8000) — 양쪽 .env에 동일한 JWT_SECRET
- 이 스크립트는 FastAPI 레포 루트에서 실행: python3 scripts/e2e_contract_check.py

검증 항목 (§7.2):
 1. Spring 실발급 토큰 → verify_access_token 통과 (sub/memberId/role)
 2. 위조 토큰 → TokenError
 3. WS 무토큰 연결 → close 4401
 4. WS 유효 토큰 연결 → accept + session_ready
 5. 리포트 발행 → 3초 내 GET /api/v1/members/me/reports에 sessionId 등장
 6. 같은 sessionId 재발행 → 중복 적재 없음 (멱등)
 7. 계약 외 dominant 발행 → NEUTRAL 강등 저장
 8. Redis 키 격리 — chat:* 키는 DB 0에만 존재
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

SPRING_BASE = os.getenv("E2E_SPRING_BASE", "http://localhost:8080")
FASTAPI_WS_BASE = os.getenv("E2E_FASTAPI_WS_BASE", "ws://localhost:8000")
E2E_EMAIL = os.getenv("E2E_EMAIL", "ai-e2e-test@example.com")
E2E_PASSWORD = os.getenv("E2E_PASSWORD", "Passw0rd!")
REPORT_POLL_SECONDS = 3.0


@dataclass
class CheckResult:
    number: int
    name: str
    passed: bool
    detail: str = ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_report(session_id: str, dominant: str = "TENSION") -> dict:
    return {
        "schemaVersion": 1,
        "sessionId": session_id,
        "email": E2E_EMAIL,
        "phase": "PHASE_1",
        "startedAt": _now_utc(),
        "endedAt": _now_utc(),
        "emotion": {
            "dominant": dominant,
            "scores": {"tension": 0.62, "confusion": 0.21, "neutral": 0.17},
        },
        "summary": "E2E 계약 검증용 테스트 리포트입니다.",
        "metrics": {"turnCount": 1, "speechDurationSec": 5},
    }


def signup_and_signin(client: httpx.Client) -> str:
    """회원가입(이미 있으면 무시) 후 로그인해 accessToken 반환."""
    client.post(
        f"{SPRING_BASE}/api/v1/signup",
        json={
            "email": E2E_EMAIL,
            "password": E2E_PASSWORD,
            "name": "검증",
            "gender": "MALE",
        },
    )  # 중복 가입 응답은 무시 (이미 존재 가능)

    response = client.post(
        f"{SPRING_BASE}/auth/signin",
        json={"email": E2E_EMAIL, "password": E2E_PASSWORD},
    )
    response.raise_for_status()
    return response.json()["accessToken"]


def fetch_report_sessions(client: httpx.Client, token: str) -> list[str]:
    response = client.get(
        f"{SPRING_BASE}/api/v1/members/me/reports",
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    body = response.json()
    reports = body if isinstance(body, list) else body.get("reports") or body.get("content") or []
    return [r.get("sessionId") for r in reports if isinstance(r, dict)]


def fetch_report_by_session(client: httpx.Client, token: str, session_id: str) -> dict | None:
    response = client.get(
        f"{SPRING_BASE}/api/v1/members/me/reports",
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    body = response.json()
    reports = body if isinstance(body, list) else body.get("reports") or body.get("content") or []
    for report in reports:
        if isinstance(report, dict) and report.get("sessionId") == session_id:
            return report
    return None


async def publish_and_wait(client: httpx.Client, token: str, report: dict) -> bool:
    """FastAPI 프로듀서로 리포트를 발행하고 Spring 조회 API에 나타날 때까지 폴링."""
    from app.services.kafka_producer import ReportKafkaProducer

    producer = ReportKafkaProducer()
    try:
        ok = await producer.publish_report(report)
    finally:
        await producer.stop()
    if not ok:
        return False

    deadline = time.time() + REPORT_POLL_SECONDS
    while time.time() < deadline:
        if report["sessionId"] in fetch_report_sessions(client, token):
            return True
        time.sleep(0.3)
    return report["sessionId"] in fetch_report_sessions(client, token)


async def check_websocket(token: str | None, expect_close: int | None) -> tuple[bool, str]:
    """WS 연결 검증. expect_close=None이면 session_ready 수신을 기대."""
    try:
        import websockets
    except ImportError:
        return False, "websockets 패키지 필요 (pip install websockets)"

    url = f"{FASTAPI_WS_BASE}/ws/interview"
    if token:
        url += f"?token={token}"

    try:
        async with websockets.connect(url) as ws:
            if expect_close is not None:
                try:
                    await asyncio.wait_for(ws.recv(), timeout=5)
                    return False, "메시지를 받았는데 close를 기대함"
                except websockets.exceptions.ConnectionClosed as e:
                    return e.code == expect_close, f"close code={e.code}"
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if message.get("type") == "session_ready" and message.get("email") == E2E_EMAIL:
                await ws.send("session_end")
                return True, f"session_ready (session_id={message.get('session_id')})"
            return False, f"예상 외 첫 메시지: {message}"
    except websockets.exceptions.ConnectionClosed as e:
        if expect_close is not None:
            return e.code == expect_close, f"close code={e.code}"
        return False, f"연결이 닫힘 (code={e.code})"
    except Exception as e:
        return False, f"연결 실패: {e}"


def check_redis_isolation() -> tuple[bool, str]:
    import redis

    db0 = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    db1 = redis.Redis.from_url("redis://localhost:6379/1", decode_responses=True)
    chat_db0 = db0.keys("chat:*")
    chat_db1 = db1.keys("chat:*")
    if chat_db1:
        return False, f"DB 1에 FastAPI 키 존재(금지): {chat_db1[:3]}"
    return True, f"DB 0 chat:* {len(chat_db0)}개, DB 1 오염 없음"


async def run_checks(skip_ws: bool) -> list[CheckResult]:
    from app.core.security import TokenError, verify_access_token

    results: list[CheckResult] = []
    client = httpx.Client(timeout=10)

    # 준비 — 실제 토큰
    token = signup_and_signin(client)

    # 1. 실토큰 검증
    try:
        payload = verify_access_token(token)
        ok = payload.get("sub") == E2E_EMAIL and payload.get("memberId") is not None
        detail = f"sub={payload.get('sub')}, memberId={payload.get('memberId')}, role={payload.get('role')}"
    except TokenError as e:
        ok, detail = False, f"TokenError({e.reason}) — JWT_SECRET 불일치 여부 확인"
    results.append(CheckResult(1, "Spring 실발급 토큰 검증", ok, detail))

    # 2. 위조 토큰 거부
    forged = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    try:
        verify_access_token(forged)
        results.append(CheckResult(2, "위조 토큰 거부", False, "통과되면 안 됨"))
    except TokenError as e:
        results.append(CheckResult(2, "위조 토큰 거부", e.reason == "invalid", f"TokenError({e.reason})"))

    # 3~4. WebSocket
    if skip_ws:
        results.append(CheckResult(3, "WS 무토큰 4401", False, "--skip-ws로 생략"))
        results.append(CheckResult(4, "WS 유효 토큰 accept", False, "--skip-ws로 생략"))
    else:
        ok, detail = await check_websocket(None, expect_close=4401)
        results.append(CheckResult(3, "WS 무토큰 4401", ok, detail))
        ok, detail = await check_websocket(token, expect_close=None)
        results.append(CheckResult(4, "WS 유효 토큰 accept", ok, detail))

    # 5. 리포트 발행 → 조회
    session_id = f"e2e-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}"
    report = _make_report(session_id)
    ok = await publish_and_wait(client, token, report)
    results.append(CheckResult(5, "리포트 발행→조회", ok, f"sessionId={session_id}"))

    # 6. 멱등 재발행
    await publish_and_wait(client, token, report)
    count = fetch_report_sessions(client, token).count(session_id)
    results.append(CheckResult(6, "동일 sessionId 멱등", count == 1, f"조회 결과 {count}건"))

    # 7. 미지 enum → NEUTRAL 강등
    unknown_session = f"e2e-unknown-{uuid4().hex[:6]}"
    unknown_report = _make_report(unknown_session, dominant="EXCITED")
    ok = await publish_and_wait(client, token, unknown_report)
    stored = fetch_report_by_session(client, token, unknown_session) or {}
    dominant = stored.get("dominantEmotion")
    results.append(
        CheckResult(7, "미지 enum NEUTRAL 강등", ok and dominant == "NEUTRAL", f"저장값={dominant}")
    )

    # 8. Redis DB 격리
    try:
        ok, detail = check_redis_isolation()
    except Exception as e:
        ok, detail = False, f"Redis 접속 실패: {e}"
    results.append(CheckResult(8, "Redis DB 0/1 격리", ok, detail))

    client.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="연동 가이드 §7 E2E 계약 검증")
    parser.add_argument(
        "--skip-ws",
        action="store_true",
        help="FastAPI 서버 미기동 시 WS 항목(3~4) 생략",
    )
    args = parser.parse_args()

    if not os.getenv("JWT_SECRET"):
        print("⚠️  JWT_SECRET 환경변수가 없습니다. Spring과 동일 값을 주입하세요.")
        return 1

    results = asyncio.run(run_checks(skip_ws=args.skip_ws))

    print("\n# | 결과 | 항목 | 상세")
    print("--|------|------|-----")
    for r in results:
        mark = "PASS ✅" if r.passed else "FAIL ❌"
        print(f"{r.number} | {mark} | {r.name} | {r.detail}")

    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
