# Spring 백엔드 연동 가이드 — AI 서버(FastAPI) 레포용

> '결(結)' 프로젝트의 **AI Streaming Server(FastAPI + AI 모델 서빙)** 가
> **BFF/Main Server(Spring Boot, [Gyul-BackEnd-Repository](https://github.com/MaTuna01/Gyul-BackEnd-Repository))** 와
> 완벽 호환되도록 구현하기 위한 **실무 가이드**다.
>
> - 이 문서는 FastAPI 레포에 복사해 두고 읽는 **자립형(self-contained) 문서**다.
> - 통신 규약의 **단일 소스 오브 트루스는 Spring 레포의 `docs/integration-spec.md`** 이며,
>   규약 변경은 그 문서의 PR로 양 팀 합의 후 이 가이드를 동기화한다.
> - 기준 시점: Spring 백엔드 Phase 1 완료(main 승격) 버전.

---

## 0. 전체 그림 — 두 서버의 접점은 3가지뿐

| # | 접점 | 방향 | FastAPI가 할 일 |
|---|---|---|---|
| 1 | **JWT 교차 검증** | Spring 발급 → FastAPI 검증 | 공유 시크릿으로 서명·만료 검증 (인증 위임) |
| 2 | **WebSocket 인증** | 클라이언트 → FastAPI | 핸드셰이크에서 토큰 검증 후 세션 바인딩 |
| 3 | **Kafka 리포트 발행** | FastAPI → Spring | 대화 종료 시 분석 리포트를 정해진 스키마로 발행 |

```
                 ┌────────── (1) POST /auth/signin ──────────┐
   [Client] ─────┤                                            ▼
      │          │                                     [Spring Boot]
      │          └──── JWT(Access/Refresh) ◀───────────────┘
      │
      │  (2) WS 연결 + Access Token
      ▼
 [FastAPI] ── JWT 서명 검증(공유 시크릿) ── 대화/STT/감정분석
      │
      │  (3) 대화 종료 → 분석 리포트
      ▼
   [Kafka] ── topic: interview.analysis-report ──▶ [Spring Boot] → DB 적재
```

**FastAPI는 Spring의 DB(MySQL)에 절대 직접 접근하지 않는다.** 회원/리포트 데이터는
모두 위 3가지 접점으로만 오간다.

---

## 1. JWT 교차 검증 (필수 구현)

### 1.1 계약

| 항목 | 값 |
|---|---|
| 알고리즘 | `HS256` (대칭키) |
| 시크릿 | 환경변수 **`JWT_SECRET`** — **양 서버에 동일한 값** 주입, 32바이트 이상 |
| 검증 항목 | 서명 유효성 + `exp` 만료 (둘 다 실패 시 인증 거부) |
| Access 만료 | 30분 |
| Refresh 만료 | 7일 — **FastAPI는 Refresh Token을 다루지 않는다** (재발급은 Spring 몫) |

### 1.2 Access Token Claims

| Claim | 타입 | 설명 |
|---|---|---|
| `sub` | string | **사용자 이메일** (기본 식별자) |
| `memberId` | number | **회원 PK** — 이메일이 바뀌어도 불변인 안정 식별자 |
| `role` | string | `MEMBER` / `ADMIN` |
| `iat` / `exp` | number | 발급/만료 시각 (epoch seconds) |

- 세션·캐시 키 등 **내부 저장 키는 `sub`(이메일) 기준**으로 잡는다 (Kafka 발행 시 email이 필요하므로).
- 영속 데이터를 남긴다면 `memberId`를 함께 저장해 두는 것을 권장.
- **Refresh Token에는 `memberId`/`role`이 없다** — FastAPI가 받을 일도 없지만, 받았다면 거부할 것.

### 1.3 구현 예시 (PyJWT)

```python
import os
import jwt  # pip install PyJWT

JWT_SECRET = os.environ["JWT_SECRET"]  # Spring과 동일 값

class TokenError(Exception): ...

def verify_access_token(token: str) -> dict:
    """서명+만료 검증 후 payload 반환. 실패 시 TokenError."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise TokenError("expired")
    except jwt.InvalidTokenError:
        raise TokenError("invalid")
    if "sub" not in payload:
        raise TokenError("invalid")
    return payload  # {"sub": email, "memberId": 42, "role": "MEMBER", ...}
```

주의:
- `algorithms=["HS256"]`를 **반드시 명시**한다 (alg 혼동 공격 방지).
- 시크릿은 `.env`/시크릿 매니저로만 주입하고 **레포에 커밋 금지**.
- 시계 오차로 인한 만료 오판이 잦으면 `leeway=10` 정도까지만 허용.

---

## 2. WebSocket 인증 (필수 구현)

### 2.1 연결 규약

브라우저 WebSocket은 커스텀 헤더가 어려우므로 **쿼리 파라미터**로 Access Token을 받는다.

```
wss://<ai-server-host>/ws/interview?token=<ACCESS_TOKEN>
```

### 2.2 처리 순서

1. 핸드셰이크에서 `token` 추출 → §1 방식으로 검증
2. 실패 시 **연결 거부** — 합의된 close code 사용
3. 성공 시 `sub`(이메일)를 세션에 바인딩하고 대화 시작

| close code | 의미 |
|---|---|
| `4401` | 토큰 없음 / 유효하지 않음 / 만료 |
| `4403` | 권한 없음 |
| `1000` | 정상 종료 |

### 2.3 구현 예시 (FastAPI)

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket, token: str | None = None):
    if not token:
        await websocket.close(code=4401)
        return
    try:
        payload = verify_access_token(token)
    except TokenError:
        await websocket.close(code=4401)
        return

    email = payload["sub"]          # 세션 식별자
    member_id = payload.get("memberId")
    await websocket.accept()
    # ... 대화 세션 시작 (email 기준으로 RedisChatMessageHistory 바인딩)
```

### 2.4 토큰 만료 관련 클라이언트 흐름 (참고)

- Access Token이 만료되면 클라이언트가 Spring `POST /auth/reissue`로 재발급받아 **재연결**한다.
- **세션 도중 만료 처리 정책(강제 종료 vs 유예)은 미확정(계약 §6 TODO)** — 구현 전 Spring 팀과 합의하고
  `integration-spec.md`에 반영할 것. 합의 전 기본 동작은 "연결 시점에만 검증, 세션 중 재검증 안 함"을 권장(단순).

---

## 3. Kafka 분석 리포트 발행 (필수 구현)

대화 종료 시 FastAPI가 최종 분석 리포트를 발행하면 Spring이 소비해 DB에 적재한다.

### 3.1 발행 계약

| 항목 | 값 |
|---|---|
| 토픽 | `interview.analysis-report` |
| Key | **`email`** (UTF-8 문자열) — 파티셔닝/순서 보장 기준 |
| Value | JSON (UTF-8) — 아래 스키마 v1 |
| Producer | FastAPI (이 레포) |
| Consumer | Spring (`group-id: gyul-backend`) |

### 3.2 메시지 스키마 (v1)

```json
{
  "schemaVersion": 1,
  "sessionId": "conv-20260722-abc123",
  "email": "user@example.com",
  "phase": "PHASE_1",
  "startedAt": "2026-07-22T10:00:00Z",
  "endedAt":   "2026-07-22T10:12:34Z",
  "emotion": {
    "dominant": "TENSION",
    "scores": { "tension": 0.62, "confusion": 0.21, "calm": 0.17 }
  },
  "summary": "사용자는 전반적으로 긴장 상태를 보였으며...",
  "metrics": { "speechDurationSec": 540, "turnCount": 18 }
}
```

| 필드 | 타입 | 필수 | 규칙 |
|---|---|:---:|---|
| `schemaVersion` | int | ✅ | 현재 `1` 고정 |
| `sessionId` | string | ✅ | **전역 유일** — Spring의 멱등 키. 재발행 시 같은 값이면 중복 적재 안 됨 |
| `email` | string | ✅ | **JWT `sub`와 동일한 값** 그대로 사용 (가공 금지) |
| `phase` | string | ✅ | `PHASE_1`(자기분석) / `PHASE_2`(기술면접) |
| `startedAt`/`endedAt` | string | ✅ | **ISO-8601 UTC, `Z` 접미** (예: `2026-07-22T10:00:00Z`) |
| `emotion.dominant` | string | ✅ | §3.3 enum 값 중 하나 |
| `emotion.scores` | object | ✅ | 감정명(소문자) → 확률(0~1) 맵 |
| `summary` | string | ✅ | LLM 생성 요약 (TEXT 저장 — 64KB 이내 권장) |
| `metrics` | object | ⬜ | 자유 확장 (Spring은 현재 무시하지만 하위호환 유지) |

### 3.3 감정 enum (합의값)

`TENSION`(긴장) · `CONFUSION`(당황) · `CALM`(평온) · `CONFIDENCE`(자신감) · `NEUTRAL`(중립)

- **신규 감정 추가 시 반드시 `integration-spec.md` PR로 합의** 후 양쪽 코드 동기화.
- 합의 없이 미지의 값을 보내면 Spring이 `NEUTRAL`로 **강등 처리**한다 (에러는 아니지만 데이터 왜곡).

### 3.4 Spring 컨슈머의 동작 (발행 측이 알아야 할 것)

| 상황 | Spring 동작 | FastAPI에 주는 의미 |
|---|---|---|
| 정상 수신 | DB 적재 | — |
| **같은 `sessionId` 재수신** | 무시 (멱등) | **재시도/재발행이 안전하다** — at-least-once로 편하게 발행 |
| 존재하지 않는 `email` | 로그 후 스킵 | 적재 안 됨. email은 반드시 JWT `sub` 그대로 쓸 것 |
| **JSON 파싱 불가(poison)** | 재시도 없이 **DLQ**(`interview.analysis-report.dlq`)로 이동 | 스키마를 지키지 않으면 데이터가 DLQ로 빠진다 |
| 일시 오류(DB 순단 등) | 1초 간격 2회 재시도 후 DLQ | — |
| 미지의 JSON 필드 | 무시 | **필드 추가는 하위호환** — 자유롭게 확장 가능 |

> 발행 실패(브로커 다운 등)에 대비해 FastAPI 쪽에서도 **최소 1회 재시도 + 실패 로그**를 남길 것.
> `sessionId` 멱등 덕분에 중복 발행 걱정 없이 재시도하면 된다.

### 3.5 구현 예시 (aiokafka)

```python
import json, os
from aiokafka import AIOKafkaProducer  # pip install aiokafka

TOPIC = "interview.analysis-report"

producer = AIOKafkaProducer(
    bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    acks="all",                      # 유실 방지
    enable_idempotence=True,         # 브로커 레벨 중복 방지 (선택이지만 권장)
)

async def publish_report(report: dict) -> None:
    """대화 종료 시 호출. report는 §3.2 스키마 v1 dict."""
    await producer.send_and_wait(
        TOPIC,
        key=report["email"].encode("utf-8"),
        value=json.dumps(report, ensure_ascii=False).encode("utf-8"),
    )
```

체크: `startedAt`/`endedAt`은 `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")` 형태로
**UTC + `Z`** 를 지킬 것 (KST 오프셋 `+09:00` 형태도 Spring이 파싱하지 못하면 null 저장됨 — `Z`로 통일).

---

## 4. Redis 사용 영역 분리 (필수 준수)

두 서버가 **같은 Redis 인스턴스**를 쓰므로 DB 인덱스로 격리한다.

| DB | 사용 주체 | 용도 | 키 패턴 |
|---|---|---|---|
| **`0`** | **FastAPI (이 레포)** | 대화 세션 (LangChain `RedisChatMessageHistory` 등) | FastAPI 자율 (예: `chat:{email}:{sessionId}`) |
| `1` | Spring | Refresh Token + 로그인 실패 카운터 | `RT:{email}`, `login-fail:{email}` |

- FastAPI는 **반드시 DB 0만** 사용한다 (`redis://host:6379/0`).
- **DB 1에 읽기/쓰기 금지** — Spring의 토큰 저장소를 오염시키면 전체 인증이 깨진다.
- 대화 세션 키에는 TTL을 걸어 세션 잔존물이 쌓이지 않게 할 것 (권장: 대화 종료 후 24h 이내).

---

## 5. 환경변수 통일표

FastAPI 레포의 `.env.example`에 아래를 포함시키고, **공유 값은 Spring과 동일하게** 주입한다.

| 환경변수 | 공유 여부 | 값 규칙 |
|---|---|---|
| `JWT_SECRET` | ✅ **Spring과 반드시 동일** | 32바이트 이상. 불일치 시 모든 토큰 검증 실패 |
| `KAFKA_BOOTSTRAP_SERVERS` | ✅ 동일 브로커 | 로컬 `localhost:9092` |
| `REDIS_HOST` / `REDIS_PORT` | ✅ 동일 인스턴스 | 로컬 `localhost` / `6379` |
| `REDIS_DB` | ❌ FastAPI 고유 | **`0` 고정** (§4) |
| 모델/추론 관련 키 (OpenAI 등) | ❌ FastAPI 고유 | FastAPI 레포에서 관리 |

배포 시(docker-compose/K8s) 두 컨테이너에 같은 `JWT_SECRET`을 넣는 것이 **신뢰의 기반**이다.

---

## 6. 로컬 개발 인프라 공유

인프라는 **Spring 레포의 `docker-compose.yml` 하나로 통일**한다 (plan.md Ground Rule #2).
FastAPI 레포에 별도 MySQL/Redis/Kafka compose를 만들지 말 것 (포트 충돌 + 상태 분열).

```bash
# Spring 레포에서 1회
git clone https://github.com/MaTuna01/Gyul-BackEnd-Repository
cd Gyul-BackEnd-Repository
cp .env.example .env     # 실제 값 입력 (JWT_SECRET을 FastAPI .env에도 복사)
docker compose up -d     # MySQL / Redis(6379) / Kafka(9092) 기동
./gradlew bootRun        # Spring 서버 :8080
```

| 서비스 | 로컬 접속점 | 비고 |
|---|---|---|
| Spring API | `http://localhost:8080` | |
| Kafka | `localhost:9092` | KRaft 단일 노드, 토픽 auto-create 켜짐 |
| Redis | `localhost:6379` | FastAPI는 DB 0 |
| MySQL | `localhost:${DB_PORT}` (기본 3306) | **FastAPI 접근 금지** — 참고용 |

FastAPI 서버 포트는 **8000**을 권장 (8080=Spring과 충돌 방지).

---

## 7. E2E 자가 검증 체크리스트

FastAPI 구현이 계약을 지키는지 **Spring을 상대로** 검증하는 절차. CI/수동 어느 쪽이든
아래가 전부 통과해야 "호환"이다.

### 7.1 준비 — 실제 토큰 얻기

```bash
# 회원가입 (1회)
curl -s -X POST http://localhost:8080/api/v1/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"ai-test@example.com","password":"Passw0rd!","name":"검증","gender":"MALE"}'

# 로그인 → accessToken 획득
curl -s -X POST http://localhost:8080/auth/signin \
  -H 'Content-Type: application/json' \
  -d '{"email":"ai-test@example.com","password":"Passw0rd!"}'
# → {"accessToken":"...", "refreshToken":"..."}
```

### 7.2 검증 항목

| # | 검증 | 기대 결과 |
|---|---|---|
| 1 | Spring 발급 accessToken을 `verify_access_token()`에 통과 | `sub`/`memberId`/`role` 추출 성공 |
| 2 | 위조 토큰(마지막 글자 변조) 검증 | `TokenError("invalid")` |
| 3 | WS 연결: `?token=` 없이 | close `4401` |
| 4 | WS 연결: 유효 토큰 | accept + 세션 바인딩 |
| 5 | 리포트 발행 → 3초 내 Spring 조회 | `GET /api/v1/members/me/reports` (Bearer 토큰)에 해당 `sessionId` 등장 |
| 6 | **같은 `sessionId`로 재발행** | 조회 결과 중복 없음 (멱등 확인) |
| 7 | `emotion.dominant`에 계약 외 값 발행 | 적재는 되되 `NEUTRAL`로 저장됨 확인 |
| 8 | Redis 키 확인: `redis-cli -n 0 keys '*'` / `-n 1 keys '*'` | FastAPI 키는 DB 0에만 존재 |

리포트 조회 예시:
```bash
curl -s http://localhost:8080/api/v1/members/me/reports \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

---

## 8. FastAPI 레포 브랜치 구조에 대한 지침

이 레포는 **브랜치별로 AI 모델과 FastAPI 서버가 분리**되어 있다. 이 구조에서 계약 호환을
유지하기 위한 규칙:

1. **이 가이드는 default 브랜치(및 모든 장수 브랜치)에 둔다** — 어느 브랜치를 열어도 계약이 보이도록.
2. **연동 계약(§1~§5)을 구현하는 주체는 "FastAPI 서버" 브랜치**다. AI 모델 브랜치는 계약과 무관하며,
   모델 교체가 계약(스키마·감정 enum)에 영향을 주는 경우에만 §3.3 절차(스펙 PR 합의)를 거친다.
3. **배포 단위 브랜치**(실제 서빙되는 조합)를 하나 명시적으로 정하고, §7 체크리스트는 그 브랜치에서 통과시킨다.
4. 감정 enum·스키마에 영향 주는 변경은 **먼저 Spring 레포 `docs/integration-spec.md` PR로 합의**하고,
   머지 후 이 가이드와 FastAPI 코드를 동기화한다 (계약 §5 변경 관리).
5. 장기적으로는 브랜치 분리 대신 **디렉터리 분리(`model/`, `server/`) + 단일 main**을 권장한다 —
   브랜치 조합 검증 비용이 사라지고 CI가 단순해진다. (강제는 아님, 팀 판단)

---

## 9. 구현 순서 제안 (Phase 1 잔여 작업)

Spring 측은 완료 상태이므로, FastAPI는 아래 순서를 권장한다. **①이 끝나는 즉시 양 서버가
연결 가능**해지고, ④까지 끝나면 Phase 1 파이프라인이 완성된다.

| 순서 | 작업 | 계약 접점 | 완료 판정 |
|---|---|---|---|
| ① | JWT 검증 모듈 + WS 엔드포인트 개통 | §1, §2 | 체크리스트 1~4 통과 |
| ② | LangChain + RedisChatMessageHistory 대화 세션 (Redis DB 0) | §4 | 체크리스트 8 통과 |
| ③ | STT(Whisper) → LangChain → TTS 파이프라인 | (내부) | — |
| ④ | 감정 추출 + 대화 종료 시 리포트 발행 | §3 | 체크리스트 5~7 통과 |

미결 협의 항목 (구현 중 만나면 Spring 팀과 스펙 PR로 확정):
- [ ] WebSocket 세션 중 Access Token 만료 처리 정책 (§2.4)
- [ ] `metrics` 필드의 표준 키 목록 (현재 자유 확장)

---

## 부록 A. 빠른 참조 카드

```
JWT     : HS256, secret=JWT_SECRET(공유), claims: sub(email)/memberId/role, Access 30m
WS      : wss://host/ws/interview?token=..., 실패 close 4401/4403, 정상 1000
Kafka   : topic interview.analysis-report, key=email, JSON v1, sessionId=멱등키
          DLQ interview.analysis-report.dlq (파싱실패는 Spring이 자동 이동)
Emotion : TENSION | CONFUSION | CALM | CONFIDENCE | NEUTRAL (그 외 → NEUTRAL 강등)
Time    : ISO-8601 UTC + 'Z' (예: 2026-07-22T10:00:00Z)
Redis   : FastAPI=DB 0, Spring=DB 1 (DB 1 접근 금지)
Ports   : Spring 8080, FastAPI 8000(권장), Kafka 9092, Redis 6379
금지     : MySQL 직접 접근, Redis DB 1 접근, 합의 없는 enum/스키마 변경
```

## 부록 B. Spring 측 관련 소스 (계약 근거)

| 접점 | Spring 파일 |
|---|---|
| JWT 발급/claims | `global/jwt/JwtProvider.java` |
| Kafka 컨슈머/멱등/스킵 | `domain/report/messaging/AnalysisReportListener.java`, `service/AnalysisReportService.java` |
| DLQ/재시도 정책 | `global/config/KafkaConsumerConfig.java` |
| 감정 enum | `domain/report/model/entity/Emotion.java` |
| 리포트 조회 API | `domain/report/controller/ReportController.java` |
| Redis DB 분리 | `application.yml` (`spring.data.redis.database: 1`) |
