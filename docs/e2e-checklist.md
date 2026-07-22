# E2E 계약 검증 절차 (연동 가이드 §7)

Spring 서버를 상대로 FastAPI가 연동 계약을 지키는지 검증하는 절차.
**8항목 전부 PASS해야 "호환"이며, 이것이 dev → main 승격 조건이다.**

## 1. 사전 조건

로컬 인프라는 **Spring 레포의 docker-compose 하나로 통일**한다 (이 레포에 compose를 만들지 않는다).

```bash
# ① Spring 레포에서 인프라 + 서버 기동
cd ../Gyul-BackEnd-Repository
cp .env.example .env          # 실제 값 입력
docker compose up -d          # MySQL / Redis(6379) / Kafka(9092)
./gradlew bootRun             # Spring :8080

# ② FastAPI 레포에서 서버 기동
cd ../Gyul-AI-Repository
cp .env.example .env          # JWT_SECRET을 Spring .env와 동일하게!
ln -s <모델 경로>/model.safetensors model/model.safetensors
uvicorn ser_api:app --port 8000
```

> 🔑 **JWT_SECRET이 양쪽 `.env`에서 동일**해야 한다. 불일치 시 항목 1·4가 실패한다.

## 2. 실행

```bash
# FastAPI 레포 루트에서 (.env의 JWT_SECRET이 셸에 로드된 상태)
export $(grep -v '^#' .env | xargs)   # 또는 direnv 등
python3 scripts/e2e_contract_check.py

# FastAPI 서버를 띄우지 않고 토큰/Kafka/Redis 항목만 검증할 때
python3 scripts/e2e_contract_check.py --skip-ws
```

환경변수로 대상 주소를 바꿀 수 있다: `E2E_SPRING_BASE`(기본 `http://localhost:8080`),
`E2E_FASTAPI_WS_BASE`(기본 `ws://localhost:8000`), `E2E_EMAIL`, `E2E_PASSWORD`.

## 3. 검증 항목 (§7.2 대응)

| # | 검증 | 기대 결과 |
|---|---|---|
| 1 | Spring 실발급 accessToken → `verify_access_token()` | `sub`/`memberId`/`role` 추출 성공 |
| 2 | 위조 토큰(마지막 글자 변조) | `TokenError("invalid")` |
| 3 | WS 연결: `?token=` 없이 | close `4401` |
| 4 | WS 연결: 유효 토큰 | accept + `session_ready` |
| 5 | 리포트 발행 → 3초 내 조회 | `GET /api/v1/members/me/reports`에 해당 `sessionId` 등장 |
| 6 | 같은 `sessionId` 재발행 | 조회 결과 중복 없음 (멱등) |
| 7 | `emotion.dominant`에 계약 외 값 발행 | `NEUTRAL`로 강등 저장 |
| 8 | Redis 키 격리 | `chat:*` 키는 DB 0에만, DB 1 오염 없음 |

## 4. 수동 보조 확인

```bash
# Redis 키 직접 확인
redis-cli -n 0 keys 'chat:*'     # FastAPI 세션 키
redis-cli -n 1 keys '*'          # Spring 전용 (RT:*, login-fail:* 만 있어야 함)

# 리포트 조회
curl -s http://localhost:8080/api/v1/members/me/reports \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 실제 대화 E2E: 프론트 데모(./start.sh) 대신 유효 토큰으로 /ws/interview 접속
# → 발화 → session_end → 위 리포트 조회에 세션 등장 확인
```

## 5. 전 항목 PASS 후

`dev` → `main` 승격 PR을 생성하고, 본문에 포함된 모든 이슈를 `Closes #N`으로 나열한다
(CLAUDE.md 브랜치 전략 참고).
