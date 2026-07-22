# CLAUDE.md

AI 기반 자기분석 가상 면접 솔루션 **'결(結)'** 의 AI 서버(FastAPI + AI 모델 서빙) 레포지토리.
전체 기획/로드맵은 백엔드 레포 `src/main/resources/plan.md`, Spring 연동 계약은 이 레포의
`docs/fastapi-integration-guide.md` 참고 — **연동 관련 작업 전 반드시 읽을 것.**

## 프로젝트 개요

- MSA 기반 BFF 이중 서버: Spring Boot(회원/인증/리포트 적재, [Gyul-BackEnd-Repository](https://github.com/MaTuna01/Gyul-BackEnd-Repository)) + **FastAPI(이 레포, 실시간 대화/감정 분석/리포트 발행)**
- 기술 스택: FastAPI, PyTorch/transformers(SER·얼굴 감정), LangChain(OpenAI/Kanana 라우팅), Fish Audio TTS, Redis, Kafka(aiokafka)
- 파이프라인: 브라우저 STT + PCM 오디오 → WebSocket → VAD → SER → LLM 응답 → (선택) TTS → 대화 종료 시 Kafka 분석 리포트 발행

## 실행 / 테스트

```bash
pip install -r requirements.txt
cp .env.example .env          # 최초 1회, 실제 값 입력 (JWT_SECRET은 Spring과 동일 값)
ln -s <실제 모델 경로>/model.safetensors model/model.safetensors  # SER 모델 (레포에 미포함)

uvicorn ser_api:app --reload  # FastAPI :8000 (8080=Spring과 충돌 방지)
./start.sh                    # API(8000) + React 데모(5173) 일괄 기동
pytest tests/                 # 단위 테스트 (모델/인프라 없이 동작)
```

- 로컬 인프라(MySQL/Redis/Kafka)는 **Spring 레포의 `docker-compose.yml` 하나로 통일**한다.
  이 레포에 별도 compose를 만들지 않는다 (포트 충돌 + 상태 분열 방지).
- 환경변수(시크릿)는 `.env`로만 관리하며 **절대 커밋하지 않는다.** 새 환경변수 추가 시 `.env.example`에 템플릿을 함께 추가할 것.

## Spring 연동 계약 (요약 — 원문: docs/fastapi-integration-guide.md)

| 항목 | 값 |
|---|---|
| JWT | HS256, `JWT_SECRET`(Spring과 동일 값), claims `sub`(email)/`memberId`/`role` |
| WebSocket | `/ws/interview?token=<ACCESS_TOKEN>` — 실패 close `4401`/`4403`, 정상 `1000` |
| Kafka | topic `interview.analysis-report`, key=email, 스키마 v1, `sessionId`=멱등키 |
| 감정 enum | `TENSION`/`CONFUSION`/`CALM`/`CONFIDENCE`/`NEUTRAL` (그 외 값은 Spring이 NEUTRAL 강등) |
| 시간 | ISO-8601 UTC + `Z` 접미 (예: `2026-07-22T10:00:00Z`) |
| Redis | **FastAPI는 DB 0만 사용** — DB 1(Spring: Refresh Token)은 읽기/쓰기 금지 |
| 금지 | Spring MySQL 직접 접근, Redis DB 1 접근, 합의 없는 enum/스키마 변경 |

- 계약의 단일 소스는 Spring 레포 `docs/integration-spec.md`. **enum/스키마에 영향 주는 변경은
  그 문서의 PR로 양 팀 합의를 먼저 거친 뒤** 이 레포의 가이드 사본과 코드를 동기화한다.

## 패키지 구조

```
app/
  main.py          # FastAPI 앱 + HTTP/WS 라우트 (엔트리는 루트 ser_api.py)
  core/            # config(환경변수), security(JWT 검증), emotion_mapping
  ws/              # WebSocket 핸들러 (predict=데모, interview=계약 구현)
  services/        # 모델 서비스 싱글톤 (ser, llm, fish_tts, face, kafka_producer 등)
  chains/          # persona 프롬프트, LangChain 어댑터
  llm/             # LLM 라우터 (OpenAI ↔ Kanana 로컬 폴백)
  memory/          # 대화 세션 메모리 (InMemory/Redis)
  routes/          # HTTP 라우터 (session_report)
  vision/          # 얼굴 검출/표정 분석
scripts/           # 테스트/검증 스크립트, session_report 생성 패키지
tests/             # pytest 단위 테스트
```

- 새 기능은 위 계층에 맞춰 추가한다. 모델 서비스는 `app/services/`의 싱글톤 패턴을 따른다.

## 브랜치 전략 (중요)

```
main ← dev ← feat/[#이슈번호] | bug/[#이슈번호]
```

| 브랜치 | 역할 | 규칙 |
|---|---|---|
| `main` | 운영(프로덕션) | **단위 기능이 완성되어 실제 운영 가능할 때에만** `dev`에서 merge |
| `dev` | 개발 통합 | 기능/버그 브랜치의 merge 대상. 테스트 통과 후에만 merge |
| `feat/[#이슈번호]` | 기능 개발 | GitHub 이슈 번호 기준으로 `dev`에서 생성 (예: `feat/#1`) |
| `bug/[#이슈번호]` | 버그 수정 | 버그용 이슈를 발급받아 `dev`에서 생성 (예: `bug/#2`) |

> ⚠️ 기능 브랜치에 `dev/` 접두사를 쓰지 않는다. `dev` 브랜치가 이미 존재하면 Git ref 구조상
> `dev/feat/...` 브랜치를 만들 수 없다(`cannot lock ref ... 'dev' exists`).

### 작업 흐름

1. 기능(또는 버그) 단위로 GitHub **이슈를 먼저 발급**한다 (레포: `novvvv/Gyul-AI-Repository`).
2. `dev` 브랜치에서 `feat/[#이슈번호]`(기능) 또는 `bug/[#이슈번호]`(버그 수정) 브랜치를 생성한다.
3. 기능 브랜치에 커밋/푸시하며 개발한다.
4. **테스트 완료 후** `dev` 브랜치에 merge한다 (PR 권장).
5. 단위 기능이 완성되어 실제 운영이 가능한 상태일 때에만 `dev` → `main` merge를 수행한다.
6. `main`에 직접 커밋/푸시하지 않는다.
7. 이슈 자동 종료(`Closes #N`)는 base가 `main`(기본 브랜치)인 PR에서만 동작한다.
   - `feat/[#이슈]` → `dev` PR 본문에는 `Closes`를 쓰지 않는다(어차피 닫히지 않음).
   - **`dev` → `main` 승격 PR을 생성할 때, 그 승격에 포함되는 모든 관련 이슈를 PR 본문에
     `Closes #N`으로 나열한다.** `main` 머지 시 일괄 자동 종료되므로 수동 close가 불필요하다.
8. **머지된 브랜치는 삭제하지 않는다** (로컬·원격 모두 보존).

### 브랜치 구조 특이사항 (이 레포 한정)

- `MaTuna/tts`(Fish Speech 1.5 파인튜닝 TTS), `MaTuna/01~03`(구 SER/대화 계열) 브랜치는
  main과 **git 히스토리가 분리**되어 있다(merge-base 없음). 이들 브랜치의 코드를 가져올 때는
  cherry-pick이 아닌 **파일 단위 포팅**으로 하고, 커밋 메시지에 출처 브랜치를 명기한다.
- 배포 단위 브랜치는 `main`이며, 계약 E2E 체크리스트(가이드 §7)는 main 계열에서 통과시킨다.

## 커밋 컨벤션

```
[#이슈번호] 한글로 작업 내용 요약
```

예: `[#1] 워크플로우 템플릿 및 CLAUDE.md 추가`

- 하나의 커밋은 하나의 논리적 변경만 담는다.
- 커밋 메시지에 `Co-Authored-By` 트레일러를 **넣지 않는다.**
