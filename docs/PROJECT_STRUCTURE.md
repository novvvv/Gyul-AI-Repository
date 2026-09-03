# 프로젝트 구조 요약 & 프론트엔드 개선 계획

> 작성일: 2026-09-02 · 브랜치 `fe`
> 상세 문서: [파이프라인 상세](MODEL_AND_CONVERSATION_PIPELINE_OVERVIEW.md) ·
> [Spring 연동 계약](fastapi-integration-guide.md) · [FE 개선 원안](frontend-overhaul-plan.md)

---

## 1. 한눈에 보는 시스템

AI 기반 자기분석 가상 면접 솔루션 **'결(結)'** 의 AI 서버 레포. MSA BFF 이중 서버 구조.

```
[브라우저 React]
   │  브라우저 STT(텍스트) + PCM 오디오 + 웹캠 프레임
   ▼
[FastAPI :8000]  ── 이 레포 ──
   VAD → SER(음성감정) → LLM 응답 → (선택) Fish TTS
   얼굴 표정 분류(ViT)
   대화 종료 → 세션 리포트 생성 → Kafka 발행
   │                                    │
   ├── Redis DB 0 (세션 메모리)          └── topic: interview.analysis-report
   ▼
[Spring Boot :8080]  회원/인증(JWT)/리포트 적재 · MySQL · Redis DB 1
```

| 축 | 내용 |
|---|---|
| 백엔드 | FastAPI, PyTorch/transformers, LangChain, aiokafka, Redis |
| 프론트 | React 19 + TypeScript 5.8 + Vite 6 + react-router-dom 7 (상태관리/차트 라이브러리 **없음**) |
| 규모 | 백엔드 약 2.2k LOC(py) / 프론트 약 4.6k LOC(tsx+ts+css) |

---

## 2. 백엔드 (`app/`) 구조

```
app/
  main.py                 # FastAPI 앱 + 라우트 등록 (엔트리는 루트 ser_api.py)
  core/
    config.py             # 모든 환경변수 단일 정의 지점
    security.py           # JWT 검증 (HS256 계열, Spring과 시크릿 공유)
    emotion_mapping.py    # 내부 라벨 → 계약 enum(TENSION/CONFUSION/CALM/CONFIDENCE/NEUTRAL)
  ws/
    interview.py          # 계약 구현 WS 핸들러 (인증 + 세션 루프)
    interview_session.py  # 세션 상태 홀더
    pipeline.py           # 발화 조립(UtteranceAssembler) + 턴 처리(run_final_turn)
    predict.py            # 데모용 WS 핸들러 (인증 없음)
  services/               # 모델/외부연동 싱글톤
    ser_service.py            음성 감정 인식
    face_service.py           표정 분류 (ViT, lazy load)
    face_detect_service.py    얼굴 검출
    llm_service.py            LLM 응답 서비스
    local_llm_service.py      Kanana 로컬 폴백
    fish_tts_service.py       Fish Audio TTS
    kafka_producer.py         리포트 발행
    report_builder.py         리포트 조립
  llm/
    text_generator.py     # 백엔드 라우팅(openai ↔ kanana) + lazy load
    kanana_support.py
  chains/                 # persona 프롬프트, LangChain 어댑터
  memory/session_memory.py  # InMemory / Redis 세션 메모리 (TTL 24h, 최대 20턴)
  routes/session_report.py  # POST /session/report
  vision/predict.py         # 이미지 → 표정/검출 결과
```

### 노출 인터페이스

| 종류 | 경로 | 비고 |
|---|---|---|
| HTTP | `GET /health` | SER/얼굴/LLM 백엔드/TTS 로드 상태 |
| HTTP | `POST /predict` | 오디오 파일 → 음성 감정 |
| HTTP | `POST /predict_face` | base64 이미지 → 표정 |
| HTTP | `POST /detect_face` | base64 이미지 → 얼굴 박스 |
| HTTP | `POST /session/report` | 세션 스냅샷 → 리포트(JSON + MD) |
| WS | `/ws/predict` | 데모용 (`user_id`/`session_id`/`persona_id` 쿼리) |
| WS | `/ws/interview?token=...` | **계약 구현**. 실패 close `4401`/`4403`, 정상 `1000` |

- 리포트 생성 본체는 `app/`이 아니라 **`scripts/session_report/`** 에 있고 라우터가 이를 호출한다
  (`aggregate` / `prompt` / `generate` / `format` / `schema` 분리).
- 테스트 `tests/` 9개 파일 — 모델·인프라 없이 동작하도록 구성됨 (`pytest tests/`).

---

## 3. 프론트엔드 (`frontend/`) 현재 구조

```
frontend/src/
  App.tsx                     # 라우팅 (22줄)
  layouts/AppLayout.tsx
  pages/
    LandingPage.tsx      166   # HIGHLIGHTS 카드 패턴
    DemoPage.tsx         130   # 실시간 대화 화면
    ReportLoadingPage.tsx 128
    ReportPage.tsx       466   # ★ 개선 1순위
  components/                 # 데모 화면 전용 부품 7개
    AiPersonaPanel / CameraPanel / ChatThread / ConnectionStatus
    EmotionFace / EmotionPanel / MicControls
  hooks/
    useDemoSession.ts    354   # WS 연결 + STT + 턴 상태 (실질적 도메인 코어)
    useFaceDetect.ts     154
    useHealth.ts
  services/api.ts             # fetchHealth / detectFaces / predictFaceExpression
                              # / requestSessionReport / buildWsUrl
  types/{ser,sessionReport}.ts
  utils/reportMetrics.ts      # labelKo, countsToSlices, buildRankedQuotes 등
  styles/                     # 페이지 단위 CSS (컴포넌트 단위 아님)
    index.css 193 / layout 172 / landing 434 / demo 644 / report 594 / report-loading 102
```

### 라우팅

| 경로 | 페이지 |
|---|---|
| `/` | LandingPage |
| `/demo` | DemoPage |
| `/demo/report/loading` | ReportLoadingPage |
| `/demo/report` | ReportPage |
| `*` | `/`로 리다이렉트 |

### 리포트 데이터 흐름

```
DemoPage(세션 종료) → requestSessionReport(snapshot) → POST /session/report
   → ReportLoadingPage → location.state.report 또는 sessionStorage("report:*")
   → ReportPage 렌더
```

타입: `types/sessionReport.ts`의 `SessionSnapshot` → `SessionReportResponse`
(`report_json` + `report_md` + `llm_backend`).

---

## 4. 프론트엔드 개선 계획 (요약)

원안: [frontend-overhaul-plan.md](frontend-overhaul-plan.md)

### 진단 — 지금 무엇이 문제인가

1. **`ReportPage.tsx` 466줄 단일 파일**에 데이터 가공 + 차트 구현(`PieChart`, `TurnCompareChart`) + 마크업이 전부 인라인.
2. **디자인 토큰 없음** — `CHART_COLORS`, `VOICE_COLOR`, `FACE_COLOR`가 페이지 파일에 하드코딩.
3. **CSS가 페이지 단위** — 재사용 가능한 컴포넌트 단위 스타일이 없음.
4. **기업 컬쳐핏 기능은 프론트·백엔드 양쪽 모두 부재** (완전 신규).

### 목표

① 리포트 출력단 구조 개선 ② 기업 컬쳐핏 추천 UI 신규 추가 ③ 전체 컴포넌트 구조 정리

### 4단계 로드맵

| 단계 | 작업 | 산출물 | 리스크 |
|---|---|---|---|
| **1** | 공용 UI 프리미티브 + 토큰 추출 | `components/ui/` (PieChart, BarCompareChart, RadarChart, StatCard, Badge, SectionTitle), `styles/tokens.css` | 없음 (화면 100% 동일) |
| **2** | 리포트 페이지 섹션 분리 | `components/report/` 7개 컴포넌트, ReportPage는 데이터 로딩/오케스트레이션만 | 낮음 |
| **3** | 컬쳐핏 탭 UI (목데이터) | `types/cultureFit.ts`, 랭킹 카드 + 레이더 차트 | 낮음 |
| **4** | 백엔드 연동 | `POST /session/culture-fit` (가칭) | 백엔드 작업 별도 필요 |

**1~3단계는 백엔드·모델 없이 프론트 단독 진행 가능.**

#### 1단계 — 공용 프리미티브 분리 (순수 리팩터링)

```
components/ui/
  PieChart.tsx         ← ReportPage의 PieChart 추출
  BarCompareChart.tsx  ← TurnCompareChart 일반화 (음성/표정 비교)
  RadarChart.tsx       ← 신규, 3단계 컬쳐핏용
  StatCard.tsx / Badge.tsx / SectionTitle.tsx
styles/tokens.css      ← 색상·spacing·타이포 CSS 변수 통합
```

이 단계 이후 리포트 화면과 컬쳐핏 화면이 **같은 부품을 공유**하게 되는 것이 핵심.

#### 2단계 — 리포트 섹션 분리

```
components/report/
  ReportMasthead / NarrativeSection / EmotionChartSection
  QuotesSection / PatternsStrengthsSection / ReflectionSection / TurnsTable
```

추가 개선안:
- 최상단 **핵심 지표 요약 스트립** (발화 수 · 주요 감정 · 미스매치 횟수) — `LandingPage`의 `HIGHLIGHTS` 톤과 통일
- 페이지가 길어지므로 **sticky in-page 앵커 네비게이션** 검토
- `utils/reportMetrics.ts`는 수정 없이 그대로 재사용

#### 3단계 — 기업 컬쳐핏 추천 UI (신규)

**배치 결정**: 별도 페이지가 아니라 리포트 화면 상단 **탭** `[종합 리포트] / [기업 컬쳐핏 추천]`.
→ 같은 세션 데이터를 다른 관점으로 보는 것뿐이라, 페이지를 나누면 이탈만 늘어남. 두 화면은 공통 `ReportShell` 레이아웃 공유.

**화면**: 매칭 기업/문화유형 랭킹 카드(매치율 % + 매칭 이유 태그) + 성향 vs 기업문화 레이더 차트(협업/자율성/속도/안정성 등).

**데이터 전략** — 백엔드 대기 없이:
1. 1차: 기존 `strengths` / `patterns` / 감정 aggregate를 **클라이언트 룰 기반 목데이터**로 가공해 UI 완성
2. 이후 실제 API로 교체 — **목데이터 생성 함수와 API 호출 함수가 동일한 반환 타입**을 쓰도록 `types/cultureFit.ts`를 먼저 확정

#### 4단계 — 백엔드 연동

- 스키마/서비스 설계 별도 논의 필요: `app/services/report_builder.py`에 필드 추가 vs 별도 서비스 분리
- 프론트는 3단계 인터페이스만 맞추면 되므로 **병렬 진행 가능**
- ⚠️ 감정 enum·리포트 스키마에 영향이 가면 Spring 레포 `docs/integration-spec.md` PR로 **양 팀 합의 선행** (CLAUDE.md 계약)

---

## 5. 정리 필요 항목 (working tree)

| 항목 | 상태 | 조치 |
|---|---|---|
| `MODEL_AND_CONVERSATION_PIPELINE_OVERVIEW.md` | 루트 → `docs/`로 이동 중 (미커밋) | 커밋 |
| `docs/KYUL_종합요약_수정체크리스트.md` | 루트/`docs/` 중복 존재 | 하나로 정리 |
| `frontend/gyul-frontend@0.1.0`, `frontend/vite` | **0바이트 빈 파일** (명령어 리다이렉트 오타 흔적) | 삭제 |
| 루트의 `.eml` / `.pdf` / `Eyedia_*.md` | 공모전 자료, 미추적 | `.gitignore` 또는 별도 보관 |
