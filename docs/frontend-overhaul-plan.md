# 프론트엔드 개선 작업 계획

> 브랜치: `fe` (← `dev`에서 분기)
> 목적: ① 리포트 출력단 구조 개선 ② 기업 컬쳐핏 추천 UI 신규 추가 ③ 전체 컴포넌트 구조 정리

## 배경 / 현재 상태 진단

- `frontend/src/pages/ReportPage.tsx` 하나에 데이터 가공 + 차트(PieChart, TurnCompareChart) + 마크업이 전부 들어있음 (약 400줄, 차트 컴포넌트도 파일 내부에 인라인 정의됨)
- 색상값(`CHART_COLORS` 등)이 `ReportPage.tsx`에 하드코딩, 공용 디자인 토큰 없음
- 페이지별로 `styles/*.css`가 각각 존재하고(`report.css`, `landing.css`, `demo.css` 등) 재사용 가능한 컴포넌트 단위 CSS가 아님
- **기업 컬쳐핏 관련 데이터/로직은 프론트·백엔드 어디에도 존재하지 않음** (완전 신규 기능)
- 현재 라우팅 (`App.tsx`):
  - `/` LandingPage
  - `/demo` DemoPage (실시간 음성/표정 감정 분석 + 대화)
  - `/demo/report/loading` ReportLoadingPage
  - `/demo/report` ReportPage
- 리포트 데이터 타입: `frontend/src/types/sessionReport.ts` (`SessionReportJson`, `SessionReportResponse`)
- 리포트 데이터 조회 흐름: `services/api.ts`의 `requestSessionReport()` → `ReportPage`가 `location.state.report` 또는 `sessionStorage` 캐시(`report:*` 키)에서 읽음

## 작업 순서 (단계별, 앞 단계부터 순서대로 진행 권장)

### 1단계 — 공용 UI 프리미티브 분리 (시각적 변화 없는 순수 리팩터링)

기존 `ReportPage.tsx`에 인라인으로 박혀있던 것들을 추출:

```
frontend/src/components/ui/
  PieChart.tsx         ← ReportPage.tsx의 PieChart 함수 추출
  BarCompareChart.tsx  ← TurnCompareChart 일반화 (음성/표정 비교 바 차트)
  RadarChart.tsx        ← 신규. 3단계 컬쳐핏에서 사용 (성향 vs 기업문화 축 비교)
  StatCard.tsx
  Badge.tsx
  SectionTitle.tsx

frontend/src/styles/tokens.css   ← 색상/spacing/타이포 CSS 변수로 통합
                                     (CHART_COLORS, VOICE_COLOR, FACE_COLOR 등 여기로 이동)
```

이 단계가 끝나면 리포트 페이지와 이후 만들 컬쳐핏 화면이 같은 부품을 공유하게 됨.

### 2단계 — 리포트 페이지 섹션 분리

`ReportPage.tsx`를 오케스트레이션(라우팅/데이터 로딩)만 남기고 섹션별 컴포넌트로 분리:

```
frontend/src/components/report/
  ReportMasthead.tsx        ← 헤더/툴바(다운로드, 새 대화 버튼)
  NarrativeSection.tsx      ← "나를 돌아보며" 줄글 영역 (comprehensiveParagraphs)
  EmotionChartSection.tsx   ← PieChart 2개 + TurnCompareChart
  QuotesSection.tsx         ← 핵심 발화(rankedQuotes)
  PatternsStrengthsSection.tsx
  ReflectionSection.tsx
  TurnsTable.tsx
```

추가 개선안:
- 최상단에 핵심 지표 요약 스트립 추가 (발화 수 · 주요 감정 · 미스매치 횟수 — `LandingPage.tsx`의 `HIGHLIGHTS` 패턴과 톤 통일)
- 섹션이 많아 길어지는 페이지이므로 인앵커 네비게이션(sticky in-page nav) 검토

관련 유틸: `utils/reportMetrics.ts` (`buildRankedQuotes`, `countsToSlices`, `formatSessionRange`, `labelKo`) — 그대로 재사용 가능.

### 3단계 — 기업 컬쳐핏 추천 UI (신규)

**배치 결정**: 별도 페이지가 아니라, 리포트 화면 상단에 탭 `[종합 리포트] / [기업 컬쳐핏 추천]`으로 구성.
→ 이유: 같은 세션 데이터를 다른 관점에서 보여주는 것뿐이라 페이지를 분리하면 이탈만 늘어남. 두 화면이 같은 `ReportShell` 레이아웃을 공유하도록 설계.

**화면 구성**:
- 매칭 기업/문화유형 랭킹 카드 리스트 (매치율 %, 매칭 이유 태그)
- 성향 vs 기업문화 축 비교 레이더 차트 (예: 협업 / 자율성 / 속도 / 안정성 등의 축)

**데이터 전략 (중요)**: 백엔드에 컬쳐핏 로직이 아직 없음. 백엔드를 기다리지 않기 위해:
1. 1차: 세션 리포트에 이미 있는 `strengths` / `patterns` / 감정 aggregate 데이터를 **클라이언트에서 룰 기반으로 가공한 목데이터**로 UI부터 완성
2. 이후 백엔드에 컬쳐핏 엔드포인트(가칭 `POST /session/culture-fit`)가 준비되면, 목데이터와 동일한 타입 인터페이스로 실제 API 응답으로 교체
   → 타입을 미리 `types/cultureFit.ts`에 정의해두고, 목데이터 생성 함수와 실제 API 호출 함수가 같은 반환 타입을 쓰도록 설계할 것

### 4단계 — 백엔드 연동

- 백엔드 쪽 스키마/서비스 설계는 별도 논의 필요 (`app/services/report_builder.py`에 필드 추가 vs 별도 서비스 분리)
- 프론트는 3단계에서 잡아둔 인터페이스만 맞추면 되므로 병렬 진행 가능

## 리스크 정리

| 단계 | 내용 | 리스크 |
|---|---|---|
| 1 | UI 프리미티브 + 토큰 추출 | 없음 (기존 화면 100% 동일 유지) |
| 2 | 리포트 섹션 분리 + 요약 스트립 | 낮음 |
| 3 | 컬쳐핏 탭 UI (목데이터) | 낮음, 백엔드 독립적으로 진행 가능 |
| 4 | 백엔드 연동 | 백엔드 작업 별도 필요 |

1~3단계는 백엔드 없이 프론트 단독으로 진행 가능. 모델 로드 여부와도 무관.

## 참고 — 현재 관련 파일 목록

```
frontend/src/App.tsx
frontend/src/layouts/AppLayout.tsx
frontend/src/pages/{LandingPage,DemoPage,ReportLoadingPage,ReportPage}.tsx
frontend/src/types/sessionReport.ts
frontend/src/services/api.ts
frontend/src/utils/reportMetrics.ts
frontend/src/styles/{layout,landing,demo,report,report-loading}.css
```
