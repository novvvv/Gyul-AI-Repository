# Kyul Frontend (Vite + React)

## 실행

1. 백엔드 (레포 루트):

```bash
uvicorn ser_api:app --reload
```

2. 프론트:

```bash
cd frontend
npm install
npm run dev
```

브라우저: http://127.0.0.1:5173/

- `/` — 소개 랜딩
- `/demo` — 실시간 데모 (`web_test.html`과 동일 프로토콜)

개발 시 `vite.config.ts`가 `/api` → `8000`, `/ws` → WebSocket으로 프록시합니다.

## 프로덕션 빌드

```bash
npm run build
```

`dist/`를 정적 호스팅. API 주소는 `.env`에 `VITE_API_BASE`, `VITE_WS_URL` 설정.

## 레거시

`web_test.html`은 그대로 두었습니다. 동작 비교·폴백용.
