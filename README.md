# SER API Quick Guide

## 1) 실행 순서
```bash
export OPENAI_API_KEY="여기에_진짜_API키"
uvicorn ser_api:app --reload
python3 -m http.server 5500
```

프론트 테스트 페이지:
- `http://127.0.0.1:5500/web_test.html`

## 2) API 요약

### `GET /health`
- 용도: 서버/모델 로드 상태 확인
- 응답 예시:
```json
{"ok": true}
```

### `WebSocket /ws/predict`
- 용도: 실시간 스트리밍 감정 추론
- 입력: 클라이언트가 PCM16 오디오 바이트 전송
- 처리: 서버가 VAD(음성/무음)로 발화 단위 분석
- 출력: `partial`/`final` 결과, `final`에서 감정 및 AI 답변 반환

### `POST /predict` (Debug)
- 용도: 파일 업로드 기반 단건 감정 추론
- 입력: 오디오 파일 (`file`)
- 응답 필드: `label`, `confidence`, `probs`, `filename`
