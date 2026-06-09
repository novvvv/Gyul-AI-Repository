# SER API Quick Guide

## 1) 구조

```
web_test.html → WS /ws/predict (오디오 + utterance_text)
ser_api.py → app/main.py
  startup: ser_service + (local 모드면) local_llm_service
  final: SER(오디오) → llm_service(텍스트+감정) → reply
```

|     | 입력        | 서버                                                   |
| --- | ----------- | ------------------------------------------------------ |
| SER | 오디오      | `ser_service` / `model/`                               |
| STT | -           | 브라우저만 (`web_test.html`)                           |
| TER | -           | 미연동 (`train_code/`만)                               |
| LLM | 텍스트+감정 | `llm_service` → OpenAI/Gemini 또는 `local_llm_service` |

- 감정 `label`: SER(오디오). 텍스트: STT → LLM 답변용.
- `llm_service`: 프롬프트·분기. `local_llm_service`: 로컬 HF 전용 (`ser_service`와 같은 역할 분리).

`LLM_PROVIDER`: `openai`(기본), `gemini`, `local`/`exaone`/`huggingface`(API 키 없음, EXAONE 기본).

---

## 2) 실행 순서

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="여기에_진짜_API키"
uvicorn ser_api:app --reload
python3 -m http.server 5500
```

로컬 LLM(EXAONE 등, API 키 불필요)을 사용할 경우:

```bash
export LLM_PROVIDER="local"
export LOCAL_LLM_MODEL_ID="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
uvicorn ser_api:app --reload
```

- `transformers>=5.0` 필요 (EXAONE 3.5 remote code 호환).
- 첫 실행 시 Hugging Face에서 모델을 받으므로 시간·디스크·VRAM이 필요합니다.
- GPU가 있으면 자동으로 CUDA를 사용하고, Mac은 MPS → CPU 순으로 시도합니다.

Gemini API를 사용할 경우:

```bash
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="여기에_Gemini_API키"
export GEMINI_MODEL="gemini-2.5-flash"
uvicorn ser_api:app --reload
```

프론트 테스트 페이지:

- `http://127.0.0.1:5500/web_test.html`

## 3) API 요약

### `GET /health`

- 용도: 서버/모델 로드 상태 확인
- 응답 예시 (OpenAI/Gemini 모드):

```json
{ "ok": true }
```

- 응답 예시 (로컬 LLM 모드):

```json
{ "ok": true, "llm_loaded": true, "llm_provider": "local" }
```

### `WebSocket /ws/predict`

- 용도: 실시간 스트리밍 감정 추론
- 입력: 클라이언트가 PCM16 오디오 바이트 전송
- 처리: 서버가 VAD(음성/무음)로 발화 단위 분석
- 출력: `partial`/`final` 결과, `final`에서 감정·`reply` 반환
- MVP 세션 파라미터: `user_id`, `session_id`, `persona_id`
  - 예: `ws://127.0.0.1:8000/ws/predict?user_id=u1&session_id=s1&persona_id=gyul`
  - 같은 `user_id/session_id/persona_id` 조합에서는 서버 프로세스 메모리 안에서 최근 대화가 유지됩니다.

세션 시작 메시지 예시:

```json
{
  "type": "session_start",
  "user_id": "u1",
  "session_id": "s1",
  "persona_id": "gyul"
}
```

사용자 발화 텍스트 메시지 예시:

```json
{ "type": "utterance_text", "text": "오늘 면접 생각만 하면 좀 긴장돼요." }
```

세션 종료 메시지 예시:

```text
session_end
```

### `POST /predict` (Debug)

- 용도: 파일 업로드 기반 단건 감정 추론
- 입력: 오디오 파일 (`file`)
- 응답 필드: `label`, `confidence`, `probs`, `filename`

### WS 테스트 스크립트

```bash
python scripts/ws_chat_test.py
```

### 한 번에 실행 (권장)

```bash
./start.sh
```

- API `http://127.0.0.1:8000` + 데모 `http://127.0.0.1:5173/demo` 동시 기동
- `OPENAI_API_KEY` 있으면 gpt-4o-mini, 없으면 Kanana
- 종료: `Ctrl+C` 또는 `./stop.sh`

수동 실행:

```bash
uvicorn ser_api:app --reload
cd frontend && npm run dev
```
