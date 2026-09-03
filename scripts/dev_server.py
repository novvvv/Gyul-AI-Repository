"""
개발용 대체 서버 — SER 모델 없이 프론트를 돌리기 위한 것.

실제 서버(`ser_api.py`)는 SER 가중치(`model/model.safetensors`)가 있어야
WebSocket 이 열린다. 그 파일이 없는 환경에서 **대화와 리포트만이라도
진짜 LLM으로** 확인할 수 있게 만든 서버다.

  - 대화 응답 : OpenAI 실제 호출 (app/chains/conversation_chain.py 의 페르소나 프롬프트 사용)
  - 리포트    : OpenAI 실제 호출 (scripts/session_report 와 같은 스키마로 생성)
  - 음성 감정 : **가짜** — SER 모델이 없으므로 무작위
  - 표정      : **가짜**

메시지 계약은 app/ws/pipeline.py 와 동일하다.

    pip install fastapi "uvicorn[standard]" openai python-dotenv
    python scripts/dev_server.py          # 127.0.0.1:8000
"""

from __future__ import annotations

import base64
import json
import os
import random
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel

# ── .env 로드 (python-dotenv 없이도 동작) ──
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "shimmer")
TTS_ON = os.environ.get("ENABLE_OPENAI_TTS", "1").strip().lower() not in ("0", "false", "no")
HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))

try:
    from openai import OpenAI

    _client = OpenAI() if HAS_KEY else None
except Exception:  # openai 미설치
    _client = None

app = FastAPI(title="Gyul DEV")

LABELS = ["neutral", "happiness", "sadness", "fear", "angry", "surprise", "disgust"]

# app/chains/conversation_chain.py 의 프롬프트를 그대로 가져온다
PERSONA = {
    "gyul": (
        "너는 결이라는 이름의 자기분석 대화 상대다. 사용자가 스스로를 더 잘 이해하도록 "
        "부드럽게 되묻고, 단정적인 진단은 피한다. 사용자가 한 말에 공감하고 위로를 건네며 "
        "대화를 이어간다. 계속해서 사용자의 반응을 이끌어낸다.\n"
        "규칙: 존댓말 '~해요'체로 따뜻하게. 2~3문장. 마지막은 반드시 질문으로 끝낸다. "
        "평가하거나 조언하지 않는다."
    ),
    "interviewer": (
        "너는 차분한 AI 면접관이다. 답변을 경청하되 면접 흐름을 유지하고, "
        "필요하면 짧은 꼬리 질문을 한다.\n"
        "규칙: 존댓말 '~습니다'체로 격식 있게. 2~3문장. 방금 답변에서 더 확인할 지점을 "
        "하나 골라 구체적으로 되묻는다. 칭찬이나 평가는 하지 않는다."
    ),
}


def fake_emotion() -> dict:
    """SER 모델이 없으므로 무작위. 실제 모델 연결 시 이 함수만 교체하면 된다."""
    label = random.choice(LABELS)
    conf = round(random.uniform(0.45, 0.92), 4)
    rest = round((1 - conf) / (len(LABELS) - 1), 4)
    return {
        "label": label,
        "confidence": conf,
        "probs": {x: (conf if x == label else rest) for x in LABELS},
    }


# 사용자 SER 감정 → 낭독 톤 (app/services/openai_tts_service.py 와 동일)
SER_TONE = {
    "sadness": "차분하고 낮은 톤으로, 조금 느리게. 위로하듯이 읽어라.",
    "fear": "부드럽고 안심시키는 톤으로, 천천히 읽어라.",
    "angry": "감정을 누그러뜨리는 차분한 톤으로 읽어라.",
    "disgust": "담담하고 중립적인 톤으로 읽어라.",
    "happiness": "밝고 가벼운 톤으로, 미소 띤 목소리로 읽어라.",
    "surprise": "약간 올라간 톤으로 반가움을 담아 읽어라.",
    "neutral": "차분하고 담담한 톤으로 읽어라.",
}


def speak_b64(text: str, emotion: str) -> str | None:
    """OpenAI audio/speech 로 낭독해 base64(mp3) 반환. 실패하면 None."""
    if not (TTS_ON and _client and text.strip()):
        return None
    try:
        kwargs: dict = {
            "model": TTS_MODEL,
            "voice": TTS_VOICE,
            "input": text.strip(),
            "response_format": "mp3",
        }
        if "gpt-4o" in TTS_MODEL:
            kwargs["instructions"] = SER_TONE.get(emotion, SER_TONE["neutral"])
        res = _client.audio.speech.create(**kwargs)
        return base64.b64encode(res.read()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        print(f"[dev] TTS 실패: {e}")
        return None


def chat(messages: list[dict], *, temperature: float = 0.7, json_mode: bool = False) -> str:
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY 가 없거나 openai 패키지가 설치되지 않았습니다.")
    kwargs: dict = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    res = _client.chat.completions.create(**kwargs)
    return (res.choices[0].message.content or "").strip()


# ────────────────────────── HTTP ──────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "face_enabled": True,
        "face_loaded": True,
        "text_llm_backend": "openai" if _client else "none",
        "text_llm_loaded": bool(_client),
        "fish_tts_enabled": False,
        "openai_tts_enabled": bool(TTS_ON and _client),
        "ser_loaded": False,  # 개발 서버는 SER 없음
    }


class ImageRequest(BaseModel):
    image: str


@app.post("/detect_face")
def detect_face(_: ImageRequest) -> dict:
    return {
        "faces": [{"x": 180, "y": 90, "w": 220, "h": 260, "confidence": 0.94}],
        "width": 640,
        "height": 480,
    }


@app.post("/predict_face")
def predict_face(_: ImageRequest) -> dict:
    return fake_emotion()


class TtsRequest(BaseModel):
    text: str
    emotion: str = "neutral"


@app.post("/tts")
def tts(body: TtsRequest) -> dict:
    """임의의 문장을 낭독한다. 고정 문항(첫 질문) 읽어주기에 쓴다."""
    audio_b64 = speak_b64(body.text, body.emotion)
    if not audio_b64:
        return {"audio_b64": None, "format": None}
    return {"audio_b64": audio_b64, "format": "mp3"}


class ReportRequest(BaseModel):
    session: dict = {}
    turns: list = []


REPORT_SYSTEM = """너는 음성 면접 코치다. 아래 대화 기록을 읽고 리포트를 JSON으로만 출력한다.

각 발화에는 목소리에서 읽힌 감정(voice_emotion)과 표정(face_emotion)이 붙어 있다.
둘이 다른 구간은 "말과 표정이 어긋난 지점"으로 해석한다.

반드시 아래 키를 가진 JSON 객체 하나만 출력한다:
{
  "comprehensive_report": "3~4문단. 첫 문단은 이 대화 전체를 한 문장으로 요약하는 결론.",
  "summary": "한 문장 요약",
  "topics": ["다룬 주제 3~5개"],
  "quotes": ["기억할 만한 발화 2~3개 (원문 그대로)"],
  "patterns": ["관찰된 말버릇·태도 2~3개"],
  "strengths": ["잘하고 있는 점 2~3개. 근거를 함께"],
  "reflection_questions": ["다음에 생각해볼 질문 2~3개"],
  "next_topics": ["더 다뤄볼 주제 1~2개"]
}

문체 규칙:
- 자가분석(persona_id=gyul) 대화면: 평가하지 말고 관찰만. '~했어요' 체로 따뜻하게.
- 면접(persona_id=interviewer) 대화면: 면접 코치 어투. '~합니다' 체로 구체적이고 실용적으로.
- 근거 없는 단정 금지. 관찰된 발화를 인용해 뒷받침한다.
"""


def fallback_report(turns: list, persona: str) -> dict:
    """LLM 실패 시에도 화면이 비지 않도록 집계만으로 채운다."""
    n = len(turns)
    return {
        "comprehensive_report": (
            f"{n}번의 답변이 기록됐습니다. 리포트 본문 생성에 실패해 집계만 표시합니다."
        ),
        "summary": "리포트 생성 실패",
        "topics": [],
        "quotes": [t.get("user_text", "") for t in turns[:3] if t.get("user_text")],
        "patterns": [],
        "strengths": [],
        "reflection_questions": [],
        "next_topics": [],
    }


@app.post("/session/report")
def session_report(body: ReportRequest) -> dict:
    turns = body.turns
    persona = body.session.get("persona_id", "gyul")

    voice_counts: dict[str, int] = {}
    face_counts: dict[str, int] = {}
    mismatch = 0
    for t in turns:
        v = (t.get("voice_emotion") or {}).get("label")
        f = (t.get("face_emotion") or {}).get("label")
        if v:
            voice_counts[v] = voice_counts.get(v, 0) + 1
        if f:
            face_counts[f] = face_counts.get(f, 0) + 1
        if v and f and v != f:
            mismatch += 1

    transcript = "\n".join(
        f"{i + 1}. 사용자: {t.get('user_text', '')}\n"
        f"   (음성 {(t.get('voice_emotion') or {}).get('label', '?')}, "
        f"표정 {(t.get('face_emotion') or {}).get('label', '?')})\n"
        f"   AI: {t.get('bot_reply', '')}"
        for i, t in enumerate(turns)
    )

    generated: dict
    error: str | None = None
    try:
        raw = chat(
            [
                {"role": "system", "content": REPORT_SYSTEM},
                {
                    "role": "user",
                    "content": f"persona_id: {persona}\n\n대화 기록:\n{transcript}",
                },
            ],
            temperature=0.5,
            json_mode=True,
        )
        generated = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        error = str(e)
        generated = fallback_report(turns, persona)

    dominant = sorted(voice_counts, key=lambda k: voice_counts[k], reverse=True)
    face_dominant = sorted(face_counts, key=lambda k: face_counts[k], reverse=True)

    report_json = {
        "meta": {
            "session_id": body.session.get("session_id", "dev-session"),
            "persona_id": persona,
            "started_at": body.session.get("started_at"),
            "ended_at": body.session.get("ended_at"),
            "llm_backend": "openai" if _client else "none",
            "user_turn_count": len(turns),
            "bot_turn_count": len(turns),
            "voice_dominant": dominant[:2],
            "face_dominant": face_dominant[:2],
            "emotion_shifts": dominant[:3],
            "mismatch_count": mismatch,
        },
        **{k: generated.get(k, []) for k in
           ("topics", "quotes", "patterns", "strengths", "reflection_questions", "next_topics")},
        "comprehensive_report": generated.get("comprehensive_report", ""),
        "summary": generated.get("summary", ""),
        "disclaimer": (
            "이 리포트는 심리 검사나 진단이 아닙니다. 대화에서 관찰된 신호를 정리한 기록이며, "
            "감정 분석 결과에는 오차가 있을 수 있습니다. "
            "(개발 서버 — 음성 감정은 실제 SER 모델이 아닌 임시 값입니다)"
        ),
        "turns": turns,
        "aggregates": {"voice_counts": voice_counts, "face_counts": face_counts},
    }
    if error:
        report_json["generation_error"] = error

    return {
        "report_json": report_json,
        "report_md": f"# 리포트\n\n{report_json['comprehensive_report']}\n",
        "llm_backend": report_json["meta"]["llm_backend"],
    }


# ────────────────────────── WebSocket ──────────────────────────
async def run_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    session_id = websocket.query_params.get("session_id", "default")
    persona_id = websocket.query_params.get("persona_id", "gyul")
    history: list[dict] = []
    frames = 0

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_msg = message.get("text")
            if text_msg:
                if text_msg in ("flush",):
                    continue
                if text_msg == "session_end":
                    await websocket.send_json(
                        {"type": "session_closed", "session_id": session_id,
                         "turn_count": len(history) // 2}
                    )
                    continue
                try:
                    payload = json.loads(text_msg)
                except json.JSONDecodeError:
                    continue

                kind = payload.get("type")
                if kind in ("session", "session_start", "session_context"):
                    session_id = payload.get("session_id", session_id)
                    persona_id = payload.get("persona_id", persona_id)
                    history = []
                    await websocket.send_json(
                        {"type": "session_ready", "session_id": session_id,
                         "persona_id": persona_id}
                    )

                elif kind == "utterance_text":
                    text = (payload.get("text") or "").strip()
                    if not text:
                        continue

                    emo = fake_emotion()
                    system = PERSONA.get(persona_id, PERSONA["gyul"])

                    try:
                        reply = chat(
                            [{"role": "system", "content": system}]
                            + history
                            + [{
                                "role": "user",
                                "content": (
                                    f"{text}\n"
                                    f"[목소리에서 읽힌 감정: {emo['label']}]\n"
                                    "위 감정 신호는 답변의 톤을 고르는 데만 참고해줘. "
                                    "감정을 직접 언급하지는 마."
                                ),
                            }]
                        )
                    except Exception as e:  # noqa: BLE001
                        await websocket.send_json({"error": f"LLM 호출 실패: {e}"})
                        continue

                    history += [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": reply},
                    ]
                    history = history[-12:]  # 최근 6턴만 유지

                    payload = {
                        "type": "final",
                        "session_id": session_id,
                        "persona_id": persona_id,
                        "text": text,
                        "reply": reply,
                        **emo,
                    }
                    audio_b64 = speak_b64(reply, emo["label"])
                    if audio_b64:
                        payload["reply_audio_b64"] = audio_b64
                        payload["reply_audio_format"] = "mp3"

                    await websocket.send_json(payload)
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue
            frames += 1
            if frames % 8 == 0:
                await websocket.send_json({"type": "partial", **fake_emotion()})

    except WebSocketDisconnect:
        return


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    await run_socket(websocket)


@app.websocket("/ws/interview")
async def ws_interview(websocket: WebSocket) -> None:
    await run_socket(websocket)


if __name__ == "__main__":
    import uvicorn

    print(f"[dev] LLM: {'OpenAI ' + OPENAI_MODEL if _client else '없음 (키 확인 필요)'}")
    print(f"[dev] TTS: {TTS_MODEL + ' / ' + TTS_VOICE if (TTS_ON and _client) else '꺼짐'}")
    print("[dev] SER: 없음 — 음성 감정은 임시 값입니다")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
