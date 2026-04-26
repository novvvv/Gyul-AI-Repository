import asyncio
import json
import os
from collections import deque
from io import BytesIO
from urllib import error, request

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import WebSocket, WebSocketDisconnect

# Force Transformers to use PyTorch only in this service.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_DIR = "model"
TARGET_SR = 16000
MIN_CHUNK_SECONDS = 1.0
VAD_RMS_THRESHOLD = 0.01
VAD_MIN_SPEECH_SECONDS = 0.4
VAD_SILENCE_SECONDS = 0.7
OPENAI_MODEL = "gpt-4o-mini"

app = FastAPI(title="SER API")

feature_extractor = None
model = None


@app.on_event("startup")
def load_model() -> None:
    global feature_extractor, model
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
    model.eval()


@app.get("/health")
def health() -> dict:
    return {"ok": model is not None}


def _predict_from_audio(audio: np.ndarray) -> dict:
    inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id].item())
    all_probs = {
        model.config.id2label[i]: float(probs[i].item()) for i in range(len(probs))
    }
    return {"label": label, "confidence": confidence, "probs": all_probs}


def _generate_ai_reply(text: str, emotion: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"OPENAI_API_KEY가 없어서 감정({emotion})만 분석했어요."

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 한국어 대화 어시스턴트다. "
                    "사용자의 문장과 감정 분석 결과를 반영해 짧고 자연스럽게 답해라."
                ),
            },
            {
                "role": "user",
                "content": f"사용자 발화: {text}\n감정: {emotion}\n한두 문장으로 답변해줘.",
            },
        ],
        "temperature": 0.7,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        return f"AI 답변 호출 실패(HTTP {e.code}): {detail[:180]}"
    except Exception as e:
        return f"AI 답변 호출 실패: {e}"


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        audio_bytes = await file.read()
        audio, _ = librosa.load(BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    if len(audio) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    result = _predict_from_audio(audio)

    return {
        **result,
        "filename": file.filename,
    }


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    await websocket.accept()

    if model is None or feature_extractor is None:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close(code=1011)
        return

    min_samples = int(TARGET_SR * MIN_CHUNK_SECONDS)
    vad_min_speech_samples = int(TARGET_SR * VAD_MIN_SPEECH_SECONDS)
    vad_silence_samples = int(TARGET_SR * VAD_SILENCE_SECONDS)
    stream_buffer = np.array([], dtype=np.float32)
    utterance_buffer = np.array([], dtype=np.float32)
    pending_texts = deque()
    speech_active = False
    silence_samples = 0

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                text_msg = message["text"]
                if text_msg == "flush":
                    if len(utterance_buffer) >= vad_min_speech_samples:
                        final_result = _predict_from_audio(utterance_buffer)
                        text_for_reply = (
                            pending_texts.popleft() if pending_texts else "방금 발화"
                        )
                        ai_reply = await asyncio.to_thread(
                            _generate_ai_reply,
                            text_for_reply,
                            final_result["label"],
                        )
                        await websocket.send_json(
                            {
                                "type": "final",
                                "text": text_for_reply,
                                "reply": ai_reply,
                                **final_result,
                            }
                        )
                    utterance_buffer = np.array([], dtype=np.float32)
                    speech_active = False
                    silence_samples = 0
                else:
                    try:
                        text_payload = json.loads(text_msg)
                    except json.JSONDecodeError:
                        text_payload = None
                    if (
                        isinstance(text_payload, dict)
                        and text_payload.get("type") == "utterance_text"
                        and isinstance(text_payload.get("text"), str)
                    ):
                        utterance_text = text_payload["text"].strip()
                        if utterance_text:
                            pending_texts.append(utterance_text)
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue

            # Expect 16-bit PCM mono audio bytes from client.
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            stream_buffer = np.concatenate([stream_buffer, pcm])

            rms = float(np.sqrt(np.mean(np.square(pcm)) + 1e-12))
            is_speech = rms >= VAD_RMS_THRESHOLD

            if is_speech:
                speech_active = True
                silence_samples = 0
                utterance_buffer = np.concatenate([utterance_buffer, pcm])
            elif speech_active:
                silence_samples += len(pcm)
                utterance_buffer = np.concatenate([utterance_buffer, pcm])

            if len(stream_buffer) >= min_samples:
                audio = stream_buffer[-min_samples:]
                partial_result = _predict_from_audio(audio)
                await websocket.send_json({"type": "partial", **partial_result})

            if (
                speech_active
                and silence_samples >= vad_silence_samples
                and len(utterance_buffer) >= vad_min_speech_samples
            ):
                final_result = _predict_from_audio(utterance_buffer)
                text_for_reply = pending_texts.popleft() if pending_texts else "방금 발화"
                ai_reply = await asyncio.to_thread(
                    _generate_ai_reply,
                    text_for_reply,
                    final_result["label"],
                )
                await websocket.send_json(
                    {
                        "type": "final",
                        "text": text_for_reply,
                        "reply": ai_reply,
                        **final_result,
                    }
                )
                utterance_buffer = np.array([], dtype=np.float32)
                speech_active = False
                silence_samples = 0

    except WebSocketDisconnect:
        return
