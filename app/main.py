import asyncio

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import ENABLE_FACE, LOCAL_LLM_PROVIDERS, LLM_PROVIDER
from app.memory.session_memory import session_memory
from app.services.face_detect_service import face_detect_service
from app.services.face_service import face_service
from app.services.llm_service import llm_reply_service
from app.services.local_llm_service import local_llm_service
from app.services.ser_service import ser_service
from app.vision.predict import (
    InvalidImageError,
    analyze_face_image,
    detect_faces_in_image,
)
from app.ws.predict import predict_websocket

app = FastAPI(title="SER API")

# 정적 페이지(http.server:5500)에서 /predict_face 등 HTTP 호출을 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FaceRequest(BaseModel):
    image: str


@app.on_event("startup")
def load_model() -> None:
    ser_service.load()
    if LLM_PROVIDER in LOCAL_LLM_PROVIDERS:
        local_llm_service.load()


@app.get("/health")
def health() -> dict:
    payload: dict = {"ok": ser_service.is_loaded}
    if ENABLE_FACE:
        payload["face_enabled"] = True
        payload["face_loaded"] = face_service.is_loaded
    if LLM_PROVIDER in LOCAL_LLM_PROVIDERS:
        payload["llm_loaded"] = local_llm_service.is_loaded
        payload["llm_provider"] = LLM_PROVIDER
    return payload


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not ser_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = ser_service.predict_from_file_bytes(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    return {
        **result,
        "filename": file.filename,
    }


@app.post("/predict_face")
async def predict_face(body: FaceRequest) -> dict:
    if not ENABLE_FACE:
        raise HTTPException(status_code=503, detail="Face emotion disabled")

    # 모델은 무겁고 최초 1회 다운로드가 필요하므로 첫 요청 시 lazy load.
    if not face_service.is_loaded:
        try:
            await asyncio.to_thread(face_service.load)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Face model load failed: {e}")

    try:
        return await asyncio.to_thread(analyze_face_image, face_service, body.image)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face inference failed: {e}")


@app.post("/detect_face")
async def detect_face(body: FaceRequest) -> dict:
    if not ENABLE_FACE:
        raise HTTPException(status_code=503, detail="Face detection disabled")

    if not face_detect_service.is_loaded:
        try:
            await asyncio.to_thread(face_detect_service.load)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Detector load failed: {e}")

    try:
        return await asyncio.to_thread(
            detect_faces_in_image, face_detect_service, body.image
        )
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face detection failed: {e}")


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    await predict_websocket(websocket, ser_service, llm_reply_service, session_memory)
