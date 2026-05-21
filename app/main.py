from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket

from app.memory.session_memory import session_memory
from app.services.llm_service import llm_reply_service
from app.services.ser_service import ser_service
from app.ws.predict import predict_websocket

app = FastAPI(title="SER API")


@app.on_event("startup")
def load_model() -> None:
    ser_service.load()


@app.get("/health")
def health() -> dict:
    return {"ok": ser_service.is_loaded}


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


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    await predict_websocket(websocket, ser_service, llm_reply_service, session_memory)
