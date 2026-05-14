import os

# Force Transformers to use PyTorch only in this service.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

MODEL_DIR = os.getenv("GYUL_MODEL_DIR", "model")
TARGET_SR = int(os.getenv("GYUL_TARGET_SR", "16000"))
MIN_CHUNK_SECONDS = float(os.getenv("GYUL_MIN_CHUNK_SECONDS", "1.0"))
VAD_RMS_THRESHOLD = float(os.getenv("GYUL_VAD_RMS_THRESHOLD", "0.01"))
VAD_MIN_SPEECH_SECONDS = float(os.getenv("GYUL_VAD_MIN_SPEECH_SECONDS", "0.4"))
VAD_SILENCE_SECONDS = float(os.getenv("GYUL_VAD_SILENCE_SECONDS", "0.7"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
