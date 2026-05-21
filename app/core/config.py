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


# LLM
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Local LLM 
LOCAL_LLM_PROVIDERS = frozenset({"local", "exaone", "huggingface"})
LOCAL_LLM_MODEL_ID = os.getenv(
    "LOCAL_LLM_MODEL_ID",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
)

# LOCAL_LLM_MAX_NEW_TOKENS : 최대 생성 토큰 수 
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
# LOCAL_LLM_TEMPERATURE : 샘플링 무작위성 
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7"))

# Conversation memory
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_MEMORY_TTL_SECONDS = int(os.getenv("SESSION_MEMORY_TTL_SECONDS", "86400"))
SESSION_MEMORY_MAX_TURNS = int(os.getenv("SESSION_MEMORY_MAX_TURNS", "20"))
