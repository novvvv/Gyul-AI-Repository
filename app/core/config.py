import os

# Force Transformers to use PyTorch only in this service.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

# Spring 연동 (docs/fastapi-integration-guide.md §1) — Spring과 동일 값 주입 필수
JWT_SECRET = os.getenv("JWT_SECRET", "")

# 대화 세션 메모리 (가이드 §4 — FastAPI는 Redis DB 0만 사용, DB 1은 Spring 전용)
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "memory").strip().lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_MEMORY_TTL_SECONDS = int(os.getenv("SESSION_MEMORY_TTL_SECONDS", "86400"))
SESSION_MEMORY_MAX_TURNS = int(os.getenv("SESSION_MEMORY_MAX_TURNS", "20"))

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

# Face emotion (vision)
# PyTorch(transformers) 기반 얼굴 표정 분류기. FER 7감정 라벨셋(kyul_mvp와 동일 계열).
FACE_MODEL_ID = os.getenv("FACE_MODEL_ID", "trpakov/vit-face-expression")
ENABLE_FACE = os.getenv("ENABLE_FACE", "1").strip().lower() not in ("0", "false", "no")


# Local LLM (Kanana — OPENAI_API_KEY 없을 때 기본 폴백)
KANANA_MODEL_ID = os.getenv(
    "KANANA_MODEL_ID",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
)
LOCAL_LLM_PROVIDERS = frozenset({"local", "exaone", "huggingface", "kanana"})
LOCAL_LLM_MODEL_ID = os.getenv("LOCAL_LLM_MODEL_ID", KANANA_MODEL_ID)

# LOCAL_LLM_MAX_NEW_TOKENS : 최대 생성 토큰 수 
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
# LOCAL_LLM_TEMPERATURE : 샘플링 무작위성 
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7"))

# Fish Audio TTS (https://fish.audio) — FISH_AUDIO_API_KEY 있을 때 활성화
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY", "")
# Fish Audio 문서 예시 voice (playground 샘플). .env로 교체 가능.
FISH_AUDIO_VOICE_ID = os.getenv(
    "FISH_AUDIO_VOICE_ID",
    "7f92f8afb8ec43bf81429cc1c9199cb1",
)
FISH_AUDIO_MODEL = os.getenv("FISH_AUDIO_MODEL", "s2-pro")
ENABLE_FISH_TTS = os.getenv("ENABLE_FISH_TTS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
