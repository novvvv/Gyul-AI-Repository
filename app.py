"""
SER + STT 감정 인식 테스트 프론트엔드
실행: python -m streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path
import importlib.util

# Anaconda의 깨진 tensorflow 바이너리로 인해 transformers import가 실패하는 환경 우회
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "tensorflow" or name.startswith("tensorflow."):
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

import numpy as np
import streamlit as st
import torch
import soundfile as sf
from scipy.signal import resample
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

MODEL_NAME = "superb/wav2vec2-base-superb-er"
TARGET_SR = 16000
EMOTION_EMOJI = {
    "neu": "😐", "hap": "😊", "ang": "😡", "sad": "😢",
    "dis": "🤢", "fea": "😨", "sur": "😮",
    "neutral": "😐", "happy": "😊", "angry": "😡", "sadness": "😢",
    "disgust": "🤢", "fear": "😨", "surprise": "😮",
}
EMOTION_KO = {
    "neu": "중립", "hap": "행복", "ang": "분노", "sad": "슬픔",
    "dis": "혐오", "fea": "공포", "sur": "놀람",
    "neutral": "중립", "happy": "행복", "angry": "분노", "sadness": "슬픔",
    "disgust": "혐오", "fear": "공포", "surprise": "놀람",
}
BAR_COLORS = {
    "neu": "#888888", "hap": "#4CAF50", "ang": "#F44336", "sad": "#2196F3",
    "dis": "#9C27B0", "fea": "#FF9800", "sur": "#00BCD4",
    "neutral": "#888888", "happy": "#4CAF50", "angry": "#F44336", "sadness": "#2196F3",
    "disgust": "#9C27B0", "fear": "#FF9800", "surprise": "#00BCD4",
}


@st.cache_resource
def load_ser_model():
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return extractor, model


@st.cache_resource
def load_whisper_model():
    import whisper
    return whisper.load_model("base")


def load_audio(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        num_samples = int(len(data) * TARGET_SR / sr)
        data = resample(data, num_samples).astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def transcribe(whisper_model, wav_path: str) -> str:
    audio = load_audio(wav_path)
    audio = np.asarray(audio, dtype=np.float32)
    result = whisper_model.transcribe(audio, language="ko")
    return result["text"].strip()


def predict(extractor, model, audio: np.ndarray, segment_sec: float = 3.0) -> dict:
    seg_len = int(TARGET_SR * segment_sec)
    segments = []
    for i in range(0, len(audio), seg_len):
        seg = audio[i : i + seg_len]
        if len(seg) >= int(TARGET_SR * 0.5):
            segments.append(seg)

    if not segments:
        segments = [audio]

    all_probs = []
    for seg in segments:
        inputs = extractor(seg, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    id2label = model.config.id2label
    labels = [id2label.get(i, f"label_{i}") for i in range(len(avg_probs))]
    pred_idx = int(np.argmax(avg_probs))

    return {
        "emotion": labels[pred_idx],
        "confidence": float(avg_probs[pred_idx]),
        "scores": {labels[i]: float(avg_probs[i]) for i in range(len(avg_probs))},
        "segments": len(segments),
        "duration": len(audio) / TARGET_SR,
    }


def render_result(result: dict, segment_sec: float, stt_text: str = None):
    """감정 분석 결과 렌더링 (공용)"""
    emotion = result["emotion"]
    conf = result["confidence"]
    emoji = EMOTION_EMOJI.get(emotion, "❓")
    ko = EMOTION_KO.get(emotion, emotion)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{ko} ({emotion})</div>
            <div class="result-conf">신뢰도 {conf:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if stt_text:
        st.markdown(
            f"""
            <div class="stt-box">
                <div class="stt-title">📝 음성 인식 (Whisper)</div>
                <div class="stt-text">"{stt_text}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**감정별 점수**")
    sorted_scores = sorted(result["scores"].items(), key=lambda x: -x[1])
    for label, score in sorted_scores:
        color = BAR_COLORS.get(label, "#555")
        lko = EMOTION_KO.get(label, label)
        pct = score * 100
        st.markdown(
            f"""
            <div class="bar-label">{EMOTION_EMOJI.get(label,'')} {lko} ({label})</div>
            <div class="bar-container">
                <div class="bar-fill" style="width:{max(pct,2):.1f}%;background:{color};">
                    {pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="info-box">
            <b>오디오 길이:</b> {result['duration']:.1f}초 &nbsp;|&nbsp;
            <b>세그먼트:</b> {result['segments']}개 × {segment_sec}초 &nbsp;|&nbsp;
            <b>모델:</b> {MODEL_NAME}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

st.set_page_config(page_title="SER 감정 인식", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #0E0E0E;}
    .stApp {background-color: #0E0E0E; color: #F2F2F2;}
    h1, h2, h3 {color: #F2F2F2 !important;}
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px; padding: 28px; text-align: center;
        border: 1px solid #262424; margin-bottom: 16px;
    }
    .result-emoji {font-size: 64px;}
    .result-label {font-size: 28px; font-weight: 700; color: #FAFAFA; margin-top: 8px;}
    .result-conf {font-size: 16px; color: #AAAAAA;}
    .stt-box {
        background: linear-gradient(135deg, #1a2e1a 0%, #162e21 100%);
        border-radius: 14px; padding: 20px; text-align: center;
        border: 1px solid #2a4a2a; margin-bottom: 16px;
    }
    .stt-title {font-size: 14px; color: #88CC88; margin-bottom: 8px; font-weight: 600;}
    .stt-text {font-size: 20px; color: #F2F2F2; font-weight: 500; line-height: 1.5;}
    .bar-container {
        background: #1A1A1A; border-radius: 8px; height: 28px;
        margin-bottom: 6px; overflow: hidden; position: relative;
    }
    .bar-fill {
        height: 100%; border-radius: 8px;
        display: flex; align-items: center; padding-left: 10px;
        font-size: 13px; font-weight: 600; color: #FFF;
        transition: width 0.6s ease;
    }
    .bar-label {font-size: 13px; color: #CCCCCC; margin-bottom: 2px;}
    .info-box {
        background: #151515; border: 1px solid #2A2A2A;
        border-radius: 12px; padding: 16px; margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎙️ SER 감정 인식 테스트")
st.caption("Wav2Vec2 음성 감정 분석 + Whisper 음성 인식")

tab_record, tab_file, tab_sample = st.tabs(["🎤 실시간 녹음", "📁 파일 업로드", "📂 Sample 데이터"])

segment_sec = st.sidebar.slider("세그먼트 길이 (초)", 1.0, 5.0, 3.0, 0.5)
use_stt = st.sidebar.checkbox("Whisper STT 활성화", value=True)

# ── 탭 1: 실시간 녹음 ──
with tab_record:
    st.subheader("마이크로 녹음하기")
    st.caption("아래 버튼을 눌러 녹음 → 자동으로 감정 분석 + 음성 인식")

    recorded = st.audio_input("녹음 시작")

    if recorded:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(recorded.read())
        tmp.flush()
        wav_path = tmp.name

        st.audio(wav_path, format="audio/wav")

        with st.spinner("SER 모델 로딩 중..."):
            extractor, model = load_ser_model()

        with st.spinner("감정 분석 중..."):
            audio = load_audio(wav_path)
            result = predict(extractor, model, audio, segment_sec)

        stt_text = None
        if use_stt:
            with st.spinner("Whisper 음성 인식 중..."):
                whisper_model = load_whisper_model()
                stt_text = transcribe(whisper_model, wav_path)

        render_result(result, segment_sec, stt_text)

# ── 탭 2: 파일 업로드 ──
with tab_file:
    st.subheader("WAV 파일 업로드")

    uploaded = st.file_uploader("WAV 파일 선택", type=["wav"])
    run_file = st.button("분석 실행", key="run_file", type="primary", use_container_width=True)

    if run_file and uploaded:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(uploaded.read())
        tmp.flush()
        wav_path = tmp.name

        st.audio(wav_path, format="audio/wav")

        with st.spinner("SER 모델 로딩 중..."):
            extractor, model = load_ser_model()

        with st.spinner("감정 분석 중..."):
            audio = load_audio(wav_path)
            result = predict(extractor, model, audio, segment_sec)

        stt_text = None
        if use_stt:
            with st.spinner("Whisper 음성 인식 중..."):
                whisper_model = load_whisper_model()
                stt_text = transcribe(whisper_model, wav_path)

        render_result(result, segment_sec, stt_text)

    elif run_file:
        st.warning("먼저 WAV 파일을 업로드하세요.")

# ── 탭 3: Sample 데이터 ──
with tab_sample:
    st.subheader("Sample 데이터 선택")

    sample_dir = Path("Sample/01.원천데이터")
    if sample_dir.exists():
        wav_files = sorted(sample_dir.glob("*.wav"))
        if wav_files:
            selected = st.selectbox("Sample 파일", wav_files, format_func=lambda x: x.name)
            run_sample = st.button("분석 실행", key="run_sample", type="primary", use_container_width=True)

            if run_sample:
                wav_path = str(selected)
                st.audio(wav_path, format="audio/wav")

                with st.spinner("SER 모델 로딩 중..."):
                    extractor, model = load_ser_model()

                with st.spinner("감정 분석 중..."):
                    audio = load_audio(wav_path)
                    result = predict(extractor, model, audio, segment_sec)

                stt_text = None
                if use_stt:
                    with st.spinner("Whisper 음성 인식 중..."):
                        whisper_model = load_whisper_model()
                        stt_text = transcribe(whisper_model, wav_path)

                render_result(result, segment_sec, stt_text)
        else:
            st.warning("Sample 폴더에 wav 파일이 없습니다.")
    else:
        st.warning("Sample/01.원천데이터 폴더가 없습니다.")
