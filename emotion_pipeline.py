"""
멀티모달 감정 인식 파이프라인

SER (Wav2Vec2 음성 기반) + TER (Whisper STT + Claude 텍스트 기반) 결합
→ 노이즈 환경에서는 TER 가중치를 높이는 적응형 로직 포함

감정 클래스: happiness, angry, disgust, fear, neutral, sadness, surprise
"""

import numpy as np
import librosa
from pathlib import Path

from ser_wav2vec import SERModel
from ter_whisper_claude import TERPipeline

EMOTION_LABELS = ["happiness", "angry", "disgust", "fear", "neutral", "sadness", "surprise"]

SER_TO_UNIFIED = {
    "hap": "happiness",
    "happy": "happiness",
    "ang": "angry",
    "angry": "angry",
    "dis": "disgust",
    "disgust": "disgust",
    "fea": "fear",
    "fear": "fear",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "sur": "surprise",
    "surprise": "surprise",
}


def estimate_snr(wav_path: str, sr: int = 16000) -> float:
    """
    간이 SNR 추정
    노이즈가 심하면 낮은 값 → TER 가중치를 높이는 데 사용
    """
    y, _ = librosa.load(wav_path, sr=sr)
    rms = librosa.feature.rms(y=y).flatten()
    if len(rms) == 0:
        return 0.0

    signal_power = np.percentile(rms, 90) ** 2
    noise_power = np.percentile(rms, 10) ** 2 + 1e-10

    return float(10 * np.log10(signal_power / noise_power))


def adaptive_weights(snr_db: float, base_ser: float = 0.5) -> tuple[float, float]:
    """
    SNR 기반 적응형 가중치
    - SNR 높음(깨끗한 음성): SER 가중치 ↑
    - SNR 낮음(노이즈 심함): TER 가중치 ↑
    """
    if snr_db >= 20:
        ser_w = 0.6
    elif snr_db >= 10:
        ser_w = 0.5
    elif snr_db >= 5:
        ser_w = 0.35
    else:
        ser_w = 0.2

    return ser_w, 1.0 - ser_w


def normalize_label(label: str) -> str:
    return SER_TO_UNIFIED.get(label.lower().strip(), "neutral")


class EmotionPipeline:
    """SER + TER 멀티모달 감정 인식"""

    def __init__(
        self,
        ser_model_name: str = "superb/wav2vec2-base-superb-er",
        whisper_size: str = "base",
        claude_api_key: str = None,
        claude_model: str = "claude-3-5-haiku-latest",
        segment_sec: float = 3.0,
    ):
        self.ser = SERModel(model_name=ser_model_name)
        self.ter = TERPipeline(
            whisper_size=whisper_size,
            claude_api_key=claude_api_key,
            claude_model=claude_model,
        )
        self.segment_sec = segment_sec

    def analyze(self, wav_path: str) -> dict:
        snr = estimate_snr(wav_path)
        ser_w, ter_w = adaptive_weights(snr)

        ser_result = self.ser.predict_file(wav_path, segment_sec=self.segment_sec)
        ter_result = self.ter.analyze(wav_path)

        ser_emotion = normalize_label(ser_result["predicted_emotion"])
        ter_emotion = normalize_label(ter_result.get("emotion", "neutral"))

        ser_conf = ser_result["confidence"]
        ter_conf = ter_result.get("confidence", 0.0) or 0.0

        scores = {}
        for label in EMOTION_LABELS:
            ser_score = ser_result["all_scores"].get(
                ser_emotion if label == ser_emotion else "", 0.0
            )
            for k, v in ser_result["all_scores"].items():
                if normalize_label(k) == label:
                    ser_score = max(ser_score, v)

            ter_score = 1.0 if label == ter_emotion else 0.0
            scores[label] = ser_w * ser_score + ter_w * ter_score * ter_conf

        final_emotion = max(scores, key=scores.get)

        return {
            "final_emotion": final_emotion,
            "final_confidence": scores[final_emotion],
            "combined_scores": scores,
            "ser": {
                "emotion": ser_emotion,
                "confidence": ser_conf,
                "weight": ser_w,
                "raw_scores": ser_result["all_scores"],
            },
            "ter": {
                "emotion": ter_emotion,
                "confidence": ter_conf,
                "weight": ter_w,
                "transcribed_text": ter_result.get("transcribed_text", ""),
                "reason": ter_result.get("reason", ""),
            },
            "snr_db": snr,
            "duration_sec": ser_result.get("duration_sec", 0),
            "segments": ser_result.get("segments", 0),
        }


def batch_analyze(pipeline: EmotionPipeline, wav_dir: str, limit: int = None) -> list[dict]:
    wav_files = sorted(Path(wav_dir).glob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]

    results = []
    for wf in wav_files:
        print(f"  분석 중: {wf.name}")
        try:
            result = pipeline.analyze(str(wf))
            result["filename"] = wf.name
            results.append(result)
        except Exception as e:
            print(f"  [에러] {wf.name}: {e}")
            results.append({"filename": wf.name, "error": str(e)})
    return results


if __name__ == "__main__":
    print("=== 멀티모달 감정 인식 파이프라인 ===\n")
    print("사용법:")
    print("  from emotion_pipeline import EmotionPipeline")
    print()
    print("  pipeline = EmotionPipeline(")
    print("      whisper_size='base',")
    print("      claude_api_key='your-key',")
    print("      segment_sec=3.0,")
    print("  )")
    print()
    print("  result = pipeline.analyze('Sample/01.원천데이터/A-A2-A-009-0101.wav')")
    print("  print(result['final_emotion'])")
    print("  print(result['ser'])  # 음성 기반 결과")
    print("  print(result['ter'])  # 텍스트 기반 결과")
    print()
    print("배치 분석:")
    print("  from emotion_pipeline import batch_analyze")
    print("  results = batch_analyze(pipeline, 'Sample/01.원천데이터', limit=10)")
