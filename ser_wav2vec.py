"""
SER (Speech Emotion Recognition) - Wav2Vec2 기반 감정 인식 모듈

감정 클래스: happiness, angry, disgust, fear, neutral, sadness, surprise
모델: superb/wav2vec2-base-superb-er (영어 기반 baseline)
     → 한국어 파인튜닝 시 wav2vec-large-xlsr-korean 기반으로 전이학습 권장
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

EMOTION_LABELS = ["neutral", "happy", "angry", "sad", "disgust", "fear", "surprise"]
MODEL_NAME = "superb/wav2vec2-base-superb-er"
TARGET_SR = 16000


class SERModel:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def load_audio(self, wav_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
        y, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        return np.asarray(y, dtype=np.float32)

    def segment_audio(
        self, audio: np.ndarray, sr: int = TARGET_SR, segment_sec: float = 3.0
    ) -> list[np.ndarray]:
        """오디오를 segment_sec 단위로 분할 (1~3초 최적 구간 탐색용)"""
        segment_len = int(sr * segment_sec)
        segments = []
        for start in range(0, len(audio), segment_len):
            seg = audio[start : start + segment_len]
            if len(seg) >= int(sr * 0.5):
                segments.append(seg)
        return segments

    def predict(self, audio: np.ndarray) -> dict:
        inputs = self.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))

        id2label = self.model.config.id2label
        labels = [id2label.get(i, f"label_{i}") for i in range(len(probs))]

        return {
            "predicted_emotion": labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_scores": {labels[i]: float(probs[i]) for i in range(len(probs))},
        }

    def predict_file(self, wav_path: str, segment_sec: float = 3.0) -> dict:
        """
        wav 파일 → 감정 예측
        segment_sec: 분석 단위 (초). 1~3초 사이 권장
        """
        audio = self.load_audio(wav_path)
        total_duration = len(audio) / TARGET_SR

        if total_duration <= segment_sec:
            result = self.predict(audio)
            result["duration_sec"] = total_duration
            result["segments"] = 1
            return result

        segments = self.segment_audio(audio, segment_sec=segment_sec)
        all_scores = []

        for seg in segments:
            res = self.predict(seg)
            all_scores.append(res["all_scores"])

        labels = list(all_scores[0].keys())
        avg_scores = {}
        for label in labels:
            avg_scores[label] = float(np.mean([s[label] for s in all_scores]))

        pred_emotion = max(avg_scores, key=avg_scores.get)

        return {
            "predicted_emotion": pred_emotion,
            "confidence": avg_scores[pred_emotion],
            "all_scores": avg_scores,
            "duration_sec": total_duration,
            "segments": len(segments),
        }


def find_optimal_segment_length(
    model: SERModel, wav_path: str, candidates: list[float] = None
) -> dict:
    """1~3초 사이에서 최적 세그먼트 길이를 탐색"""
    if candidates is None:
        candidates = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = {}
    for sec in candidates:
        res = model.predict_file(wav_path, segment_sec=sec)
        results[sec] = {
            "emotion": res["predicted_emotion"],
            "confidence": res["confidence"],
            "segments": res["segments"],
        }
    return results


if __name__ == "__main__":
    sample_dir = Path("Sample/01.원천데이터")
    wav_files = sorted(sample_dir.glob("*.wav"))[:3]

    if not wav_files:
        print("Sample wav 파일이 없습니다.")
    else:
        print("=== SER (Wav2Vec2) 감정 인식 ===\n")
        model = SERModel()

        for wf in wav_files:
            print(f"[{wf.name}]")
            result = model.predict_file(str(wf), segment_sec=3.0)
            print(f"  감정: {result['predicted_emotion']} ({result['confidence']:.2%})")
            print(f"  길이: {result['duration_sec']:.1f}초, 세그먼트: {result['segments']}")
            print(f"  전체: {result['all_scores']}")
            print()

        print("=== 최적 세그먼트 길이 탐색 ===\n")
        opt = find_optimal_segment_length(model, str(wav_files[0]))
        for sec, info in opt.items():
            print(f"  {sec}초 → {info['emotion']} (conf={info['confidence']:.2%}, seg={info['segments']})")
