"""
TER (Text Emotion Recognition) - Whisper STT + Claude API 기반 감정 분석 모듈

흐름: wav → Whisper(STT) → 텍스트 → Claude(감정 분류) → 감정 라벨
감정 클래스: happiness, angry, disgust, fear, neutral, sadness, surprise
"""

import json
from pathlib import Path

import whisper
import anthropic

EMOTION_LABELS = ["happiness", "angry", "disgust", "fear", "neutral", "sadness", "surprise"]

EMOTION_PROMPT = """당신은 한국어 감정 분석 전문가입니다.
아래 대화 텍스트의 감정을 반드시 다음 7개 중 하나로 분류하세요:
happiness, angry, disgust, fear, neutral, sadness, surprise

반드시 아래 JSON 형식으로만 응답하세요:
{{"emotion": "감정라벨", "confidence": 0.0~1.0, "reason": "판단 근거 한 줄"}}

텍스트: {text}"""


class WhisperSTT:
    def __init__(self, model_size: str = "base"):
        """model_size: tiny, base, small, medium, large"""
        self.model = whisper.load_model(model_size)

    def transcribe(self, wav_path: str, language: str = "ko") -> dict:
        import librosa
        import numpy as np

        audio = librosa.load(wav_path, sr=16000, mono=True)[0].astype(np.float32)
        result = self.model.transcribe(audio, language=language)
        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }


class ClaudeEmotionClassifier:
    def __init__(self, api_key: str = None, model: str = "claude-3-5-haiku-latest"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "reason": "빈 텍스트",
            }

        prompt = EMOTION_PROMPT.format(text=text)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        try:
            result = json.loads(raw)
            if result.get("emotion") not in EMOTION_LABELS:
                result["emotion"] = "neutral"
                result["confidence"] = 0.0
            return result
        except json.JSONDecodeError:
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "reason": f"파싱 실패: {raw[:100]}",
            }


class TERPipeline:
    """Whisper STT → Claude 감정 분류 통합 파이프라인"""

    def __init__(
        self,
        whisper_size: str = "base",
        claude_api_key: str = None,
        claude_model: str = "claude-3-5-haiku-latest",
    ):
        self.stt = WhisperSTT(model_size=whisper_size)
        self.classifier = ClaudeEmotionClassifier(
            api_key=claude_api_key, model=claude_model
        )

    def analyze(self, wav_path: str) -> dict:
        stt_result = self.stt.transcribe(wav_path)
        text = stt_result["text"]

        emotion_result = self.classifier.classify(text)

        return {
            "wav_path": wav_path,
            "transcribed_text": text,
            "stt_segments": stt_result["segments"],
            "emotion": emotion_result.get("emotion"),
            "confidence": emotion_result.get("confidence"),
            "reason": emotion_result.get("reason", ""),
        }

    def analyze_from_text(self, text: str) -> dict:
        """이미 텍스트가 있는 경우 (라벨 데이터의 tr 필드 등)"""
        emotion_result = self.classifier.classify(text)
        return {
            "input_text": text,
            "emotion": emotion_result.get("emotion"),
            "confidence": emotion_result.get("confidence"),
            "reason": emotion_result.get("reason", ""),
        }


if __name__ == "__main__":
    sample_dir = Path("Sample/01.원천데이터")
    wav_files = sorted(sample_dir.glob("*.wav"))[:3]

    if not wav_files:
        print("Sample wav 파일이 없습니다.")
    else:
        print("=== TER (Whisper + Claude) 감정 분석 ===\n")

        print("[1] Whisper STT 테스트")
        stt = WhisperSTT(model_size="base")
        for wf in wav_files:
            result = stt.transcribe(str(wf))
            print(f"  {wf.name} → \"{result['text']}\"")

        print("\n[2] Claude 감정 분류 (API 키 필요)")
        print("  사용법:")
        print("    export ANTHROPIC_API_KEY='your-key'")
        print("    pipeline = TERPipeline()")
        print("    result = pipeline.analyze('audio.wav')")
        print("    print(result)")
