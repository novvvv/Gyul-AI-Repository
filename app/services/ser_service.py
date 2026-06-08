from io import BytesIO
from typing import Any

from app.core.config import MODEL_DIR, TARGET_SR


class SERService:
    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.feature_extractor: Any = None
        self.model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.feature_extractor is not None and self.model is not None

    def load(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_dir)
        self.model.eval()

    def predict_from_audio(self, audio: Any) -> dict:
        import torch

        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        inputs = self.feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())

        label = self.model.config.id2label[pred_id]
        confidence = float(probs[pred_id].item())
        all_probs = {
            self.model.config.id2label[i]: float(probs[i].item())
            for i in range(len(probs))
        }
        return {"label": label, "confidence": confidence, "probs": all_probs}

    def predict_from_file_bytes(self, audio_bytes: bytes) -> dict:
        import librosa

        audio, _ = librosa.load(BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
        if len(audio) == 0:
            raise ValueError("Empty audio")
        return self.predict_from_audio(audio)


ser_service = SERService()
