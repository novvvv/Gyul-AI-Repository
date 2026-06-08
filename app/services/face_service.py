from io import BytesIO
from typing import Any

from app.core.config import FACE_MODEL_ID


class FaceEmotionService:
    """PyTorch(transformers) 기반 얼굴 표정 감정 인식 서비스.

    kyul_mvp의 DeepFace(TensorFlow) 파이프라인을 대체해, 기존 torch/transformers
    스택과 동일하게 동작하는 ViT 표정 분류기를 사용한다. (FER 7감정)
    """

    def __init__(self, model_id: str = FACE_MODEL_ID) -> None:
        self.model_id = model_id
        self.processor: Any = None
        self.model: Any = None
        self.device: Any = None

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.device = _select_device(torch)
        self.model.to(self.device)
        self.model.eval()

    def predict_from_image_bytes(self, image_bytes: bytes) -> dict:
        import torch
        from PIL import Image

        if not self.is_loaded:
            raise RuntimeError("Face model not loaded")

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = _center_square_crop(image)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())

        id2label = self.model.config.id2label
        label = id2label[pred_id]
        confidence = float(probs[pred_id].item())
        all_probs = {id2label[i]: float(probs[i].item()) for i in range(len(probs))}
        return {"label": label, "confidence": confidence, "probs": all_probs}


def _select_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _center_square_crop(image: Any) -> Any:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


face_service = FaceEmotionService()
