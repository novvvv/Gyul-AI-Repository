import base64
import binascii

from app.services.face_detect_service import FaceDetectService
from app.services.face_service import FaceEmotionService


class InvalidImageError(ValueError):
    pass


def decode_data_url(image: str) -> bytes:
    """data URL(`data:image/jpeg;base64,...`) 또는 raw base64 문자열을 디코딩한다."""
    raw = image.split(",", 1)[1] if "," in image else image
    try:
        return base64.b64decode(raw)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(str(exc)) from exc


def analyze_face_image(service: FaceEmotionService, image: str) -> dict:
    image_bytes = decode_data_url(image)
    if not image_bytes:
        raise InvalidImageError("empty image")
    return service.predict_from_image_bytes(image_bytes)


def detect_faces_in_image(service: FaceDetectService, image: str) -> dict:
    image_bytes = decode_data_url(image)
    if not image_bytes:
        raise InvalidImageError("empty image")
    return service.detect_from_image_bytes(image_bytes)
