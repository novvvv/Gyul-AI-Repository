from typing import Any

_insight_app: Any = None


class FaceDetectService:
    """InsightFace 기반 얼굴 검출 (kyul_mvp interview_engine과 동일 계열).

    DeepFace opencv 백엔드보다 정확도가 높다. bbox만 반환한다.
    """

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        global _insight_app
        if _insight_app is None:
            from insightface.app import FaceAnalysis

            _insight_app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection"],
            )
            _insight_app.prepare(ctx_id=-1, det_size=(640, 640))
        self._loaded = True

    def detect_from_image_bytes(self, image_bytes: bytes) -> dict:
        import cv2
        import numpy as np

        if not self.is_loaded:
            self.load()

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid image")

        height, width = frame.shape[:2]
        faces_raw = _insight_app.get(frame)

        faces: list[dict[str, Any]] = []
        for face in faces_raw:
            x1, y1, x2, y2 = face.bbox.astype(int)
            w = int(x2 - x1)
            h = int(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            score = float(getattr(face, "det_score", 0) or 0)
            faces.append(
                {
                    "x": int(x1),
                    "y": int(y1),
                    "w": w,
                    "h": h,
                    "confidence": score,
                }
            )

        return {"faces": faces, "width": width, "height": height}


face_detect_service = FaceDetectService()
