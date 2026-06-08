import type { RefObject } from "react";
import type {
  DetectStatus,
  FaceExpressionSnapshot,
} from "../hooks/useFaceDetect";

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  overlayRef: RefObject<HTMLCanvasElement | null>;
  running: boolean;
  status: DetectStatus;
  faceCount: number;
  faceExpression: FaceExpressionSnapshot;
};

const STATUS_TEXT: Record<DetectStatus, string> = {
  idle: "카메라 꺼짐",
  loading: "카메라 준비 중",
  live: "얼굴 인식 중",
  error: "연결 오류",
};

export function CameraPanel({
  videoRef,
  overlayRef,
  running,
  status,
  faceCount,
  faceExpression,
}: Props) {
  const expressionText =
    running && faceExpression
      ? `${faceExpression.label} · ${(faceExpression.confidence * 100).toFixed(0)}%`
      : running
        ? "—"
        : "";
  return (
    <section className="camera-panel card">
      <div className="camera-head">
        <h2>카메라 · 얼굴 인식</h2>
        <span className={`camera-status ${status}`}>
          {STATUS_TEXT[status]}
          {running && status === "live" ? ` · ${faceCount}명` : ""}
        </span>
      </div>

      <div className="camera-stage">
        <video ref={videoRef} playsInline muted className="camera-video" />
        <canvas ref={overlayRef} className="camera-overlay" />
        {!running && (
          <div className="camera-empty">
            <span className="camera-empty-mark">◎</span>
            <p>
              <strong>시작</strong>을 누르면 카메라가 켜지고
              <br />
              인식된 얼굴에 사각형이 표시됩니다.
            </p>
          </div>
        )}
      </div>

      {running && (
        <p className="camera-expression" aria-live="polite">
          {expressionText}
        </p>
      )}
    </section>
  );
}
