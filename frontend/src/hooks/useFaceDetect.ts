import { useCallback, useRef, useState } from "react";
import {
  detectFaces,
  predictFaceExpression,
  type FaceBox,
} from "../services/api";

export type FaceExpressionSnapshot = {
  label: string;
  confidence: number;
} | null;

const DETECT_INTERVAL_MS = 500;
const JPEG_QUALITY = 0.82;

export type DetectStatus = "idle" | "loading" | "live" | "error";

/**
 * 웹캠 프레임을 주기적으로 캡처해 /detect_face(InsightFace)로 보내고,
 * 검출된 얼굴 bounding box를 오버레이 캔버스에 사각형으로 그린다.
 * 감정 분류 UI 없이 "얼굴 위치 표시"만 담당. 음성 세션과 분리.
 */
export function useFaceDetect() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<DetectStatus>("idle");
  const [faceCount, setFaceCount] = useState(0);
  const [faceExpression, setFaceExpression] =
    useState<FaceExpressionSnapshot>(null);

  const drawBoxes = useCallback(
    (boxes: FaceBox[], srcW: number, srcH: number) => {
      const overlay = overlayRef.current;
      if (!overlay) return;
      if (overlay.width !== srcW || overlay.height !== srcH) {
        overlay.width = srcW;
        overlay.height = srcH;
      }
      const ctx = overlay.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      ctx.lineWidth = Math.max(2, Math.round(srcW / 240));
      ctx.strokeStyle = "#22c55e";
      ctx.lineJoin = "miter";

      for (const b of boxes) {
        const pad = 4;
        const side = Math.max(b.w, b.h) + pad * 2;
        const cx = b.x + b.w / 2;
        const cy = b.y + b.h / 2;
        const x = Math.round(cx - side / 2);
        const y = Math.round(cy - side / 2);
        ctx.strokeRect(x, y, side, side);
      }
    },
    [],
  );

  const captureAndDetect = useCallback(async () => {
    const video = videoRef.current;
    if (busyRef.current || !video || !video.videoWidth) return;
    busyRef.current = true;
    try {
      const canvas = captureCanvasRef.current ?? document.createElement("canvas");
      captureCanvasRef.current = canvas;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", JPEG_QUALITY);

      const result = await detectFaces(image);
      drawBoxes(result.faces, result.width, result.height);
      setFaceCount(result.faces.length);

      if (result.faces.length > 0) {
        try {
          const expr = await predictFaceExpression(image);
          setFaceExpression({
            label: expr.label,
            confidence: expr.confidence,
          });
        } catch {
          setFaceExpression(null);
        }
      } else {
        setFaceExpression(null);
      }

      setStatus("live");
    } catch {
      setStatus("error");
    } finally {
      busyRef.current = false;
    }
  }, [drawBoxes]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setRunning(true);
      setStatus("loading");
      timerRef.current = window.setInterval(() => {
        void captureAndDetect();
      }, DETECT_INTERVAL_MS);
    } catch {
      setStatus("error");
    }
  }, [captureAndDetect]);

  const stop = useCallback(() => {
    if (timerRef.current != null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const overlay = overlayRef.current;
    const ctx = overlay?.getContext("2d");
    if (overlay && ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    busyRef.current = false;
    setRunning(false);
    setStatus("idle");
    setFaceCount(0);
    setFaceExpression(null);
  }, []);

  return {
    videoRef,
    overlayRef,
    running,
    status,
    faceCount,
    faceExpression,
    start,
    stop,
  };
}
