import { API_BASE } from "../config";
import type { SessionReportResponse, SessionSnapshot } from "../types/sessionReport";

export type HealthResponse = {
  ok: boolean;
  llm_loaded?: boolean;
  llm_provider?: string;
  face_enabled?: boolean;
  face_loaded?: boolean;
  text_llm_backend?: string;
  text_llm_loaded?: boolean;
};

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`health ${res.status}`);
  }
  return res.json() as Promise<HealthResponse>;
}

export type FaceBox = {
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
};

export type FaceDetectResponse = {
  faces: FaceBox[];
  width: number;
  height: number;
};

export async function detectFaces(
  imageDataUrl: string,
): Promise<FaceDetectResponse> {
  const res = await fetch(`${API_BASE}/detect_face`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl }),
  });
  if (!res.ok) {
    throw new Error(`detect ${res.status}`);
  }
  return res.json() as Promise<FaceDetectResponse>;
}

export type FaceExpressionResponse = {
  label: string;
  confidence: number;
  probs?: Record<string, number>;
};

export async function predictFaceExpression(
  imageDataUrl: string,
): Promise<FaceExpressionResponse> {
  const res = await fetch(`${API_BASE}/predict_face`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl }),
  });
  if (!res.ok) {
    throw new Error(`predict_face ${res.status}`);
  }
  return res.json() as Promise<FaceExpressionResponse>;
}

export async function requestSessionReport(
  snapshot: SessionSnapshot,
): Promise<SessionReportResponse> {
  const res = await fetch(`${API_BASE}/session/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(snapshot),
  });
  if (!res.ok) {
    throw new Error(`report ${res.status}`);
  }
  return res.json() as Promise<SessionReportResponse>;
}

export function buildWsUrl(params: {
  userId: string;
  sessionId: string;
  personaId: string;
}): string {
  const base = import.meta.env.VITE_WS_URL ?? `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/predict`;
  const url = new URL(base);
  url.searchParams.set("user_id", params.userId);
  url.searchParams.set("session_id", params.sessionId);
  url.searchParams.set("persona_id", params.personaId);
  return url.toString();
}
