const devDefaults = {
  apiBase: "/api",
  wsUrl: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/predict`,
};

export const API_BASE = import.meta.env.VITE_API_BASE ?? devDefaults.apiBase;
export const WS_URL = import.meta.env.VITE_WS_URL ?? devDefaults.wsUrl;

export const DEMO_SESSION = {
  userId: "demo-user",
  sessionId: `session-${Date.now()}`,
  personaId: "gyeol",
};

export const EMOTION_LABELS = [
  "happiness",
  "angry",
  "disgust",
  "fear",
  "neutral",
  "sadness",
  "surprise",
] as const;

export type EmotionLabel = (typeof EMOTION_LABELS)[number];

export const EMOTION_LABELS_KO: Record<EmotionLabel, string> = {
  happiness: "기쁨",
  angry: "분노",
  disgust: "혐오",
  fear: "두려움",
  neutral: "중립",
  sadness: "슬픔",
  surprise: "놀람",
};
