import { EMOTION_LABELS_KO, type EmotionLabel } from "../config";
import type { SessionReportJson, SessionTurnRecord } from "../types/sessionReport";

const FACE_LABELS_KO: Record<string, string> = {
  angry: "분노",
  disgust: "혐오",
  fear: "두려움",
  happy: "기쁨",
  neutral: "중립",
  sad: "슬픔",
  surprise: "놀람",
};

export function labelKo(label: string): string {
  if (label in EMOTION_LABELS_KO) {
    return EMOTION_LABELS_KO[label as EmotionLabel];
  }
  return FACE_LABELS_KO[label] ?? label;
}

export function formatSessionRange(start?: string, end?: string): string {
  if (!start || !end) return "-";
  const s = new Date(start);
  const e = new Date(end);
  if (Number.isNaN(s.getTime()) || Number.isNaN(e.getTime())) {
    return `${start} ~ ${end}`;
  }
  const mins = Math.max(1, Math.round((e.getTime() - s.getTime()) / 60000));
  return `${s.toLocaleString("ko-KR")} ~ ${e.toLocaleString("ko-KR", { hour: "2-digit", minute: "2-digit" })} · ${mins}분`;
}

export function buildBarSeries(turns: SessionTurnRecord[]) {
  return turns.map((turn, i) => ({
    id: `${i}-${turn.at}`,
    label: `${i + 1}`,
    value: Math.round((turn.voice_emotion?.confidence ?? 0) * 100),
    emotion: turn.voice_emotion?.label ?? "neutral",
  }));
}

export function countsToSlices(
  counts: Record<string, number> | undefined,
  totalFallback: number,
) {
  const entries = Object.entries(counts ?? {});
  const total = entries.reduce((sum, [, v]) => sum + v, 0) || totalFallback || 1;
  return entries
    .sort((a, b) => b[1] - a[1])
    .map(([label, count], index) => ({
      label,
      labelKo: labelKo(label),
      count,
      pct: Math.round((count / total) * 100),
      color: DONUT_COLORS[index % DONUT_COLORS.length],
    }));
}

const DONUT_COLORS = [
  "#3B7EA1",
  "#6BA3C7",
  "#A3C1D4",
  "#5C6B7A",
  "#8E99A4",
  "#B8C5CE",
  "#D4DCE3",
];

export function dominantLabel(
  report: SessionReportJson,
  kind: "voice" | "face",
): string {
  const list =
    kind === "voice"
      ? report.meta.voice_dominant
      : report.meta.face_dominant;
  return list?.[0] ? labelKo(list[0]) : "-";
}

export function buildRankedQuotes(quotes: string[], turns: SessionTurnRecord[]) {
  if (quotes.length) {
    return quotes.map((text, i) => ({ rank: i + 1, text }));
  }
  return turns
    .filter((t) => t.user_text)
    .slice(0, 5)
    .map((t, i) => ({ rank: i + 1, text: t.user_text }));
}
