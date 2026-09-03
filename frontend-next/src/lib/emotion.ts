/**
 * 감정 라벨 ↔ 색·한글 이름.
 *
 * 색은 **라벨에 고정**한다. 정렬 순서나 인덱스로 배정하지 않는다 —
 * 그렇게 하면 음성 차트와 표정 차트에서 같은 감정이 다른 색으로 칠해진다.
 * (기존 frontend 의 `countsToSlices` + `CHART_COLORS[i]` 가 그 문제를 갖고 있다.)
 *
 * 원문: docs/design.md §1.6
 */

export const EMOTION_ORDER = [
  "happiness",
  "fear",
  "sadness",
  "disgust",
  "angry",
  "surprise",
  "neutral",
] as const;

export type EmotionKey = (typeof EMOTION_ORDER)[number];

const KO: Record<EmotionKey, string> = {
  happiness: "기쁨",
  fear: "두려움",
  sadness: "슬픔",
  disgust: "혐오",
  angry: "분노",
  surprise: "놀람",
  neutral: "중립",
};

/** FER 계열 라벨(표정 모델)을 SER 라벨로 정규화 */
const ALIAS: Record<string, EmotionKey> = {
  happy: "happiness",
  happiness: "happiness",
  sad: "sadness",
  sadness: "sadness",
  angry: "angry",
  anger: "angry",
  fear: "fear",
  fearful: "fear",
  disgust: "disgust",
  surprise: "surprise",
  surprised: "surprise",
  neutral: "neutral",
};

export function normalize(label: string | undefined | null): EmotionKey {
  if (!label) return "neutral";
  return ALIAS[label.toLowerCase()] ?? "neutral";
}

/** 감정 이름 (한글) */
export function labelKo(label: string | undefined | null): string {
  if (!label) return "—";
  const key = ALIAS[label.toLowerCase()];
  return key ? KO[key] : label;
}

/** 감정 색 (CSS 변수 참조 — 테마를 따라간다) */
export function emotionColor(label: string | undefined | null): string {
  return `var(--e-${normalize(label)})`;
}

export function pct(confidence: number | undefined | null): string {
  if (confidence == null) return "—";
  return `${Math.round(confidence * 100)}%`;
}

/** counts 를 비중 슬라이스로. 색은 라벨에서 나온다. */
export type Slice = {
  label: string;
  ko: string;
  color: string;
  count: number;
  pct: number;
};

export function toSlices(
  counts: Record<string, number> | undefined,
  totalFallback = 0,
): Slice[] {
  const entries = Object.entries(counts ?? {});
  const total =
    entries.reduce((sum, [, v]) => sum + v, 0) || totalFallback || 1;

  return entries
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => ({
      label,
      ko: labelKo(label),
      color: emotionColor(label),
      count,
      pct: Math.round((count / total) * 100),
    }));
}
