import { EMOTION_LABELS_KO } from "../config";
import type { EmotionSnapshot } from "../hooks/useDemoSession";
import type { EmotionLabel } from "../config";

type Props = {
  emotion: EmotionSnapshot;
};

function labelKo(key: string): string {
  if (key in EMOTION_LABELS_KO) {
    return EMOTION_LABELS_KO[key as EmotionLabel];
  }
  return key;
}

export function EmotionPanel({ emotion }: Props) {
  const isActive = emotion.phase === "final";

  const text = isActive
    ? `${labelKo(emotion.label)} · ${(emotion.confidence * 100).toFixed(0)}%`
    : "말이 끝나면 감정이 표시됩니다";

  return (
    <p className={`emotion-line ${isActive ? "is-active" : ""}`}>{text}</p>
  );
}
