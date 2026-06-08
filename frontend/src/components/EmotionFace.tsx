import type { EmotionLabel } from "../config";
import { EMOTION_LABELS_KO } from "../config";

type Props = {
  emotion: EmotionLabel | string;
  size?: "sm" | "md" | "lg";
  className?: string;
  showLabel?: boolean;
};

const SIZES = { sm: 40, md: 72, lg: 96 } as const;

function isEmotionLabel(v: string): v is EmotionLabel {
  return v in EMOTION_LABELS_KO;
}

export function EmotionFace({
  emotion,
  size = "md",
  className = "",
  showLabel = false,
}: Props) {
  const px = SIZES[size];
  const label = isEmotionLabel(emotion) ? EMOTION_LABELS_KO[emotion] : emotion;
  const key = isEmotionLabel(emotion) ? emotion : "neutral";

  return (
    <div
      className={`emotion-face emotion-face--${key} emotion-face--${size} ${className}`.trim()}
      aria-label={`공감 표정: ${label}`}
    >
      <svg
        width={px}
        height={px}
        viewBox="0 0 80 80"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden
      >
        <circle cx="40" cy="40" r="38" className="face-bg" />
        <g className="face-features">{renderFeatures(key)}</g>
      </svg>
      {showLabel && <span className="emotion-face-label">{label}</span>}
    </div>
  );
}

function renderFeatures(emotion: EmotionLabel | "neutral") {
  switch (emotion) {
    case "happiness":
      return (
        <>
          <path d="M22 38c2-4 8-4 12 0" className="face-eye-happy" />
          <path d="M46 38c2-4 8-4 12 0" className="face-eye-happy" />
          <path
            d="M26 52c4 8 10 12 14 12s10-4 14-12"
            className="face-mouth face-mouth-happy"
          />
        </>
      );
    case "angry":
      return (
        <>
          <path d="M22 28l12 4" className="face-brow face-brow-angry-l" />
          <path d="M58 28l-12 4" className="face-brow face-brow-angry-r" />
          <circle cx="28" cy="40" r="3" className="face-eye-dot" />
          <circle cx="52" cy="40" r="3" className="face-eye-dot" />
          <path d="M28 56c6-4 18-4 24 0" className="face-mouth face-mouth-angry" />
        </>
      );
    case "disgust":
      return (
        <>
          <path d="M22 30l14 2" className="face-brow" />
          <path d="M58 28l-10 6" className="face-brow face-brow-disgust" />
          <circle cx="28" cy="42" r="3" className="face-eye-dot" />
          <circle cx="52" cy="40" r="3.5" className="face-eye-dot" />
          <path d="M30 54c4-6 16-6 20 0" className="face-mouth face-mouth-disgust" />
        </>
      );
    case "fear":
      return (
        <>
          <path d="M20 26c4 2 10 2 16 0" className="face-brow face-brow-fear" />
          <path d="M44 26c6 2 12 2 16 0" className="face-brow face-brow-fear" />
          <ellipse cx="28" cy="40" rx="5" ry="6" className="face-eye-wide" />
          <ellipse cx="52" cy="40" rx="5" ry="6" className="face-eye-wide" />
          <ellipse cx="40" cy="56" rx="6" ry="8" className="face-mouth face-mouth-fear" />
        </>
      );
    case "sadness":
      return (
        <>
          <path d="M22 30c4-2 10-2 14 0" className="face-brow face-brow-sad" />
          <path d="M44 30c4-2 10-2 14 0" className="face-brow face-brow-sad" />
          <circle cx="28" cy="42" r="3" className="face-eye-dot" />
          <circle cx="52" cy="42" r="3" className="face-eye-dot" />
          <path d="M28 58c6 6 18 6 24 0" className="face-mouth face-mouth-sad" />
          <circle cx="56" cy="48" r="2" className="face-tear" />
        </>
      );
    case "surprise":
      return (
        <>
          <path d="M22 28h16" className="face-brow" />
          <path d="M42 28h16" className="face-brow" />
          <circle cx="28" cy="40" r="5" className="face-eye-wide" />
          <circle cx="52" cy="40" r="5" className="face-eye-wide" />
          <ellipse cx="40" cy="56" rx="7" ry="9" className="face-mouth face-mouth-surprise" />
        </>
      );
    case "neutral":
    default:
      return (
        <>
          <path d="M24 30h12" className="face-brow" />
          <path d="M44 30h12" className="face-brow" />
          <circle cx="28" cy="42" r="3" className="face-eye-dot" />
          <circle cx="52" cy="42" r="3" className="face-eye-dot" />
          <path d="M30 54h20" className="face-mouth face-mouth-neutral" />
        </>
      );
  }
}

export function resolveEmotionLabel(raw: string | undefined): EmotionLabel {
  if (raw && isEmotionLabel(raw)) return raw;
  return "neutral";
}
