import { EmotionFace, resolveEmotionLabel } from "./EmotionFace";
import { EMOTION_LABELS_KO } from "../config";
import type { EmotionLabel } from "../config";

type Props = {
  emotion: EmotionLabel;
  speaking?: boolean;
};

export function AiPersonaPanel({ emotion, speaking = false }: Props) {
  const label = EMOTION_LABELS_KO[emotion];

  return (
    <section className="ai-persona">
      <p className="ai-persona-tag">AI 페르소나 · 결</p>
      <div className={`ai-persona-face-wrap ${speaking ? "is-speaking" : ""}`}>
        <EmotionFace emotion={emotion} size="lg" />
      </div>
      <p className="ai-persona-emotion">
        공감 표정 · <strong>{label}</strong>
      </p>
      <p className="ai-persona-desc">
        {speaking
          ? "지금 말한 감정에 맞춰 표정을 바꿔 응답하고 있어요."
          : "대화가 이어지면 감정에 맞는 표정으로 공감해 드려요."}
      </p>
    </section>
  );
}

export { resolveEmotionLabel };
