export type SerPartialMessage = {
  type: "partial";
  label: string;
  confidence: number;
  probs: Record<string, number>;
};

export type SerFinalMessage = {
  type: "final";
  label: string;
  confidence: number;
  probs: Record<string, number>;
  reply?: string;
  reply_audio_b64?: string;
  reply_audio_format?: string;
  text?: string;
  session_id?: string;
  persona_id?: string;
};

/** 서버가 보내는 단독 오류 객체 (type 필드 없음) */
export type SerErrorMessage = {
  error: string;
};

export type SerWsMessage =
  | SerPartialMessage
  | SerFinalMessage
  | SerErrorMessage;

export function isSerErrorMessage(data: SerWsMessage): data is SerErrorMessage {
  return "error" in data && !("type" in data);
}

export type ChatMessage =
  | {
      id: string;
      role: "bot";
      text: string;
      meta?: string;
      emotion?: string;
    }
  | {
      id: string;
      role: "user";
      text: string;
      /** 이 발화에서 읽힌 음성 감정 — final 수신 시 채워진다 */
      voice?: { label: string; confidence: number };
      /** 같은 시점의 표정 감정 */
      face?: { label: string; confidence: number } | null;
    };
