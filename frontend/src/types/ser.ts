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
  | { id: string; role: "user"; text: string };
