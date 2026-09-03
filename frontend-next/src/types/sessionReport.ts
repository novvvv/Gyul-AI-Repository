export type EmotionSignal = {
  label: string;
  confidence: number;
  probs?: Record<string, number>;
};

export type SessionTurnRecord = {
  user_text: string;
  voice_emotion: EmotionSignal | null;
  face_emotion: EmotionSignal | null;
  bot_reply: string;
  at: string;
};

export type SessionSnapshot = {
  session: {
    user_id: string;
    session_id: string;
    persona_id: string;
    started_at: string;
    ended_at: string;
  };
  turns: SessionTurnRecord[];
};

export type SessionReportJson = {
  meta: {
    session_id: string;
    persona_id: string;
    started_at?: string;
    ended_at?: string;
    llm_backend: string;
    user_turn_count: number;
    bot_turn_count: number;
    voice_dominant: string[];
    face_dominant: string[];
    emotion_shifts: string[];
    mismatch_count: number;
  };
  comprehensive_report: string;
  summary: string;
  topics: string[];
  quotes: string[];
  patterns: string[];
  strengths: string[];
  reflection_questions: string[];
  next_topics: string[];
  disclaimer: string;
  turns: SessionTurnRecord[];
  aggregates: Record<string, unknown>;
  generation_error?: string;
};

export type SessionReportResponse = {
  report_json: SessionReportJson;
  report_md: string;
  llm_backend: string;
};
