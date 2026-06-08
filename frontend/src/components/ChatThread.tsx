import { useEffect, useRef } from "react";
import { EmotionFace, resolveEmotionLabel } from "./EmotionFace";
import type { ChatMessage } from "../types/ser";

type Props = {
  messages: ChatMessage[];
  liveText: string;
};

export function ChatThread({ messages, liveText }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <section className="chat-thread card">
      <h2>대화 기록</h2>
      <div className="messages">
        {messages.map((m) =>
          m.role === "bot" ? (
            <div key={m.id} className="bubble bot">
              <div className="bubble-row">
                <EmotionFace
                  emotion={resolveEmotionLabel(m.emotion)}
                  size="sm"
                />
                <div className="bubble-content">
                  {m.text}
                  {m.meta && <div className="meta">{m.meta}</div>}
                </div>
              </div>
            </div>
          ) : (
            <div key={m.id} className="bubble user">
              {m.text}
            </div>
          ),
        )}
        <div ref={endRef} />
      </div>
      <footer className="live">말하는 중 · {liveText}</footer>
    </section>
  );
}
