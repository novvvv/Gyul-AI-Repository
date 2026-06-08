import { useCallback, useRef, useState } from "react";
import { buildWsUrl } from "../services/api";
import { EMOTION_LABELS_KO, type EmotionLabel } from "../config";
import {
  isSerErrorMessage,
  type ChatMessage,
  type SerWsMessage,
} from "../types/ser";

function floatToInt16Bytes(float32Array: Float32Array): ArrayBuffer {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out.buffer;
}

type SpeechRecognitionCtor = new () => SpeechRecognition;

function getSpeechRecognition(): SpeechRecognitionCtor | null {
  const w = window as Window & {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export type EmotionSnapshot = {
  label: string;
  confidence: number;
  probs: Record<string, number>;
  phase: "idle" | "partial" | "final";
};

export function useDemoSession(session: {
  userId: string;
  sessionId: string;
  personaId: string;
}) {
  const [status, setStatus] = useState("대기 중");
  const [running, setRunning] = useState(false);
  const [liveText, setLiveText] = useState("...");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "bot",
      text: "안녕하세요, 결입니다. 편하게 말씀해 주세요. 목소리에서 읽은 감정과 함께 대화를 이어갈게요.",
      emotion: "neutral",
    },
  ]);
  const [botEmotion, setBotEmotion] = useState<EmotionLabel>("neutral");
  const [emotion, setEmotion] = useState<EmotionSnapshot>({
    label: "neutral",
    confidence: 0,
    probs: {},
    phase: "idle",
  });

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const pendingSentencesRef = useRef<string[]>([]);

  const sendUtteranceText = useCallback((text: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "utterance_text", text }));
  }, []);

  const startSpeechToText = useCallback(() => {
    const SR = getSpeechRecognition();
    if (!SR) {
      setLiveText("(브라우저 음성인식 미지원)");
      return;
    }

    const recognition = new SR();
    recognition.lang = "ko-KR";
    recognition.interimResults = true;
    recognition.continuous = true;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          const sentence = t.trim();
          if (sentence) {
            pendingSentencesRef.current.push(sentence);
            setMessages((prev) => [
              ...prev,
              { id: `u-${Date.now()}-${i}`, role: "user", text: sentence },
            ]);
            sendUtteranceText(sentence);
          }
        } else {
          interim += t;
        }
      }
      setLiveText(interim || "...");
    };

    recognition.onerror = () => {};
    recognition.onend = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        recognition.start();
      }
    };
    recognition.start();
    recognitionRef.current = recognition;
  }, [sendUtteranceText]);

  const start = useCallback(async () => {
    try {
      const ws = new WebSocket(
        buildWsUrl({
          userId: session.userId,
          sessionId: session.sessionId,
          personaId: session.personaId,
        }),
      );
      ws.binaryType = "arraybuffer";

      ws.onopen = () => setStatus("연결됨 / 청취 중");
      ws.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        const data = JSON.parse(event.data) as SerWsMessage;

        if (isSerErrorMessage(data)) {
          setStatus(`오류: ${data.error}`);
          return;
        }

        // partial(중간 결과)은 무시 — 감정은 발화 완료(final) 시 한 번만 갱신
        if (data.type === "partial") return;

        if (data.type !== "final") return;

        setEmotion({
          label: data.label,
          confidence: data.confidence,
          probs: data.probs,
          phase: "final",
        });

        const sentence = pendingSentencesRef.current.shift();
        if (!sentence) return;

        setMessages((prev) => [
          ...prev,
          {
            id: `b-${Date.now()}`,
            role: "bot",
            text: data.reply ?? "응답을 생성하지 못했어요.",
            emotion: data.label,
            meta: `감정 ${EMOTION_LABELS_KO[data.label as keyof typeof EMOTION_LABELS_KO] ?? data.label} · ${((data.confidence ?? 0) * 100).toFixed(0)}%`,
          },
        ]);
        setBotEmotion(
          data.label in EMOTION_LABELS_KO
            ? (data.label as EmotionLabel)
            : "neutral",
        );
      };

      ws.onerror = () => setStatus("WebSocket 오류");
      ws.onclose = () => setStatus("WebSocket 종료");

      wsRef.current = ws;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioCtx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);
        wsRef.current.send(floatToInt16Bytes(input));
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      processorRef.current = processor;

      setRunning(true);
      startSpeechToText();
    } catch (err) {
      setStatus(`시작 실패: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [session, startSpeechToText]);

  const stop = useCallback(async () => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    if (audioCtxRef.current) await audioCtxRef.current.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("flush");
    }
    wsRef.current?.close();

    recognitionRef.current?.stop();

    wsRef.current = null;
    streamRef.current = null;
    audioCtxRef.current = null;
    processorRef.current = null;
    sourceRef.current = null;
    recognitionRef.current = null;
    pendingSentencesRef.current = [];

    setLiveText("...");
    setRunning(false);
    setStatus("중지됨");
    setEmotion((e) => ({ ...e, phase: "idle" }));
  }, []);

  return {
    status,
    running,
    liveText,
    messages,
    emotion,
    botEmotion,
    start,
    stop,
  };
}
