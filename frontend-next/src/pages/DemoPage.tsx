import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Character, type Mood } from "../components/Character";
import { DEMO_SESSION } from "../config";
import { TTS_RATE_OPTIONS, useDemoSession } from "../hooks/useDemoSession";
import { useFaceDetect } from "../hooks/useFaceDetect";
import { useHealth } from "../hooks/useHealth";
import { synthesizeSpeech } from "../services/api";
import { questionsFor, type Stage as QStage } from "../lib/questions";

const MIC_ICON = (
  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <rect x="9" y="3" width="6" height="11" rx="3" stroke="currentColor" strokeWidth={1.8} />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" />
  </svg>
);

const STOP_ICON = (
  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <rect x="7.5" y="7.5" width="9" height="9" rx="1.5" fill="currentColor" />
  </svg>
);

/** 브라우저 음성인식 지원 여부 — Firefox·Safari 는 미지원 */
function sttSupported(): boolean {
  const w = window as Window & { SpeechRecognition?: unknown; webkitSpeechRecognition?: unknown };
  return Boolean(w.SpeechRecognition ?? w.webkitSpeechRecognition);
}

export type Stage = QStage;

const STAGE = {
  self: {
    persona: "gyul",
    kind: "gyul" as const,
    who: "결",
    eyebrow: "1단계 · 자가분석 대화",
    title: "편하게 이야기해요",
    hd: "이야기하는 중",
    finish: "여기까지 하고 리포트 보기",
  },
  interview: {
    persona: "interviewer",
    kind: "interviewer" as const,
    who: "면접관",
    eyebrow: "2단계 · 면접 대화",
    title: "면접처럼 물어볼게요",
    hd: "면접 보는 중",
    finish: "여기까지 하고 리포트 보기",
  },
};

export function DemoPage({ stage = "self" }: { stage?: Stage }) {
  const navigate = useNavigate();
  const logRef = useRef<HTMLDivElement>(null);
  const cfg = STAGE[stage];

  const session = useMemo(
    () => ({
      userId: DEMO_SESSION.userId,
      sessionId: `${DEMO_SESSION.sessionId}-${stage}`,
      personaId: cfg.persona,
    }),
    [stage, cfg.persona],
  );

  const { health, error: healthError } = useHealth();

  /* 첫 질문만 고정. 그 뒤로는 LLM이 대화를 이어간다 */
  const opener = questionsFor(stage)[0];

  const {
    videoRef,
    overlayRef,
    running: camRunning,
    status: camStatus,
    faceCount,
    faceExpression,
    start: startCam,
    stop: stopCam,
  } = useFaceDetect();

  const faceRef = useRef(faceExpression);
  faceRef.current = faceExpression;

  /* 첫 질문 낭독 — 다 읽은 뒤에 마이크를 켠다.
     동시에 켜면 스피커 소리가 마이크로 되돌아 들어가 인식이 엉킨다. */
  const [greeting, setGreeting] = useState(false);
  const greetAudioRef = useRef<HTMLAudioElement | null>(null);

  const {
    status,
    running,
    liveText,
    messages,
    botSpeaking,
    ttsRate,
    setTtsRate,
    start,
    stop,
    buildSnapshot,
    getTurnCount,
  } = useDemoSession(session, { getFaceExpression: () => faceRef.current });

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, liveText]);

  const beginSession = useCallback(() => {
    void start();
    void startCam();
  }, [start, startCam]);

  /** 첫 질문을 읽어준 뒤 세션을 시작한다. 낭독이 안 되면 바로 시작. */
  const handleStart = useCallback(async () => {
    setGreeting(true);
    try {
      const { audio_b64, format } = await synthesizeSpeech(opener.text);
      if (audio_b64) {
        const mime = format === "mp3" ? "audio/mpeg" : `audio/${format ?? "mpeg"}`;
        const audio = new Audio(`data:${mime};base64,${audio_b64}`);
        audio.preservesPitch = true;
        audio.playbackRate = ttsRate;
        greetAudioRef.current = audio;
        await new Promise<void>((resolve) => {
          audio.onended = () => resolve();
          audio.onerror = () => resolve();
          void audio.play().catch(() => resolve());
        });
      }
    } catch {
      /* 낭독 실패는 넘어간다 — 대화는 진행돼야 한다 */
    }
    setGreeting(false);
    beginSession();
  }, [beginSession, opener.text, ttsRate]);

  /* 낭독 중에 배속을 바꾸면 첫 질문 오디오에도 바로 반영한다. */
  useEffect(() => {
    if (greetAudioRef.current) greetAudioRef.current.playbackRate = ttsRate;
  }, [ttsRate]);


  useEffect(
    () => () => {
      greetAudioRef.current?.pause();
    },
    [],
  );

  const handleStop = useCallback(async () => {
    await stop();
    stopCam();
  }, [stop, stopCam]);

  const handleFinish = useCallback(async () => {
    if (running) {
      await stop();
      stopCam();
    }
    const snapshot = buildSnapshot();
    if (!snapshot) return;
    navigate("/demo/report/loading", { state: { snapshot, stage } });
  }, [buildSnapshot, navigate, running, stop, stopCam, stage]);

  /* ── 상태 해석 ── */
  const apiOk = health?.ok === true;
  const wsOk = status.includes("연결") || status.includes("청취");
  const listening = running && !botSpeaking;
  const hearing = listening && liveText !== "..." && liveText.length > 0;
  const noStt = !sttSupported();

  const answers = messages.filter((m) => m.role === "user");
  const lastAns = answers[answers.length - 1];
  const waitingReply = running && Boolean(lastAns) && !lastAns.voice;

  const mood: Mood = botSpeaking || greeting ? "speak" : listening ? "listen" : "idle";

  const failed =
    status.startsWith("시작 실패") || status.startsWith("오류") || status.includes("종료");

  const barState = greeting
    ? "인사 중"
    : failed
    ? "시작하지 못했어요"
    : !running
    ? "대기"
    : botSpeaking
      ? "읽어주는 중"
      : hearing
        ? "인식 중"
        : "듣는 중";

  const barSay = greeting
    ? "첫 질문을 읽어드리고 있어요. 끝나면 마이크가 켜집니다."
    : failed
    ? status.replace(/^시작 실패: /, "")
    : !running
    ? "시작을 누르면 마이크랑 카메라가 켜져요."
    : botSpeaking
      ? "지금은 듣고 계세요. 끝나면 다시 열려요."
      : hearing
        ? liveText
        : noStt
          ? "이 브라우저에선 음성 인식이 안 돼요. 크롬에서 열어 주세요."
          : "편하게 말씀하세요. 바로 알아들어요.";

  const barHint = failed
    ? "권한을 허용한 뒤 시작을 다시 눌러 주세요."
    : !running
    ? "카메라는 표정만 봐요. 영상은 저장하지 않아요."
    : botSpeaking
      ? "다 읽으면 알아서 다시 들을게요."
      : "말 끝내고 잠깐 쉬면 알아서 넘어가요.";

  const turns = getTurnCount();

  return (
    <main className="session">
      <div className="s-top">
        <div className="t">
          <span className="eyebrow">{cfg.eyebrow}</span>
          <h1>{cfg.title}</h1>
        </div>
        <span className={`st${apiOk && wsOk ? " ok" : ""}`}>
          <i />
          {healthError ? "서버 끊김" : apiOk ? status : "서버 확인 중"}
        </span>
        <div className="acts">
          <div className="rate" role="group" aria-label="낭독 배속">
            {TTS_RATE_OPTIONS.map((r) => (
              <button
                key={r}
                className={`rate-b${ttsRate === r ? " on" : ""}`}
                type="button"
                aria-pressed={ttsRate === r}
                onClick={() => setTtsRate(r)}
              >
                {r}x
              </button>
            ))}
          </div>
          {running ? (
            <button className="btn sm" type="button" onClick={() => void handleStop()}>
              일시정지
            </button>
          ) : (
            <button
              className="btn sm"
              type="button"
              disabled={greeting}
              onClick={() => void handleStart()}
            >
              {greeting ? "읽어주는 중" : "시작"}
            </button>
          )}
          <button
            className="btn sm solid"
            type="button"
            disabled={turns === 0}
            onClick={() => void handleFinish()}
          >
            종료 · 리포트
          </button>
        </div>
      </div>

      <div className="track">
        <div className={`leg ${stage === "self" ? "now" : "done"}`}>
          <span className="k">01 · {stage === "self" ? `답변 ${answers.length}개` : "완료"}</span>
          <span className="nm">자가분석 대화</span>
        </div>
        <div className={`leg ${stage === "interview" ? "now" : ""}`}>
          <span className="k">02 · {stage === "interview" ? `답변 ${answers.length}개` : "대기"}</span>
          <span className="nm">면접 대화</span>
        </div>
        <div className="leg">
          <span className="k">03 · 대기</span>
          <span className="nm">{stage === "self" ? "자가분석 리포트" : "면접 리포트"}</span>
        </div>
      </div>

      <div className="s-grid">
        <div className="chat">
          <div className="chat-hd">
            <span className="who">{cfg.who}</span>
            <span className="r">{running ? `${answers.length}번 답함` : cfg.hd}</span>
          </div>

          <div className="log" ref={logRef}>
            {/* 첫 인사만 고정. 그 뒤로는 LLM이 대화를 이어간다 */}
            <div className="turn ai">
              <div className="bub">
                <span className="qtext">{opener.text}</span>
              </div>
            </div>

            {/* 오간 순서 그대로 그린다 — 인덱스로 짝지으면 한 칸씩 밀린다 */}
            {messages.map((m) =>
              m.role === "bot" ? (
                <div className="turn ai" key={m.id}>
                  <div className="bub">{m.text}</div>
                </div>
              ) : (
                <div className="turn me" key={m.id}>
                  <div className="bub">{m.text}</div>
                </div>
              ),
            )}

            {hearing && (
              <div className="turn me pending">
                <div className="bub">{liveText}</div>
              </div>
            )}

            {waitingReply && (
              <div className="turn ai">
                <div className="bub">
                  <span className="dots3">
                    <i />
                    <i />
                    <i />
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="bar">
            <div className="bar-main two">
              <button
                className={`mic${listening ? " on" : ""}`}
                type="button"
                disabled={botSpeaking || greeting}
                onClick={running ? () => void handleStop() : () => void handleStart()}
                aria-label={running ? "중지" : "시작"}
              >
                {running ? STOP_ICON : MIC_ICON}
              </button>

              <div className="bar-txt">
                <p className="bar-state">{barState}</p>
                <p className={`bar-say${hearing ? " live" : ""}`}>
                  {barSay}
                  {hearing && (
                    <span className="wave">
                      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                        <i key={i} style={{ animationDelay: `${(i * 0.09).toFixed(2)}s` }} />
                      ))}
                    </span>
                  )}
                </p>
              </div>

            </div>
            <p className="bar-hint">{barHint}</p>
          </div>
        </div>

        <aside className="side">
          <div className="stage">
            <div className="stage-fig">
              <Character kind={cfg.kind} mood={mood} />
            </div>
            <div className="stage-ft">
              <span className="nm">{cfg.who}</span>
              <span className="st">
                {!running
                  ? "대기 중"
                  : botSpeaking
                    ? "읽어주는 중"
                    : waitingReply
                      ? "생각하는 중"
                      : "듣고 있어요"}
              </span>
            </div>
          </div>

          <div>
            <div className="camwrap">
              <video ref={videoRef} playsInline muted />
              <canvas ref={overlayRef} />
              {!camRunning && <div className="off">카메라 꺼짐</div>}
            </div>
            <div className="cam-meta">
              <span>
                {camStatus === "live" ? `얼굴 ${faceCount}` : camStatus === "loading" ? "준비 중" : camStatus === "error" ? "연결 오류" : "대기"}
              </span>
              <span>{camRunning ? "표정 분석 중" : ""}</span>
            </div>
          </div>

          {noStt && (
            <p className="hint">
              <b>여기선 음성 인식이 안 돼요.</b>
              <br />
              크롬이나 엣지에서 열면 말이 글로 옮겨져요. 목소리 감정은 여기서도 읽힙니다.
            </p>
          )}
        </aside>
      </div>
    </main>
  );
}
