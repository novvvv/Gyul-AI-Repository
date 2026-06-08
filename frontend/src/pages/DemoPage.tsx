import { useCallback, useMemo } from "react";
import { AiPersonaPanel } from "../components/AiPersonaPanel";
import { CameraPanel } from "../components/CameraPanel";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { ChatThread } from "../components/ChatThread";
import { EmotionPanel } from "../components/EmotionPanel";
import { MicControls } from "../components/MicControls";
import { DEMO_SESSION } from "../config";
import { useDemoSession } from "../hooks/useDemoSession";
import { useFaceDetect } from "../hooks/useFaceDetect";
import { useHealth } from "../hooks/useHealth";
import "../styles/demo.css";

export function DemoPage() {
  const session = useMemo(
    () => ({
      userId: DEMO_SESSION.userId,
      sessionId: DEMO_SESSION.sessionId,
      personaId: DEMO_SESSION.personaId,
    }),
    [],
  );

  const { health, error: healthError } = useHealth();
  const {
    status,
    running,
    liveText,
    messages,
    emotion,
    botEmotion,
    start,
    stop,
  } = useDemoSession(session);

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

  const handleStart = useCallback(() => {
    void start();
    void startCam();
  }, [start, startCam]);

  const handleStop = useCallback(() => {
    void stop();
    stopCam();
  }, [stop, stopCam]);

  const aiSpeaking = useMemo(() => {
    const last = messages[messages.length - 1];
    return running && last?.role === "bot";
  }, [messages, running]);

  return (
    <div className="demo-page">
      <header className="demo-header">
        <div className="demo-title">
          <span className="demo-badge">체험하기</span>
          <h1>음성 + 표정 대화 체험</h1>
          <p className="sub">
            편하게 말해 보세요. 목소리에서 읽은 감정과 말한 내용을 바탕으로 AI가
            공감형 답변을 이어갑니다. 카메라로는 얼굴 위치를 함께 인식합니다.
          </p>
        </div>
        <ConnectionStatus
          health={health}
          healthError={healthError}
          wsStatus={status}
        />
        <MicControls
          running={running}
          onStart={handleStart}
          onStop={handleStop}
        />
      </header>

      <div className="demo-grid">
        <aside className="demo-left">
          <div className="persona-block card">
            <AiPersonaPanel emotion={botEmotion} speaking={aiSpeaking} />
            <EmotionPanel emotion={emotion} />
          </div>
        </aside>
        <div className="demo-main">
          <ChatThread messages={messages} liveText={liveText} />
        </div>
        <aside className="demo-camera-col">
          <CameraPanel
            videoRef={videoRef}
            overlayRef={overlayRef}
            running={camRunning}
            status={camStatus}
            faceCount={faceCount}
            faceExpression={faceExpression}
          />
        </aside>
      </div>
    </div>
  );
}
