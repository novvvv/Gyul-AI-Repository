import type { HealthResponse } from "../services/api";

type Props = {
  health: HealthResponse | null;
  healthError: string | null;
  wsStatus: string;
};

export function ConnectionStatus({ health, healthError, wsStatus }: Props) {
  const apiOk = health?.ok === true;
  const wsOk = wsStatus.includes("연결") || wsStatus.includes("청취");
  const llmHint =
    health?.llm_provider != null
      ? health.llm_loaded
        ? "AI 준비됨"
        : "AI 로딩 중"
      : null;

  return (
    <div className="connection-status">
      <span className={`pill ${apiOk ? "ok" : "bad"}`}>
        {apiOk ? "서버 연결" : healthError ?? "서버 끊김"}
      </span>
      <span className={`pill ${wsOk ? "ok" : "idle"}`}>{wsStatus}</span>
      {llmHint && <span className="pill muted">{llmHint}</span>}
    </div>
  );
}
