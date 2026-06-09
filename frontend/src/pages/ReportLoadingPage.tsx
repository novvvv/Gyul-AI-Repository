import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { fetchHealth, requestSessionReport } from "../services/api";
import type { SessionSnapshot } from "../types/sessionReport";
import "../styles/report-loading.css";

type LocationState = {
  snapshot?: SessionSnapshot;
};

const PHASES = [
  "오늘 나눈 대화를 살펴보고 있어요",
  "감정의 흐름을 읽고 있어요",
  "당신을 위한 레포트를 쓰고 있어요",
];

function estimateSeconds(backend: string, turnCount: number): number {
  const base = backend === "openai" ? 15 : 75;
  return base + Math.min(turnCount, 8) * (backend === "openai" ? 2 : 8);
}

export function ReportLoadingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const snapshot = (location.state as LocationState | null)?.snapshot;
  const startedRef = useRef(false);

  const [elapsed, setElapsed] = useState(0);
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [backend, setBackend] = useState<string>("kanana");
  const [error, setError] = useState<string | null>(null);

  const turnCount = snapshot?.turns.length ?? 0;
  const etaSec = useMemo(
    () => estimateSeconds(backend, turnCount),
    [backend, turnCount],
  );

  useEffect(() => {
    if (!snapshot) {
      navigate("/demo", { replace: true });
      return;
    }

    void fetchHealth()
      .then((h) => setBackend(h.text_llm_backend ?? "kanana"))
      .catch(() => setBackend("kanana"));
  }, [snapshot, navigate]);

  useEffect(() => {
    const tick = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    const phaseTimer = window.setInterval(
      () => setPhaseIdx((i) => (i + 1) % PHASES.length),
      6000,
    );
    return () => {
      window.clearInterval(tick);
      window.clearInterval(phaseTimer);
    };
  }, []);

  useEffect(() => {
    if (!snapshot || startedRef.current) return;
    startedRef.current = true;

    void (async () => {
      try {
        const report = await requestSessionReport(snapshot);
        sessionStorage.setItem(
          `report:${snapshot.session.session_id}`,
          JSON.stringify(report),
        );
        navigate("/demo/report", { state: { report }, replace: true });
      } catch (e) {
        setError(e instanceof Error ? e.message : "레포트 생성 실패");
      }
    })();
  }, [snapshot, navigate]);

  const progress = Math.min(95, Math.round((elapsed / etaSec) * 100));
  const remaining = Math.max(0, etaSec - elapsed);

  if (!snapshot) return null;

  return (
    <div className="rpt-loading-page">
      <div className="rpt-loading-card">
        <p className="rpt-loading-brand">Gyul Session Report</p>
        <h1>레포트를 작성하고 있어요</h1>
        <p className="rpt-loading-phase">{PHASES[phaseIdx]}</p>

        <div className="rpt-loading-bar-track">
          <div
            className="rpt-loading-bar-fill"
            style={{ width: `${error ? 100 : progress}%` }}
          />
        </div>

        <div className="rpt-loading-meta">
          <span>경과 {elapsed}초</span>
          {!error && (
            <span>
              예상 {remaining > 0 ? `약 ${remaining}초 남음` : "곧 완료"}
              {" · "}
              {backend === "openai" ? "GPT-4o-mini" : "Kanana"}
            </span>
          )}
        </div>

        <p className="rpt-loading-hint">
          {error
            ? `오류가 발생했어요: ${error}`
            : `발화 ${turnCount}건 기준, 보통 ${Math.ceil(etaSec / 10) * 10}초 안에 완료돼요. 창을 닫지 말고 잠시만 기다려 주세요.`}
        </p>

        {error && (
          <button
            type="button"
            className="rpt-loading-retry"
            onClick={() => navigate("/demo")}
          >
            대화로 돌아가기
          </button>
        )}
      </div>
    </div>
  );
}
