import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { fetchHealth, requestSessionReport } from "../services/api";
import type { SessionSnapshot } from "../types/sessionReport";

type LocationState = { snapshot?: SessionSnapshot; stage?: "self" | "interview" };

const PHASES = [
  "오늘 나눈 대화를 살펴보는 중",
  "감정의 흐름을 읽는 중",
  "당신을 위한 리포트를 쓰는 중",
];

/** 백엔드에 따라 걸리는 시간이 달라 예상치만 조정한다. 화면에 모델명을 띄우지는 않는다. */
function estimateSeconds(backend: string, turnCount: number): number {
  const base = backend === "openai" ? 15 : 75;
  return base + Math.min(turnCount, 8) * (backend === "openai" ? 2 : 8);
}

export function ReportLoadingPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const st = location.state as LocationState | null;
  const snapshot = st?.snapshot;
  const stage = st?.stage ?? "self";
  const startedRef = useRef(false);

  const [elapsed, setElapsed] = useState(0);
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [backend, setBackend] = useState("kanana");
  const [error, setError] = useState<string | null>(null);

  const turnCount = snapshot?.turns.length ?? 0;
  const etaSec = useMemo(() => estimateSeconds(backend, turnCount), [backend, turnCount]);

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
    const phase = window.setInterval(
      () => setPhaseIdx((i) => (i + 1) % PHASES.length),
      6000,
    );
    return () => {
      window.clearInterval(tick);
      window.clearInterval(phase);
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
          JSON.stringify({ ...report, stage }),
        );
        navigate("/demo/report", { state: { report, stage }, replace: true });
      } catch (e) {
        setError(e instanceof Error ? e.message : "리포트 생성 실패");
      }
    })();
  }, [snapshot, navigate, stage]);

  if (!snapshot) return null;

  const progress = Math.min(95, Math.round((elapsed / etaSec) * 100));
  const remaining = Math.max(0, etaSec - elapsed);

  return (
    <main className="wait">
      <span className="eyebrow">리포트 생성</span>
      <h1>
        {error
          ? "리포트를 만들지 못했어요"
          : stage === "self"
            ? "자가분석 대화를 정리하고 있어요"
            : "면접 대화를 정리하고 있어요"}
      </h1>
      <p className="phase">{error ? "" : PHASES[phaseIdx]}</p>

      <div className="barline">
        <i style={{ width: `${error ? 100 : progress}%` }} />
      </div>

      <div className="meta">
        <span>{elapsed}초</span>
        {!error && (
          <span>{remaining > 0 ? `약 ${remaining}초 남음` : "곧 완료"}</span>
        )}
      </div>

      {error ? (
        <div className="err">
          <p>
            <b>{error}</b>
          </p>
          <p style={{ marginTop: 10, color: "var(--ink-2)" }}>
            서버가 응답하지 않았습니다. 대화 내용은 그대로 남아 있으니 다시 시도해 보세요.
          </p>
          <p style={{ marginTop: 22 }}>
            <Link className="btn" to="/demo">
              대화로 돌아가기
            </Link>
          </p>
        </div>
      ) : (
        <p className="err" style={{ color: "var(--ink-3)", fontSize: 12.5 }}>
          발화 {turnCount}건 기준, 보통 {Math.ceil(etaSec / 10) * 10}초 안에 끝납니다.
          창을 닫지 말고 잠시만 기다려 주세요.
        </p>
      )}
    </main>
  );
}
