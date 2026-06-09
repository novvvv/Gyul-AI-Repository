type Props = {
  running: boolean;
  loading?: boolean;
  canFinishReport?: boolean;
  onStart: () => void;
  onStop: () => void;
  onFinishReport: () => void;
};

export function MicControls({
  running,
  loading = false,
  canFinishReport = false,
  onStart,
  onStop,
  onFinishReport,
}: Props) {
  return (
    <div className="mic-controls">
      <button
        type="button"
        className="primary"
        disabled={running || loading}
        onClick={onStart}
      >
        시작
      </button>
      <button
        type="button"
        disabled={!running || loading}
        onClick={onStop}
      >
        중지
      </button>
      <button
        type="button"
        className="report-finish"
        disabled={!canFinishReport || loading}
        onClick={onFinishReport}
      >
        {loading ? "리포트 생성 중..." : "종료(레포트)"}
      </button>
    </div>
  );
}
