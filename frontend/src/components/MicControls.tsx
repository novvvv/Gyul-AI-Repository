type Props = {
  running: boolean;
  onStart: () => void;
  onStop: () => void;
};

export function MicControls({ running, onStart, onStop }: Props) {
  return (
    <div className="mic-controls">
      <button type="button" className="primary" disabled={running} onClick={onStart}>
        시작
      </button>
      <button type="button" disabled={!running} onClick={onStop}>
        중지
      </button>
    </div>
  );
}
