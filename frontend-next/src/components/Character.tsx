/**
 * AI 캐릭터 — 인라인 SVG 선화.
 *
 * 결은 머리를 7도 기울인 듣는 자세 + 가로 결 무늬,
 * 정(면접관)은 수직 자세 + 안경·넥타이 + 세로 규칙선.
 * 원본 SVG: frontend/mockups/characters/
 */

export type CharacterKind = "gyul" | "interviewer";
export type Mood = "idle" | "listen" | "think" | "speak";

type Props = {
  kind: CharacterKind;
  mood?: Mood;
  className?: string;
  title?: string;
};

const S = {
  bg: "var(--bg)",
  sand: "var(--sand)",
  fill: "var(--sand-lite)",
};

function GyulFace({ mood }: { mood: Mood }) {
  return (
    <>
      <g stroke={S.sand} strokeWidth={4}>
        <path d="M6 40h108M6 55h108M6 70h108" />
      </g>
      <path d="M14 128c0-25 20-40 46-40s46 15 46 40" fill={S.fill} />
      <g transform="rotate(-7 60 56)">
        <ellipse cx={60} cy={52} rx={30} ry={31} fill={S.bg} />
        <path
          d="M32 46c2-15 13-24 28-24s26 9 28 24"
          fill={S.sand}
          stroke="none"
        />
        <path d="M32 46c2-15 13-24 28-24s26 9 28 24" />
        <path d="M30 55q-5 2-4 7t6 4M90 55q5 2 4 7t-6 4" />

        {mood === "listen" && <path d="M45 52q6-8 12 0M63 52q6-8 12 0" />}
        {mood === "think" && <path d="M46 49q5-7 11 0M64 49q5-7 11 0" />}
        {(mood === "idle" || mood === "speak") && (
          <>
            <circle cx={51} cy={mood === "speak" ? 51 : 52} r={3.3} fill="currentColor" stroke="none" />
            <circle cx={69} cy={mood === "speak" ? 51 : 52} r={3.3} fill="currentColor" stroke="none" />
          </>
        )}

        <path d="M60 58v4" />

        {mood === "speak" ? (
          <ellipse cx={60} cy={69} rx={6.4} ry={4.8} fill={S.sand} />
        ) : mood === "think" ? (
          <path d="M54 69h12" />
        ) : (
          <path d="M50 67q10 8 20 0" />
        )}
      </g>
      {mood === "think" && (
        <>
          <circle cx={98} cy={26} r={3} fill="currentColor" stroke="none" opacity={0.45} />
          <circle cx={107} cy={17} r={2} fill="currentColor" stroke="none" opacity={0.3} />
        </>
      )}
    </>
  );
}

function InterviewerFace({ mood }: { mood: Mood }) {
  const eyeY = mood === "think" ? 50 : 52;
  const eyeX = mood === "think" ? [51.5, 72.5] : [49.5, 70.5];
  const brow =
    mood === "think"
      ? "M41 36h15M64 38h15"
      : mood === "listen"
        ? "M41 38.5h15M64 38.5h15"
        : "M41 37.5h15M64 37.5h15";

  return (
    <>
      <path d="M110 6v118" stroke={S.sand} strokeWidth={4} />
      <path d="M12 128c1-23 19-36 48-36s47 13 48 36" fill={S.fill} />
      <path d="M44 94l16 15 16-15" fill={S.bg} />
      <path d="M60 109l5 7-5 12-5-12z" fill="currentColor" stroke="none" />
      <rect x={31} y={19} width={58} height={64} rx={21} fill={S.bg} />
      <path
        d="M31 43c0-17 12-26 29-26s29 9 29 26h-7c-2-9-10-14-22-14s-20 5-22 14z"
        fill={S.sand}
        stroke="none"
      />
      <path d="M31 43c0-17 12-26 29-26s29 9 29 26" />
      <path d={brow} />
      <rect x={40} y={45} width={19} height={15} rx={7} />
      <rect x={61} y={45} width={19} height={15} rx={7} />
      <path d="M59 52h2" />
      <circle cx={eyeX[0]} cy={eyeY} r={2.7} fill="currentColor" stroke="none" />
      <circle cx={eyeX[1]} cy={eyeY} r={2.7} fill="currentColor" stroke="none" />
      {mood === "speak" ? (
        <ellipse cx={60} cy={71} rx={5.4} ry={4} fill={S.sand} />
      ) : (
        <path d="M52 71h16" />
      )}
    </>
  );
}

export function Character({ kind, mood = "idle", className, title }: Props) {
  return (
    <svg
      viewBox="0 0 120 128"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={2.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: "var(--ink)", display: "block", width: "100%", height: "100%" }}
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      {kind === "gyul" ? <GyulFace mood={mood} /> : <InterviewerFace mood={mood} />}
    </svg>
  );
}
