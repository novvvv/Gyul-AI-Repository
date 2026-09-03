/**
 * 여섯 축 비교 — 두 대화(편할 때 / 면접에서)를 겹쳐 본다.
 *
 * 채움(모래색) = 기준선, 윤곽(검정) = 비교 대상.
 * 면접 값이 안쪽으로 줄면 그만큼 모래색이 밖으로 드러나고,
 * 그 드러난 면적이 곧 "닫힌 정도"가 된다.
 */

export type Axis = {
  name: string;
  base: number; // 편할 때 (0–100)
  cmp: number; // 면접에서 (0–100)
};

type Props = {
  axes: Axis[];
  size?: number;
  labels?: boolean;
};

export function Hexagon({ axes, size = 300, labels = true }: Props) {
  const cx = 150;
  const cy = 140;
  const R = 92;
  const n = axes.length;

  const point = (i: number, v: number): [number, number] => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const r = (R * v) / 100;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  };
  const poly = (values: number[]) =>
    values.map((v, i) => point(i, v).map((x) => x.toFixed(1)).join(",")).join(" ");

  const desc = axes
    .map((a) => `${a.name} ${a.base}대 ${a.cmp}`)
    .join(", ");

  return (
    <svg
      viewBox="0 0 300 286"
      width={size}
      style={{ width: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label={`육각형 비교 차트. 편할 때 대비 면접에서. ${desc}.`}
    >
      {[25, 50, 75, 100].map((step) => (
        <polygon
          key={step}
          points={poly(axes.map(() => step))}
          fill="none"
          stroke={step === 100 ? "var(--ink-3)" : "var(--rule)"}
          strokeWidth={1}
        />
      ))}

      {axes.map((_, i) => {
        const [x, y] = point(i, 100);
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={x.toFixed(1)}
            y2={y.toFixed(1)}
            stroke="var(--rule)"
            strokeWidth={1}
          />
        );
      })}

      <polygon
        points={poly(axes.map((a) => a.base))}
        fill="var(--sand)"
        stroke="var(--ink-3)"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
      <polygon
        points={poly(axes.map((a) => a.cmp))}
        fill="var(--bg)"
        fillOpacity={0.4}
        stroke="var(--ink)"
        strokeWidth={2}
        strokeLinejoin="round"
      />

      {axes.map((a, i) => {
        const [x, y] = point(i, a.cmp);
        return <circle key={a.name} cx={x.toFixed(1)} cy={y.toFixed(1)} r={3.2} fill="var(--ink)" />;
      })}

      {labels &&
        axes.map((a, i) => {
          const [x, y] = point(i, 125);
          const anchor = x > cx + 6 ? "start" : x < cx - 6 ? "end" : "middle";
          const d = a.cmp - a.base;
          return (
            <g key={`l-${a.name}`}>
              <text
                x={x.toFixed(1)}
                y={(y + 3).toFixed(1)}
                textAnchor={anchor}
                fill="var(--ink-2)"
                fontSize={12}
                fontWeight={600}
              >
                {a.name}
              </text>
              <text
                x={x.toFixed(1)}
                y={(y + 16).toFixed(1)}
                textAnchor={anchor}
                fill="var(--ink-3)"
                fontSize={10.5}
                fontFamily="ui-monospace, monospace"
              >
                {d > 0 ? `+${d}` : d}
              </text>
            </g>
          );
        })}
    </svg>
  );
}

/** 0을 가운데 두고 좌우로 뻗는 발산 막대 */
export function DivergingBars({ axes, max = 30 }: { axes: Axis[]; max?: number }) {
  const sorted = [...axes].sort((a, b) => a.cmp - a.base - (b.cmp - b.base));

  return (
    <div>
      <div className="div-head">
        <span />
        <span className="l">면접에서 줄어듦</span>
        <span className="r">늘어남</span>
        <span className="e">차이</span>
      </div>

      {sorted.map((a) => {
        const d = a.cmp - a.base;
        const w = Math.min(100, (Math.abs(d) / max) * 100);
        return (
          <div className="div-row" key={a.name} title={`${a.name} — 편할 때 ${a.base} / 면접에서 ${a.cmp}`}>
            <span className="nm">{a.name}</span>
            <span className="neg">{d < 0 && <i style={{ width: `${w}%` }} />}</span>
            <span className="pos">{d > 0 && <i style={{ width: `${w}%` }} />}</span>
            <span className="d">{d > 0 ? `+${d}` : d}</span>
          </div>
        );
      })}

      <div className="div-foot">
        <span />
        <span className="l">−{max}</span>
        <span className="r">+{max}</span>
        <span />
      </div>
    </div>
  );
}
