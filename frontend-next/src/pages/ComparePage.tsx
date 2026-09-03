import { Link } from "react-router-dom";
import { DivergingBars, Hexagon, type Axis } from "../components/Hexagon";
import { AxisGuide } from "../components/AxisGuide";
import { axisDef } from "../lib/axes";

/**
 * 두 대화 비교 — 아직 백엔드가 없다.
 *
 * 현재 파이프라인은 세션 1개만 만든다. 자가분석/면접 2단계와 여섯 축 산출이
 * 서버에 생기면 이 상수를 API 응답으로 갈아끼우면 된다.
 * 타입(Axis[])은 그대로 쓸 수 있게 맞춰 두었다.
 */
const AXES: Axis[] = [
  { name: "자기개방", base: 82, cmp: 54 },
  { name: "구체성", base: 58, cmp: 79 },
  { name: "일관성", base: 84, cmp: 61 },
  { name: "안정감", base: 71, cmp: 52 },
  { name: "주도성", base: 63, cmp: 45 },
  { name: "감정 표현", base: 76, cmp: 48 },
];

const PAIRS = [
  {
    topic: "실패한 프로젝트",
    gap: "감정 표현 −31",
    base: "그때 진짜 무서웠어요. 팀원들이 저를 어떻게 볼지가 제일 겁났어요.",
    cmp: "커뮤니케이션 부족이 원인이었고, 이후 주간 싱크를 도입해 재발을 막았습니다.",
  },
  {
    topic: "팀에서의 역할",
    gap: "자기개방 −26",
    base: "사실 저는 나서는 걸 잘 못해요. 그래서 늘 뒤에서 정리하는 쪽이었어요.",
    cmp: "팀 내에서는 조율과 문서화를 맡아 왔습니다.",
  },
];

export function ComparePage() {
  const drops = AXES.filter((a) => a.cmp < a.base);
  const rises = AXES.filter((a) => a.cmp > a.base);
  const worst = [...AXES].sort((a, b) => a.cmp - a.base - (b.cmp - b.base))[0];
  const best = [...AXES].sort((a, b) => b.cmp - b.base - (a.cmp - a.base))[0];

  return (
    <main className="doc">
      <p className="crumbs">
        <Link to="/">결</Link>
        <span className="sep">/</span>두 대화의 차이
      </p>

      <div className="doc-head">
        <h1>두 대화의 차이</h1>
        <p className="sub">편하게 이야기할 때와 면접에서, 같은 잣대로 재어 비교합니다</p>
      </div>

      <section className="sec">
        <div className="compare">
          <div>
            <Hexagon axes={AXES} />
            <div className="hex-key">
              <div className="r">
                <span className="sw a" />
                <b>편할 때</b>&nbsp;— 자가분석 대화
              </div>
              <div className="r">
                <span className="sw b" />
                <b>면접에서</b>&nbsp;— 면접 대화
              </div>
              <p className="hex-note">
                검은 선이 안으로 들어온 만큼 모래색이 드러나요. 그게 면접에서 줄어든 부분입니다.
              </p>
            </div>
          </div>
          <DivergingBars axes={AXES} />
        </div>
      </section>

      <section className="sec">
        <div className="figures">
          <div className="fig mark">
            <div className="k">가장 많이 줄어든 것</div>
            <div className="v">{worst.cmp - worst.base}</div>
            <div className="s">{worst.name} · {axisDef(worst.name)?.what}</div>
          </div>
          <div className="fig">
            <div className="k">유일하게 늘어난 것</div>
            <div className="v">+{best.cmp - best.base}</div>
            <div className="s">{best.name} · {axisDef(best.name)?.what}</div>
          </div>
          <div className="fig">
            <div className="k">줄어든 항목</div>
            <div className="v">
              {drops.length}
              <small>/ {AXES.length}</small>
            </div>
            <div className="s">늘어난 항목 {rises.length}개</div>
          </div>
          <div className="fig">
            <div className="k">평균 차이</div>
            <div className="v">
              {Math.round(AXES.reduce((s, a) => s + Math.abs(a.cmp - a.base), 0) / AXES.length)}
              <small>점</small>
            </div>
            <div className="s">작을수록 평소 모습에 가까워요</div>
          </div>
        </div>

      </section>

      <section className="sec">
        <h2>이 여섯 가지는 무엇인가요</h2>
        <AxisGuide />
      </section>

      <section className="sec">
        <h2>같은 주제, 두 답변</h2>
        <p className="cap">두 대화에서 같은 주제가 나온 지점을 짝지었습니다.</p>

        {PAIRS.map((p) => (
          <div key={p.topic} style={{ padding: "26px 0", borderBottom: "1px solid var(--rule-soft)" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 16,
                marginBottom: 18,
                alignItems: "baseline",
              }}
            >
              <span style={{ fontSize: 14, fontWeight: 600 }}>{p.topic}</span>
              <span className="mono" style={{ fontSize: 12, color: "var(--ink-2)" }}>
                {p.gap}
              </span>
            </div>
            <div className="seam-two">
              <div className="col">
                <h3>편할 때</h3>
                <p>“{p.base}”</p>
              </div>
              <div className="seam" style={{ background: "var(--ink)" }} />
              <div className="col mine">
                <h3>면접에서</h3>
                <p>“{p.cmp}”</p>
              </div>
            </div>
          </div>
        ))}
      </section>

      <div className="handoff">
        <div>
          <h3>컬쳐핏 추천</h3>
          <p>두 대화에서 읽힌 성향으로 잘 맞을 조직 문화를 살펴봅니다.</p>
        </div>
        <Link className="btn solid" to="/culture-fit">
          컬쳐핏 보기
        </Link>
      </div>

      <p className="tail">
        이 화면은 심리 검사나 진단이 아닙니다. 두 대화에서 관찰된 신호를 비교한 기록이며,
        감정 분석 결과에는 오차가 있을 수 있습니다.
      </p>
    </main>
  );
}
