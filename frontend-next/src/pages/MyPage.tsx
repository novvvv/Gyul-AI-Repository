import { Link } from "react-router-dom";
import { Hexagon, type Axis } from "../components/Hexagon";
import { AxisGuide } from "../components/AxisGuide";
import { axisDef } from "../lib/axes";

/**
 * 마이페이지 — 계정과 세션 이력은 Spring 이 적재한다.
 * 조회 API 가 붙으면 아래 상수를 응답으로 교체한다.
 */

/** 가장 최근 세션의 여섯 축 */
const LATEST: Axis[] = [
  { name: "자기개방", base: 82, cmp: 67 },
  { name: "구체성", base: 58, cmp: 79 },
  { name: "일관성", base: 84, cmp: 70 },
  { name: "안정감", base: 71, cmp: 60 },
  { name: "주도성", base: 63, cmp: 52 },
  { name: "감정 표현", base: 76, cmp: 61 },
];

const TREND = [
  { when: "7월 21일", gap: 23 },
  { when: "8월 16일", gap: 19 },
  { when: "9월 2일", gap: 14 },
];

const SESSIONS = [
  {
    draft: true,
    date: "2026-09-05",
    time: "진행 중",
    title: "자가분석 대화",
    tag: "이어하기",
    facts: ["답변 6개", "면접 대화 남음"],
    to: "/demo",
    cta: "이어서 대화",
    solid: true,
  },
  {
    date: "2026-09-02",
    time: "14:20",
    title: "세 번째 대화",
    facts: ["평균 차이 14점", "자기개방 −15", "말과 표정 어긋남 3회"],
    to: "/compare",
    cta: "자세히",
  },
  {
    date: "2026-08-16",
    time: "20:05",
    title: "두 번째 대화",
    facts: ["평균 차이 19점", "자기개방 −21", "말과 표정 어긋남 4회"],
    to: "/compare",
    cta: "자세히",
  },
  {
    date: "2026-07-21",
    time: "11:32",
    title: "첫 대화",
    facts: ["평균 차이 23점", "자기개방 −28", "말과 표정 어긋남 5회"],
    to: "/compare",
    cta: "자세히",
  },
];

const MAX = 28;

export function MyPage() {
  const avg = Math.round(
    LATEST.reduce((s, a) => s + Math.abs(a.cmp - a.base), 0) / LATEST.length,
  );
  const worst = [...LATEST].sort((a, b) => a.cmp - a.base - (b.cmp - b.base))[0];

  return (
    <main className="doc">
      <p className="crumbs">
        <Link to="/">결</Link>
        <span className="sep">/</span>내 기록
      </p>

      <div className="acct">
        <div>
          <p className="em">nov@gyul.kr</p>
          <p className="sub">가입 2026-06-14 · 대화 3회</p>
        </div>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <button className="btn sm" type="button">
            계정 설정
          </button>
          <Link className="btn sm solid" to="/demo">
            새 대화 시작
          </Link>
        </div>
      </div>

      {/* ── 한눈에 ── */}
      <div className="tally">
        <div className="c">
          <div className="k">대화한 횟수</div>
          <div className="v">
            3<small>번</small>
          </div>
          <div className="s">가장 최근 9월 2일</div>
        </div>
        <div className="c">
          <div className="k">평균 차이</div>
          <div className="v">
            {avg}
            <small>점</small>
          </div>
          <div className="s">첫 대화 23점에서 9점 좁혀졌어요</div>
        </div>
        <div className="c">
          <div className="k">가장 많이 달라진 것</div>
          <div className="v" style={{ fontSize: 22 }}>
            {worst.name}
          </div>
          <div className="s">
            {worst.cmp - worst.base}점 줄었어요 · {axisDef(worst.name)?.what}
          </div>
        </div>
      </div>

      {/* ── 격차 육각형 ── */}
      <section className="sec">
        <h2>가장 최근 대화 — 편할 때와 면접에서</h2>
        <p className="cap">
          모래색이 <b>편하게 이야기할 때</b>, 검은 선이 <b>면접에서</b>입니다. 검은 선이 안으로
          들어온 만큼 모래색이 밖으로 드러나는데, 그게 면접에서 줄어든 부분이에요.
        </p>

        <div className="compare">
          <Hexagon axes={LATEST} />

          <div className="gapfull">
            {[...LATEST]
              .sort((a, b) => a.cmp - a.base - (b.cmp - b.base))
              .map((a) => {
                const d = a.cmp - a.base;
                const def = axisDef(a.name);
                return (
                  <div className="gaprow" key={a.name}>
                    <div className="top">
                      <span className="nm">{a.name}</span>
                      <span className="nums">
                        <b>{a.base}</b> → <b>{a.cmp}</b>
                        <em className={d < 0 ? "down" : "up"}>{d > 0 ? `+${d}` : d}</em>
                      </span>
                    </div>
                    <p className="wt">{def?.what}</p>
                  </div>
                );
              })}
          </div>
        </div>
      </section>

      {/* ── 여섯 축 안내 ── */}
      <section className="sec">
        <h2>이 여섯 가지는 무엇인가요</h2>
        <AxisGuide />
      </section>

      {/* ── 추이 ── */}
      <section className="sec">
        <h2>세 번의 대화, 간격이 줄고 있어요</h2>
        <p className="cap">
          막대가 낮을수록 편할 때와 면접에서의 모습이 가깝다는 뜻입니다.
        </p>
        <div className="trend">
          <div className="trend-plot">
            {TREND.map((t) => (
              <div className="tp" key={t.when}>
                <span className="val">{t.gap}점</span>
                <span className="bar2" style={{ height: Math.round((t.gap / MAX) * 108) }} />
                <span className="lbl">{t.when}</span>
              </div>
            ))}
          </div>
          <p className="trend-key">
            <b>처음보다 9점 가까워졌어요.</b>
            면접에서도 평소처럼 말하게 되고 있다는 뜻입니다. 특히 자기개방이 −28에서 −15로
            절반 가까이 회복됐어요.
          </p>
        </div>
      </section>

      {/* ── 기록 ── */}
      <section className="sec">
        <h2>지난 대화</h2>
        {SESSIONS.map((s) => (
          <article className={`ses${s.draft ? " draft" : ""}`} key={s.date}>
            <div className="when">
              {s.date}
              <em>{s.time}</em>
            </div>
            <div>
              <div className="ttl">
                {s.title}
                {s.tag && <span className="pill">{s.tag}</span>}
              </div>
              <div className="facts">
                {s.facts.map((f) => (
                  <span key={f}>{f}</span>
                ))}
              </div>
            </div>
            <div className="go">
              <Link className={`btn sm${s.solid ? " solid" : ""}`} to={s.to}>
                {s.cta}
              </Link>
            </div>
          </article>
        ))}
      </section>

      <p className="tail">
        대화 기록과 리포트는 계정에 저장됩니다. 음성과 영상 원본은 저장하지 않고, 분석된 수치와
        글로 옮긴 답변만 남습니다. 계정 설정에서 전체 삭제할 수 있어요.
      </p>
    </main>
  );
}
