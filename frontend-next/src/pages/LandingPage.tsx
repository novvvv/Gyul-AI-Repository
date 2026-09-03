import { Link } from "react-router-dom";
import { Hexagon, type Axis } from "../components/Hexagon";

/** 히어로 예시 — 실제 세션 데이터가 아니라 제품 설명용 표본 */
const SAMPLE_AXES: Axis[] = [
  { name: "자기개방", base: 82, cmp: 54 },
  { name: "구체성", base: 58, cmp: 79 },
  { name: "일관성", base: 84, cmp: 61 },
  { name: "안정감", base: 71, cmp: 52 },
  { name: "주도성", base: 63, cmp: 45 },
  { name: "감정 표현", base: 76, cmp: 48 },
];

const STEPS = [
  {
    n: "01",
    title: "자가분석 대화",
    desc: "요즘 어떤지, 뭐가 신경 쓰이는지 편하게 이야기해요. 평가하지 않고 듣기만 합니다.",
    t: "10분 · 자유 대화",
  },
  {
    n: "02",
    title: "면접 대화",
    desc: "같은 주제를 면접에서 다시 물어요. 말투도 질문도 면접 그대로입니다.",
    t: "15분 · 8문항",
  },
  {
    n: "03",
    title: "차이 리포트",
    desc: "두 대화를 여섯 축으로 겹쳐 봐요. 어디가 얼마나 달라졌는지 한 장에 담깁니다.",
    t: "컬쳐핏 포함",
  },
];

export function LandingPage() {
  return (
    <main className="lp">
      <div className="lp-inner">
        <section className="hero">
          <div>
            <span className="eyebrow">두 번의 대화 · 하나의 차이</span>
            <h1>
              편할 때의 나와
              <span className="low">면접에서의 나</span>
            </h1>
            <p className="lead">
              같은 사람인데 상황에 따라 말이 달라져요. 결은 두 번 대화합니다.
              먼저 편하게 이야기하고, 그다음에 면접을 봐요. 그 사이의 간격이 리포트가 됩니다.
            </p>
            <div className="acts">
              <Link className="btn solid" to="/demo">
                대화 시작하기
              </Link>
              <Link className="btn bare" to="/compare">
                리포트 예시 보기
              </Link>
            </div>
            <p className="fine">브라우저에서 바로 · 두 대화 합쳐 약 25분</p>
          </div>

          <figure className="hero-fig">
            <Hexagon axes={SAMPLE_AXES} />
            <figcaption>
              여섯 축 가운데 <b>다섯이 줄고 하나가 늘었습니다.</b> 면접에서 더 구체적으로 말한 대신,
              자기개방과 감정 표현이 28점씩 내려갔습니다.
            </figcaption>
          </figure>
        </section>

        <section className="band">
          <div className="band-grid">
            <h2>
              <span className="eyebrow">왜 두 번인가</span>
              한 번만 보면 그게 원래 모습인지 알 수 없다
            </h2>
            <div className="seam-two">
              <div className="col">
                <h3>한 번만 보는 진단</h3>
                <p>
                  면접 결과 하나로 “표현이 적은 사람”이라고 말해요. 평소엔 잘 말하던 사람이
                  그 자리에서만 굳었을 수도 있는데, 그건 보이지 않습니다.
                </p>
              </div>
              <div className="seam" />
              <div className="col mine">
                <h3>결의 두 대화</h3>
                <p>
                  편한 대화를 기준선으로 먼저 잡고, 같은 사람의 면접 대화를 겹칩니다.{" "}
                  <b>무엇이 얼마나 달라졌는지</b>가 숫자로 남습니다.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="band">
          <div className="band-grid">
            <h2>
              <span className="eyebrow">진행 방식</span>세 단계
            </h2>
            <div className="steps">
              {STEPS.map((s) => (
                <div className="step" key={s.n}>
                  <span className="n">{s.n}</span>
                  <div>
                    <h3>{s.title}</h3>
                    <p>{s.desc}</p>
                  </div>
                  <span className="t">{s.t}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="closing">
          <div>
            <h2>기준선부터 만들어 볼까요</h2>
            <p>첫 대화는 면접이 아니에요. 편하게 말하는 것부터 시작해요.</p>
          </div>
          <Link className="btn solid" to="/demo">
            대화 시작
          </Link>
        </section>

        <div className="foot">
          <span>결 — 두 대화의 차이로 읽는 자기분석</span>
          <span>2026</span>
        </div>
      </div>
    </main>
  );
}
