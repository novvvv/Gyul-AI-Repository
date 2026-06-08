import { Link } from "react-router-dom";
import "../styles/landing.css";

const OFFERS = [
  {
    num: "01",
    title: "실시간 음성 감정 분석",
    desc: "말하는 순간 목소리를 분석합니다. SER 모델이 7가지 감정과 확률을 실시간으로 보여, 텍스트만으로는 드러나지 않는 단서를 포착합니다.",
    tag: "기능 1",
  },
  {
    num: "02",
    title: "감정을 반영한 AI 대화",
    desc: "발화 텍스트와 분석된 감정을 함께 전달합니다. 선택형 문항이 아닌, 편하게 말하는 대화 속 반응에 맞춰 공감형 답변을 받습니다.",
    tag: "기능 2",
  },
  {
    num: "03",
    title: "맥락을 기억하는 탐색 세션",
    desc: "세션·페르소나 단위로 대화 흐름을 유지합니다. 말과 감정 패턴을 차분히 이어가며 스스로를 살펴볼 수 있습니다.",
    tag: "기능 3",
  },
];

const FLOW = [
  { n: "01", title: "음성으로 말하기", desc: "마이크로 편하게 발화" },
  { n: "02", title: "감정 추론", desc: "SER 7종 실시간 분석" },
  { n: "03", title: "내용 인식", desc: "문장 단위 음성 인식" },
  { n: "04", title: "공감 응답", desc: "감정 반영 AI 답변" },
];

const HIGHLIGHTS = [
  { value: "7", label: "감정 클래스" },
  { value: "실시간", label: "음성 분석" },
  { value: "음성", label: "우선 대화" },
];

export function LandingPage() {
  return (
    <div className="landing-page">
      <section className="landing-hero">
        <div className="landing-hero-grid">
          <div className="landing-hero-copy">
            <span className="landing-badge">AI 음성 자기 탐색</span>
            <h1>
              글자가 아닌
              <br />
              <em>목소리</em>에서 나를 읽다
            </h1>
            <p className="landing-lead">
              망설임과 떨림, 말투 속에 있는 진짜 감정. 결은 음성 대화와 실시간
              감정 분석으로 포장되지 않은 반응을 함께 들여다봅니다.
            </p>
            <div className="landing-hero-actions">
              <Link to="/demo" className="landing-btn landing-btn-primary">
                음성 대화 체험하기
              </Link>
              <a href="#features" className="landing-btn landing-btn-secondary">
                기능 살펴보기
              </a>
            </div>
          </div>
          <div className="landing-hero-visual" aria-hidden>
            <div className="visual-card">
              <p className="visual-label">실시간 감정</p>
              <p className="visual-emotion">neutral</p>
              <div className="visual-bars">
                {[40, 72, 28, 55, 88, 36, 48].map((h, i) => (
                  <span key={i} style={{ height: `${h}%` }} />
                ))}
              </div>
              <p className="visual-caption">목소리 파형 · SER 분석</p>
            </div>
          </div>
        </div>
        <ul className="landing-highlights">
          {HIGHLIGHTS.map((item) => (
            <li key={item.label}>
              <strong>{item.value}</strong>
              <span>{item.label}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="landing-section landing-why">
        <div className="landing-section-head">
          <span className="landing-section-tag">왜 결인가</span>
          <h2>텍스트만으로는 닿지 않는 감정</h2>
          <p>
            문항을 고르는 순간, 바람직한 답을 무의식적으로 선택하는 편향이
            생깁니다. 결은 말하는 순간의 목소리를 분석해 텍스트 뒤 반응을
            드러냅니다.
          </p>
        </div>
        <div className="landing-why-grid">
          <article className="landing-quote-card">
            <h3>기존 텍스트 진단</h3>
            <p>
              “나는 외향적인가요?”처럼 의도된 선택지는 이상적인 자아를 그리게
              만듭니다. 억양·망설임·감정이 섞인 목소리는 반영되지 않습니다.
            </p>
          </article>
          <article className="landing-quote-card landing-quote-card-accent">
            <h3>결의 접근</h3>
            <p>
              AI와 음성으로 대화하며 SER로 발화 감정을 분석합니다. 감정과 말한
              내용을 함께 반영해, 스스로도 몰랐던 반응을 마주할 수 있습니다.
            </p>
          </article>
        </div>
      </section>

      <section className="landing-section landing-features" id="features">
        <div className="landing-section-head">
          <span className="landing-section-tag">핵심 기능</span>
          <h2>지금 경험할 수 있는 것</h2>
          <p>마이크만 켜면 바로 체험할 수 있는 Web 기반 음성 자기 탐색입니다.</p>
        </div>
        <div className="landing-features-grid">
          {OFFERS.map((item) => (
            <article key={item.tag} className="landing-feature-card">
              <div className="landing-feature-top">
                <span className="landing-feature-num">{item.num}</span>
                <span className="landing-feature-tag">{item.tag}</span>
              </div>
              <h3>{item.title}</h3>
              <p>{item.desc}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-section landing-flow">
        <div className="landing-section-head">
          <span className="landing-section-tag">이용 흐름</span>
          <h2>한 번의 발화가 지나가는 길</h2>
        </div>
        <ol className="landing-flow-track">
          {FLOW.map((step) => (
            <li key={step.n}>
              <span className="flow-num">{step.n}</span>
              <div>
                <strong>{step.title}</strong>
                <span>{step.desc}</span>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="landing-cta-band">
        <div className="landing-cta-inner">
          <h2>지금, 목소리로 나를 만나보세요</h2>
          <p>
            선택지에 답하지 않아도 됩니다. 편하게 말하면 결이 감정을 읽고,
            대화로 이어갑니다.
          </p>
          <Link to="/demo" className="landing-btn landing-btn-primary landing-btn-light">
            음성 대화 시작
          </Link>
        </div>
      </section>
    </div>
  );
}
