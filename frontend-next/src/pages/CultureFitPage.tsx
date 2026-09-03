import { Link } from "react-router-dom";

/**
 * 컬쳐핏 추천 — 백엔드 미구현.
 * `POST /session/culture-fit` 이 생기면 이 상수를 응답으로 교체한다.
 * 회사는 모두 가상이며, 실제 기업명을 쓰지 않는다.
 */
export type CultureFitCompany = {
  rank: number;
  name: string;
  sub: string;
  score: number;
  fits: string[];
  watch: string;
  tags: string;
};

const COMPANIES: CultureFitCompany[] = [
  {
    rank: 1,
    name: "오르빗랩",
    sub: "AI 인프라 · 62명 · 시리즈 B · 수평적 협업형",
    score: 89,
    fits: [
      "온보딩 첫 달은 페어로 일합니다. 낯선 자리에서 표현이 위축되는 폭을 줄여 줍니다.",
      "주 단위 회고가 제도로 자리 잡았습니다. 평소 값이 높은 회고 지향이 그대로 발휘됩니다.",
    ],
    watch:
      "기술 결정을 문서로 공개 논의합니다. 자기개방 차이가 큰 편이라 초반 몇 주는 부담이 될 수 있습니다.",
    tags: "페어 온보딩 · 주간 회고 · 원격 병행",
  },
  {
    rank: 2,
    name: "세움소프트",
    sub: "엔터프라이즈 SaaS · 180명 · 시리즈 C · 안정 지향형",
    score: 84,
    fits: [
      "역할과 책임 범위가 문서로 정해져 있어, 말로 자기 몫을 주장하지 않아도 됩니다.",
      "면접 대화에서 높게 나온 구체성이 리뷰 문화와 잘 맞습니다.",
    ],
    watch: "의사결정이 느린 편입니다. 빠른 실행을 기대하면 답답할 수 있습니다.",
    tags: "문서 우선 · 리뷰 필수 · 재택 주 2회",
  },
  {
    rank: 3,
    name: "필드노트",
    sub: "에듀테크 · 34명 · 시리즈 A · 수평적 협업형",
    score: 76,
    fits: [
      "실패한 실험을 공유하는 자리가 정기적으로 있습니다. 평소 값의 자기개방이 살아날 여지가 큽니다.",
    ],
    watch: "인원이 적어 한 사람이 여러 역할을 겸합니다. 범위가 자주 바뀝니다.",
    tags: "소규모 · 실험 공유 · 전원 출근",
  },
  {
    rank: 4,
    name: "마루커넥트",
    sub: "커머스 플랫폼 · 240명 · 시리즈 D · 자율 실행형",
    score: 68,
    fits: ["목표만 맞추면 방식은 자유입니다. 단독 실행 선호와 맞습니다."],
    watch:
      "초반 정렬 없이 바로 실행에 들어갑니다. 주도성이 면접에서 줄어드는 편이라 첫 분기에 방향이 흔들릴 수 있습니다.",
    tags: "목표 중심 · 높은 자율성 · 빠른 배포",
  },
  {
    rank: 5,
    name: "넥스트폴드",
    sub: "핀테크 · 410명 · 상장 준비 · 속도 우선형",
    score: 61,
    fits: ["성장 기회가 많고 직무 이동이 자유롭습니다."],
    watch:
      "주간 단위로 우선순위가 바뀌고 회고보다 다음 실행을 먼저 요구합니다. 되짚어 정리하는 성향과 리듬이 다릅니다.",
    tags: "빠른 전환 · 성과 중심 · 야근 있음",
  },
];

export function CultureFitPage() {
  return (
    <main className="doc">
      <p className="crumbs">
        <Link to="/">결</Link>
        <span className="sep">/</span>
        <Link to="/demo/report">리포트</Link>
        <span className="sep">/</span>컬쳐핏
      </p>

      <div className="doc-head">
        <h1>컬쳐핏 추천</h1>
        <p className="sub">대화 기준 · 5곳</p>
      </div>

      <section className="sec">
        <h2>매칭 기준</h2>
        <p className="cap">
          두 대화의 평균이 아니라 <b>차이</b>를 씁니다. 평소 값은 그 사람이 가진 폭이고,
          면접 값은 낯선 상황에서 실제로 나오는 값입니다.
        </p>
        <ul className="rules">
          <li>
            <b>평소 값이 높은 축</b>은 그 문화에서 발휘될 수 있는 잠재력으로 봅니다.
          </li>
          <li>
            <b>두 대화의 차이가 큰 항목</b>은 적응 기간에 부담이 될 수 있는 지점으로 봅니다.
          </li>
          <li>
            자기개방·감정 표현의 차이가 커서 <b>초반 심리적 안전감을 갖춘 조직</b>에 가중치를 뒀습니다.
          </li>
        </ul>
      </section>

      <section className="sec">
        <h2>잘 맞을 회사</h2>
        {COMPANIES.map((c) => (
          <article className="co" key={c.name}>
            <div className="co-hd">
              <div>
                <div className="co-name">
                  <span className="rk">{String(c.rank).padStart(2, "0")}</span>
                  {c.name}
                </div>
                <div className="co-sub">{c.sub}</div>
              </div>
              <div className="co-score">
                <div className="v">{c.score}%</div>
                <div className="k">매치율</div>
              </div>
            </div>
            <div className="co-bar">
              <i style={{ width: `${c.score}%` }} />
            </div>
            <div className="co-why">
              {c.fits.map((f, i) => (
                <div className="r" key={f}>
                  <span className="lbl">{i === 0 ? "맞음" : ""}</span>
                  <span>{f}</span>
                </div>
              ))}
              <div className="r watch">
                <span className="lbl">확인</span>
                <span>{c.watch}</span>
              </div>
            </div>
            <div className="co-tags">{c.tags}</div>
          </article>
        ))}
      </section>

      <p className="tail">
        컬쳐핏 해석은 참고용이며 채용 결과를 예측하지 않습니다. 회사 문화는 팀과 시기에 따라 크게
        다를 수 있으므로, 실제 지원 전에 직접 확인하시길 권합니다.
      </p>
    </main>
  );
}
