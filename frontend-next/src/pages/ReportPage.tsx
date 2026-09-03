import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { emotionColor, labelKo, normalize } from "../lib/emotion";
import type { SessionReportResponse, SessionTurnRecord } from "../types/sessionReport";
import { questionsFor, counterpart } from "../lib/questions";

type Stage = "self" | "interview";
type LocationState = { report?: SessionReportResponse; stage?: Stage };

/* ────────────────────────────────────────────────
   면접 평가 축 — 음성 면접에서 실제로 관찰 가능한 것만 쓴다.
   서버가 rubric 을 내려주면 그 값으로 대체한다.
   ──────────────────────────────────────────────── */
type Rubric = { key: string; name: string; score: number; note: string };

/* ── 자가분석: 평가가 아니라 관찰. 축도 문구도 다르다 ── */
const OPEN_WORDS = /(무서|겁|불안|답답|아쉽|속상|힘들|기쁘|뿌듯|편하|좋았|싫)/;
const SELF_WORDS = /(저는|제가|나는|내가)/;

function buildSelfRubric(turns: SessionTurnRecord[]): Rubric[] {
  const texts = turns.map((t) => t.user_text ?? "").filter(Boolean);
  const n = Math.max(1, texts.length);
  const feeling = texts.filter((t) => OPEN_WORDS.test(t)).length / n;
  const first = texts.filter((t) => SELF_WORDS.test(t)).length / n;
  const avgLen = texts.reduce((a, t) => a + t.length, 0) / n;
  const paired = turns.filter((t) => t.voice_emotion && t.face_emotion);
  const agree = paired.length
    ? paired.filter((t) => normalize(t.voice_emotion?.label) === normalize(t.face_emotion?.label)).length /
      paired.length
    : 1;
  const clamp = (v: number) => Math.max(12, Math.min(96, Math.round(v)));

  return [
    {
      key: "feeling",
      name: "감정 언어",
      score: clamp(28 + feeling * 66),
      note:
        feeling >= 0.5
          ? "감정을 그대로 말하는 단어가 자주 나왔어요."
          : "사실 위주로 말씀하셔서 감정 단어는 적었어요.",
    },
    {
      key: "ownership",
      name: "1인칭",
      score: clamp(30 + first * 62),
      note:
        first >= 0.6
          ? "「저는」으로 시작하는 문장이 많아요. 본인 이야기로 읽혀요."
          : "주어가 빠지거나 팀·상황으로 바뀌는 문장이 있었어요.",
    },
    {
      key: "depth",
      name: "깊이",
      score: clamp(avgLen >= 40 ? 80 : 40 + avgLen),
      note: `답변 평균 ${Math.round(avgLen)}자 — 짧아도 괜찮지만, 이유가 붙으면 다음 단계에서 비교가 쉬워집니다.`,
    },
    {
      key: "congruence",
      name: "말과 표정",
      score: clamp(agree * 100),
      note:
        agree >= 0.7
          ? "말과 표정이 대체로 같은 쪽이었어요."
          : "말과 표정이 어긋난 데가 있어요. 면접에서 더 벌어지는지 봐야 할 지점이에요.",
    },
  ];
}

const TENSE = new Set(["fear", "angry", "sadness"]);

function buildRubric(turns: SessionTurnRecord[]): Rubric[] {
  const texts = turns.map((t) => t.user_text ?? "").filter(Boolean);
  const n = Math.max(1, texts.length);

  const avgSent =
    texts.reduce(
      (sum, t) => sum + Math.max(1, t.split(/[.?!]|다\.|요\./).filter(Boolean).length),
      0,
    ) / n;
  const leadFirst = texts.filter((t) => /^(결과|결론|먼저|핵심|가장)/.test(t.trim())).length / n;
  const concrete = texts.filter((t) => /\d|퍼센트|%|개월|주간|명|번|년/.test(t)).length / n;
  const tense = turns.filter((t) => TENSE.has(normalize(t.voice_emotion?.label))).length / n;

  const paired = turns.filter((t) => t.voice_emotion && t.face_emotion);
  const agree = paired.length
    ? paired.filter(
        (t) => normalize(t.voice_emotion?.label) === normalize(t.face_emotion?.label),
      ).length / paired.length
    : 1;

  const clamp = (v: number) => Math.max(12, Math.min(96, Math.round(v)));

  return [
    {
      key: "structure",
      name: "구조",
      score: clamp(34 + leadFirst * 58),
      note:
        leadFirst >= 0.5
          ? "결론을 앞에 두는 답변이 절반을 넘습니다."
          : "상황 설명이 먼저 나오고 결과가 뒤에 옵니다.",
    },
    {
      key: "concrete",
      name: "구체성",
      score: clamp(30 + concrete * 62),
      note:
        concrete >= 0.4
          ? "수치와 기간이 답변에 실려 있습니다."
          : "사례는 있으나 수치·기간이 빠져 있습니다.",
    },
    {
      key: "length",
      name: "분량",
      score: clamp(avgSent >= 2 && avgSent <= 4 ? 84 : 46),
      note: `답변 평균 ${avgSent.toFixed(1)}문장 — 면접 권장 구간은 2~4문장입니다.`,
    },
    {
      key: "stability",
      name: "안정감",
      score: clamp(92 - tense * 68),
      note:
        tense >= 0.4
          ? "긴장 신호가 읽힌 답변이 적지 않습니다."
          : "전반적으로 안정된 톤을 유지했습니다.",
    },
    {
      key: "congruence",
      name: "일치도",
      score: clamp(agree * 100),
      note:
        agree >= 0.7
          ? "말과 표정이 대체로 같은 방향입니다."
          : "말과 표정이 어긋나는 구간이 있습니다.",
    },
  ];
}

type Fix = { title: string; why: string; now: string; better: string };

const FIX_LIB: Record<string, Fix> = {
  structure: {
    title: "결론을 첫 문장에 두기",
    why: "면접관은 첫 10초로 답변의 방향을 잡습니다. 두괄식으로 열면 뒤따르는 설명이 근거로 읽힙니다.",
    now: "상황을 설명하다가 마지막에 결과가 나옵니다. 듣는 쪽은 어디로 가는 이야기인지 끝까지 모릅니다.",
    better:
      "결과부터 말씀드리면, 배포 장애를 월 4건에서 1건으로 줄였습니다. 원인은 팀 간 공유 주기가 없었던 것이었고, 주간 싱크를 도입해 해결했습니다.",
  },
  concrete: {
    title: "숫자와 기간을 하나씩 넣기",
    why: "같은 경험도 수치가 붙으면 검증 가능한 주장이 됩니다. 규모·기간·변화량 중 하나면 충분합니다.",
    now: "“열심히 했다” “많이 개선됐다”처럼 크기를 알 수 없는 표현이 반복됩니다.",
    better:
      "3개월간 주 1회 코드 리뷰를 운영했고, 그 기간 동안 재작업 비율이 30%에서 12%로 내려갔습니다.",
  },
  length: {
    title: "한 답변을 세 문장으로 끊기",
    why: "길게 말할수록 신뢰가 오르지는 않습니다. 짧게 맺고 꼬리 질문을 유도하는 편이 유리합니다.",
    now: "한 답변이 길어지며 핵심 문장이 중간에 묻힙니다.",
    better: "결론 한 문장, 근거 한 문장, 배운 점 한 문장. 더 궁금하면 면접관이 물어봅니다.",
  },
  stability: {
    title: "첫 3초를 확보하기",
    why: "짧은 침묵은 감점 요인이 아닙니다. 정리되지 않은 문장이 감점 요인입니다.",
    now: "질문이 끝나자마자 말을 시작해, 첫 문장이 도중에 끊기고 다시 시작되는 구간이 있습니다.",
    better:
      "“잠시 정리해서 말씀드리겠습니다.” 한 마디를 두고 3초 쉰 뒤, 결론부터 시작합니다.",
  },
  congruence: {
    title: "내용과 태도를 맞추기",
    why: "면접관은 답변 내용보다 태도의 일관성을 먼저 기억합니다.",
    now: "아쉬운 경험을 말할 때 표정은 중립을 유지해, 내용과 태도가 따로 놉니다.",
    better:
      "실패를 말할 때는 담담하게, 거기서 배운 점으로 넘어갈 때 표정과 톤을 함께 올립니다.",
  },
};

function formatRange(start?: string, end?: string): string {
  if (!start || !end) return "-";
  const s = new Date(start);
  const e = new Date(end);
  if (Number.isNaN(s.getTime()) || Number.isNaN(e.getTime())) return `${start} ~ ${end}`;
  const mins = Math.max(1, Math.round((e.getTime() - s.getTime()) / 60000));
  return `${s.toLocaleDateString("ko-KR")} · ${mins}분`;
}

export function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const nav = location.state as LocationState | null;
  const stateReport = nav?.report;
  const [cached, setCached] = useState<SessionReportResponse | null>(null);
  const [cachedStage, setCachedStage] = useState<Stage>("self");
  const stage = nav?.stage ?? cachedStage;

  useEffect(() => {
    if (stateReport) return;
    const keys = Object.keys(sessionStorage).filter((k) => k.startsWith("report:"));
    const latest = keys.sort().at(-1);
    if (!latest) return;
    try {
      const parsed = JSON.parse(sessionStorage.getItem(latest) ?? "") as SessionReportResponse & {
        stage?: Stage;
      };
      setCached(parsed);
      if (parsed.stage) setCachedStage(parsed.stage);
    } catch {
      /* 손상된 캐시는 무시 */
    }
  }, [stateReport]);

  const report = stateReport ?? cached;
  const json = report?.report_json;
  const turns = useMemo(() => json?.turns ?? [], [json]);
  const rubric = useMemo(
    () => (turns.length ? (stage === "self" ? buildSelfRubric(turns) : buildRubric(turns)) : []),
    [turns, stage],
  );
  const fixes = useMemo(
    () =>
      stage === "interview"
        ? [...rubric]
            .sort((a, b) => a.score - b.score)
            .slice(0, 3)
            .map((r) => FIX_LIB[r.key])
            .filter(Boolean)
        : [],
    [rubric, stage],
  );
  const overall = rubric.length
    ? Math.round(rubric.reduce((s, r) => s + r.score, 0) / rubric.length)
    : 0;

  if (!report || !json) {
    return (
      <main className="empty">
        <h1>아직 리포트가 없습니다</h1>
        <p>대화를 마치면 여기에 정리해 드려요.</p>
        <Link className="btn solid" to="/demo">
          대화 시작하기
        </Link>
      </main>
    );
  }

  const strengths = (json.strengths ?? []).slice(0, 3);
  const mismatch = (t: SessionTurnRecord) =>
    Boolean(
      t.voice_emotion &&
        t.face_emotion &&
        normalize(t.voice_emotion.label) !== normalize(t.face_emotion.label),
    );

  const download = () => {
    const blob = new Blob([report.report_md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${json.meta.session_id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="doc">
      <p className="crumbs">
        <Link to="/">결</Link>
        <span className="sep">/</span>
        {stage === "self" ? "자가분석 기록" : "면접 피드백"}
      </p>

      <div className="doc-head">
        <h1>{stage === "self" ? "자가분석 기록" : "면접 피드백"}</h1>
        <p className="sub">
          {formatRange(json.meta.started_at, json.meta.ended_at)} · 답변 {json.meta.user_turn_count}개
        </p>
        <div className="acts">
          <button className="btn sm" type="button" onClick={download}>
            다운로드
          </button>
          <button className="btn sm" type="button" onClick={() => navigate("/demo")}>
            다시 연습
          </button>
        </div>
      </div>

      <section className="sec">
        <h2 style={{ marginBottom: 22 }}>
          {stage === "self" ? "이번 대화에서 보인 것" : "평가"}
        </h2>
        <div className={`score${stage === "self" ? " noscore" : ""}`}>
          {stage === "interview" && (
            <div className="score-n">
              <span className="v">{overall}</span>
              <span className="k">종합</span>
            </div>
          )}
          <div className="rubric">
            {rubric.map((r) => (
              <div className="rb" key={r.key}>
                <span className="nm">{r.name}</span>
                <span className="tr">
                  <i style={{ width: `${r.score}%` }} />
                </span>
                <span className="sc">{r.score}</span>
                <span className="nt">{r.note}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {stage === "self" && turns.length > 0 && (
        <section className="sec">
          <h2>문항별로 하신 말씀</h2>
          <p className="cap">
            면접에서 같은 주제를 다시 물어요. 그때 답변이랑 나란히 놓고 뭐가 달라지는지 봅니다.
          </p>
          {questionsFor("self").map((q, i) => {
            const t = turns[i];
            const pair = counterpart(q, "self");
            if (!t) return null;
            return (
              <article className="qa" key={q.id}>
                <div className="qa-hd">
                  <span className="no">Q{i + 1}</span>
                  <h3>{q.topicName}</h3>
                </div>
                <p className="qa-a">“{t.user_text}”</p>
                <div className="qa-ft">
                  {t.voice_emotion && (
                    <span className="emo">
                      <span
                        className="dot-sw"
                        style={{ background: emotionColor(t.voice_emotion.label) }}
                      />
                      음성 {labelKo(t.voice_emotion.label)}
                    </span>
                  )}
                  {t.face_emotion && (
                    <span className="emo">
                      <span
                        className="dot-sw"
                        style={{ background: emotionColor(t.face_emotion.label) }}
                      />
                      표정 {labelKo(t.face_emotion.label)}
                    </span>
                  )}
                  {pair && <span className="qa-next">면접 문항 → {pair.topicName}</span>}
                </div>
              </article>
            );
          })}
        </section>
      )}

      {fixes.length > 0 && (
        <section className="sec">
          <h2>다음 면접에서 고칠 세 가지</h2>
          <p className="cap">점수가 낮은 것부터 골랐어요. 오른쪽 문장은 그대로 쓰셔도 돼요.</p>
          {fixes.map((f, i) => (
            <article className="fix" key={f.title}>
              <div className="fix-hd">
                <span className="no">{String(i + 1).padStart(2, "0")}</span>
                <h3>{f.title}</h3>
              </div>
              <p className="fix-why">{f.why}</p>
              <div className="ba">
                <div className="col">
                  <span className="lb">지금</span>
                  <p>{f.now}</p>
                </div>
                <div className="seam" />
                <div className="col to">
                  <span className="lb">이렇게</span>
                  <p>{f.better}</p>
                </div>
              </div>
            </article>
          ))}
        </section>
      )}

      {strengths.length > 0 && (
        <section className="sec">
          <h2>{stage === "self" ? "눈에 띈 점" : "유지할 것"}</h2>
          <ul className="keep">
            {strengths.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </section>
      )}

      {stage === "interview" && turns.length > 0 && (
        <section className="sec">
          <h2>답변별 기록</h2>
          <p className="cap">음영이 진 줄은 말과 표정이 서로 달랐던 답변이에요.</p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>답변</th>
                  <th>길이</th>
                  <th>톤</th>
                </tr>
              </thead>
              <tbody>
                {turns.map((t, i) => (
                  <tr key={`${i}-${t.at}`} className={mismatch(t) ? "gap-row" : undefined}>
                    <td>{i + 1}</td>
                    <td>{t.user_text}</td>
                    <td className="mono">{(t.user_text ?? "").length}자</td>
                    <td>
                      {t.voice_emotion ? (
                        <span className="emo">
                          <span
                            className="dot-sw"
                            style={{ background: emotionColor(t.voice_emotion.label) }}
                          />
                          {labelKo(t.voice_emotion.label)}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="handoff">
        {stage === "self" ? (
          <>
            <div>
              <h3>이제 면접 대화 차례입니다</h3>
              <p>방금 이야기를 기준으로, 같은 주제를 면접에서 다시 물어볼게요.</p>
            </div>
            <Link className="btn solid" to="/demo/interview">
              면접 시작
            </Link>
          </>
        ) : (
          <>
            <div>
              <h3>편할 때와 얼마나 달랐을까요</h3>
              <p>아까 편하게 한 이야기랑 방금 면접을 나란히 놓고 봐요.</p>
            </div>
            <Link className="btn solid" to="/compare">
              차이 보기
            </Link>
          </>
        )}
      </div>

      <p className="tail">{json.disclaimer}</p>
    </main>
  );
}
