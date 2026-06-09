import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import type { SessionReportResponse } from "../types/sessionReport";
import type { SessionTurnRecord } from "../types/sessionReport";
import {
  buildRankedQuotes,
  countsToSlices,
  formatSessionRange,
  labelKo,
} from "../utils/reportMetrics";
import "../styles/report.css";

type LocationState = {
  report?: SessionReportResponse;
};

type Slice = { labelKo: string; pct: number; color: string };

const CHART_COLORS = ["#3d342c", "#8a5d52", "#c4a882", "#9a8b7a", "#6b5d4f", "#ddd4c8"];
const VOICE_COLOR = "#3d342c";
const FACE_COLOR = "#8a5d52";

const FAILURE_MARKERS = [
  "자동 요약 생성에 실패",
  "기본 리포트를 제공",
  "기본 리포트",
];

function isFailureBoilerplate(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return true;
  return FAILURE_MARKERS.some((marker) => trimmed.includes(marker));
}

function buildClientNarrative(
  reportJson: NonNullable<SessionReportResponse["report_json"]>,
): string {
  const n = reportJson.meta.user_turn_count;
  if (n === 0) return "오늘은 아직 나눈 이야기가 없어요. 다음에 편하게 말씀해 주시면 함께 돌아볼게요.";

  const voice = reportJson.meta.voice_dominant.map(labelKo).join(", ") || "파악되지 않음";
  const face = reportJson.meta.face_dominant.map(labelKo).join(", ") || "파악되지 않음";
  const quote = reportJson.turns.find((turn) => turn.user_text)?.user_text ?? "";

  const parts = [
    "오늘 대화를 돌아보면, 당신은 자신의 마음을 꺼내려는 용기를 보여주셨어요. 말로 감정을 표현하는 일은 쉽지 않은데, 그럼에도 이야기를 이어가신 점이 인상적이에요.",
  ];
  if (quote) {
    parts.push(
      `특히 「${quote}」라고 하셨을 때, 그 안에 담긴 마음이 느껴졌어요. 겉으로 드러낸 말 너머에도 더 깊은 생각이 있었을 수 있어요.`,
    );
  }
  if (voice !== "파악되지 않음" || face !== "파악되지 않음") {
    parts.push(
      `목소리에는 ${voice}의 결이, 표정에는 ${face}의 여운이 스며 있었어요. 마음은 한 가지 색으로만 보이지 않을 때가 많거든요.`,
    );
  }
  if (reportJson.meta.mismatch_count > 0) {
    parts.push(
      "목소리와 표정이 조금 다르게 느껴진 순간도 있었어요. 그건 괜찮아요. 겉과 속이 항상 같을 필요는 없으니까요.",
    );
  }
  parts.push(
    "오늘 나눈 이야기 속에서, 당신이 무엇을 소중히 여기고 있는지 조금 더 선명해진 것 같아요. 스스로를 돌아보는 이 시간 자체가 이미 의미 있는 걸음이에요.",
  );
  parts.push(
    "다음에 마음이 편할 때, 오늘 가장 마음에 남는 순간을 조금 더 천천히 들여다보는 것도 좋겠어요.",
  );
  return parts.join(" ");
}

function splitParagraphs(text: string): string[] {
  const byNewline = text.split(/\n+/).map((s) => s.trim()).filter(Boolean);
  if (byNewline.length > 1) return byNewline;
  const bySentence = text
    .split(/(?<=[.!?。])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 2);
  if (bySentence.length <= 3) return [text];
  const mid = Math.ceil(bySentence.length / 2);
  return [
    bySentence.slice(0, mid).join(" "),
    bySentence.slice(mid).join(" "),
  ];
}

function PieChart({ title, slices }: { title: string; slices: Slice[] }) {
  const r = 52;
  const stroke = 16;
  const c = 2 * Math.PI * r;
  let offset = 0;
  const primary = slices[0];

  if (!primary) {
    return (
      <div className="rpt-pie-card">
        <h3>{title}</h3>
        <p className="rpt-muted">기록 없음</p>
      </div>
    );
  }

  return (
    <div className="rpt-pie-card">
      <h3>{title}</h3>
      <div className="rpt-pie-body">
        <div className="rpt-pie-svg-wrap">
          <svg viewBox="0 0 136 136" className="rpt-pie-svg" aria-hidden>
            <circle
              cx="68"
              cy="68"
              r={r}
              fill="none"
              stroke="rgba(61, 52, 44, 0.1)"
              strokeWidth={stroke}
            />
            {slices.map((slice) => {
              const dash = (c * slice.pct) / 100;
              const circle = (
                <circle
                  key={slice.labelKo}
                  cx="68"
                  cy="68"
                  r={r}
                  fill="none"
                  stroke={slice.color}
                  strokeWidth={stroke}
                  strokeDasharray={`${dash} ${c - dash}`}
                  strokeDashoffset={-offset + c * 0.25}
                  strokeLinecap="round"
                />
              );
              offset += dash;
              return circle;
            })}
          </svg>
          <div className="rpt-pie-center">
            <strong>{primary.pct}%</strong>
            <span>{primary.labelKo}</span>
          </div>
        </div>
        <ul className="rpt-pie-legend">
          {slices.map((slice) => (
            <li key={slice.labelKo}>
              <span className="rpt-pie-dot" style={{ background: slice.color }} />
              <span className="rpt-pie-legend-label">{slice.labelKo}</span>
              <span className="rpt-pie-legend-pct">{slice.pct}%</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function TurnCompareChart({ turns }: { turns: SessionTurnRecord[] }) {
  if (!turns.length) return null;

  const maxPct = 100;

  return (
    <div className="rpt-turn-chart">
      <div className="rpt-turn-chart-legend">
        <span><i style={{ background: VOICE_COLOR }} />음성</span>
        <span><i style={{ background: FACE_COLOR }} />표정</span>
      </div>
      {turns.map((turn, index) => {
        const voicePct = Math.round((turn.voice_emotion?.confidence ?? 0) * 100);
        const facePct = Math.round((turn.face_emotion?.confidence ?? 0) * 100);
        return (
          <div className="rpt-turn-row" key={`${index}-${turn.at}`}>
            <span className="rpt-turn-num">{index + 1}</span>
            <div className="rpt-turn-bars">
              <div className="rpt-turn-bar-line">
                <div className="rpt-turn-bar-track">
                  <div
                    className="rpt-turn-bar-fill voice"
                    style={{ width: `${(voicePct / maxPct) * 100}%` }}
                  />
                </div>
                <span className="rpt-turn-bar-meta">
                  {turn.voice_emotion ? labelKo(turn.voice_emotion.label) : "—"} {voicePct}%
                </span>
              </div>
              <div className="rpt-turn-bar-line">
                <div className="rpt-turn-bar-track">
                  <div
                    className="rpt-turn-bar-fill face"
                    style={{ width: `${(facePct / maxPct) * 100}%` }}
                  />
                </div>
                <span className="rpt-turn-bar-meta">
                  {turn.face_emotion ? labelKo(turn.face_emotion.label) : "—"} {facePct}%
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const stateReport = (location.state as LocationState | null)?.report;
  const [cachedReport, setCachedReport] = useState<SessionReportResponse | null>(
    null,
  );

  useEffect(() => {
    if (stateReport) return;
    const keys = Object.keys(sessionStorage).filter((k) => k.startsWith("report:"));
    const latestKey = keys.sort().at(-1);
    if (!latestKey) return;
    try {
      const parsed = JSON.parse(
        sessionStorage.getItem(latestKey) ?? "",
      ) as SessionReportResponse;
      setCachedReport(parsed);
    } catch {
      /* ignore corrupt cache */
    }
  }, [stateReport]);

  const report = stateReport ?? cachedReport;
  const reportJson = report?.report_json;

  const voiceSlices = useMemo(
    () =>
      countsToSlices(
        reportJson?.aggregates?.voice_counts as Record<string, number> | undefined,
        reportJson?.meta.user_turn_count ?? 0,
      ).map((s, i) => ({ ...s, color: CHART_COLORS[i % CHART_COLORS.length] })),
    [reportJson],
  );
  const faceSlices = useMemo(
    () =>
      countsToSlices(
        reportJson?.aggregates?.face_counts as Record<string, number> | undefined,
        reportJson?.meta.user_turn_count ?? 0,
      ).map((s, i) => ({ ...s, color: CHART_COLORS[i % CHART_COLORS.length] })),
    [reportJson],
  );
  const rankedQuotes = useMemo(
    () => buildRankedQuotes(reportJson?.quotes ?? [], reportJson?.turns ?? []),
    [reportJson],
  );

  const comprehensiveParagraphs = useMemo(() => {
    if (!reportJson) return [];

    const candidates = [
      reportJson.comprehensive_report,
      reportJson.summary,
    ].filter((text): text is string => Boolean(text?.trim()));

    const text =
      candidates.find((item) => !isFailureBoilerplate(item)) ??
      buildClientNarrative(reportJson);

    return splitParagraphs(text);
  }, [reportJson]);

  if (!report || !reportJson) {
    return (
      <div className="rpt-page">
        <div className="rpt-sheet rpt-empty">
          <h1>리포트가 없습니다</h1>
          <p>대화 후 <strong>종료(레포트)</strong>를 눌러주세요.</p>
          <Link className="rpt-btn" to="/demo">
            대화 시작하기
          </Link>
        </div>
      </div>
    );
  }

  const hasPatterns = reportJson.patterns.length > 0;
  const hasStrengths = reportJson.strengths.length > 0;
  const hasReflection =
    reportJson.reflection_questions.length > 0 && !reportJson.generation_error;
  const hasEmotionMix = voiceSlices.length > 0 || faceSlices.length > 0;

  const downloadMarkdown = () => {
    const blob = new Blob([report.report_md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `session_report_${reportJson.meta.session_id}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="rpt-page">
      <article className="rpt-sheet">
        <header className="rpt-masthead">
          <div className="rpt-masthead-top">
            <span className="rpt-brand">Gyul Session Report</span>
            <div className="rpt-toolbar">
              <button type="button" className="rpt-btn outline" onClick={downloadMarkdown}>
                다운로드
              </button>
              <button type="button" className="rpt-btn" onClick={() => navigate("/demo")}>
                새 대화
              </button>
            </div>
          </div>
          <p className="rpt-eyebrow">대화 종합 레포트</p>
          <h1>오늘 나눈 이야기</h1>
          <p className="rpt-deck">
            {formatSessionRange(reportJson.meta.started_at, reportJson.meta.ended_at)}
            {" · "}
            발화 {reportJson.meta.user_turn_count}회
            {reportJson.meta.mismatch_count > 0 &&
              ` · 음성·표정 차이 ${reportJson.meta.mismatch_count}회`}
          </p>
        </header>

        {/* ── 줄글 영역 ── */}
        <section className="rpt-section rpt-prose-zone">
          <h2 className="rpt-section-title">나를 돌아보며</h2>
          <div className="rpt-lined-block">
            {comprehensiveParagraphs.map((para) => (
              <p key={para.slice(0, 48)}>{para}</p>
            ))}
          </div>
        </section>

        {(hasEmotionMix || reportJson.turns.length > 0) && (
          <section className="rpt-section rpt-chart-zone">
            <h2 className="rpt-section-title">감정 분석</h2>
            {hasEmotionMix && (
              <div className="rpt-pie-grid">
                <PieChart title="음성 감정" slices={voiceSlices} />
                <PieChart title="표정" slices={faceSlices} />
              </div>
            )}
            {reportJson.turns.length > 0 && (
              <>
                <h3 className="rpt-sub-title">발화별 비교</h3>
                <TurnCompareChart turns={reportJson.turns} />
              </>
            )}
          </section>
        )}

        {/* ── 텍스트 보조 영역 ── */}
        {reportJson.topics.length > 0 && (
          <section className="rpt-section rpt-text-zone">
            <h2 className="rpt-section-title">주제</h2>
            <div className="rpt-topic-row">
              {reportJson.topics.map((topic) => (
                <span className="rpt-topic" key={topic}>
                  {topic}
                </span>
              ))}
            </div>
          </section>
        )}

        {rankedQuotes.length > 0 && (
          <section className="rpt-section rpt-text-zone">
            <h2 className="rpt-section-title">핵심 발화</h2>
            <ol className="rpt-quotes">
              {rankedQuotes.map((item) => (
                <li key={item.rank}>
                  <p>「{item.text}」</p>
                </li>
              ))}
            </ol>
          </section>
        )}

        {(hasPatterns || hasStrengths) && (
          <section className="rpt-section rpt-text-zone rpt-section-cols">
            {hasPatterns && (
              <div className="rpt-note-card">
                <h2 className="rpt-section-title">관찰된 패턴</h2>
                <ul className="rpt-bullets">
                  {reportJson.patterns.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
            {hasStrengths && (
              <div className="rpt-note-card">
                <h2 className="rpt-section-title">잘 하고 있는 점</h2>
                <ul className="rpt-bullets">
                  {reportJson.strengths.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        {hasReflection && (
          <section className="rpt-section rpt-text-zone">
            <h2 className="rpt-section-title">돌아볼 질문</h2>
            <ol className="rpt-reflect-list">
              {reportJson.reflection_questions.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ol>
          </section>
        )}

        {reportJson.turns.length > 0 && (
          <section className="rpt-section rpt-text-zone">
            <h2 className="rpt-section-title">대화 기록</h2>
            <div className="rpt-table-wrap">
              <table className="rpt-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>발화</th>
                    <th>음성</th>
                    <th>표정</th>
                  </tr>
                </thead>
                <tbody>
                  {reportJson.turns.map((turn, index) => (
                    <tr key={`${index}-${turn.at}`}>
                      <td>{index + 1}</td>
                      <td>{turn.user_text}</td>
                      <td>
                        {turn.voice_emotion ? (
                          <span className="rpt-badge voice">
                            {labelKo(turn.voice_emotion.label)}{" "}
                            {(turn.voice_emotion.confidence * 100).toFixed(0)}%
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>
                        {turn.face_emotion ? (
                          <span className="rpt-badge face">
                            {labelKo(turn.face_emotion.label)}{" "}
                            {(turn.face_emotion.confidence * 100).toFixed(0)}%
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

        <footer className="rpt-footer">
          <p>{reportJson.disclaimer}</p>
          <span>Gyul · Session Report</span>
        </footer>
      </article>
    </div>
  );
}
