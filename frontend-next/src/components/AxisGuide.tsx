import { AXIS_DEFS, AXIS_SOURCE, SCORE_GUIDE, type AxisDef } from "../lib/axes";

/**
 * 여섯 축 안내 — 여러 화면이 같은 설명을 쓰도록 한 곳에 둔다.
 *
 * 좌우 두 열로 가른다. 왼쪽은 STAR에서 온 셋(무엇을 말했는가),
 * 오른쪽은 결이 더한 셋(어떻게 말했는가). 열 제목이 경계를 대신한다.
 */

function Column({ title, sub, items }: { title: string; sub: string; items: AxisDef[] }) {
  return (
    <div className="ax-col">
      <div className="ax-col-hd">
        <h4>{title}</h4>
        <p>{sub}</p>
      </div>
      {items.map((a) => (
        <article className="ax" key={a.key}>
          <div className="ax-hd">
            <span className={`ax-tag ${a.origin}`}>{a.origin === "star" ? a.star : "결"}</span>
            <h5>{a.name}</h5>
          </div>
          <p className="l1">{a.what}</p>
          <p className="l2">
            <em>높으면</em> {a.high}
          </p>
          <p className="l3">
            <em>낮으면</em> {a.low}
          </p>
          <p className="detail">{a.detail}</p>
        </article>
      ))}
    </div>
  );
}

export function AxisGuide({ compact = false }: { compact?: boolean }) {
  const star = AXIS_DEFS.filter((a) => a.origin === "star");
  const gyul = AXIS_DEFS.filter((a) => a.origin === "gyul");

  return (
    <div className="guide">
      <div className="guide-src">
        <h3>{AXIS_SOURCE.headline}</h3>
        <p>{AXIS_SOURCE.body}</p>
        <p className="caveat">{AXIS_SOURCE.caveat}</p>
      </div>

      <div className="ax-grid">
        <Column title="STAR에서 온 셋" sub="무엇을 말했는가" items={star} />
        <Column title="결이 더한 셋" sub="어떻게 말했는가" items={gyul} />
      </div>

      {!compact && (
        <div className="guide-score">
          <h3>점수는 이렇게 읽어요</h3>
          <ul>
            <li>{SCORE_GUIDE.scale}</li>
            <li>{SCORE_GUIDE.gap}</li>
            <li>
              <b>{SCORE_GUIDE.meaning}</b>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
