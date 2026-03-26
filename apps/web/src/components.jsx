import { escapeValue, statusClass } from "./utils";

export function MetricCard({ title, value, note }) {
  return (
    <div className="metric">
      <div className="title">{title}</div>
      <div className="value">{escapeValue(value)}</div>
      <div className="detail-note">{note}</div>
    </div>
  );
}

export function StatusPill({ status }) {
  return <span className={`status ${statusClass(status)}`}>{escapeValue(status)}</span>;
}

export function EmptyState({ icon, title, note, wide = false }) {
  return (
    <div className={`empty-state ${wide ? "empty-state-wide" : ""}`}>
      <div className="empty-icon">{icon}</div>
      <div className="empty-title">{title}</div>
      {note ? <div className="empty-note">{note}</div> : null}
    </div>
  );
}

export function SectionHeader({ kicker, title, note, right }) {
  return (
    <div className="section-head">
      <div>
        {kicker ? <div className="section-kicker">{kicker}</div> : null}
        <h2>{title}</h2>
        {note ? <div className="detail-note">{note}</div> : null}
      </div>
      {right}
    </div>
  );
}
