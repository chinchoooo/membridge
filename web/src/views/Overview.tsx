import { useEffect, useState } from "react";
import { api, Rec, when } from "../api";

export default function Overview() {
  const [d, setD] = useState<Rec | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    const load = () => api("/api/overview").then(setD).catch((e) => setErr(String(e.message)));
    load();
    const t = setInterval(load, 5000); // live progress for imports/consolidation
    return () => clearInterval(t);
  }, []);

  if (err) return <p className="error">{err}</p>;
  if (!d) return <p>Loading…</p>;
  const s = d.stats;
  const g = typeof s.graph === "object" ? s.graph : null;
  const totalUser = d.messages_extracted ?? 0;
  const card = (label: string, value: number | string, sub?: string) => (
    <div className="card" key={label}>
      <div className="big">{value}</div>
      <div>{label}</div>
      {sub && <div className="muted">{sub}</div>}
    </div>
  );
  return (
    <div>
      <div className="cards">
        {card("Conversations", s.conversations)}
        {card("Messages", s.messages, `${d.messages_with_limitations} with omitted content`)}
        {card("Memories", s.memories, `${s.memories_current} current · ${s.memories_superseded} superseded · ${s.memories_unresolved} unresolved`)}
        {card("Entities", s.entities)}
        {card("Graph nodes", g ? Object.entries(g).filter(([k]) => k.startsWith("nodes_")).reduce((a, [, v]) => a + (v as number), 0) : "offline")}
        {card("Graph relationships", g ? (g as Rec).edges : "offline", g ? "" : "start FalkorDB: docker compose up -d")}
      </div>
      <div className="row">
        <section>
          <h3>Memories by verification</h3>
          {Object.entries(s.by_verification as Rec).map(([k, v]) => (
            <span key={k} className={`badge ${k}`}>{k.replace("_", " ")}: {v as number}</span>
          ))}
          <h3>By category</h3>
          {Object.entries(s.by_category as Rec).map(([k, v]) => (
            <span key={k} className="badge">{k}: {v as number}</span>
          ))}
          <h3>Extraction progress</h3>
          <p className="muted">{totalUser} messages processed by an extractor</p>
        </section>
        <section>
          <h3>Imports</h3>
          <table>
            <thead><tr><th>Job</th><th>Source</th><th>Status</th><th>Started</th></tr></thead>
            <tbody>
              {d.imports.map((j: Rec) => (
                <tr key={j.id}><td>{j.id}</td><td>{j.adapter}</td><td>{j.status}</td><td>{when(j.started_at)}</td></tr>
              ))}
              {d.live_jobs.map((j: Rec) => (
                <tr key={j.id}><td>{j.id}</td><td>ui{j.dry_run ? " (dry run)" : ""}</td><td>{j.status}</td><td>{when(j.started)}</td></tr>
              ))}
            </tbody>
          </table>
          <h3>Failed or incomplete records ({s.import_errors})</h3>
          {d.import_errors.length === 0 && <p className="muted">None.</p>}
          <ul>
            {d.import_errors.map((e: Rec, i: number) => (
              <li key={i}><code>{e.ref ?? `item ${e.item_index}`}</code>: {e.error}</li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
