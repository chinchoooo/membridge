import { useState } from "react";
import { api, CATEGORIES, Rec, when } from "../api";

type Props = { onOpenMemory: (id: string) => void; onGraph: (id: string) => void };

export default function Retrieval({ onOpenMemory, onGraph }: Props) {
  const [query, setQuery] = useState("");
  const [budget, setBudget] = useState(1500);
  const [limit, setLimit] = useState(8);
  const [useGraph, setUseGraph] = useState(true);
  const [useVector, setUseVector] = useState(false);
  const [asOf, setAsOf] = useState("");
  const [knownAt, setKnownAt] = useState("");
  const [category, setCategory] = useState("");
  const [res, setRes] = useState<Rec | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const run = () => {
    setBusy(true); setErr("");
    api("/api/retrieve", {
      method: "POST",
      body: { query, token_budget: budget, limit, use_graph: useGraph, use_vector: useVector, as_of: asOf || null, known_at: knownAt || null, category: category || null },
    })
      .then(setRes)
      .catch((e) => { setRes(null); setErr(String(e.message)); })
      .finally(() => setBusy(false));
  };

  return (
    <div>
      <h3>Retrieval test</h3>
      <p className="muted">Exactly what an AI client would receive for this question, with the evidence and reasoning behind each result.</p>
      <div className="filters">
        <input style={{ flex: "1 1 340px" }} placeholder="Ask a question…" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && query.trim() && run()} />
        <label>token budget <input type="number" min={200} max={8000} step={100} value={budget} onChange={(e) => setBudget(Number(e.target.value))} /></label>
        <label>results <input type="number" min={1} max={50} value={limit} onChange={(e) => setLimit(Number(e.target.value))} /></label>
        <button disabled={!query.trim() || busy} onClick={run}>{busy ? "Searching…" : "Search"}</button>
      </div>
      <div className="filters">
        <label><input type="checkbox" checked={useGraph} onChange={() => setUseGraph(!useGraph)} /> use FalkorDB graph</label>
        <label><input type="checkbox" checked={useVector} onChange={() => setUseVector(!useVector)} /> semantic channel</label>
        <label>true as of <input type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} /></label>
        <label>known at <input type="date" value={knownAt} onChange={(e) => setKnownAt(e.target.value)} /></label>
        <select value={category} onChange={(e) => setCategory(e.target.value)}>
          <option value="">any category</option>
          {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>
      {err && <p className="error">{err}</p>}
      {res && (
        <div>
          <div className="cards">
            <div className="card"><div className="big">{res.stats.latency_ms} ms</div>latency ({res.stats.mode})</div>
            <div className="card"><div className="big">{res.stats.tokens_estimate}</div>of {res.stats.token_budget} tokens (estimate)</div>
            <div className="card"><div className="big">{res.memories.length}</div>memories · {res.source_messages.length} exact messages</div>
            <div className="card"><div className="big">{res.stats.temporal_intent}</div>temporal intent{res.stats.global_intent ? " · global question" : ""}</div>
          </div>
          <p>
            {res.sufficient_evidence ? <span className="badge user_stated">enough evidence</span> : <span className="badge superseded">insufficient evidence</span>}{" "}
            channels: {Object.entries(res.stats.channels as Rec).map(([k, v]) => <span key={k} className="badge">{k}: {v as number}</span>)}
            {" "}PPR pushes {res.stats.ppr_pushes}, graph nodes loaded {res.stats.graph_nodes_loaded}
          </p>
          {res.note && <p className="muted">{res.note}</p>}
          {res.matched_entities.length > 0 && <p>Entities recognised in the question: {res.matched_entities.join(", ")}</p>}
          {res.community_summaries.length > 0 && (
            <div className="confirm">
              <b>Hierarchical summaries</b>
              {res.community_summaries.map((c: Rec) => <pre key={c.id}>{`L${c.level} (${c.method})\n${c.summary}`}</pre>)}
            </div>
          )}
          <h4>Retrieved memories</h4>
          {res.memories.map((m: Rec) => (
            <div key={m.id} className={`item ${m.status}`}>
              <div onClick={() => onOpenMemory(m.id)} style={{ cursor: "pointer" }}><b>{m.statement}</b></div>
              <div className="meta">
                <span className="badge">score {m.score}</span>
                <span className={`badge ${m.status}`}>{m.status}</span>
                <span className={`badge ${m.verification}`}>{m.verification}</span>
                {m.channels.map((c: string) => <span key={c} className="badge">{c}</span>)}
                {m.ppr != null && <span className="badge">PPR {m.ppr}</span>}
                <a onClick={() => onGraph(m.id)}>graph</a>
              </div>
              <ul className="muted">
                {m.why_selected.map((w: string, i: number) => <li key={i}>{w}</li>)}
              </ul>
              {m.uncertainty !== "none" && <p className="muted">⚠ {m.uncertainty}</p>}
              {m.evidence.map((e: Rec, i: number) => (
                <blockquote key={i}>{e.snippet}<div className="muted">{e.kind === "message" ? `${e.conversation} · ${when(e.ts)}` : e.source}</div></blockquote>
              ))}
            </div>
          ))}
          {res.graph_paths.length > 0 && (<><h4>Graph paths used</h4><ul>{res.graph_paths.map((p: string, i: number) => <li key={i}><code>{p}</code></li>)}</ul></>)}
          {res.source_messages.length > 0 && (
            <>
              <h4>Original messages that matched (untrusted text)</h4>
              {res.source_messages.map((x: Rec) => <blockquote key={x.message_id}>{x.snippet}<div className="muted">{x.conversation} · {x.role} · {when(x.ts)}</div></blockquote>)}
            </>
          )}
        </div>
      )}
    </div>
  );
}
