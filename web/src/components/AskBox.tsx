import { useState } from "react";
import { api, Rec } from "../api";
import { VERIF_LABEL } from "../views/Memories";

// "support:lexical:0.65" -> a sentence a person can read
export function whyText(w: string): string | null {
  const m = /^support:(\w+)(?::([\d.]+))?/.exec(w);
  if (!m) return null;
  const pct = m[2] ? ` (${Math.round(parseFloat(m[2]) * 100)}%)` : "";
  switch (m[1]) {
    case "lexical": return `matches the words you used${pct}`;
    case "entity": return "it is about something you named";
    case "message": return `found through an original message that matched${pct}`;
    case "vector": return `similar in meaning${pct}`;
    case "supersession": return "the current value that replaced what matched";
    default: return null;
  }
}

type Props = { examples: string[]; onOpenMemory?: (id: string) => void; initial?: string };

/** Ask a question the way an assistant would, and show the answer, why it was given, and (just as importantly) when it is withheld. */
export default function AskBox({ examples, onOpenMemory, initial = "" }: Props) {
  const [q, setQ] = useState(initial);
  const [res, setRes] = useState<Rec | null>(null);
  const [asked, setAsked] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const ask = (text: string) => {
    const query = text.trim();
    if (!query) return;
    setQ(query); setBusy(true); setErr("");
    api("/api/retrieve", { method: "POST", body: { query, limit: 3, token_budget: 1500, use_graph: true } })
      .then((r) => { setRes(r); setAsked(query); })
      .catch((e) => { setRes(null); setErr(String(e.message)); })
      .finally(() => setBusy(false));
  };

  return (
    <section className="ask">
      <h2>Ask your memory</h2>
      <p className="muted">This is exactly what an assistant would be told. If nothing is stored, it says so instead of guessing.</p>
      <div className="askrow">
        <input
          placeholder="Ask anything about yourself, your work, your decisions…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") ask(q); }}
          aria-label="Ask your memory"
        />
        <button className="primary" disabled={!q.trim() || busy} onClick={() => ask(q)}>{busy ? "Looking…" : "Ask"}</button>
      </div>
      <div className="chips">
        {examples.map((e) => <button key={e} className="chip" onClick={() => ask(e)}>{e}</button>)}
      </div>
      {err && <p className="error">{err}</p>}
      {res && res.sufficient_evidence && (
        <div className="answer ok">
          <div className="verdict">Answered from {res.memories.length} {res.memories.length === 1 ? "memory" : "memories"}</div>
          <div className="muted">“{asked}” · {res.stats.latency_ms} ms · {res.stats.mode === "graph" ? "graph" : "local database"}</div>
          {res.memories.map((m: Rec) => (
            <div key={m.id} className={`ans ${m.status}`}>
              <div className="stmt" onClick={() => onOpenMemory?.(m.id)}>{m.statement}</div>
              <div className="meta">
                <span className={`badge ${m.status}`}>{m.status === "superseded" ? "older value, kept as history" : m.status}</span>
                <span className={`badge ${m.verification}`}>{VERIF_LABEL[m.verification] ?? m.verification}</span>
                {m.why_selected.map((w: string) => whyText(w)).filter(Boolean).slice(0, 1).map((t: string) => <span key={t} className="why">why: {t}</span>)}
              </div>
              {m.evidence[0] && <blockquote>{m.evidence[0].snippet}<div className="muted">{m.evidence[0].kind === "message" ? `from “${m.evidence[0].conversation}”` : m.evidence[0].source}</div></blockquote>}
            </div>
          ))}
        </div>
      )}
      {res && !res.sufficient_evidence && (
        <div className="answer none">
          <div className="verdict">Nothing stored about that, so it will not guess.</div>
          <div className="muted">“{asked}” · {res.stats.latency_ms} ms · no memory has enough support, and no unrelated memory was returned.</div>
        </div>
      )}
      <p className="muted lock">🔒 Private memories (health, money, family, identity documents) are never part of an answer unless you grant access.</p>
    </section>
  );
}
