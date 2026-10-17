import { useCallback, useEffect, useState } from "react";
import { api, CATEGORIES, Rec, when } from "../api";

type Props = { selected: string | null; onSelect: (id: string | null) => void; onGraph: (id: string) => void };

export const VERIF_LABEL: Record<string, string> = {
  user_stated: "user-stated",
  user_verified: "verified by you",
  client_asserted: "added by an assistant for you",
  provider_stated: "from an AI provider's memory",
  model_inferred: "inferred by a model",
};

export default function Memories({ selected, onSelect, onGraph }: Props) {
  const [q, setQ] = useState("");
  const [category, setCategory] = useState("");
  const [status, setStatus] = useState("");
  const [verification, setVerification] = useState("");
  const [showPrivate, setShowPrivate] = useState(false);
  const [list, setList] = useState<Rec | null>(null);
  const [err, setErr] = useState("");

  const load = useCallback(() => {
    const p = new URLSearchParams({ q, category, status, verification, limit: "100", show_private: String(showPrivate) });
    api("/api/memories?" + p).then((d) => { setList(d); setErr(""); }).catch((e) => setErr(String(e.message)));
  }, [q, category, status, verification, showPrivate]);
  useEffect(() => {
    const t = setTimeout(load, 200);
    return () => clearTimeout(t);
  }, [load]);

  return (
    <div className="split">
      <section className="list">
        <div className="filters">
          <input placeholder="Search memories…" value={q} onChange={(e) => setQ(e.target.value)} />
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">all categories</option>
            {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
          </select>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">any status</option>
            {["current", "superseded", "unresolved"].map((c) => <option key={c}>{c}</option>)}
          </select>
          <select value={verification} onChange={(e) => setVerification(e.target.value)}>
            <option value="">any verification</option>
            {Object.keys(VERIF_LABEL).map((c) => <option key={c} value={c}>{VERIF_LABEL[c]}</option>)}
          </select>
          <label title="Private memories (health, money, family...) are hidden from assistants and from this list unless you ask">
            <input type="checkbox" checked={showPrivate} onChange={() => setShowPrivate(!showPrivate)} /> show private
          </label>
        </div>
        {err && <p className="error">{err}</p>}
        <p className="muted">{list ? `${list.total} memories` : "Loading…"}</p>
        {list?.items.map((m: Rec) => (
          <div key={m.id} className={`item ${selected === m.id ? "sel" : ""} ${m.status}`} onClick={() => onSelect(m.id)}>
            <div>{m.statement}</div>
            <div className="meta">
              <span className="badge">{m.category}</span>
              <span className={`badge ${m.status}`}>{m.status}</span>
              <span className={`badge ${m.verification}`}>{VERIF_LABEL[m.verification] ?? m.verification}</span>
            </div>
          </div>
        ))}
        <NewMemory onSaved={(id) => { load(); onSelect(id); }} />
      </section>
      <section className="detail">
        {selected ? (
          <Detail id={selected} onChanged={load} onGraph={onGraph} onSelect={onSelect} />
        ) : (
          <p className="muted">Select a memory to inspect its evidence, relationships and history.</p>
        )}
      </section>
    </div>
  );
}

function NewMemory({ onSaved }: { onSaved: (id: string) => void }) {
  const [text, setText] = useState("");
  const [msg, setMsg] = useState("");
  return (
    <details>
      <summary>Add a memory manually</summary>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} placeholder="Something you want remembered" />
      <button
        disabled={text.trim().length < 3}
        onClick={() =>
          api("/api/memories", { method: "POST", body: { content: text, category: "fact", source: "user_stated" } })
            .then((r) => { setText(""); setMsg(r.result); onSaved(r.memory_id); })
            .catch((e) => setMsg(e.message))
        }
      >
        Save
      </button>
      <span className="muted"> {msg}</span>
    </details>
  );
}

function Detail({ id, onChanged, onGraph, onSelect }: { id: string; onChanged: () => void; onGraph: (id: string) => void; onSelect: (id: string | null) => void }) {
  const [d, setD] = useState<Rec | null>(null);
  const [err, setErr] = useState("");
  const [edit, setEdit] = useState<Rec | null>(null);
  const [token, setToken] = useState<Rec | null>(null);
  const [conv, setConv] = useState<Rec | null>(null);

  const load = useCallback(() => {
    setErr(""); setToken(null); setEdit(null);
    api(`/api/memories/${id}`).then(setD).catch((e) => { setD(null); setErr(String(e.message)); });
  }, [id]);
  useEffect(load, [load]);

  if (err) return <p className="error">{err}</p>;
  if (!d) return <p>Loading…</p>;
  const m = d.memory, prov = d.provenance;
  return (
    <div>
      <h3>{m.statement}</h3>
      <div className="meta">
        <span className="badge">{m.category}</span>
        <span className={`badge ${m.status}`}>{m.status}</span>
        <span className={`badge ${m.verification}`}>{VERIF_LABEL[m.verification] ?? m.verification}</span>
        <span className="badge">confidence {m.confidence}</span>
        <span className="badge">mentioned {m.mention_count}×</span>
        <span className="badge">{m.extraction_method}</span>
        {m.sensitivity !== "normal" && <span className="badge superseded" title="Hidden from assistants unless you grant access">private</span>}
      </div>
      <table className="kv">
        <tbody>
          <tr><td>true from</td><td>{when(m.valid_at ?? m.valid_from)}</td></tr>
          <tr><td>true until</td><td>{m.invalid_at ? when(m.invalid_at) : "still true"}</td></tr>
          <tr><td>learned</td><td>{when(m.recorded_at)}</td></tr>
          <tr><td>retracted</td><td>{m.expired_at ? `${when(m.expired_at)} (later corrected as never true)` : "no"}</td></tr>
          <tr><td>importance / activation</td><td>{Number(m.importance).toFixed(2)} / {Number(m.activation).toFixed(2)}</td></tr>
        </tbody>
      </table>
      <div className="actions">
        <button onClick={() => onGraph(m.id)}>Show in graph</button>
        <button onClick={() => setEdit({ statement: m.statement, category: m.category, verification: m.verification, confidence: m.confidence, reason: "" })}>Edit</button>
        {m.verification !== "user_verified" && (
          <button onClick={() => api(`/api/memories/${id}`, { method: "PATCH", body: { changes: { verification: "user_verified" }, reason: "verified in dashboard" } }).then(() => { load(); onChanged(); })}>
            Mark verified
          </button>
        )}
        <button title="Private memories are hidden from every assistant unless you grant access" onClick={() => api(`/api/memories/${id}/private`, { method: "POST", body: { private: m.sensitivity === "normal" } }).then(() => { load(); onChanged(); })}>
          {m.sensitivity === "normal" ? "Mark private" : "Make visible to assistants"}
        </button>
        <button className="danger" onClick={() => api(`/api/memories/${id}`, { method: "DELETE" }).then(setToken)}>Delete…</button>
      </div>
      {token && (
        <div className="confirm">
          <p>Permanently forget “{token.will_forget}”? Its text and evidence links are erased; a tombstone stops it being re-extracted.</p>
          <button className="danger" onClick={() => api(`/api/memories/${id}?confirm_token=${token.confirm_token}`, { method: "DELETE" }).then(() => { onChanged(); onSelect(null); })}>
            Yes, forget it
          </button>
          <button onClick={() => setToken(null)}>Cancel</button>
        </div>
      )}
      {edit && (
        <div className="confirm">
          <textarea rows={3} value={edit.statement} onChange={(e) => setEdit({ ...edit, statement: e.target.value })} />
          <select value={edit.category} onChange={(e) => setEdit({ ...edit, category: e.target.value })}>
            {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
          </select>
          <input type="number" min={0} max={1} step={0.05} value={edit.confidence} onChange={(e) => setEdit({ ...edit, confidence: Number(e.target.value) })} />
          <input placeholder="Reason for the edit (kept in history)" value={edit.reason} onChange={(e) => setEdit({ ...edit, reason: e.target.value })} />
          <button
            disabled={!edit.reason}
            onClick={() => {
              const { reason, ...changes } = edit;
              api(`/api/memories/${id}`, { method: "PATCH", body: { changes, reason } }).then(() => { load(); onChanged(); }).catch((e) => setErr(e.message));
            }}
          >
            Save
          </button>
          <button onClick={() => setEdit(null)}>Cancel</button>
        </div>
      )}
      <h4>Evidence ({prov.sources.length})</h4>
      {prov.sources.map((s: Rec, i: number) => (
        <div className="evidence" key={i}>
          <div className="muted">
            {s.kind === "message" ? `${s.conversation} · ${s.role} · ${when(s.ts)}` : s.source}
            {s.kind === "message" && (s.quote_found_in_original ? " · quote verified in the original" : " · quote NOT found in the original")}
          </div>
          <blockquote>{s.quote}</blockquote>
          {s.kind === "message" && <button onClick={() => api(`/api/conversations/${s.conversation_id}`).then(setConv)}>Open source conversation</button>}
        </div>
      ))}
      {conv && (
        <div className="conv">
          <h4>{conv.title} <button onClick={() => setConv(null)}>close</button></h4>
          <p className="muted">Archived text, shown as untrusted data.</p>
          {conv.messages.map((x: Rec) => (
            <p key={x.id} className={x.alternate_branch ? "alt" : ""}>
              <b>{x.role}</b>{x.alternate_branch ? " (alternate branch)" : ""}: {x.text}
            </p>
          ))}
        </div>
      )}
      <h4>Links & history</h4>
      {m.links.length === 0 && <p className="muted">No links.</p>}
      {m.links.map((l: Rec, i: number) => (
        <p key={i}>
          {l.src_id === id ? "this" : <a onClick={() => onSelect(l.src_id)}>{l.src_id.slice(0, 12)}</a>} <b>{l.rel}</b>{" "}
          {l.dst_id === id ? "this" : <a onClick={() => onSelect(l.dst_id)}>{l.dst_id.slice(0, 12)}</a>} <span className="muted">({l.reason})</span>
        </p>
      ))}
      <h4>Related memories</h4>
      {d.related.related.map((r: Rec) => (
        <div key={r.id} className="item" onClick={() => onSelect(r.id)}>
          {r.statement}
          <div className="muted">{r.why_related.join(", ")} · {r.status}</div>
        </div>
      ))}
      <h4>Entities</h4>
      <p>{m.entities.join(", ") || "—"}</p>
    </div>
  );
}
