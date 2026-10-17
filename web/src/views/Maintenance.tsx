import { useEffect, useState } from "react";
import { api, Rec } from "../api";

type Props = { onOpenMemory: (id: string) => void };

function useLoad<T = Rec>(path: string, deps: unknown[] = []): [T | null, () => void, string] {
  const [d, setD] = useState<T | null>(null);
  const [err, setErr] = useState("");
  const [n, setN] = useState(0);
  useEffect(() => { api<T>(path).then((r) => { setD(r); setErr(""); }).catch((e) => setErr(String(e.message))); }, [path, n, ...deps]); // eslint-disable-line react-hooks/exhaustive-deps
  return [d, () => setN((x) => x + 1), err];
}

export default function Maintenance({ onOpenMemory }: Props) {
  return (
    <div>
      <Import />
      <Consolidate />
      <EntityReview />
      <Contradictions onOpenMemory={onOpenMemory} />
      <PrivateAccess onOpenMemory={onOpenMemory} />
      <ExportBackup />
      <Clients />
      <Health />
    </div>
  );
}

function PrivateAccess({ onOpenMemory }: Props) {
  const [st, reload] = useLoad("/api/private/status");
  const [fr, reloadFr] = useLoad("/api/forget-requests");
  const [oldPin, setOldPin] = useState("");
  const [newPin, setNewPin] = useState("");
  const [pin, setPin] = useState("");
  const [viewPin, setViewPin] = useState("");
  const [client, setClient] = useState("claude-code");
  const [minutes, setMinutes] = useState(10);
  const [uses, setUses] = useState(25);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const done = (ok: string) => () => { setMsg(ok); setErr(""); setPin(""); setOldPin(""); setNewPin(""); reload(); reloadFr(); };
  const fail = (e: Error) => { setErr(e.message); setMsg(""); };
  return (
    <section>
      <h3>Private memories</h3>
      <p className="muted">
        Health, money, identity-document and family facts are hidden from every assistant. To let one assistant read them for a few minutes, set a PIN once, then grant it.
        The PIN is stored hashed and never shown to an assistant.
      </p>
      <div className="filters">
        <b>{st?.pin_set ? "PIN is set" : "No PIN yet"}</b>
        {st?.pin_set && <input type="password" placeholder="current PIN" value={oldPin} onChange={(e) => setOldPin(e.target.value)} />}
        <input type="password" placeholder="new PIN (6+ characters)" value={newPin} onChange={(e) => setNewPin(e.target.value)} />
        <button disabled={newPin.length < 6} onClick={() => api("/api/private/pin", { method: "POST", body: { new_pin: newPin, old_pin: oldPin || null } }).then(done("PIN saved")).catch(fail)}>
          {st?.pin_set ? "Change PIN" : "Set PIN"}
        </button>
      </div>
      <div className="filters">
        <input placeholder="client name" value={client} onChange={(e) => setClient(e.target.value)} title="the MEMBRIDGE_CLIENT_NAME the assistant connects with" />
        <input type="number" min={1} max={120} style={{ width: 90 }} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} title="minutes" />
        <input type="number" min={1} max={200} style={{ width: 90 }} value={uses} onChange={(e) => setUses(Number(e.target.value))} title="private reads allowed" />
        <input type="password" placeholder="PIN" value={pin} onChange={(e) => setPin(e.target.value)} />
        <button disabled={!st?.pin_set || !pin || !client.trim()} onClick={() => api("/api/private/grant", { method: "POST", body: { pin, client: client.trim(), minutes, uses } }).then(done("Access granted")).catch(fail)}>
          Grant access
        </button>
        <button onClick={() => api("/api/private/revoke", { method: "POST", body: {} }).then(done("All access revoked")).catch(fail)}>Revoke all</button>
      </div>
      <div className="filters">
        <b>View private memories here</b>
        <input type="password" placeholder="PIN" value={viewPin} onChange={(e) => setViewPin(e.target.value)} />
        <button disabled={!st?.pin_set || !viewPin} onClick={() => api("/api/private/unlock", { method: "POST", body: { pin: viewPin } }).then(() => { setViewPin(""); setMsg("Private memories unlocked in this dashboard for 10 minutes. Tick “show private” in Memories."); setErr(""); }).catch(fail)}>
          Unlock for 10 minutes
        </button>
        <button onClick={() => api("/api/private/lock", { method: "POST", body: {} }).then(() => { setMsg("Private view locked."); setErr(""); }).catch(fail)}>Lock now</button>
      </div>
      {msg && <p className="muted">{msg}</p>}
      {err && <p className="error">{err}</p>}
      {(st?.grants?.length ?? 0) > 0 && st && (
        <table className="kv">
          <tbody>
            {st.grants.slice(0, 8).map((g: Rec) => (
              <tr key={g.id}>
                <td>{g.client}</td>
                <td>{g.active ? `active · ${g.uses_left} reads left · until ${new Date(g.expires_at * 1000).toLocaleTimeString()}` : g.revoked ? "revoked" : "ended"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {(fr?.requests?.length ?? 0) > 0 && fr && (
        <div className="confirm">
          <b>An assistant asked to forget:</b>
          {fr.requests.map((r: Rec) => (
            <p key={r.id}>
              “{String(r.statement).slice(0, 120)}” <button onClick={() => onOpenMemory(r.id)}>Review</button>
            </p>
          ))}
          <p className="muted">Open the memory and choose Delete… to confirm; nothing is forgotten until you do.</p>
        </div>
      )}
    </section>
  );
}

function Import() {
  const [path, setPath] = useState("");
  const [dry, setDry] = useState(true);
  const [job, setJob] = useState<Rec | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setInterval(() => api(`/api/imports/${job.id}`).then(setJob).catch((e) => setErr(e.message)), 1000);
    return () => clearInterval(t);
  }, [job]);

  return (
    <section>
      <h3>Import an export</h3>
      <p className="muted">Path to a ChatGPT or Claude export (.zip, folder or conversations.json) on this computer. A dry run validates without writing.</p>
      <div className="filters">
        <input style={{ flex: "1 1 360px" }} placeholder="C:\Users\you\Downloads\conversations-000.zip" value={path} onChange={(e) => setPath(e.target.value)} />
        <label><input type="checkbox" checked={dry} onChange={() => setDry(!dry)} /> dry run</label>
        <button disabled={!path.trim()} onClick={() => { setErr(""); api("/api/imports", { method: "POST", body: { path, dry_run: dry } }).then(setJob).catch((e) => setErr(e.message)); }}>Start</button>
      </div>
      {err && <p className="error">{err}</p>}
      {job && (
        <div className="confirm">
          <b>{job.id}</b> — {job.status}{job.error ? `: ${job.error}` : ""}
          {job.manifest && (
            <pre>{JSON.stringify({ adapter: job.manifest.adapter, conversations_new: job.manifest.conversations_new, conversations_unchanged: job.manifest.conversations_unchanged, messages_new: job.manifest.messages_new, failures: job.manifest.failures.length, notes: job.manifest.notes.slice(0, 3) }, null, 2)}</pre>
          )}
        </div>
      )}
    </section>
  );
}

function Consolidate() {
  const [out, setOut] = useState<Rec | null>(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const run = (scope: string) => { setBusy(scope); setErr(""); api("/api/consolidate", { method: "POST", body: { scope } }).then(setOut).catch((e) => setErr(e.message)).finally(() => setBusy("")); };
  return (
    <section>
      <h3>Consolidation</h3>
      <div className="actions">
        <button disabled={!!busy} onClick={() => run("incremental")}>{busy === "incremental" ? "Running…" : "Extract new messages"}</button>
        <button disabled={!!busy} onClick={() => run("summaries")}>{busy === "summaries" ? "Running…" : "Refresh summaries"}</button>
        <button disabled={!!busy} onClick={() => run("graph")}>{busy === "graph" ? "Running…" : "Rebuild graph"}</button>
      </div>
      {err && <p className="error">{err}</p>}
      {out && <pre>{JSON.stringify(out, null, 2)}</pre>}
    </section>
  );
}

function EntityReview() {
  const [d, reload, err] = useLoad("/api/entity-review");
  const [msg, setMsg] = useState("");
  const act = (p: Promise<Rec>) => p.then((r) => { setMsg(JSON.stringify(r)); reload(); }).catch((e) => setMsg(e.message));
  return (
    <section>
      <h3>Entity resolution</h3>
      <p className="muted">Aliases such as Postgres / PostgreSQL / pg become one node. Certain matches merge automatically; uncertain ones wait here. Every merge can be undone.</p>
      <div className="actions">
        <button onClick={() => act(api("/api/resolve", { method: "POST", body: { dry_run: true } }))}>Preview merges</button>
        <button onClick={() => act(api("/api/resolve", { method: "POST", body: { dry_run: false } }))}>Run resolution</button>
      </div>
      {err && <p className="error">{err}</p>}
      {msg && <pre>{msg}</pre>}
      <h4>Needs your decision ({d?.pending.length ?? 0})</h4>
      {d?.pending.length === 0 && <p className="muted">No uncertain pairs.</p>}
      {d?.pending.map((p: Rec) => (
        <div key={p.a_id + p.b_id} className="item">
          <b>{p.a_name}</b> ↔ <b>{p.b_name}</b> <span className="muted">similarity {Number(p.score).toFixed(2)}</span>
          <div className="actions">
            <button onClick={() => act(api("/api/entity-review/decide", { method: "POST", body: { a_id: p.a_id, b_id: p.b_id, accept: true } }))}>Same thing — merge</button>
            <button onClick={() => act(api("/api/entity-review/decide", { method: "POST", body: { a_id: p.a_id, b_id: p.b_id, accept: false } }))}>Different — never merge</button>
          </div>
        </div>
      ))}
      <h4>Merges ({d?.merges.length ?? 0})</h4>
      {d?.merges.map((m: Rec) => (
        <div key={m.id} className="item">
          {m.absorbed} → {m.into_name} <span className="muted">{m.method} · {Number(m.score).toFixed(2)} · by {m.decided_by}</span>
          <button onClick={() => act(api("/api/entity-review/split", { method: "POST", body: { merge_id: m.id } }))}>Undo</button>
        </div>
      ))}
    </section>
  );
}

function Contradictions({ onOpenMemory }: Props) {
  const [d, reload, err] = useLoad("/api/contradictions");
  const [msg, setMsg] = useState("");
  const keep = (k: string, x: string) => api("/api/contradictions/resolve", { method: "POST", body: { keep_id: k, drop_id: x } }).then(() => { setMsg("resolved"); reload(); }).catch((e) => setMsg(e.message));
  return (
    <section>
      <h3>Unresolved contradictions ({d?.contradictions.length ?? 0})</h3>
      {err && <p className="error">{err}</p>}
      {d?.contradictions.length === 0 && <p className="muted">None. Changed decisions are recorded as history automatically; only genuinely incompatible claims appear here.</p>}
      {d?.contradictions.map((c: Rec, i: number) => (
        <div key={i} className="item">
          <div><a onClick={() => onOpenMemory(c.src_id)}>A</a>: {c.a_text}</div>
          <div><a onClick={() => onOpenMemory(c.dst_id)}>B</a>: {c.b_text}</div>
          <div className="actions">
            <button onClick={() => keep(c.src_id, c.dst_id)}>Keep A (B becomes history)</button>
            <button onClick={() => keep(c.dst_id, c.src_id)}>Keep B (A becomes history)</button>
          </div>
        </div>
      ))}
      {msg && <p className="muted">{msg}</p>}
    </section>
  );
}

function ExportBackup() {
  const [fmt, setFmt] = useState("json");
  const [raw, setRaw] = useState(false);
  const [sens, setSens] = useState(false);
  const [out, setOut] = useState<Rec | null>(null);
  const [err, setErr] = useState("");
  return (
    <section>
      <h3>Export & backup</h3>
      <div className="filters">
        <select value={fmt} onChange={(e) => setFmt(e.target.value)}><option value="json">JSON</option><option value="markdown">Markdown</option></select>
        <label><input type="checkbox" checked={raw} onChange={() => setRaw(!raw)} /> include raw archive</label>
        <label><input type="checkbox" checked={sens} onChange={() => setSens(!sens)} /> include sensitive memories</label>
        <button onClick={() => api("/api/export", { method: "POST", body: { format: fmt, include_raw: raw, include_sensitive: sens } }).then(setOut).catch((e) => setErr(e.message))}>Export</button>
        <button onClick={() => api("/api/backup", { method: "POST" }).then(setOut).catch((e) => setErr(e.message))}>Back up database</button>
      </div>
      {err && <p className="error">{err}</p>}
      {out && (
        <div className="confirm">
          <pre>{JSON.stringify(out, null, 2)}</pre>
          {out.path && /membridge-export-/.test(out.path) && <a href={`/api/exports/${out.path.split(/[\\/]/).pop()}`}>Download</a>}
        </div>
      )}
    </section>
  );
}

function Clients() {
  const [d, , err] = useLoad("/api/clients");
  const [copied, setCopied] = useState("");
  return (
    <section>
      <h3>Connect AI clients (MCP)</h3>
      {d && <p className="muted">{d.note}</p>}
      {err && <p className="error">{err}</p>}
      {d?.clients.map((c: Rec) => (
        <details key={c.id}>
          <summary>
            {c.name} {c.installed === true && <span className="badge user_stated">installed</span>} {c.installed === false && <span className="badge">not found</span>}
            <span className={`badge ${String(c.status).startsWith("Verified") ? "user_stated" : "superseded"}`}>{String(c.status).startsWith("Verified") ? "verified" : "unverified"}</span>
          </summary>
          <p className="muted">{c.status}</p>
          <p>{c.how}</p>
          <pre>{JSON.stringify(c.config, null, 2)}</pre>
          <button onClick={() => navigator.clipboard.writeText(JSON.stringify(c.config, null, 2)).then(() => setCopied(c.id))}>{copied === c.id ? "Copied" : "Copy config"}</button>
        </details>
      ))}
    </section>
  );
}

function Health() {
  const [d, reload, err] = useLoad("/api/health");
  return (
    <section>
      <h3>Health <button onClick={reload}>refresh</button></h3>
      {err && <p className="error">{err}</p>}
      {d && (
        <ul>
          <li>SQLite: {d.sqlite.ok ? "ok" : "ERROR"} ({Math.round((d.sqlite.bytes ?? 0) / 1024)} KB)</li>
          <li>FalkorDB: {d.graph.ok ? "reachable" : "not reachable (retrieval runs in degraded SQLite mode)"}</li>
          <li>Process memory: {d.process.rss_mb} MB · system RAM free {d.system.ram_free_mb} MB of {d.system.ram_total_mb} MB · disk free {d.system.disk_free_gb} GB</li>
        </ul>
      )}
    </section>
  );
}
