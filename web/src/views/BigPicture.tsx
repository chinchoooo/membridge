import { useEffect, useState } from "react";
import { api, Rec } from "../api";

export default function BigPicture() {
  const [d, setD] = useState<Rec | null>(null);
  useEffect(() => { api("/api/overview/global").then(setD).catch(() => setD(null)); }, []);
  if (!d || (d.areas.length === 0 && d.themes.length === 0)) return null;
  return (
    <section>
      <h3>The big picture</h3>
      <p className="muted">Hierarchical summaries built bottom-up from your facts. Each theme lists the facts it came from.</p>
      {d.areas.map((a: Rec) => <pre key={a.id}>{a.summary}</pre>)}
      {d.themes.map((t: Rec) => <details key={t.id}><summary>{String(t.summary).split("\n")[0]}</summary><pre>{t.summary}</pre></details>)}
    </section>
  );
}
