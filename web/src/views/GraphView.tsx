import { useEffect, useMemo, useRef, useState } from "react";
import cytoscape from "cytoscape";
import { api, Rec } from "../api";

type Props = { center: string | null; onCenter: (id: string | null) => void; onOpenMemory: (id: string) => void };

const COLORS: Record<string, string> = {
  Memory: "#4c78a8", Entity: "#54a24b", Conversation: "#b279a2", Message: "#9d9d9d", Source: "#eeca3b", User: "#e45756", Community: "#f58518",
};

export default function GraphView({ center, onCenter, onOpenMemory }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const cy = useRef<cytoscape.Core | null>(null);
  const [data, setData] = useState<Rec | null>(null);
  const [err, setErr] = useState("");
  const [depth, setDepth] = useState(1);
  const [labels, setLabels] = useState<Record<string, boolean>>({ Memory: true, Entity: true, Conversation: true, Message: false, Source: true, User: true, Community: true });
  const [hideHistory, setHideHistory] = useState(false);
  const [search, setSearch] = useState("");
  const [found, setFound] = useState<Rec[]>([]);
  const [picked, setPicked] = useState<Rec | null>(null);

  useEffect(() => {
    if (search.trim().length < 2) { setFound([]); return; }
    const t = setTimeout(() => api(`/api/entities?q=${encodeURIComponent(search)}`).then((r) => setFound(r.items)).catch(() => setFound([])), 200);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    if (!center) return;
    setErr(""); setPicked(null);
    api(`/api/graph/neighborhood?center=${encodeURIComponent(center)}&depth=${depth}`).then(setData).catch((e) => { setData(null); setErr(String(e.message)); });
  }, [center, depth]);

  const view = useMemo(() => {
    if (!data) return { nodes: [] as Rec[], edges: [] as Rec[] };
    const nodes = (data.nodes as Rec[]).filter((n) => labels[n.label] !== false && !(hideHistory && n.status === "superseded"));
    const ids = new Set(nodes.map((n) => n.id));
    return { nodes, edges: (data.edges as Rec[]).filter((e) => ids.has(e.source) && ids.has(e.target)) };
  }, [data, labels, hideHistory]);

  useEffect(() => {
    if (!box.current) return;
    cy.current?.destroy();
    const c = cytoscape({
      container: box.current,
      elements: [
        ...view.nodes.map((n) => ({ data: { id: n.id, label: (n.text || n.id).slice(0, 28), kind: n.label, status: n.status } })),
        ...view.edges.map((e, i) => ({ data: { id: `e${i}`, source: e.source, target: e.target, label: e.type } })),
      ],
      layout: { name: "cose", animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 90 },
      style: [
        { selector: "node", style: { label: "data(label)", "font-size": 9, "text-wrap": "wrap", "text-max-width": "90px", color: "#888", "background-color": (el: cytoscape.NodeSingular) => COLORS[el.data("kind")] ?? "#999", width: 18, height: 18 } },
        { selector: "node[status = 'superseded']", style: { opacity: 0.45, "border-width": 2, "border-style": "dashed", "border-color": "#999" } },
        { selector: "edge", style: { label: "data(label)", "font-size": 7, color: "#999", "curve-style": "bezier", "target-arrow-shape": "triangle", width: 1, "line-color": "#bbb", "target-arrow-color": "#bbb" } },
        { selector: `node[id = "${center ?? ""}"]`, style: { width: 28, height: 28, "border-width": 3, "border-color": "#e45756" } },
      ],
    });
    c.on("tap", "node", (ev) => setPicked(view.nodes.find((n) => n.id === ev.target.id()) ?? null));
    cy.current = c;
    return () => c.destroy();
  }, [view, center]);

  return (
    <div className="split">
      <section className="list">
        <h3>Graph explorer</h3>
        <p className="muted">Shows a bounded neighbourhood (never the whole graph). Pick an entity, or open a memory and choose “Show in graph”.</p>
        <input placeholder="Find an entity…" value={search} onChange={(e) => setSearch(e.target.value)} />
        {found.map((e) => (
          <div key={e.id} className="item" onClick={() => { onCenter(e.id); setSearch(""); }}>
            {e.name} <span className="muted">{e.type} · {e.memories} memories</span>
          </div>
        ))}
        <label>Depth <select value={depth} onChange={(e) => setDepth(Number(e.target.value))}><option value={1}>1 hop</option><option value={2}>2 hops</option></select></label>
        <div className="filters">
          {Object.keys(labels).map((l) => (
            <label key={l}><input type="checkbox" checked={labels[l]} onChange={() => setLabels({ ...labels, [l]: !labels[l] })} /> <span style={{ color: COLORS[l] }}>●</span> {l}</label>
          ))}
          <label><input type="checkbox" checked={hideHistory} onChange={() => setHideHistory(!hideHistory)} /> hide superseded</label>
        </div>
        {data?.truncated && <p className="muted">Truncated: showing the strongest connections only.</p>}
        {picked && (
          <div className="confirm">
            <b>{picked.label}</b>
            <p>{picked.text}</p>
            <div className="actions">
              {picked.label === "Memory" && <button onClick={() => onOpenMemory(picked.id)}>Open memory</button>}
              <button onClick={() => onCenter(picked.id)}>Recentre here</button>
            </div>
          </div>
        )}
      </section>
      <section className="detail">
        {err && <p className="error">{err}</p>}
        {!center && !err && <p className="muted">Nothing selected yet.</p>}
        <div ref={box} className="graphbox" />
      </section>
    </div>
  );
}
