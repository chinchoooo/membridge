import { lazy, Suspense, useEffect, useState } from "react";
import Overview from "./views/Overview";
import Memories from "./views/Memories";
import Maintenance from "./views/Maintenance";
import Retrieval from "./views/Retrieval";
import BigPicture from "./views/BigPicture";
import { api, Rec } from "./api";

// The graph library is large; load it only when the Graph tab is opened.
const GraphView = lazy(() => import("./views/GraphView"));

export type Tab = "overview" | "memories" | "graph" | "retrieval" | "maintenance";
const TABS: [Tab, string][] = [
  ["overview", "Overview"],
  ["memories", "Memories"],
  ["graph", "Graph"],
  ["retrieval", "Retrieval test"],
  ["maintenance", "Maintenance"],
];

function fromHash(): { tab: Tab; arg: string | null } {
  const [t, a] = window.location.hash.replace(/^#/, "").split("/");
  const tab = TABS.some(([id]) => id === t) ? (t as Tab) : "overview";
  return { tab, arg: a ? decodeURIComponent(a) : null };
}

export default function App() {
  const init = fromHash();
  const [tab, setTab] = useState<Tab>(init.tab);
  const [memoryId, setMemoryId] = useState<string | null>(init.tab === "memories" ? init.arg : null);
  const [graphCenter, setGraphCenter] = useState<string | null>(init.tab === "graph" ? init.arg : null);
  const [sandbox, setSandbox] = useState<Rec | null>(null);
  useEffect(() => { api("/api/sandbox").then((d) => d.sandbox && setSandbox(d)).catch(() => undefined); }, []);

  useEffect(() => {
    const arg = tab === "memories" ? memoryId : tab === "graph" ? graphCenter : null;
    window.history.replaceState(null, "", `#${tab}${arg ? "/" + encodeURIComponent(arg) : ""}`);
  }, [tab, memoryId, graphCenter]);

  const openMemory = (id: string) => {
    setMemoryId(id);
    setTab("memories");
  };
  const openGraph = (id: string) => {
    setGraphCenter(id);
    setTab("graph");
  };

  return (
    <div className="app">
      {sandbox && (
        <div className="confirm">
          <b>Public sandbox.</b> Everything here is synthetic and read-only. Try the <b>Retrieval test</b> tab (ask “What is my favourite pizza topping?” and watch it decline),
          open <b>Memories</b> and tick “show private” (it needs the PIN, which is <code>{sandbox.pin}</code> here). To use MemBridge on your own history, install it locally:
          <code> uvx --from git+https://github.com/chinchoooo/membridge membridge quickstart</code>
        </div>
      )}
      <header>
        <h1>MemBridge</h1>
        <nav>
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
        <span className="muted">local-first · your data stays on this machine</span>
      </header>
      <main>
        {tab === "overview" && (
          <>
            <Overview />
            <BigPicture />
          </>
        )}
        {tab === "memories" && <Memories selected={memoryId} onSelect={setMemoryId} onGraph={openGraph} />}
        {tab === "graph" && (
          <Suspense fallback={<p>Loading graph…</p>}>
            <GraphView center={graphCenter} onCenter={setGraphCenter} onOpenMemory={openMemory} />
          </Suspense>
        )}
        {tab === "retrieval" && <Retrieval onOpenMemory={openMemory} onGraph={openGraph} />}
        {tab === "maintenance" && <Maintenance onOpenMemory={openMemory} />}
      </main>
    </div>
  );
}
