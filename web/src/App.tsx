import { useState } from "react";
import Overview from "./views/Overview";
import Memories from "./views/Memories";
import GraphView from "./views/GraphView";
import Maintenance from "./views/Maintenance";
import Retrieval from "./views/Retrieval";

export type Tab = "overview" | "memories" | "graph" | "retrieval" | "maintenance";
const TABS: [Tab, string][] = [
  ["overview", "Overview"],
  ["memories", "Memories"],
  ["graph", "Graph"],
  ["retrieval", "Retrieval test"],
  ["maintenance", "Maintenance"],
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [memoryId, setMemoryId] = useState<string | null>(null);
  const [graphCenter, setGraphCenter] = useState<string | null>(null);

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
        {tab === "overview" && <Overview />}
        {tab === "memories" && <Memories selected={memoryId} onSelect={setMemoryId} onGraph={openGraph} />}
        {tab === "graph" && <GraphView center={graphCenter} onCenter={setGraphCenter} onOpenMemory={openMemory} />}
        {tab === "retrieval" && <Retrieval onOpenMemory={openMemory} onGraph={openGraph} />}
        {tab === "maintenance" && <Maintenance onOpenMemory={openMemory} />}
      </main>
    </div>
  );
}
