# Hackathon submission: answers to paste

Fill the two `TODO`s (YouTube link, track) and read Part 2 once: edit it so it reflects what *you* actually experienced.

**Project title:** MemBridge

**Short description (one line):** An AI memory you own: import what ChatGPT/Claude know about you, keep it on your machine as a FalkorDB graph with the evidence behind every fact, and get an honest "I don't know" instead of a guess.

**Description:**
MemBridge imports a ChatGPT or Claude memory/export into a local store (SQLite as source of truth, FalkorDB as a rebuildable graph projection) and serves it to any assistant over MCP and to you through a dashboard. Every memory has a trust level (what you said beats what a provider summarised) and links back to the exact messages it came from. Retrieval has an evidence gate, so when nothing is stored the answer is "nothing stored" rather than a plausible-sounding guess. Health, money and family facts are private by default: an assistant can read them only after you grant that one assistant access with a PIN, for a few minutes and a few reads. It installs with one command and has a public read-only sandbox.

**What problem does your project solve, and who is it for?**
Every AI assistant now keeps a memory about you, but it is a list you cannot read, correct, audit or take to another assistant, and it will answer confidently about things you never told it. MemBridge is for anyone who uses AI assistants daily and wants their memory to be portable, inspectable and honest: it shows where each fact came from, keeps your own words above a provider's summary, hides sensitive facts from assistants unless you allow it, and says "I don't know" when it doesn't.

**Track(s):** TODO (paste the track options to me; the best fit is the graph/knowledge-graph or AI-agents-memory track, if there is one)

**Repository:** https://github.com/chinchoooo/membridge

**Demo video:** TODO (YouTube, public or unlisted)

**Live demo:** https://membridge-sandbox.onrender.com (synthetic data, read-only; PIN for the private view is 123456). The free instance sleeps; the first load can take about 30 s.

**How did you use FalkorDB in your project?**
FalkorDB holds a rebuildable graph projection of the memory store (SQLite stays the source of truth, so answers are identical if the graph is down). Nodes: Memory, Entity, Conversation, Message, Source, User, Community; private memories are never projected. I use it for:
- Cypher path queries that answer multi-hop questions and return the path itself (message → conversation → project → decision) so the answer is explainable. On a 48-question benchmark, text-only search answers 0/12 of the "no shared words" questions; the full pipeline with the graph answers 12/12.
- Full-text indexes on memory statements and entity names/aliases, range indexes, and an optional HNSW vector index.
- `algo.pageRank` and `algo.WCC` for the "big picture" view (most important entities, clusters).
- Bounded neighbourhood queries behind the dashboard's graph explorer.
- Batched `UNWIND`/`MERGE` for incremental sync, and a parity test that the graph path and the SQLite fallback agree.
- A breadth-first Cypher query that loads a seed neighbourhood, over which I run personalised PageRank in Python.

**Blog link:** https://chinchoooo.github.io/writing/membridge-memory-graph-falkordb/

## Part 2: feedback on FalkorDB

**What did you like about FalkorDB?**
It was easy to run and easy to reason about: one container, openCypher, and it was fast enough that the graph path costs only a few milliseconds more than my SQLite path. Having full-text, range and vector indexes in the same database as the graph meant I did not need a second system. `algo.pageRank` and `algo.WCC` gave me the "big picture" view with almost no code. Idempotent `MERGE` with `UNWIND` batches made incremental sync simple, and because dropping and rebuilding a graph is cheap, treating it as a disposable projection worked well.

**What improvements would you like to see in FalkorDB?**
- Personalised PageRank (seeded from chosen nodes). `algo.pageRank` is global, so I load a neighbourhood with a BFS query and run personalised PageRank in Python.
- Clearer, documented behaviour and examples for combining a full-text or vector hit with a graph traversal in a single query (the "find seeds, then expand" pattern is the heart of graph retrieval).
- Better error messages for Cypher syntax and index mistakes, and a documented way to test whether the server is up, with the indexes built, from a client in one call.
- A small official guide for the "graph as a rebuildable projection of another store" pattern: bulk load, incremental upsert and safe rebuild.

*(Edit these to match your own experience before submitting: the judges read the Part 2 answers for honesty, and the list above is drawn from what the code needed, not from anything you did not run into.)*

---

## YouTube

**Title:** MemBridge: an AI memory you own (FalkorDB graph, honest "I don't know")

**Description:**
Every AI assistant keeps a memory about you that you can't read, correct or take with you. MemBridge imports what ChatGPT already knows about you, stores it on your own machine as a FalkorDB graph, shows the evidence for every fact, hides private facts unless you grant access, and says "I don't know" instead of guessing.

Code: https://github.com/chinchoooo/membridge
Live sandbox: https://membridge-sandbox.onrender.com
Write-up: https://chinchoooo.github.io/writing/membridge-memory-graph-falkordb/

Install: uvx --from git+https://github.com/chinchoooo/membridge membridge quickstart

Set visibility to Public or Unlisted. Upload `submission/final/membridge_demo.mp4`, `submission/thumbnail.png`.
