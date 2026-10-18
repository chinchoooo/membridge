# MemBridge: submission summary

**What it is.** A local-first personal AI memory. Import your ChatGPT / Claude history (and what ChatGPT already "remembers" about you) once; keep a
provenance-linked, bi-temporal knowledge graph on your own machine; let any MCP-capable assistant read and update it, under rules *you* control.

**Why it is different.** Most memory features are a provider's private list you cannot inspect, move or correct. MemBridge keeps SQLite as the source of truth with a rebuildable
FalkorDB graph on top, and treats memory as something that must be *trustworthy*, not just retrievable:
- **Provenance and history:** every fact links to the exact original words; changes end a fact's validity instead of overwriting it ("what was true in November?").
- **Graph retrieval that wins where text search cannot:** on a 48-question multi-hop benchmark text-only retrieval gets 0/12 of the questions whose answer shares no words with the question; the full system answers 12/12 (`docs/EVAL.md` §1).
- **It says "I don't know":** an evidence gate returns a memory only when something about *this* query supports it (34/34 unanswerable test questions declined, 32/33 answerable answered).
- **Trust and contradictions:** `user_verified > user_stated > assistant-written > a provider's summary > model-inferred` decides conflicts; undated claims never outrank dated ones; ties wait for you; your decision is remembered.
- **Privacy by construction:** health, money, identity-document and family facts are absent from every default view (search, graph, summaries, exports); an assistant gets them only under a PIN-created, short, single-client, use-counted grant; every access is audited in a hash-chained log; `forget` really erases the text, including from the file.
- **Portability:** a prompt (`docs/CHATGPT_DUMP_PROMPT.md`) turns what ChatGPT knows about you into an importable dump while you wait for the official export.

**Try it (1 minute, nothing to install first).** Hosted read-only sandbox on synthetic data: https://membridge-sandbox.onrender.com. Locally:
`uvx --from git+https://github.com/chinchoooo/membridge membridge quickstart --demo` (guided first run; `quickstart` without `--demo` does it on your own export).

**Developer try-out (5 minutes).** `uv sync && docker compose up -d && uv run python scripts/demo.py` (six-step walkthrough, synthetic data) and `uv run python scripts/demo_trust.py` (trust, consent, forgetting; no Docker needed).
Dashboard: `uv run membridge serve` prints a link with a per-launch access token.

**Evidence.** 354 automated tests (unit, algorithm properties, real MCP over stdio in separate processes, live FalkorDB parity, API security, 60+ privacy/trust regressions), mypy `--strict` and ruff clean,
Docker image built and run (non-root, loopback-only, token required, healthcheck). Two independent adversarial reviews by a separate model found 26 + 17 issues; all were fixed and each has a regression test
(`docs/threat-model.md` lists what is and is not protected).

**Honest limits.**
- Benchmarks are synthetic; the abstention gate and retrieval quality have not been scored on real data until you judge answers (`gold-init` / `eval-real`).
- Rule extraction is conservative; recall on casual chat is low; LLM extraction is slow locally.
- Verified clients: Claude Code, the Python MCP SDK client and the official MCP Inspector CLI. Cursor / VS Code / opencode / ChatGPT / hosted connectors are unverified.
- The live database is plaintext while unlocked (`membridge lock` encrypts at rest). The PIN protects the MCP and dashboard path; a shell-capable program running as you can open the file.
- A new question costs ~14 ms on a 2,000-message archive (the evidence gate adds ~11 ms); large-scale latency was not re-measured after the gate.
- The hosted sandbox is deliberately not a service for real data: it is read-only, synthetic, rate-limited and public. Your own instance has one user and must never be exposed (`docs/DEPLOY.md`).
- Tested on Windows (developer machine and CI) and Linux (CI, with a real FalkorDB); macOS is not tested.
