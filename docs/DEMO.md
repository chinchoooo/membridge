# Demo script and video plan (about 7 minutes)

Reproduce everything: `docker compose up -d && uv run python scripts/demo.py` (add `--claude` to use the real Claude Code CLI as the first MCP client).
It uses a synthetic archive (`samples/`), a throw-away data dir (`.membridge-demo/`) and prints a transcript of all six steps.

| Time | Show | Say |
|---|---|---|
| 0:00 | Dashboard overview of a real import (counts, failed records, "big picture") | "My AI history is scattered across providers. MemBridge migrates it once into a store I own." |
| 0:40 | Step 1: `membridge import` (dry-run, real import, second import = no-op) | "Streaming, resumable, idempotent, and malformed records are reported, not hidden." |
| 1:10 | Step 2 in the Retrieval tab: *Why did I stop using Postgres for Atlas?* | "Facts come from different conversations; each links to the exact original words." Click Open source conversation. |
| 1:50 | Step 3: *fatal remaining connection slots reserved* | "No fact contains these words. The graph hops message → conversation → project → decision. Text search scores 0 here." Show the graph paths and the Graph tab. |
| 2:40 | Step 4: Claude Code records a decision; a second process reads it (audit log shows both identities) | "Two independent MCP clients, one store, no provider memory involved." |
| 3:20 | Step 5: change a decision; history and the *as-of* date picker | "The old decision is not overwritten: it ends validity. I can ask what was true in November." |
| 4:00 | Step 6: restart FalkorDB, re-ask; export; backup; restore elsewhere | "Everything survives restarts and is portable without any AI provider." |
| 4:30 | Maintenance → Entity review and client badges | "Uncertain merges wait for me; clients that I did not verify are labelled unverified." |
| 5:00 | `uv run python scripts/demo_trust.py` (synthetic, no Docker): *Where do I work?* vs *favourite pizza topping?* | "It says 'nothing stored' instead of guessing. 34 of 34 unanswerable test questions are declined." |
| 5:40 | same script, step 2: a ChatGPT memory dump next to what I said myself | "ChatGPT's summary is imported as *ChatGPT's belief*, undated, and cannot overrule what I told it. Both are kept; mine is current." |
| 6:10 | same script, step 3: private medication fact; the assistant asks; tries to grant itself access; I set a PIN and grant it 10 minutes / 2 reads | "Health, money and family facts are invisible to assistants by default. Only I can open the door, with a PIN, for one assistant, for minutes. Every access is in a tamper-evident log." |
| 6:40 | step 4: the assistant *asks* to forget; I confirm; scan the database file for the text | "An assistant can only request a deletion. When I confirm, the text is gone from the file, not just hidden." |

Tips: record at 1280×800 in dark mode; run the demo once beforehand so Docker images are warm.

## Dashboard access
`membridge serve` prints a link with this launch's access token. Open that exact link once (it sets an HttpOnly cookie); a bare `http://127.0.0.1:8787` shows nothing and the API answers 401.

## Limitations to state on camera
Rule extraction is conservative (low recall on casual chat); LLM extraction runs locally and is slow; benchmark numbers are from
synthetic data (the abstention gate has not been scored on real data until you judge answers with `gold-init`/`eval-real`); Cursor/VS Code/opencode/ChatGPT connectors are provided but unverified;
the live database is plaintext while unlocked (`membridge lock` encrypts it at rest); the PIN protects the MCP and dashboard path, not a shell-capable program that can open the file directly.
