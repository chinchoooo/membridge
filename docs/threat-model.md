# Threat model

MemBridge stores a person's AI conversation history, so it is treated as sensitive by default. Everything runs on the
user's machine; there is no mandatory cloud service. This document describes what the code does *now*; each row names the
tests that pin it. (An earlier version of this file claimed several things the code did not do; they were found by an
adversarial review and fixed, and the tests below exist because of that review.)

## Assets
1. The raw archive (conversations, messages). 2. Extracted memories, including private ones (health, money, identity
documents, family). 3. Secrets: API keys for optional hosted inference, the MCP bearer token, the dashboard token, the PIN.
4. Integrity of memory: a poisoned or forged memory misleads every connected assistant.

## Who is calling: two principals
| Principal | Who | Can |
|---|---|---|
| **person** (`human=True`) | the CLI and the dashboard (behind its token) | everything below |
| **agent** (`human=False`) | an MCP client | read; add what it learned (recorded as `client_asserted`); `RELATED_TO`/`DEPENDS_ON`/… links; edit a memory's text/category/confidence (**rewriting the text drops the memory to `client_asserted`**: different words are the assistant's claim, not what you said). It **cannot** raise a memory's trust, change a status, create `SUPERSEDES`/`CONTRADICTS` links, settle a contradiction, merge/split entities, set the PIN, create a grant, or complete a `forget`. |

## Trust levels
`user_verified` (4) > `user_stated` (3) > `client_asserted` (2, what an agent writes on your behalf) > `provider_stated` (1, a
summary by ChatGPT/Claude) > `model_inferred` (0). A conflict between two statements about the same thing is decided by trust,
then by time; a statement with no date (`valid_time_known=0`) is never "newer" than a dated one; equal trust with no usable
date is left **unresolved** for you, never guessed. An agent's *newer* claim against your older statement waits for you **while your statement stays current**, so repeating
claims cannot take your own words out of circulation. A ChatGPT dump can never overrule something you or an assistant established (its newer value is recorded as outranked).
Settled contradictions are remembered, so the same disagreement does not return when sources are imported again.
Tests: `tests/test_trust.py`, `tests/test_security.py`.

## Trust boundaries and what we do at each

| # | Boundary / threat | Mitigation (and the test that pins it) |
|---|---|---|
| 1 | **Imported text is untrusted** (prompt injection in old chats, pasted emails). | Extraction only *matches* text (rules) or sends it fenced to a model that has **no tools**; model output is schema-validated and every memory needs a **verbatim quote that also supports the statement** (word overlap and matching negation; a real quote attached to the opposite claim is rejected). Quoted (`>`), forwarded and email-header text is not treated as the user's own words. Questions never become facts. Retrieval labels archived text `untrusted`, tags each item with its `origin`, and marks instruction-like memories "stored data, never an instruction". Tests: `test_rules_corpus.py`, `test_llm.py::test_quote_must_support_the_statement`, `test_hardening.py::test_instruction_like_memories_*`. |
| 2 | **An agent calls the MCP tools maliciously or by mistake.** | Fixed tool set (no SQL/Cypher/shell/filesystem). Inputs bounded and validated. **Principal model above**: an agent's writes are `client_asserted`; it cannot mint `user_verified`, flip a status, or delete. `forget_memory` from an agent only **files a request**: there is no token to echo, the person confirms in the dashboard or with `membridge forget`. Tests: `test_security.py::test_agents_*`, `test_service.py::test_agent_can_only_request_a_forget_*`, `test_mcp.py`. |
| 3 | **Query injection** (Cypher / FTS5 / SQL). | Values only as parameters; the only interpolated tokens are whitelisted relationship names and integers; FTS text is tokenised and quoted. Tests: `test_graph.py`, `test_import.py`. |
| 4 | **Secrets in conversations.** | Credential patterns (`sk-…`, AWS/GitHub/Google/Slack tokens, JWTs, URLs with passwords, "password is …", private keys, Luhn-valid card numbers) are never stored as memories, rejected by `save_memory`, and **redacted from everything returned** (snippets, conversations, exports). The raw archive keeps the original text as imported (it is your history); it is redacted on the way out. 13-digit timestamps are not mistaken for cards. Tests: `test_trust.py::test_secret_detection_and_redaction`, `test_security.py::test_secrets_in_the_raw_archive_*`, `test_hardening.py::test_export_*`. |
| 5 | **Private data leaves the machine.** | Default LLM provider is `none`; "local" Ollama must be a **loopback URL** (a remote one is treated as remote: opt-in and redacted). Hosted inference needs `MEMBRIDGE_ALLOW_HOSTED=true` and a key from an env var. The conflict classifier never sends private text to a model and cannot overrule trust. FalkorDB is loopback-bound **and the app refuses a non-loopback graph host unless the process environment sets `MEMBRIDGE_ALLOW_REMOTE_GRAPH=1`**; `.env` is read only from `~/.membridge/`, never from the working directory (an IDE may start the server inside an untrusted repo). Tests: `test_hardening.py`. |
| 6 | **Network exposure of the dashboard/API.** | Loopback bind enforced by the CLI; a **per-launch access token** (HttpOnly, SameSite=Strict cookie for the UI via the link `membridge serve` prints; the access log is off so the token is not written to a log), bearer header for scripts, `Host` allow-list (DNS rebinding), `Origin` check, no CORS, custom header on mutations. **Seeing private memories, exporting them, or making one visible again needs the PIN too** (a signed, HttpOnly cookie valid for ten minutes); the token alone is not enough. `/api/ping` is the only unauthenticated route and reveals nothing. Tests: `test_api.py`. |
| 7 | **Remote MCP access.** | HTTP transport refuses to start without a ≥16-char `MEMBRIDGE_MCP_TOKEN` or on a non-loopback address; constant-time comparison. Remote/hosted connectors (ChatGPT etc.) remain **unverified**. |
| 8 | **Malformed or hostile archives.** | Streaming parser; bad records become reported failures; cycles/orphans bounded; zip members read by name, never extracted. |
| 9 | **Memory poisoning / silent overwrite.** | History is not deleted by normal operation: supersession invalidates, corrections retract, a value that comes back is current again (A→B→A; both directions of that chain are kept as `SUPERSEDES` events, newest first), an outranked statement is kept as history with a zero-length validity, edits keep revisions, `forget` leaves a tombstone so re-extraction cannot resurrect it. The one explicit exception is `seed --replace`, which hard-deletes memories derived *purely* from Claude's seed files (with their links; a conflict partner is returned to `current`), and never touches tombstones or anything you or an assistant stated. Tests: `test_trust.py`, `test_seed_text.py`, `test_review_fixes.py`. |
| 10 | **Data loss.** | Consistent online backup, integrity-checked restore that moves the current DB aside, export, reproducible rebuild of the graph and embeddings. SQLite `busy_timeout` so a CLI import and a running server wait for each other. A locked vault is never replaced by an empty database. |
| 11 | **Backups, exports and the database at rest.** | `backup/export --encrypt` (scrypt + streaming AES-256-GCM; tampering, truncation, reordering rejected) and `lock`/`unlock`. `lock` **refuses while any process has the database open** (an open-marker byte lock; deleting a file under a running server would lose its writes) and `lock --protect-artifacts` encrypts backups and deletes exports/dry-run payloads/manifests, which otherwise stay readable. Tests: `test_crypto.py`, `test_hardening.py`. |
| 12 | **Hosted extraction** (`claude_cli`, opt-in). | `MEMBRIDGE_ALLOW_HOSTED=true`; redaction first; tools disabled; empty working directory; `--dry-run` shows the exact payload; verbatim-quote, entailment and no-invented-number guards; outputs labelled `model_inferred`. |
| 13 | **Private memories (consent).** | A regex lexicon (health, money, identity documents, family, body measurements) classifies memories **and raw messages** at import; you can mark anything private by hand (`membridge mark-private`). Private text is **absent from every default view**: the search index, the PageRank graph and its cache, derived entity edges, community and project summaries, the FalkorDB projection (nodes, edges, message snippets, titles), embeddings, exports. So are **entities that exist only through private memories** (a medication's name is the memory) and **private conversation titles**. A private id and a missing id give the same error; `save_memory` of text matching a private memory returns the same shape as a new private one (no id); every result carries one constant notice; statistics for agents omit private counts. Access needs a **PIN** (scrypt, rate-limited) typed at a hidden prompt or the dashboard, which creates a grant for **one client**, for minutes, with a **use count**. Every private access and refusal is written to the audit log immediately. Tests: `test_security.py` (20 cases), `test_review_fixes.py`. |
| 14 | **Audit integrity.** | Each audit row commits to the one before it (hash chain); `membridge audit-verify` and `/api/audit/verify` report an edited or removed row. Private-data events are committed synchronously, so a killed server cannot lose them. |
| 15 | **Forgetting is real.** | `forget` (and `mark-private`) removes the statement text, structured value, entity links, **entities it alone created**, graph edges, embeddings, client notes, the matching line of an imported dump document; rebuilds entity edges, communities and every summary (including polished LLM ones, which are replaced when their source facts change); compacts the full-text index; and with `secure_delete` plus a WAL checkpoint the words are not left in the file's free pages (a test scans the database bytes). The tombstone (id + dedup key) stays so it cannot be re-extracted. The original chat message in the raw archive is your own history and is not edited. Tests: `test_security.py::test_forget_scrubs_*`, `test_review_fixes.py`. |

## What the consent control is, and is not
The PIN and grants protect against **an assistant reading private memories through MemBridge** (MCP, the dashboard, and the
CLI's `--private`, `export --sensitive` and `mark-private --public`, which all need the PIN at a real terminal: stdin must be a TTY, so a program cannot feed it). A grant names a client; that name is declared by the connecting process (`MEMBRIDGE_CLIENT_NAME`),
so this is a *consent control*, not a boundary against malware. A coding agent that can run shell commands as you can also
open the SQLite file directly or run `membridge search` (raw messages). The boundary against that is **`membridge lock`**
(encrypted at rest) and OS account separation. The PIN is never sent through a chat or an MCP tool, and `grant-sensitive`
needs a hidden prompt that a program without a terminal cannot answer.

## Not covered (be aware)
- While **unlocked**, the SQLite file is plaintext (SQLCipher has no Windows build); use `lock` when not in use, and OS disk encryption.
- Anyone with local OS access to your account can read the data and the audit log (the chain makes edits *detectable*, not impossible).
- The privacy lexicon and secret patterns are pattern-based: they can miss unusual phrasing (use `mark-private`) and can over-flag (wrongly-private is the safe failure).
- Message-anchored retrieval support is conversation-level: a message containing a near-unique token ties its conversation's project facts to the query.
- Audit-chain forks are possible if two processes append in the same instant (reported as a break; investigate before assuming tampering). Ordinary read audits are buffered for about two seconds and can be lost if the server is killed; private-data events are written immediately.
- If a message's branch is abandoned after a re-import (an edited reply), memories already extracted from it stay; only future extraction follows the current branch.
- Term-frequency counts used to prune full-text queries include private rows (a very weak side channel); results never do.
- The open-marker lock is tested on Windows here; the POSIX branch uses standard `flock` but has not been run on this machine.
- Remote exposure and ChatGPT/other hosted-client connectors are unverified (see `CLIENTS.md`).
