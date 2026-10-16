# Threat model

MemBridge stores a person's AI conversation history, so it is treated as sensitive by default. Everything runs on the
user's machine; there is no mandatory cloud service.

## Assets
1. The raw archive (conversations, messages). 2. Extracted memories, including sensitive ones. 3. Secrets: API keys for
optional hosted inference, the MCP HTTP bearer token. 4. Integrity of memory (a poisoned memory misleads every connected assistant).

## Trust boundaries and what we do at each

| # | Boundary / threat | Mitigation (where, and how it is tested) |
|---|---|---|
| 1 | **Imported text is untrusted** (prompt injection in old chats). | Extraction only *matches* text (rules) or sends it fenced in `<untrusted_messages>` to a model that has **no tools**; model output is schema-validated, categories whitelisted, and every memory needs a **verbatim quote** found in a user message (unsupported quotes rejected: 9 of 12 on real data with a 3.8B model). Retrieval labels all archived text `untrusted` and says "data, not instructions". Tests: `test_memory.py::test_injection_text_is_data_not_memory`, `test_llm.py::test_prompt_injection_*`, `test_service.py::test_injected_instructions_*`. |
| 2 | **An AI agent calls the MCP tools maliciously or by mistake.** | Fixed tool set; no raw Cypher/SQL/shell/filesystem tool exists. Inputs bounded and validated (`ServiceError`). `forget_memory` is **two-step** (token issued, must be echoed within 5 min). Agents cannot opt into sensitive memories. Tool annotations mark read-only vs destructive. All calls audited with the client's identity. Tests: `test_mcp.py`. |
| 3 | **Query injection** (Cypher / FTS5 / SQL). | Cypher: values only as parameters; the only interpolated tokens are whitelisted relationship names and integers. FTS: user text is tokenised and quoted. SQL: parameters throughout. Tests: `test_graph.py::test_fulltext_seed_and_safe_query_builder`, `test_import.py::test_fts_query_cannot_inject_syntax`. |
| 4 | **Secrets in conversations.** | Credential patterns (`sk-…`, AWS/GitHub/Slack tokens, private keys, `password=`, long digit runs) are never stored as memories and are rejected by `save_memory`. Hosted inference redacts emails, phones, cards, IPs, keys *before* sending. |
| 5 | **Private data leaves the machine.** | Default provider is `none`; local Ollama is next. Hosted inference needs `MEMBRIDGE_ALLOW_HOSTED=true` **and** a key from an env var (never stored); `GuardedLLM` counts characters sent. FalkorDB is bound to `127.0.0.1` only. |
| 6 | **Network exposure of the dashboard/API.** | Loopback bind enforced by the CLI; `Host` allow-list (blocks DNS rebinding); no CORS; every mutating request needs a custom header (blocks cross-site forms). Tests: `test_api.py::test_host_header_and_csrf_guards`. |
| 7 | **Remote MCP access.** | HTTP transport refuses to start without a ≥16-char `MEMBRIDGE_MCP_TOKEN` or on a non-loopback address; constant-time token comparison. For remote use, put an authenticated TLS tunnel in front; this is documented, **not verified** with any hosted client. |
| 8 | **Malformed or hostile archives.** | Streaming parser; bad records become reported failures; cycles/orphans in message trees are bounded; zip members are read by name, never extracted to disk. |
| 9 | **Memory poisoning / silent overwrite.** | History is never deleted: supersession invalidates, corrections retract, edits keep revisions, `forget` leaves a tombstone so re-extraction cannot resurrect. Provider-stated and model-inferred facts are labelled and never overwrite user-stated ones. Deleting is soft with the text erased. |
| 10 | **Data loss.** | Consistent online backup (`membridge backup`), integrity-checked restore that moves the current DB aside, JSON/Markdown export, reproducible rebuild of the graph and embeddings from SQLite. |

## Not covered (be aware)
- The SQLite file and exports are **not encrypted at rest**; rely on OS disk encryption.
- Anyone with local OS access to your account can read the data and the audit log.
- Rule-based secret detection is pattern-based and can miss unusual formats.
- Remote exposure and ChatGPT/other hosted-client connectors are unverified (see `CLIENTS.md`).
- The 5-minute `forget` token protects against accidents and agent misuse, not against a local attacker with DB access.
