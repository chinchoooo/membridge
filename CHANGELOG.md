# Changelog

## 0.2.0
Found by an adversarial architecture review; each item has a regression test.

**Trust and conflicts**
- Trust order decides conflicts (`user_verified > user_stated > client_asserted > provider_stated > model_inferred`); undated claims are never "newer"; equal-trust undated clashes become unresolved; settled contradictions are remembered.
- A value stated again later is current again (A→B→A); confirmation merges trust, sensitivity, date and entities.
- Two facts in one sentence no longer collapse; single-valued attributes (employer, school, city, name, editor…) can conflict, skills cannot.
- New: `seed-text` (ChatGPT data-portability dump, strict line format or prose), `restructure`.

**Privacy and consent**
- Private memories (and raw messages) are absent from every default view by construction; PIN + client-bound, use-counted, expiring grants; constant notice; synchronous audit; hash-chained audit log.
- Agents cannot forge trust, change status, settle contradictions or complete a forget; `forget` really scrubs; `mark-private`.
- Credentials are redacted from everything returned; stricter secret detection (Luhn).

**Retrieval**
- Evidence gate replaces the score threshold that could never fail: per-item support or nothing.
- Extraction: capital "I"/"We" rules, clause-bounded objects, pasted/quoted text ignored, questions never facts; LLM quote must entail the statement.

**Operations**
- Dashboard needs a per-launch token (+ Origin check); `/api/ping` for health checks; compose requires a token.
- `.env` only from `~/.membridge`; remote graph host needs an env-level opt-in; "local" LLM URL must be loopback.
- `lock` refuses while the database is open and can protect plaintext artifacts; locked vault is never replaced by an empty DB.
- Exports exclude private entities/messages and redact credentials; PPR cache snapshot race fixed; agent-triggered consolidation is bounded; derived rebuilds coalesced.
- Schema migration 7.

### 0.2.0, public release prep
- `membridge quickstart [--demo]` guided first run; `membridge dump-prompt`; installs with `uvx --from git+https://github.com/chinchoooo/membridge ...` (dashboard bundled in the package, synthetic demo data packaged, no Node needed).
- `membridge sandbox` / `Dockerfile.sandbox`: a public read-only demo on synthetic data (no login, rate-limited, server details hidden, security headers, Host allow-list); `fly.toml`, `render.yaml`, compose profile, CI workflow.
- `upsert` never stores text the privacy lexicon flags as ordinary, whatever the caller says.

### 0.2.0, second pass (independent re-review)
- Private-only entities and private conversation titles are invisible; FalkorDB gets no private snippet/title/entity; forget and mark-private rebuild communities and summaries, drop orphan entities, compact the index and scrub free pages (`secure_delete`).
- Dashboard: seeing/exporting private memories needs the PIN (ten-minute session); access log off; CLI `--public`/`--sensitive` need the PIN at a real terminal.
- Agent edits drop to `client_asserted`; agent claims cannot take a user's statement out of circulation; dump `CHANGE` lines cannot overrule stated facts; private text matching is not an existence oracle.
- Extraction ignores instructions, reported speech and quoted clauses; more credential shapes; remote LLM extraction never receives private messages.
- Evidence gate: id-like tokens only for message anchors, slot-word candidate channel, aspect synonyms, tie-safe focus term. Migration 8.

## 0.1.0
Initial release.
