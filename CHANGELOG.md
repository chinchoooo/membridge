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

## 0.1.0
Initial release.
