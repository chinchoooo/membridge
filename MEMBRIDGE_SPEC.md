# Build MemBridge: A Local-First, Cross-Assistant Personal AI Memory System

## 1. Your Role

Act as a principal software architect, AI/ML engineer, graph database specialist, MCP protocol engineer, security engineer, and pragmatic hackathon technical lead.

Your task is to design and implement a production-minded, local-first personal AI memory system called **MemBridge**.

Do not immediately start generating application code. First investigate the problem, compare viable architectures and models, identify the best trade-offs, and produce an implementation plan. Then execute that plan incrementally, validating each component through tests.

Make engineering decisions independently when sufficient information is available. Do not ask me to choose between technologies without explaining the trade-offs and making a recommendation yourself.

The goal is a working, demonstrable product, not a collection of mock interfaces, disconnected scripts, or architectural diagrams.

## 2. Product Vision

Build a persistent personal memory engine that migrates my accumulated AI conversation history into a portable knowledge system that I own and operate on my own computer.

The system must:

1. Import historical conversations from ChatGPT exports and support additional conversation formats through extensible import adapters.
2. Recover durable personal facts, preferences, projects, technical knowledge, decisions, recurring constraints, and historical context.
3. Preserve original conversation data and source provenance.
4. Compress and organise the information into hierarchical summaries, semantic memories, episodic records, and a graph of entities and relationships.
5. Store the resulting knowledge persistently on my computer.
6. Expose the memory engine through a standards-compliant Model Context Protocol (MCP) server.
7. Allow multiple compatible AI clients to query and update the same memory store.
8. Support incremental updates without requiring the entire archive to be processed again.
9. Provide a user interface to inspect, correct, search, export, and delete memories.
10. Operate without requiring a GPU or continuously running cloud infrastructure.

The key product principle is:

**Migrate historical memory once, maintain it continuously, and make it portable across AI assistants.**

The system must not depend on the memory features of any individual AI provider.

## 3. Core Problem

AI assistants have fragmented, session-dependent context. A user can accumulate years of conversations, research, decisions, projects, and preferences, yet that knowledge is difficult to transfer between assistants.

MemBridge should create an external memory layer that allows a compatible AI assistant to answer questions such as:

- What have I previously decided about this project?
- What was the technical problem I encountered several months ago?
- Which approaches have I already attempted, and why did they fail?
- What are the current constraints and status of each project?
- What have I learned about a topic across multiple conversations?
- Which previous conversations contain the evidence for a particular fact?
- Has my preference or decision changed over time?
- What context from my earlier work is relevant to the current task?

Answers must be based on retrieved evidence whenever possible. The system must distinguish verified information, user-stated facts, model-generated inferences, and unresolved contradictions.

## 4. Architecture Research and Decision Phase

Before implementation, evaluate the following architectural decisions.

### 4.1 Graph database

FalkorDB is the preferred graph database because the project is intended for the Graph Hacks hackathon and the graph must be a meaningful part of the system.

Evaluate:

- FalkorDB for entity relationships, memory traversal, and graph queries.
- SQLite for lightweight metadata, job tracking, checkpoints, and transactional state where appropriate.
- A vector index or vector database for semantic similarity retrieval.
- Full-text search for exact terms, names, identifiers, and technical error messages.

Do not introduce a separate database for every capability without justification.

Determine whether FalkorDB's available vector and search capabilities are sufficient for the initial implementation. Prefer the smallest architecture that satisfies the requirements.

The graph must actively participate in retrieval. It must not be a decorative visualisation over an ordinary vector database.

### 4.2 Language models

Compare appropriate models for:

- Entity extraction.
- Relationship extraction.
- Conversation summarisation.
- Personal fact extraction.
- Contradiction detection.
- Memory deduplication.
- Query interpretation.
- Context selection.
- Memory consolidation.

Evaluate model quality, cost, latency, structured-output reliability, context-window requirements, CPU performance, privacy, and implementation complexity.

Compare three execution modes:

1. Local models running on CPU.
2. Hosted LLM APIs.
3. A hybrid system with local processing by default and optional hosted inference for difficult tasks.

Do not assume that the largest model is the best model.

Do not assume that a model advertised as small will perform sufficiently well at entity resolution or contradiction detection. Evaluate these capabilities on representative data.

Recommend a default model configuration and explain why it is appropriate for a personal archive.

Use provider-independent interfaces so models can be replaced without redesigning the application.

Keep the model provider optional. Core storage, graph queries, data import, and deterministic retrieval must work without an LLM API.

Never hardcode API keys.

### 4.3 Embeddings

Evaluate whether embeddings are necessary for the MVP and which embedding model is appropriate for CPU-only execution.

Compare embedding-based retrieval with:

- Full-text search.
- Graph traversal.
- Metadata filtering.
- Hybrid retrieval.

Do not use embeddings as a substitute for structured relationships.

Support an optional embedding index that can be rebuilt from canonical stored records.

### 4.4 MCP architecture

Design a standards-compliant MCP server that exposes the same persistent memory to compatible AI clients.

Investigate current MCP SDKs, supported transports, client compatibility, authentication requirements, and configuration formats.

Support local MCP connections first, using the transport appropriate for each client.

Design remote access as an optional extension using authenticated, encrypted connections or a suitable secure tunnel.

Do not assume every ChatGPT plan or client supports arbitrary custom MCP servers. Treat client integrations as independent adapters with explicit capability checks.

### 4.5 Final architecture decision

Produce an architecture decision record containing:

- Recommended components and versions.
- Alternatives considered.
- Reasons for accepting or rejecting alternatives.
- Expected resource consumption.
- Data flow.
- Failure modes.
- Security implications.
- Approximate running costs.
- Migration and deployment strategy.

Do not begin the full implementation until this analysis and a concrete milestone plan are complete.

## 5. Historical Conversation Import

Build a robust importer for exported ChatGPT conversation archives.

The initial supported input should include the standard ChatGPT data export, especially `conversations.json`, where available.

Design the importer so additional sources, such as Claude exports or other compatible conversation formats, can be supported through independent adapters.

Do not assume that different providers have identical schemas.

### Import requirements

- Parse conversation identifiers, titles, timestamps, message roles, message content, and available metadata.
- Preserve message ordering and conversation boundaries.
- Handle missing fields, deleted or incomplete messages, malformed records, non-text content, and unexpected schema variations.
- Preserve original content without silently replacing it with a summary.
- Detect duplicate imports using stable identifiers or content hashes.
- Make imports resumable after interruption.
- Report processing progress, failures, skipped records, and extraction statistics.
- Allow dry-run validation before writing data.
- Provide an import manifest and error report.
- Never overwrite an existing archive silently.
- Never claim to have imported data that was not actually provided.

The importer must support incremental re-imports and must not create duplicate memories when the same archive is processed repeatedly.

Use streaming or chunked processing when appropriate instead of loading the entire archive into memory.

### Historical fidelity

The raw archive is the source of truth.

Store the original conversations and messages in a persistent, queryable form. Use content-addressed identifiers or equivalent stable references so extracted memories can link back to the original evidence.

Where attachments or multimodal content are not included in the available export, record that limitation instead of inventing their contents.

## 6. Memory Representation

Implement a hierarchical memory architecture.

### Layer A: Raw episodic archive

Preserve the original conversation records, including:

- Conversation ID.
- Message ID or stable derived identifier.
- Timestamp when available.
- Speaker or role.
- Original message content.
- Import source.
- Relevant metadata.

This layer should support direct retrieval of the original message.

### Layer B: Episodic memories

Represent events and experiences tied to particular conversations or periods.

Examples:

- A project was started.
- A particular debugging approach failed.
- A research hypothesis was revised.
- A deadline was changed.
- An experiment produced a specific result.

Each episodic memory should link to its supporting source records.

### Layer C: Semantic memories

Extract durable knowledge, such as:

- User preferences.
- Long-term constraints.
- Technical skills and interests.
- Project objectives and status.
- Previously made decisions.
- Known technical facts.
- Lessons learned.
- Recurring workflows.

Each memory should include, as appropriate:

- A stable ID.
- A concise statement.
- A category.
- Provenance references.
- Confidence and extraction method.
- Created and updated timestamps.
- Temporal validity.
- Sensitivity classification.
- Verification status.

Distinguish facts explicitly stated by the user from facts inferred by the model.

### Layer D: Hierarchical summaries

Create summaries at multiple levels:

1. Message or event summaries where useful.
2. Conversation summaries.
3. Project and topic summaries.
4. Long-term personal context summaries.

Summaries should be regenerated or incrementally updated when their underlying records change.

Never treat a summary as a replacement for the raw archive.

### Layer E: Relationship graph

Design a flexible graph schema containing entities such as:

- User.
- Preference.
- Project.
- Person.
- Organisation.
- Conversation.
- Message.
- Decision.
- Task.
- Event.
- Research paper.
- Technical concept.
- Experiment.
- Finding.
- Source.
- Memory.

Use typed relationships such as:

- `WORKED_ON`
- `PREFERS`
- `DECIDED`
- `SUPERSEDES`
- `DEPENDS_ON`
- `RELATED_TO`
- `LEARNED_FROM`
- `SUPPORTED_BY`
- `CONTRADICTS`
- `DERIVED_FROM`
- `MENTIONED_IN`

The final schema must be justified by actual retrieval requirements, not by a desire to maximise the number of node types.

Include temporal and provenance information in the graph model.

Use FalkorDB's supported Cypher-compatible query language for graph operations.

### Layer F: Optional vector representations

Store embeddings only when useful for semantic retrieval.

Associate each embedding with a stable canonical record ID and model version.

Support rebuilding the vector index without losing the original memory records.

## 7. Compression and Memory Consolidation

This is a core component, not an optional future feature.

Design a pipeline that converts a large conversation archive into compact, useful, queryable memory while preserving historical fidelity.

The pipeline should:

1. Segment conversations into manageable processing units.
2. Extract candidate facts, decisions, events, and relationships.
3. Identify source references for each extracted item.
4. Resolve duplicate entities and aliases.
5. Detect semantically equivalent memories.
6. Identify contradictory statements.
7. Track changes to facts over time.
8. Generate hierarchical summaries.
9. Update the graph incrementally.
10. Preserve links to the original records.

### Contradiction handling

Do not blindly overwrite older memories with newer statements.

Represent changes explicitly. For example, a previously selected technology may later be replaced by a different one.

Distinguish between:

- A changed preference.
- A correction of an earlier factual error.
- A temporary project constraint.
- Two incompatible claims that remain unresolved.
- A model's uncertain inference.

Use deterministic rules where possible and an LLM only where semantic interpretation is necessary.

Do not silently delete historical context during consolidation.

### Memory quality

Implement a memory scoring strategy based on factors such as:

- Explicitness.
- Recency.
- Repeated confirmation.
- Long-term relevance.
- Task relevance.
- Source reliability.
- Confidence.

Do not assume that frequently mentioned information is always more important.

Allow important one-time decisions to remain retrievable even when they are old.

Avoid generating excessive near-duplicate nodes or creating speculative relationships.

## 8. Retrieval and Context Assembly

Implement a retrieval engine that combines:

- Exact search.
- Full-text search.
- Metadata filtering.
- Graph traversal.
- Optional semantic vector search.
- Recency and temporal filtering.
- Source-aware ranking.

The retrieval pipeline should be able to:

1. Interpret an incoming query.
2. Identify relevant entities and topics.
3. Retrieve candidate memories.
4. Traverse related graph relationships.
5. Fetch supporting source records.
6. Rank results by relevance and evidence quality.
7. Deduplicate overlapping results.
8. Fit selected context within a configurable token budget.
9. Return a compact, structured context package.

Avoid retrieving the entire personal archive for every question.

Implement token-aware context assembly and configurable limits on traversal depth, result count, and context size.

Expose a retrieval explanation that identifies which memories and relationships caused a result to be selected.

When the evidence is insufficient, the system should report that limitation rather than fabricate a memory.

## 9. MCP Tools

Implement a clean, documented MCP tool interface.

At minimum, consider:

### Retrieval

- `search_memory(query, limit, filters)`
- `get_memory(memory_id)`
- `get_related_memories(memory_id, depth)`
- `get_conversation(conversation_id)`
- `search_conversations(query, date_range, filters)`
- `get_project_context(project)`
- `get_relevant_context(query, token_budget)`
- `trace_memory_provenance(memory_id)`

### Persistence

- `save_memory(content, category, source, confidence)`
- `update_memory(memory_id, changes, reason)`
- `record_decision(project, decision, rationale, alternatives)`
- `commit_session_summary(summary, decisions, unresolved_tasks)`
- `link_memories(source_id, relationship, target_id)`

### Maintenance

- `list_projects()`
- `get_memory_statistics()`
- `run_consolidation(scope)`
- `export_memory(format)`
- `forget_memory(memory_id)`

Refine these tool definitions based on MCP conventions and usability.

Do not expose destructive database operations directly to an AI agent.

Require appropriate validation and confirmation for destructive or bulk operations.

Tool responses should be concise, structured, and useful to language models.

Include provenance references and uncertainty information in retrieval responses.

Design the interface so additional compatible AI clients can use it without needing provider-specific database code.

## 10. Session Persistence

Support memory updates during or after an AI task.

A typical lifecycle should be:

1. The client requests relevant context.
2. The memory server returns selected memories and source references.
3. The assistant completes the task.
4. The client submits durable findings, decisions, or a session summary.
5. The memory engine validates the submission.
6. New memories and relationships are persisted.
7. Existing memories are updated when justified.
8. Future sessions can retrieve the updated state.

Implement idempotent writes and duplicate detection.

Provide a clear mechanism for committing session summaries without relying on the assumption that every AI client automatically exposes its entire transcript.

Where automatic transcript capture is unavailable, use explicit MCP calls or supported export/import integrations.

Never claim that MCP alone can intercept every conversation or force every AI client to save memories automatically.

## 11. Security and Privacy

This is a personal memory database. Treat it as sensitive by default.

Requirements:

- Local-first storage.
- No mandatory cloud dependency.
- No automatic upload of the raw archive.
- No default transmission of private conversations to third-party LLMs.
- Explicit opt-in for hosted inference.
- Configurable redaction before external inference.
- Local configuration and secret management.
- Authentication for remote access.
- Encrypted transport for remote connections.
- Restricted database network exposure.
- Input validation and query limits.
- Audit logging for memory reads and writes.
- User-controlled export and deletion.
- Clear handling of sensitive memories.
- Backup and restore functionality.
- Protection against prompt injection contained in imported conversations.
- Protection against malicious content attempting to manipulate memory extraction or tool execution.

Imported conversation content must be treated as untrusted data, not as instructions to the importer or the MCP server.

An AI agent must not be allowed to execute arbitrary database commands, filesystem operations, or shell commands through the memory interface.

Define and document the threat model.

## 12. Technology Stack

Use the following as the starting hypothesis, not an unquestionable mandate:

- Python for the core engine.
- FastMCP or the currently appropriate official MCP Python SDK.
- FalkorDB for the graph.
- SQLite for lightweight metadata and job state where needed.
- FastAPI for an optional HTTP API and dashboard backend.
- React and TypeScript for the dashboard.
- Docker Compose for reproducible local deployment.
- An optional local embedding model.
- A provider-agnostic LLM adapter.
- Automated tests and type checking.

Research current compatibility and choose stable, mutually compatible versions.

Prefer a modular monolith over premature microservices.

The MCP server, importer, retrieval engine, and consolidation pipeline should be separate modules with well-defined interfaces, even if they run in one process.

Keep the deployment lightweight enough for an ordinary Linux laptop.

Do not require a discrete GPU.

Do not introduce Kubernetes, a message broker, or a complex distributed architecture unless a demonstrated requirement justifies it.

## 13. User Interface

Build a functional dashboard, not merely a landing page.

It should include:

### Archive overview

- Number of imported conversations.
- Number of messages.
- Number of extracted memories.
- Number of graph entities and relationships.
- Import and consolidation progress.
- Failed or incomplete records.

### Memory explorer

- Search memories.
- Inspect individual memory records.
- View source conversations and supporting messages.
- Inspect related nodes and relationships.
- See when a memory was created or updated.
- Distinguish verified facts from inferred information.

### Graph explorer

- Visualise relevant portions of the graph.
- Display relationship labels.
- Navigate from a memory to its source.
- Filter by project, category, time, and source.
- Avoid attempting to render the entire graph at once.

### Maintenance

- Start and monitor imports.
- Trigger consolidation.
- Inspect unresolved contradictions.
- Edit or delete memories.
- Export and back up the database.
- View configured AI clients and MCP connection instructions.

### Retrieval testing

Provide a query interface where I can ask a question and inspect:

- The answer or returned context.
- Retrieved memories.
- Graph paths used.
- Original evidence.
- Retrieval latency.
- Token estimates when available.

## 14. Evaluation Framework

Build an evaluation suite before claiming the system works.

Create a test dataset from a small, representative subset of imported conversations.

Include questions whose answers are known from the original archive.

Evaluate:

1. Historical fact retrieval.
2. Exact message retrieval.
3. Retrieval across separate conversations.
4. Project-context reconstruction.
5. Decision-history reconstruction.
6. Temporal updates and contradictions.
7. Duplicate detection.
8. Provenance correctness.
9. Context compression quality.
10. Retrieval latency.
11. Memory storage growth.
12. CPU and RAM usage.
13. MCP tool reliability.
14. Cross-client consistency.
15. Resistance to prompt injection in imported content.

Measure retrieval precision and recall where ground-truth annotations are available.

Evaluate summary fidelity against source records rather than assuming a concise summary is a good summary.

Record failures and compare architectural alternatives on the same dataset.

Do not invent benchmark results.

## 15. Hackathon Demonstration

The final demonstration should show the following workflow.

### Step 1: Historical migration

Import a real exported conversation archive.

Show the importer processing the archive and generating persistent memories and graph relationships.

### Step 2: Cross-conversation retrieval

Ask a question whose answer requires information from more than one historical conversation.

Show the system retrieving the relevant memories and tracing them back to their sources.

### Step 3: Graph-based reasoning

Demonstrate that graph traversal connects information that would otherwise remain fragmented across conversations.

Show the graph paths used to construct the context.

### Step 4: Cross-client portability

Use one compatible client to save a new decision or finding.

Then use a second compatible client to retrieve it from the same memory server.

Do not simulate this integration. Verify that both clients access the same persistent database.

### Step 5: Update and contradiction

Change a project decision, then demonstrate that the system preserves the historical record while updating the current state.

### Step 6: Local ownership

Show that the memory persists across application restarts and can be exported and backed up independently of an AI provider.

## 16. Development Workflow

Follow these phases in order.

### Phase 0: Repository and environment inspection

- Inspect the existing repository before modifying it.
- Identify existing code, dependencies, configuration, and available tests.
- Detect the operating system and available CPU/RAM resources when possible.
- Identify existing Docker, Python, and Node.js installations.
- Avoid overwriting existing user work.
- Record assumptions and constraints.

### Phase 1: Architecture and model selection

- Compare candidate designs.
- Select the graph, metadata, vector, and search architecture.
- Select the extraction and summarisation models.
- Decide which tasks require an LLM and which can be deterministic.
- Estimate resource consumption.
- Produce an architecture decision record and milestone plan.

### Phase 2: Data model and importer

- Define canonical schemas.
- Implement the historical archive importer.
- Preserve raw records and provenance.
- Add resumability and idempotency.
- Test against representative sample archives.

### Phase 3: Memory extraction and graph construction

- Implement entity and relationship extraction.
- Add memory categorisation.
- Implement deduplication and temporal updates.
- Add hierarchical summaries.
- Persist results to FalkorDB.
- Validate graph integrity.

### Phase 4: Retrieval engine

- Implement exact, full-text, and graph-based retrieval.
- Add optional semantic retrieval.
- Implement ranking and context budgeting.
- Add source tracing.
- Create retrieval benchmarks.

### Phase 5: MCP server

- Implement and validate the MCP tools.
- Add schemas, error handling, and documentation.
- Test with at least one real compatible MCP client.
- Add configuration instructions for additional clients.

### Phase 6: Dashboard and operations

- Build the archive overview, memory explorer, graph explorer, and maintenance tools.
- Add backup and restore.
- Add health checks and resource monitoring.

### Phase 7: Integration and evaluation

- Run automated tests.
- Perform an end-to-end import.
- Test persistence across restarts.
- Test duplicate imports.
- Test contradictory memories.
- Verify provenance.
- Test MCP compatibility.
- Measure resource usage.
- Fix critical issues before adding optional features.

### Phase 8: Hackathon submission

Prepare:

- A complete README.
- Architecture documentation.
- An architecture decision record.
- A graph schema and example Cypher queries.
- Docker Compose configuration.
- Sample data that contains no private user information.
- Setup and troubleshooting instructions.
- MCP client configuration examples.
- Automated tests.
- A reproducible demo script.
- A concise demo video plan.
- A documented limitations section.

## 17. Engineering Standards

Follow these requirements throughout development:

- Write maintainable, typed Python.
- Use explicit interfaces between modules.
- Validate external input.
- Handle errors with actionable messages.
- Avoid hardcoded machine-specific paths.
- Keep secrets out of version control.
- Add tests for meaningful behaviour.
- Use structured logs without exposing sensitive conversation content.
- Document configuration variables.
- Avoid unnecessary dependencies.
- Keep operations idempotent where practical.
- Implement graceful degradation when an LLM provider is unavailable.
- Make long-running imports resumable.
- Never claim a feature works without testing it.
- Never replace real persistence with in-memory mock data in the final implementation.
- Never claim compatibility with an AI client without testing or clearly documenting the unverified configuration.

Use sensible defaults, but make important limits configurable.

## 18. Definition of Done

The MVP is complete only when:

1. A real conversation export can be imported.
2. Original messages remain recoverable.
3. The system generates semantic memories and meaningful graph relationships.
4. Memories retain provenance.
5. Historical information can be retrieved across conversations.
6. The graph materially contributes to retrieval.
7. The MCP server exposes working memory tools.
8. A compatible AI client can query and update the persistent store.
9. A second compatible client can access the same store, with actual integration verified.
10. Memory survives application and database restarts.
11. Duplicate imports do not create duplicate memories.
12. Contradictory or outdated facts are handled explicitly.
13. Private data is not transmitted externally without the configured consent and safeguards.
14. The system runs without a GPU.
15. The project can be installed from documented instructions.
16. Automated tests cover the core pipeline.
17. The user can inspect, correct, export, and delete memories.

## 19. Final Instructions

Begin by inspecting the repository and producing the architecture decision report.

Do not stop after writing a proposal. Once the design is established, implement the system in small, verifiable milestones.

At the end of each milestone, report:

- What was implemented.
- Which files changed.
- Which tests ran and their actual results.
- Which decisions remain unresolved.
- Resource requirements and known limitations.
- The next concrete milestone.

Prioritise a complete, working vertical slice over a large but unfinished collection of features.

If a component is uncertain or unavailable, investigate it and implement the best supported alternative without pretending the original component works.

The ultimate objective is to give the user permanent ownership of their accumulated AI memory, with graph-based retrieval and a transferable MCP interface that makes that memory available to compatible assistants.

Build a real product, not a demo that merely looks like one.
