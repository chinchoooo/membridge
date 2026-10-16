"""SQLite schema for the memory layers (episodic/semantic memories, entities, summaries)."""

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,               -- doc_<hash(kind, name)>
  kind TEXT NOT NULL,                -- claude_memory | claude_project | client_note | session
  name TEXT NOT NULL,
  text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  updated_at TEXT,
  import_job TEXT
);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,               -- mem_<hash(dedup_key)>
  statement TEXT NOT NULL,
  kind TEXT NOT NULL,                -- episodic | semantic
  category TEXT NOT NULL,            -- preference|decision|project|constraint|profile|skill|fact|lesson|interest|health|education|research|session
  subject TEXT,
  predicate TEXT,
  object TEXT,
  status TEXT NOT NULL DEFAULT 'current',   -- current | superseded | unresolved | deleted
  verification TEXT NOT NULL,        -- user_stated | provider_stated | model_inferred | user_verified
  confidence REAL NOT NULL,
  extraction_method TEXT NOT NULL,
  sensitivity TEXT NOT NULL DEFAULT 'normal',  -- normal | sensitive
  valid_from REAL,
  valid_to REAL,
  first_seen REAL,
  last_seen REAL,
  mention_count INTEGER NOT NULL DEFAULT 1,
  score REAL NOT NULL DEFAULT 0,
  dedup_key TEXT NOT NULL UNIQUE,
  origin TEXT NOT NULL DEFAULT 'import',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status, category);
CREATE INDEX IF NOT EXISTS idx_mem_sp ON memories(subject, predicate);

CREATE TABLE IF NOT EXISTS memory_evidence (
  memory_id TEXT NOT NULL REFERENCES memories(id),
  ref TEXT NOT NULL,                 -- message id or document id
  ref_kind TEXT NOT NULL,            -- message | document
  quote TEXT,
  PRIMARY KEY (memory_id, ref)
);

CREATE TABLE IF NOT EXISTS memory_links (
  src_id TEXT NOT NULL REFERENCES memories(id),
  rel TEXT NOT NULL,                 -- SUPERSEDES | CONTRADICTS | RELATED_TO | DEPENDS_ON | LEARNED_FROM | DERIVED_FROM
  dst_id TEXT NOT NULL REFERENCES memories(id),
  reason TEXT,
  created_at REAL NOT NULL,
  PRIMARY KEY (src_id, rel, dst_id)
);

CREATE TABLE IF NOT EXISTS memory_revisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id TEXT NOT NULL,
  ts REAL NOT NULL,
  actor TEXT NOT NULL,
  reason TEXT,
  old_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,               -- ent_<hash(canonical)>
  canonical TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'concept',      -- project | concept | person | organisation | tool
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entities (
  memory_id TEXT NOT NULL REFERENCES memories(id),
  entity_id TEXT NOT NULL REFERENCES entities(id),
  role TEXT NOT NULL DEFAULT 'about',        -- about | object | subject
  PRIMARY KEY (memory_id, entity_id, role)
);
CREATE INDEX IF NOT EXISTS idx_me_entity ON memory_entities(entity_id);

CREATE TABLE IF NOT EXISTS entity_mentions (
  entity_id TEXT NOT NULL REFERENCES entities(id),
  conversation_id TEXT NOT NULL REFERENCES conversations(id),
  count INTEGER NOT NULL,
  PRIMARY KEY (entity_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS summaries (
  id TEXT PRIMARY KEY,               -- sum_<level>_<ref>
  level TEXT NOT NULL,               -- conversation | project | personal
  ref TEXT NOT NULL,
  text TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  method TEXT NOT NULL,              -- extractive | llm
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_state (
  message_id TEXT NOT NULL,
  extractor TEXT NOT NULL,
  processed_at REAL NOT NULL,
  PRIMARY KEY (message_id, extractor)
);

CREATE TABLE IF NOT EXISTS pending_confirms (
  token TEXT PRIMARY KEY,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  expires_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  memory_id UNINDEXED, statement, tokenize = 'porter unicode61'
);
"""
