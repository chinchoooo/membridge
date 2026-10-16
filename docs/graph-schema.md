# Graph schema and example Cypher

SQLite is the system of record. FalkorDB holds a **rebuildable projection** (`membridge graph-rebuild`) used for
bounded neighbourhood extraction, path explanation, full-text/vector indexes and global algorithms. Every query
below is executed against a live graph by `tests/test_docs_cypher.py`, so this file cannot drift from the code.

## Nodes

| Label | Key properties | Meaning |
|---|---|---|
| `User` | `id='user'` | the owner of the memory |
| `Entity` | `id, name, type, canonical, aliases, pr, component` | a project, tool, person, concept (one node per resolved real-world thing) |
| `Memory` | `id, statement, category, kind, status, verification, confidence, score, importance, activation, valid_at, invalid_at, recorded_at, expired_at, sensitivity` | a fact/decision/preference (bi-temporal) |
| `Conversation` | `id, title, ts` | an archived conversation (episode) |
| `Message` | `id, role, ts, snippet` | an original message that is cited as evidence |
| `Source` | `id, name, kind` | a non-message source (provider memory file, client note, session) |
| `Community` | `id, level, method, title` | a detected group of related entities (level 1 = theme, 2 = area) |

## Relationships

| Type | From → To | Properties | Why it exists (retrieval requirement) |
|---|---|---|---|
| `ABOUT` | Memory → Entity | `role` (subject/object/about) | joins facts from different conversations at one node |
| `SUPPORTED_BY` | Memory → Message/Source | `quote` | provenance: every fact traces to evidence |
| `PART_OF` | Message → Conversation | | evidence → episode |
| `MENTIONED_IN` | Entity → Conversation | `count` | the cross-conversation hop for questions with no shared words |
| `SUPERSEDES`, `CONTRADICTS`, `RELATED_TO`, `DEPENDS_ON`, `LEARNED_FROM`, `DERIVED_FROM` | Memory → Memory | `reason` | history and explicit conflicts |
| `REL` | Entity → Entity | `predicate, fact_id, weight, valid_at, invalid_at` | typed relations derived from structured facts ("Atlas USES PostgreSQL"); gives real multi-hop structure |
| `RELATED_TO` | Entity → Entity | `weight` | weak co-occurrence signal |
| `SAME_AS` | Entity → Entity | `score, status` | uncertain alias pairs awaiting review |
| `MEMBER_OF` / `CHILD_OF` | Entity → Community / Community → Community | | hierarchy for "big picture" questions |
| `DECIDED`, `PREFERS`, `WORKED_ON` | User → Memory/Entity | `polarity` | the user's own commitments |

## Bi-temporal semantics

`valid_at`/`invalid_at` = when the fact was true in the world. `recorded_at`/`expired_at` = when we learned it, and
when we learned it was *never* true (a correction). Changing a decision sets `invalid_at`; correcting a mistake also sets
`expired_at`. Nothing is deleted, so "what was true then" and "what did we believe then" are both answerable.

## Example queries

Current facts about a project, strongest first:

```cypher
MATCH (e:Entity {canonical:'atlas'})<-[:ABOUT]-(m:Memory)
WHERE m.status = 'current'
RETURN DISTINCT m.statement, m.category ORDER BY m.score DESC
```

How a decision changed over time (supersede chain):

```cypher
MATCH (newer:Memory)-[r:SUPERSEDES]->(older:Memory)
RETURN newer.statement, r.reason, older.statement
```

What was true on a given date (`$t` is an epoch timestamp):

```cypher
MATCH (m:Memory)-[:ABOUT]->(e:Entity {canonical:'atlas'})
WHERE m.valid_at <= $t AND (m.invalid_at IS NULL OR m.invalid_at > $t)
RETURN DISTINCT m.statement
```

Provenance of one fact, down to the original conversation (`$id` is a memory id):

```cypher
MATCH (m:Memory {id:$id})-[:SUPPORTED_BY]->(g:Message)-[:PART_OF]->(c:Conversation)
RETURN g.snippet, c.title
```

Every conversation in which an entity appears (how fragments connect):

```cypher
MATCH (e:Entity {canonical:'atlas'})-[x:MENTIONED_IN]->(c:Conversation)
RETURN c.title, x.count ORDER BY x.count DESC
```

The multi-hop pattern behind "lexical-gap" questions: message → conversation → project → facts:

```cypher
MATCH (g:Message)-[:PART_OF]->(c:Conversation)<-[:MENTIONED_IN]-(e:Entity {type:'project'})<-[:ABOUT]-(m:Memory)
WHERE g.snippet CONTAINS 'connection slots'
RETURN DISTINCT m.statement, c.title
```

Typed relations with their validity (the entity-level view of history):

```cypher
MATCH (a:Entity)-[r:REL]->(b:Entity)
RETURN a.name, r.predicate, b.name, r.invalid_at
```

Hierarchy: which themes sit under which area:

```cypher
MATCH (e:Entity)-[:MEMBER_OF]->(c:Community)-[:CHILD_OF]->(p:Community)
RETURN p.title, c.title, collect(e.name) AS members
```

Uncertain alias pairs waiting for a human decision:

```cypher
MATCH (a:Entity)-[s:SAME_AS]->(b:Entity)
RETURN a.name, b.name, s.score
```

Find an entity by any of its aliases (full-text index over `name` and `aliases`):

```cypher
CALL db.idx.fulltext.queryNodes('Entity', 'atlas') YIELD node, score
RETURN node.name, score ORDER BY score DESC
```

Global importance prior (after `compute_global_metrics`):

```cypher
MATCH (e:Entity) WHERE e.pr IS NOT NULL RETURN e.name, e.pr ORDER BY e.pr DESC LIMIT 5
```

## Retrieval algorithm (what runs, in order)

1. **Entity linking** (names + aliases + FalkorDB full-text) and **temporal intent** (history / current / neutral).
2. **Channels**: facts BM25 (SQLite FTS5, same in every mode), exact-message BM25, entity facts, hop (message → conversation → project → facts),
   optional vector, and **Personalized PageRank** (forward push, hub-damped) over a bounded 2-hop neighbourhood.
3. **Fusion**: Reciprocal Rank Fusion + lexical term-coverage rerank + stored score + channel agreement; superseded facts are
   down-weighted unless the question is about the past; a superseded best match lifts its current superseder.
4. **MMR** diversity, **exact knapsack** packing into the token budget, calibrated "insufficient evidence".
