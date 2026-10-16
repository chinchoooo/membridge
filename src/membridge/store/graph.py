"""FalkorDB projection of the canonical SQLite records.

SQLite is the system of record; the graph is a rebuildable index of relationships used for
retrieval (entity -> memories, supersede chains, cross-conversation mentions). All values are
passed as query parameters; only whitelisted relationship names are ever put into Cypher text.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from typing import Any

from falkordb import FalkorDB  # type: ignore[import-untyped]

from ..memory.models import LINK_RELS
from .sqlite import Store

BATCH = 400


def _batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(rows), BATCH):
        yield rows[i : i + BATCH]


def fulltext_query(text: str, max_terms: int = 12) -> str | None:
    """Turn free text into a safe RediSearch OR-query (alphanumerics only, no operators)."""
    terms = [t for t in re.findall(r"[A-Za-z0-9]{2,}", text.lower())]
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return "|".join(seen[:max_terms]) or None


class GraphStore:
    def __init__(self, host: str = "127.0.0.1", port: int = 6390, name: str = "membridge") -> None:
        self.host, self.port, self.name = host, port, name
        self._db: FalkorDB | None = None

    @property
    def db(self) -> FalkorDB:
        if self._db is None:
            self._db = FalkorDB(host=self.host, port=self.port, socket_connect_timeout=3)
        return self._db

    @property
    def graph(self) -> Any:
        return self.db.select_graph(self.name)

    def available(self) -> bool:
        try:
            self.db.connection.ping()
            return True
        except Exception:
            self._db = None
            return False

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        result: list[list[Any]] = self.graph.query(cypher, params or {}).result_set
        return result

    def drop(self) -> None:
        with contextlib.suppress(Exception):  # the graph may not exist yet
            self.graph.delete()

    def ensure_indexes(self) -> None:
        for stmt in (
            "CALL db.idx.fulltext.createNodeIndex('Memory','statement')",
            "CALL db.idx.fulltext.createNodeIndex('Entity','name')",
            "CREATE INDEX FOR (m:Memory) ON (m.id)",
            "CREATE INDEX FOR (e:Entity) ON (e.id)",
            "CREATE INDEX FOR (c:Conversation) ON (c.id)",
            "CREATE INDEX FOR (g:Message) ON (g.id)",
        ):
            try:
                self.query(stmt)
            except Exception as e:  # already exists
                if "already" not in str(e).lower() and "exist" not in str(e).lower():
                    raise

    def rebuild(self, store: Store) -> dict[str, int]:
        self.drop()
        self.ensure_indexes()
        return self.sync(store)

    # ---- projection ------------------------------------------------------
    def sync(self, store: Store) -> dict[str, int]:
        """Idempotent MERGE of every record. Safe to call after each write batch."""
        c = store.conn
        self.ensure_indexes()
        self.query("MERGE (:User {id:'user', name:'me'})")
        n = {"entities": 0, "memories": 0, "conversations": 0, "messages": 0, "edges": 0}

        def rows(sql: str, *args: Any) -> list[dict[str, Any]]:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

        def run(cypher: str, data: list[dict[str, Any]], key: str | None = None) -> None:
            for chunk in _batches(data):
                self.query(cypher, {"rows": chunk})
            if key:
                n[key] += len(data)

        run(
            "UNWIND $rows AS r MERGE (e:Entity {id:r.id}) SET e.name=r.name, e.type=r.type, e.canonical=r.canonical",
            rows("SELECT id, name, type, canonical FROM entities"), "entities",
        )  # fmt: skip
        gone = [r["id"] for r in rows("SELECT id FROM memories WHERE status='deleted'")]
        if gone:
            self.query("MATCH (m:Memory) WHERE m.id IN $ids DETACH DELETE m", {"ids": gone})
        run(
            "UNWIND $rows AS r MERGE (m:Memory {id:r.id}) SET m.statement=r.statement, m.category=r.category, "
            "m.kind=r.kind, m.status=r.status, m.verification=r.verification, m.confidence=r.confidence, "
            "m.score=r.score, m.sensitivity=r.sensitivity, m.first_seen=r.first_seen, m.last_seen=r.last_seen, "
            "m.mentions=r.mention_count",
            rows("SELECT id, statement, category, kind, status, verification, confidence, score, sensitivity, "
                 "IFNULL(first_seen,0) AS first_seen, IFNULL(last_seen,0) AS last_seen, mention_count "
                 "FROM memories WHERE status!='deleted'"), "memories",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MERGE (v:Conversation {id:r.id}) SET v.title=r.title, v.ts=r.ts",
            rows("SELECT id, IFNULL(title,'') AS title, IFNULL(create_time,0) AS ts FROM conversations"), "conversations",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MERGE (g:Message {id:r.id}) SET g.role=r.role, g.ts=r.ts, g.snippet=r.snippet "
            "WITH g, r MATCH (v:Conversation {id:r.cid}) MERGE (g)-[:PART_OF]->(v)",
            rows("SELECT DISTINCT g.id, g.role, IFNULL(g.create_time,0) AS ts, SUBSTR(g.text,1,200) AS snippet, "
                 "g.conversation_id AS cid FROM messages g JOIN memory_evidence e ON e.ref=g.id "
                 "JOIN memories m ON m.id=e.memory_id WHERE m.status!='deleted'"), "messages",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MERGE (s:Source {id:r.id}) SET s.name=r.name, s.kind=r.kind",
            rows("SELECT d.id, d.name, d.kind FROM documents d"),
        )
        live = "m.status!='deleted'"
        run(
            "UNWIND $rows AS r MATCH (m:Memory {id:r.m}), (e:Entity {id:r.e}) MERGE (m)-[x:ABOUT]->(e) SET x.role=r.role",
            rows(f"SELECT me.memory_id AS m, me.entity_id AS e, me.role AS role FROM memory_entities me "
                 f"JOIN memories m ON m.id=me.memory_id WHERE {live}"), "edges",
        )  # fmt: skip
        for tgt, kind in (("Message", "message"), ("Source", "document")):
            run(
                f"UNWIND $rows AS r MATCH (m:Memory {{id:r.m}}), (t:{tgt} {{id:r.t}}) "
                "MERGE (m)-[x:SUPPORTED_BY]->(t) SET x.quote=r.quote",
                rows(f"SELECT ev.memory_id AS m, ev.ref AS t, IFNULL(ev.quote,'') AS quote FROM memory_evidence ev "
                     f"JOIN memories m ON m.id=ev.memory_id WHERE ev.ref_kind='{kind}' AND {live}"), "edges",
            )  # fmt: skip
        for rel in sorted(LINK_RELS):  # whitelist: relationship names are never user-controlled
            run(
                f"UNWIND $rows AS r MATCH (a:Memory {{id:r.a}}), (b:Memory {{id:r.b}}) "
                f"MERGE (a)-[x:{rel}]->(b) SET x.reason=r.reason",
                rows("SELECT l.src_id AS a, l.dst_id AS b, IFNULL(l.reason,'') AS reason FROM memory_links l "
                     "JOIN memories a ON a.id=l.src_id JOIN memories b ON b.id=l.dst_id "
                     "WHERE l.rel=? AND a.status!='deleted' AND b.status!='deleted'", rel), "edges",
            )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (e:Entity {id:r.e}), (v:Conversation {id:r.c}) "
            "MERGE (e)-[x:MENTIONED_IN]->(v) SET x.count=r.n",
            rows("SELECT entity_id AS e, conversation_id AS c, count AS n FROM entity_mentions"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (a:Entity {id:r.a}), (b:Entity {id:r.b}) MERGE (a)-[x:RELATED_TO]->(b) SET x.weight=r.w",
            rows("SELECT a.entity_id AS a, b.entity_id AS b, COUNT(*) AS w FROM memory_entities a "
                 "JOIN memory_entities b ON a.memory_id=b.memory_id AND a.entity_id<b.entity_id "
                 "JOIN memories m ON m.id=a.memory_id WHERE m.status!='deleted' GROUP BY a.entity_id, b.entity_id"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (u:User {id:'user'}), (m:Memory {id:r.m}) MERGE (u)-[x:DECIDED]->(m)",
            rows("SELECT id AS m FROM memories WHERE category='decision' AND status!='deleted'"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (u:User {id:'user'}), (m:Memory {id:r.m}) MERGE (u)-[x:PREFERS]->(m) SET x.polarity=r.p",
            rows("SELECT id AS m, predicate AS p FROM memories WHERE category='preference' AND status!='deleted'"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (u:User {id:'user'}), (e:Entity {id:r.e}) MERGE (u)-[x:WORKED_ON]->(e)",
            rows("SELECT DISTINCT e.id AS e FROM entities e WHERE e.type='project'"), "edges",
        )  # fmt: skip
        return n

    def counts(self) -> dict[str, int]:
        nodes = {r[0]: int(r[1]) for r in self.query("MATCH (n) RETURN labels(n)[0], count(n)")}
        edges = int(self.query("MATCH ()-[r]->() RETURN count(r)")[0][0])
        return {**{f"nodes_{k.lower()}": v for k, v in nodes.items()}, "edges": edges}
