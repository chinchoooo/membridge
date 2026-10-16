"""FalkorDB projection (schema v2) of the canonical SQLite records.

SQLite is the system of record; the graph is a rebuildable index used for bounded neighbourhood extraction,
Cypher path explanation, full-text/vector indexes and global algorithms (PageRank prior, WCC). All values are
passed as query parameters; only whitelisted relationship names are ever put into Cypher text.

Schema (label `Memory` is the plan's "Fact", `Conversation` its "Episode"):
  nodes  User, Entity{name,type,canonical,aliases,pr,component}, Memory{statement,category,status,verification,
         confidence,score,importance,activation,valid_at,invalid_at,recorded_at,expired_at,...}, Conversation,
         Message, Source
  edges  Memory-ABOUT{role}->Entity, Memory-SUPPORTED_BY{quote}->Message|Source, Message-PART_OF->Conversation,
         Entity-MENTIONED_IN{count}->Conversation, Memory-SUPERSEDES|CONTRADICTS|...->Memory,
         Entity-REL{predicate,fact_id,weight,valid_at,invalid_at}->Entity, Entity-RELATED_TO{weight}->Entity,
         Entity-SAME_AS{score,status}->Entity (pending review), User-DECIDED|PREFERS|WORKED_ON
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterable, Iterator
from typing import Any

from falkordb import FalkorDB  # type: ignore[import-untyped]

from ..graphalgo.adjacency import Graph, load_from_graph
from ..memory.models import LINK_RELS
from .sqlite import Store

BATCH = 400


def _batches(rows: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    for i in range(0, len(rows), BATCH):
        yield rows[i : i + BATCH]


def fulltext_query(text: str, max_terms: int = 12) -> str | None:
    """Turn free text into a safe RediSearch OR-query (alphanumerics only, no operators)."""
    terms = re.findall(r"[A-Za-z0-9]{2,}", text.lower())
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
            "CALL db.idx.fulltext.createNodeIndex('Entity','name','aliases')",
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
    def sync(self, store: Store, only: Iterable[str] | None = None) -> dict[str, int]:
        """Idempotent projection. `only` limits memory-scoped work to those memory ids (incremental sync);
        derived structures (entities, REL, RELATED_TO, SAME_AS, MENTIONED_IN) are always refreshed because
        they are cheap and a merge/split anywhere can change them."""
        c = store.conn
        self.ensure_indexes()
        self.query("MERGE (:User {id:'user', name:'me'})")
        n = {"entities": 0, "memories": 0, "conversations": 0, "messages": 0, "edges": 0}
        scoped = only is not None
        ids: list[str] = []
        if scoped:
            ids = list(only or [])
            c.execute("CREATE TEMP TABLE IF NOT EXISTS _sync_ids(id TEXT PRIMARY KEY)")
            c.execute("DELETE FROM _sync_ids")
            c.executemany("INSERT OR IGNORE INTO _sync_ids(id) VALUES (?)", [(i,) for i in ids])
        flt = " AND m.id IN (SELECT id FROM _sync_ids)" if scoped else ""

        def rows(sql: str, *args: Any) -> list[dict[str, Any]]:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

        def run(cypher: str, data: list[dict[str, Any]], key: str | None = None) -> None:
            for chunk in _batches(data):
                self.query(cypher, {"rows": chunk})
            if key:
                n[key] += len(data)

        # derived edges are wiped and re-added so merges, splits and decisions can never leave stale edges
        for rel in ("REL", "RELATED_TO", "SAME_AS", "MENTIONED_IN"):
            self.query(f"MATCH ()-[x:{rel}]->() DELETE x")
        if scoped:
            self.query("MATCH (m:Memory)-[x]-() WHERE m.id IN $ids DELETE x", {"ids": ids})
        else:
            for rel in ("ABOUT", "SUPPORTED_BY", "PART_OF", "DECIDED", "PREFERS", "WORKED_ON", *sorted(LINK_RELS)):
                self.query(f"MATCH ()-[x:{rel}]->() DELETE x")

        self.query("MATCH (c:Community) DETACH DELETE c")  # derived: re-projected below
        gone_entities = [r["id"] for r in rows("SELECT id FROM entities WHERE merged_into IS NOT NULL")]
        if gone_entities:
            self.query("MATCH (e:Entity) WHERE e.id IN $ids DETACH DELETE e", {"ids": gone_entities})
        run(
            "UNWIND $rows AS r MERGE (e:Entity {id:r.id}) SET e.name=r.name, e.type=r.type, e.canonical=r.canonical, e.aliases=r.aliases",
            rows("SELECT e.id, e.name, e.type, e.canonical, IFNULL((SELECT group_concat(alias_norm,' ') FROM entity_aliases a "
                 "WHERE a.entity_id=e.id),'') AS aliases FROM live_entities e"), "entities",
        )  # fmt: skip
        gone = [r["id"] for r in rows(f"SELECT m.id FROM memories m WHERE m.status='deleted'{flt}")]
        if gone:
            self.query("MATCH (m:Memory) WHERE m.id IN $ids DETACH DELETE m", {"ids": gone})
        run(
            "UNWIND $rows AS r MERGE (m:Memory {id:r.id}) SET m.statement=r.statement, m.category=r.category, "
            "m.kind=r.kind, m.status=r.status, m.verification=r.verification, m.confidence=r.confidence, "
            "m.score=r.score, m.importance=r.importance, m.activation=r.activation, m.sensitivity=r.sensitivity, "
            "m.first_seen=r.first_seen, m.last_seen=r.last_seen, m.mentions=r.mention_count, "
            "m.valid_at=r.valid_at, m.invalid_at=r.invalid_at, m.recorded_at=r.recorded_at, m.expired_at=r.expired_at",
            rows("SELECT m.id, m.statement, m.category, m.kind, m.status, m.verification, m.confidence, m.score, m.importance, "
                 "m.activation, m.sensitivity, IFNULL(m.first_seen,0) AS first_seen, IFNULL(m.last_seen,0) AS last_seen, "
                 "m.mention_count, m.valid_at, m.invalid_at, m.recorded_at, m.expired_at "
                 f"FROM memories m WHERE m.status!='deleted'{flt}"), "memories",
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
                 f"JOIN memories m ON m.id=e.memory_id WHERE m.status!='deleted'{flt}"), "messages",
        )  # fmt: skip
        run("UNWIND $rows AS r MERGE (s:Source {id:r.id}) SET s.name=r.name, s.kind=r.kind", rows("SELECT d.id, d.name, d.kind FROM documents d"))
        live = f"m.status!='deleted'{flt}"
        run(
            "UNWIND $rows AS r MATCH (m:Memory {id:r.m}), (e:Entity {id:r.e}) MERGE (m)-[x:ABOUT {role:r.role}]->(e)",
            rows("SELECT me.memory_id AS m, me.entity_id AS e, me.role AS role FROM memory_entities me "
                 f"JOIN memories m ON m.id=me.memory_id JOIN live_entities le ON le.id=me.entity_id WHERE {live}"), "edges",
        )  # fmt: skip
        for tgt, kind in (("Message", "message"), ("Source", "document")):
            run(
                f"UNWIND $rows AS r MATCH (m:Memory {{id:r.m}}), (t:{tgt} {{id:r.t}}) MERGE (m)-[x:SUPPORTED_BY]->(t) SET x.quote=r.quote",
                rows("SELECT ev.memory_id AS m, ev.ref AS t, IFNULL(ev.quote,'') AS quote FROM memory_evidence ev "
                     f"JOIN memories m ON m.id=ev.memory_id WHERE ev.ref_kind='{kind}' AND {live}"), "edges",
            )  # fmt: skip
        scope_links = " AND (l.src_id IN (SELECT id FROM _sync_ids) OR l.dst_id IN (SELECT id FROM _sync_ids))" if scoped else ""
        for rel in sorted(LINK_RELS):  # whitelist: relationship names are never user-controlled
            run(
                f"UNWIND $rows AS r MATCH (a:Memory {{id:r.a}}), (b:Memory {{id:r.b}}) MERGE (a)-[x:{rel}]->(b) SET x.reason=r.reason",
                rows("SELECT l.src_id AS a, l.dst_id AS b, IFNULL(l.reason,'') AS reason FROM memory_links l "
                     "JOIN memories a ON a.id=l.src_id JOIN memories b ON b.id=l.dst_id "
                     f"WHERE l.rel=? AND a.status!='deleted' AND b.status!='deleted'{scope_links}", rel), "edges",
            )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (e:Entity {id:r.e}), (v:Conversation {id:r.c}) MERGE (e)-[x:MENTIONED_IN]->(v) SET x.count=r.n",
            rows("SELECT x.entity_id AS e, x.conversation_id AS c, x.count AS n FROM entity_mentions x "
                 "JOIN live_entities le ON le.id=x.entity_id"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (a:Entity {id:r.a}), (b:Entity {id:r.b}) MERGE (a)-[x:RELATED_TO]->(b) SET x.weight=r.w",
            rows("SELECT a.entity_id AS a, b.entity_id AS b, COUNT(*) AS w FROM memory_entities a "
                 "JOIN memory_entities b ON a.memory_id=b.memory_id AND a.entity_id<b.entity_id "
                 "JOIN memories m ON m.id=a.memory_id JOIN live_entities ea ON ea.id=a.entity_id "
                 "JOIN live_entities eb ON eb.id=b.entity_id WHERE m.status!='deleted' GROUP BY a.entity_id, b.entity_id"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (a:Entity {id:r.a}), (b:Entity {id:r.b}) "
            "MERGE (a)-[x:REL {fact_id:r.f}]->(b) SET x.predicate=r.p, x.weight=r.w, x.valid_at=r.v, x.invalid_at=r.i",
            rows("SELECT src_entity AS a, dst_entity AS b, fact_id AS f, predicate AS p, weight AS w, valid_at AS v, invalid_at AS i FROM entity_edges"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (a:Entity {id:r.a}), (b:Entity {id:r.b}) MERGE (a)-[x:SAME_AS]->(b) SET x.score=r.s, x.status=r.st",
            rows("SELECT a_id AS a, b_id AS b, score AS s, status AS st FROM entity_link_candidates WHERE status='pending'"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (u:User {id:'user'}), (m:Memory {id:r.m}) MERGE (u)-[x:DECIDED]->(m)",
            rows(f"SELECT m.id AS m FROM memories m WHERE m.category='decision' AND m.status!='deleted'{flt}"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (u:User {id:'user'}), (m:Memory {id:r.m}) MERGE (u)-[x:PREFERS]->(m) SET x.polarity=r.p",
            rows(f"SELECT m.id AS m, m.predicate AS p FROM memories m WHERE m.category='preference' AND m.status!='deleted'{flt}"), "edges",
        )  # fmt: skip
        run("UNWIND $rows AS r MERGE (c:Community {id:r.id}) SET c.level=r.level, c.method=r.method", rows("SELECT id, level, method FROM communities"))
        run(
            "UNWIND $rows AS r MATCH (e:Entity {id:r.e}), (c:Community {id:r.c}) MERGE (e)-[:MEMBER_OF]->(c)",
            rows("SELECT cm.entity_id AS e, cm.community_id AS c FROM community_members cm JOIN communities c ON c.id=cm.community_id "
                 "JOIN live_entities le ON le.id=cm.entity_id WHERE c.level=1"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (a:Community {id:r.a}), (b:Community {id:r.b}) MERGE (a)-[:CHILD_OF]->(b)",
            rows("SELECT id AS a, parent_id AS b FROM communities WHERE parent_id IS NOT NULL"), "edges",
        )  # fmt: skip
        run(
            "UNWIND $rows AS r MATCH (u:User {id:'user'}), (e:Entity {id:r.e}) MERGE (u)-[x:WORKED_ON]->(e)",
            rows("SELECT DISTINCT e.id AS e FROM live_entities e WHERE e.type='project'"), "edges",
        )  # fmt: skip
        return n

    def compute_global_metrics(self) -> dict[str, Any]:
        """Run FalkorDB's built-in global algorithms and store results on Entity nodes (prior / diagnostics)."""
        out: dict[str, Any] = {}
        try:
            self.query("CALL algo.pageRank(null, null) YIELD node, score WITH node, score WHERE 'Entity' IN labels(node) SET node.pr = score")
            out["pagerank"] = int(self.query("MATCH (e:Entity) WHERE e.pr IS NOT NULL RETURN count(e)")[0][0])
        except Exception as e:
            out["pagerank_error"] = str(e)[:120]
        try:
            self.query(
                "CALL algo.WCC({nodeLabels:['Entity','Memory'], relationshipTypes:['ABOUT','REL']}) YIELD node, componentId "
                "WITH node, componentId WHERE 'Entity' IN labels(node) SET node.component = componentId"
            )
            out["components"] = int(self.query("MATCH (e:Entity) WHERE e.component IS NOT NULL RETURN count(DISTINCT e.component)")[0][0])
        except Exception as e:
            out["wcc_error"] = str(e)[:120]
        return out

    def adjacency(self, seeds: list[str], hops: int = 2, cap: int = 2000) -> Graph:
        return load_from_graph(self, seeds, hops, cap)

    def counts(self) -> dict[str, int]:
        nodes = {r[0]: int(r[1]) for r in self.query("MATCH (n) RETURN labels(n)[0], count(n)")}
        edges = int(self.query("MATCH ()-[r]->() RETURN count(r)")[0][0])
        return {**{f"nodes_{k.lower()}": v for k, v in nodes.items()}, "edges": edges}

    def neighborhood(self, center: str, depth: int = 1, limit: int = 120) -> dict[str, Any]:
        """Nodes and edges within `depth` hops of a node id. Bounded; never the whole graph."""
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        title = "coalesce(x.statement, x.name, x.title, x.snippet, x.id)"

        def add(rows: list[list[Any]]) -> None:
            for r in rows:
                for nid, label, text, extra in ((r[0], r[1], r[2], r[3]), (r[4], r[5], r[6], r[7])):
                    nodes.setdefault(nid, {"id": nid, "label": label, "text": str(text)[:140], "status": extra})
                edges[(r[8], r[9], r[10])] = {"source": r[8], "target": r[9], "type": r[10]}

        q1 = (
            "MATCH (c {id:$id})-[r]-(x) RETURN c.id, labels(c)[0], coalesce(c.statement,c.name,c.title,c.snippet,c.id), coalesce(c.status,c.type,''), "
            f"x.id, labels(x)[0], {title}, coalesce(x.status,x.type,''), startNode(r).id, endNode(r).id, type(r) LIMIT {int(limit)}"
        )
        add(self.query(q1, {"id": center}))
        if depth >= 2 and nodes:
            frontier = [n for n in nodes if n != center and not n.startswith("msg_")][:25]
            q2 = (
                "MATCH (c)-[r]-(x) WHERE c.id IN $ids AND x.id <> $id RETURN c.id, labels(c)[0], coalesce(c.statement,c.name,c.title,c.snippet,c.id), "
                f"coalesce(c.status,c.type,''), x.id, labels(x)[0], {title}, coalesce(x.status,x.type,''), startNode(r).id, endNode(r).id, type(r) "
                f"LIMIT {int(limit)}"
            )
            add(self.query(q2, {"ids": frontier, "id": center}))
        return {"center": center, "nodes": list(nodes.values())[: limit * 2], "edges": list(edges.values())[: limit * 3], "truncated": len(nodes) >= limit}
