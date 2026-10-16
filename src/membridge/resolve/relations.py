"""Derive typed entity->entity relations (`entity_edges`) from structured facts.

A fact that links a subject entity and an object entity ("Atlas uses PostgreSQL") becomes a typed edge that
carries the fact's validity interval, so as-of queries and supersede chains also hold at the relation level.
This is what gives the graph real multi-hop structure beyond co-occurrence. Always rebuilt from SQLite truth.
"""

from __future__ import annotations

from ..store.sqlite import Store

EDGE_PREDICATES = {"uses", "depends_on", "part_of", "built_with"}


def rebuild_entity_edges(store: Store) -> dict[str, int]:
    """Idempotent: regenerate every edge (and backfill subject/object entity ids) from memory_entities roles."""
    c = store.conn
    c.execute("DELETE FROM entity_edges")
    c.execute(
        """
        UPDATE memories SET subject_entity_id = (
            SELECT me.entity_id FROM memory_entities me JOIN live_entities e ON e.id=me.entity_id
            WHERE me.memory_id=memories.id AND me.role='subject' LIMIT 1),
          object_entity_id = (
            SELECT me.entity_id FROM memory_entities me JOIN live_entities e ON e.id=me.entity_id
            WHERE me.memory_id=memories.id AND me.role='object' LIMIT 1)
        WHERE status!='deleted'
        """
    )
    rows = c.execute(
        "SELECT id, subject_entity_id AS s, object_entity_id AS o, predicate, confidence, status, "
        "IFNULL(valid_at, valid_from) AS v, invalid_at AS iv FROM memories "
        "WHERE status!='deleted' AND subject_entity_id IS NOT NULL AND object_entity_id IS NOT NULL "
        "AND subject_entity_id != object_entity_id"
    ).fetchall()
    n = 0
    for r in rows:
        base = (r["predicate"] or "").split(":")[0]
        if base not in EDGE_PREDICATES:
            continue
        w = float(r["confidence"]) * (1.0 if r["status"] == "current" else 0.3)
        cur = c.execute(
            "INSERT OR IGNORE INTO entity_edges(src_entity, predicate, dst_entity, fact_id, weight, valid_at, invalid_at) VALUES (?,?,?,?,?,?,?)",
            (r["s"], base.upper(), r["o"], r["id"], w, r["v"], r["iv"]),
        )
        n += cur.rowcount
    c.execute("DELETE FROM entity_cooccurrence")
    c.execute(
        "INSERT INTO entity_cooccurrence(a_entity, b_entity, weight) SELECT a.entity_id, b.entity_id, COUNT(*) FROM memory_entities a JOIN memory_entities b ON a.memory_id=b.memory_id AND a.entity_id<b.entity_id JOIN memories m ON m.id=a.memory_id JOIN live_entities ea ON ea.id=a.entity_id JOIN live_entities eb ON eb.id=b.entity_id WHERE m.status!='deleted' GROUP BY a.entity_id, b.entity_id"
    )
    c.commit()
    return {"edges": n}
