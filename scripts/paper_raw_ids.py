"""Preserve published raw-document IDs after portable source-path substitution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAP_PATH = Path(__file__).resolve().parents[1] / "configs" / "raw_parent_id_map.json"


def apply_paper_raw_ids(
    docs: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, int]:
    """Map only known source parents; new source releases keep their portable IDs.

    The compact map contains source-relative filenames and identifiers, never raw
    contents or a developer's absolute data path. Matched-budget extraction later
    checks that every required published doc ID is present.
    """
    payload = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    parent_map = {row["portable_parent_id"]: row["paper_parent_id"] for row in payload["parents"]}

    def mapped(identifier: str | None) -> str | None:
        if not identifier:
            return identifier
        for marker in ("-BLK", "-CH"):
            if marker in identifier:
                parent, suffix = identifier.rsplit(marker, 1)
                return parent_map.get(parent, parent) + marker + suffix
        return parent_map.get(identifier, identifier)

    changed = 0
    for doc in docs:
        old = doc["doc_id"]
        doc["doc_id"] = mapped(old)
        doc["parent_id"] = mapped(doc.get("parent_id"))
        changed += old != doc["doc_id"]
    for edge in edges:
        for key in ("source_doc_id", "target_doc_id"):
            if key in edge:
                edge[key] = mapped(edge[key])
    for row in manifest:
        row["doc_id"] = mapped(row["doc_id"])
    if len({doc["doc_id"] for doc in docs}) != len(docs):
        raise ValueError("Raw ID compatibility mapping produced duplicate document IDs")
    ids = {doc["doc_id"] for doc in docs}
    if any(edge.get("source_doc_id") not in ids or edge.get("target_doc_id") not in ids for edge in edges):
        raise ValueError("Raw ID compatibility mapping produced an orphan edge")
    return {"mapped_document_count": changed, "known_parent_count": len(parent_map)}
