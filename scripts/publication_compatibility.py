"""Restore publication row order from ID-only permutation assets.

The assets contain identifiers and integer indices only.  They do not contain
source text, drug-response measurements, model outputs, or retrieval scores.
"""
from __future__ import annotations

import array
import gzip
import hashlib
import json
import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any


COMPAT_ROOT = Path(__file__).resolve().parents[1] / "configs" / "publication_compatibility"
RUNTIME_TEXT_RE = re.compile(r'("(created_at|extracted_at)"\s*:\s*")([^"\n]*)(")')


@lru_cache(maxsize=1)
def publication_manifest() -> dict[str, Any]:
    path = COMPAT_ROOT / "manifest.json"
    if not path.is_file():
        return {"files": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported publication compatibility manifest: {path}")
    return manifest


def publication_key(row: dict[str, Any], key_type: str) -> str:
    if key_type == "doc_id":
        return str(row["doc_id"])
    if key_type == "edge_triple_json":
        return json.dumps(
            [row["source_doc_id"], row["target_doc_id"], row["relation"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise ValueError(f"Unknown publication key type: {key_type}")


def runtime_values(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata", {}).get("raw_node", {})
    values = {}
    if "created_at" in raw:
        values["created_at"] = raw["created_at"]
    provenance = raw.get("provenance", {})
    if "extracted_at" in provenance:
        values["extracted_at"] = provenance["extracted_at"]
    return values


def runtime_text_occurrences(row: dict[str, Any]) -> list[list[str]]:
    values = runtime_values(row)
    return [[m[2], m[3]] for m in RUNTIME_TEXT_RE.finditer(row.get("text", ""))
            if m[2] in values and m[3] == values[m[2]]]


def nonruntime_fingerprint(row: dict[str, Any]) -> str:
    """Hash a full row with only its own runtime paths/occurrences normalized."""
    values = runtime_values(row)
    normalized = dict(row)
    metadata = dict(row.get("metadata", {}))
    raw = dict(metadata.get("raw_node", {}))
    provenance = dict(raw.get("provenance", {}))
    if "created_at" in values:
        raw["created_at"] = "<PUBLICATION_RUNTIME:created_at>"
    if "extracted_at" in values:
        provenance["extracted_at"] = "<PUBLICATION_RUNTIME:extracted_at>"
        raw["provenance"] = provenance
    metadata["raw_node"] = raw
    normalized["metadata"] = metadata
    def normalize_match(match: re.Match[str]) -> str:
        if match[2] in values and match[3] == values[match[2]]:
            return match[1] + f"<PUBLICATION_RUNTIME:{match[2]}>" + match[4]
        return match[0]
    normalized["text"] = RUNTIME_TEXT_RE.sub(normalize_match, row.get("text", ""))
    return hashlib.sha256(json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def publication_runtime_entries() -> dict[str, Any]:
    entry = publication_manifest().get("retrieval_runtime")
    if entry is None:
        return {}
    path = COMPAT_ROOT / entry["file"]
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != entry["gzip_sha256"]:
        raise ValueError(f"Publication runtime checksum mismatch: {path}")
    payload = gzip.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != entry["uncompressed_sha256"]:
        raise ValueError(f"Publication runtime payload checksum mismatch: {path}")
    data = json.loads(payload)
    if data.get("format_version") != 1 or len(data["entries"]) != entry["count"]:
        raise ValueError(f"Invalid publication runtime map: {path}")
    return data["entries"]


def restore_publication_runtime(path: Path, rows: list[dict[str, Any]]) -> dict[str, int] | None:
    """Restore historical retrieval provenance only for matching content.

    Source/native graph metadata is never written. Fresh input facts cannot
    acquire historical provenance: a whole-row non-runtime hash must match.
    """
    if path.parent.name != "multi_cancer" or path.name != "evidence_graph_complete_docs.jsonl":
        return None
    entries = publication_runtime_entries()
    report = {"eligible": 0, "restored": 0, "mismatched": 0, "unlisted": 0}
    for row in rows:
        entry = entries.get(row.get("doc_id"))
        if entry is None:
            report["unlisted"] += 1
            continue
        report["eligible"] += 1
        fingerprint, created, extracted, occurrences = entry
        current = runtime_values(row)
        if (nonruntime_fingerprint(row) != fingerprint
                or [name for name, _value in runtime_text_occurrences(row)]
                != [name for name, _value in occurrences]):
            report["mismatched"] += 1
            continue
        originals = {"created_at": created, "extracted_at": extracted}
        metadata = dict(row["metadata"])
        raw = dict(metadata["raw_node"])
        if "created_at" in current:
            raw["created_at"] = created
        if "extracted_at" in current:
            provenance = dict(raw["provenance"])
            provenance["extracted_at"] = extracted
            raw["provenance"] = provenance
        metadata["raw_node"] = raw
        row["metadata"] = metadata
        def restore_match(match: re.Match[str]) -> str:
            if match[2] in current and match[3] == current[match[2]]:
                return match[1] + originals[match[2]] + match[4]
            return match[0]
        row["text"] = RUNTIME_TEXT_RE.sub(restore_match, row.get("text", ""))
        report["restored"] += 1
    if report["mismatched"]:
        warnings.warn(
            f"{path.name}: {report['mismatched']} publication runtime fingerprints "
            "differ; retained fresh provenance for those rows.", stacklevel=2,
        )
    return report


def apply_publication_order(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the published order on an exact key-set match, else lexical order.

    Ordering applies only to the explicitly registered multi_cancer corpora.
    New corpora remain usable without applying an unrelated permutation.
    """
    if path.parent.name != "multi_cancer":
        return rows
    entry = publication_manifest().get("files", {}).get(path.name)
    if entry is None:
        return rows
    key_type = entry["key_type"]
    ordered = sorted(rows, key=lambda row: publication_key(row, key_type))
    keyset_hash = hashlib.sha256()
    previous = None
    unique = True
    for row in ordered:
        key = publication_key(row, key_type)
        if key == previous:
            unique = False
        previous = key
        keyset_hash.update(key.encode("utf-8") + b"\n")
    if (not unique or len(ordered) != entry["count"]
            or keyset_hash.hexdigest() != entry["keyset_sha256"]):
        warnings.warn(
            f"{path.name}: publication key set differs; using stable lexical order.",
            stacklevel=2,
        )
        return ordered

    permutation_path = COMPAT_ROOT / entry["permutation_file"]
    compressed = permutation_path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != entry["gzip_sha256"]:
        raise ValueError(f"Publication permutation checksum mismatch: {permutation_path}")
    payload = gzip.decompress(compressed)
    if (len(payload) != 4 * len(ordered)
            or hashlib.sha256(payload).hexdigest() != entry["permutation_sha256"]):
        raise ValueError(f"Invalid publication permutation payload: {permutation_path}")
    permutation = array.array("I")
    if permutation.itemsize != 4:
        raise RuntimeError("Publication permutations require 32-bit unsigned integers")
    permutation.frombytes(payload)
    if sys.byteorder != "little":
        permutation.byteswap()
    seen = bytearray(len(ordered))
    for index in permutation:
        if index >= len(ordered) or seen[index]:
            raise ValueError(f"Non-bijective publication permutation: {permutation_path}")
        seen[index] = 1
    return [ordered[index] for index in permutation]
