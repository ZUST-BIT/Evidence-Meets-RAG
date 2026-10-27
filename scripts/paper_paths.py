"""Publication paths and cohort checks shared by the public entry points."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl"
RESULTS = ROOT / "dataset/rag/results"
ANALYSIS = ROOT / "dataset/rag/analysis/unified_571"
FROZEN = ROOT / "dataset/frozen_contexts"
SOURCES = ("raw_observation_view", "semantic_kg", "evidence_graph_complete")
FRAMEWORKS = ("naive_rag", "dense_rag", "hybrid_rag", "graph_rag", "crag_rag", "raptor_rag", "kg2rag")
FULL = RESULTS / "fixed_context_component_ablation_571_vanilla__full.jsonl"
GRAPH_FULL = RESULTS / "unified_571/stage/R1C1G1.jsonl"


def suffix(framework: str, source: str) -> str:
    if framework in ("raptor_rag", "kg2rag"):
        return "formalv3_571_cross_raptor_kg2rag"
    return "formalv3_571_classic4_semantic_kg" if source == "semantic_kg" else "formalv3_571_classic4_raw_vs_graph"


def result_path(framework: str, source: str, *, canonical: bool = True) -> Path:
    if canonical and source == "evidence_graph_complete":
        if framework == "naive_rag":
            return FULL
        if framework == "graph_rag":
            return GRAPH_FULL
    return RESULTS / f"multi_cancer__{source}__{framework}__{suffix(framework, source)}.jsonl"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checked_rows(path: Path, *, predictions: bool = False) -> list[dict]:
    records = read_rows(BENCHMARK)
    expected = {str(r["id"]): r for r in records}
    rows = read_rows(path)
    keyed = {str((r.get("record") or r)["id"]): r for r in rows}
    if len(expected) != 571 or len(rows) != 571 or len(keyed) != 571 or set(keyed) != set(expected):
        raise ValueError(f"{path}: expected the same 571 unique benchmark IDs")
    for rid, row in keyed.items():
        if (row.get("record") or row) != expected[rid]:
            raise ValueError(f"{path}: changed benchmark record {rid}")
        if predictions and (not row.get("prediction") or row.get("normalized_prediction") in (None, "")):
            raise ValueError(f"{path}: incomplete prediction for {rid}")
    return [keyed[str(r["id"])] for r in records]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
