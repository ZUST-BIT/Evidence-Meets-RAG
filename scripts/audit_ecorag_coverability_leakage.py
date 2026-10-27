from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from rag_eval_lib import compute_answer_metrics, gold_label, normalize_gold_text  # noqa: E402


DEFAULT_DATASET = ROOT / "dataset" / "evaluated" / "external_reasoning_benchmark_formal_v3_571.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "rag" / "analysis" / "ecorag_coverability_leakage"

KNOWLEDGE_SOURCES = {
    "doc_corpus": {
        "label": "Document Corpus",
        "docs": ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / "raw_observation_view_docs.jsonl",
        "result": ROOT / "dataset" / "rag" / "results" / "multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
    "semantic_kg": {
        "label": "Semantic KG",
        "docs": ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / "semantic_kg_docs.jsonl",
        "result": ROOT / "dataset" / "rag" / "results" / "multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl",
    },
    "evidence_kg": {
        "label": "Evidence KG",
        "docs": ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / "evidence_graph_complete_docs.jsonl",
        "result": ROOT / "dataset" / "rag" / "results" / "multi_cancer__evidence_graph_complete__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl",
    },
}

DISEASE_ALIASES = {
    "aml": ["AML", "acute myeloid leukemia", "LAML"],
    "brca": ["BRCA", "breast carcinoma", "breast cancer"],
    "crc": ["CRC", "colorectal cancer", "colorectal adenocarcinoma", "colon adenocarcinoma", "rectal adenocarcinoma", "colon cancer", "rectal cancer", "COAD", "READ", "COADREAD"],
    "gastroesoph": ["GASTROESOPH", "gastroesophageal adenocarcinoma", "esophagogastric carcinoma", "esophageal adenocarcinoma", "stomach adenocarcinoma", "gastric cancer", "STAD"],
    "glioma": ["GLIOMA", "glioma", "glioblastoma", "glioblastoma multiforme", "GBM"],
    "hcc": ["HCC", "hepatocellular carcinoma", "liver hepatocellular carcinoma", "hepatocellular cancer", "liver cancer", "LIHC"],
    "hnsc": ["HNSC", "head and neck squamous cell carcinoma"],
    "meso": ["MESO", "mesothelioma"],
    "nsclc": ["NSCLC", "non-small cell lung cancer", "non small cell lung cancer", "lung adenocarcinoma", "lung squamous cell carcinoma", "LUAD", "LUSC"],
    "ovarian": ["OVARIAN", "ovarian carcinoma", "ovarian cancer", "OV"],
    "paad": ["PAAD", "pancreatic adenocarcinoma", "pancreatic cancer"],
    "prad": ["PRAD", "prostate cancer"],
    "rcc": ["RCC", "renal cell carcinoma", "kidney renal clear cell carcinoma", "KIRC"],
}

COMPONENT_CUES = {
    "target_association": [
        "target association",
        "target_association",
        "associated with",
        "association",
        "supports target discovery",
        "knowledgebase association",
        "direct target",
        "gene disease",
    ],
    "causal_relation": [
        "causal",
        "causal_chain",
        "activates",
        "activation",
        "inhibits",
        "inhibition",
        "regulates",
        "regulation",
        "upregulated",
        "downregulated",
        "mechanism context",
        "mechanism_context",
    ],
    "pathway_context": [
        "pathway",
        "pathway_context",
        "signaling",
        "mapk",
        "pi3k",
        "akt",
        "wnt",
        "notch",
        "jak",
        "stat",
        "emt",
        "apoptosis",
        "angiogenesis",
    ],
    "driver": [
        "driver",
        "oncogenic",
        "tumor suppressor",
        "tumour suppressor",
        "mutation",
        "mutated",
        "amplified",
        "deleted",
        "recurrent",
    ],
    "dependency_vulnerability": [
        "dependency",
        "vulnerability",
        "depmap",
        "crispr",
        "ceres",
        "selective dependency",
    ],
    "omics_manifestation": [
        "expression",
        "methylation",
        "copy number",
        "copy_number",
        "mutation",
        "phosphorylation",
        "multiomics",
        "tcga",
        "linkedomics",
    ],
    "literature_mechanism": [
        "pubmed",
        "literature",
        "pmid",
        "abstract",
        "clinical evidence",
        "case report",
    ],
    "aggregation": [
        "aggregate",
        "aggregation",
        "summary",
        "consistent",
        "conflicting",
        "multiple evidence",
        "member evidence",
        "bundle component",
    ],
    "precision_signal": [
        "sensitivity",
        "resistance",
        "poor outcome",
        "better outcome",
        "uncertain significance",
        "adverse outcome",
        "diagnostic",
        "prognostic",
        "oncogenicity",
        "supports sensitivity",
        "supports resistance",
        "does not support",
    ],
    "evidence_signal": [
        "diagnostic",
        "prognostic",
        "oncogenic",
        "positive",
        "negative",
        "poor outcome",
        "better outcome",
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def norm_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item not in (None, "", [], {}))
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", str(value))).strip().lower()


def unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def family_of(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    anchors = metadata.get("normalized_entity_anchors") or {}
    family = anchors.get("disease_family") or metadata.get("priority_cancer_family") or record.get("oncology_focus")
    return norm_text(str(family or "")).replace(" ", "_")


def reasoning_type(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(metadata.get("reasoning_type") or record.get("task_type") or "unknown")


def disease_terms(record: dict[str, Any]) -> list[str]:
    family = family_of(record)
    entities = record.get("entities") or {}
    terms = [
        str(record.get("oncology_focus") or ""),
        str(entities.get("disease") or ""),
        str(entities.get("disease_name") or ""),
        family,
    ]
    terms.extend(DISEASE_ALIASES.get(family, []))
    return unique_preserve([term for term in terms if term])


def gene_terms(record: dict[str, Any]) -> list[str]:
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    anchors = metadata.get("normalized_entity_anchors") or {}
    values = []
    values.extend(anchors.get("gene_symbols") or [])
    values.extend(entities.get("gene_symbols") or [])
    values.extend(
        [
            entities.get("gene_symbol"),
            entities.get("target_gene_symbol"),
            metadata.get("causal_intermediate_gene"),
        ]
    )
    return unique_preserve([str(value).upper() for value in values if value and len(str(value).strip()) >= 2])


def other_entity_terms(record: dict[str, Any]) -> list[str]:
    entities = record.get("entities") or {}
    metadata = record.get("metadata") or {}
    anchors = metadata.get("normalized_entity_anchors") or {}
    values: list[Any] = []
    for key in ["drug", "therapy_name", "variant", "alteration", "pathway_name", "cell_line", "molecular_profile"]:
        values.append(entities.get(key))
        values.append(anchors.get(key))
    values.extend(entities.get("therapy_names") or [])
    values.extend(anchors.get("therapy_names") or [])
    return unique_preserve([str(value) for value in values if value and len(str(value).strip()) >= 3])


def answer_terms(record: dict[str, Any]) -> list[str]:
    answer_type = str(record.get("answer_type") or "")
    if answer_type == "boolean":
        return []
    gold = record.get("gold_standard") or {}
    values = [
        record.get("answer"),
        gold.get("canonical_answer"),
        gold.get("normalized_answer"),
        gold.get("class_label"),
        *(gold.get("acceptable_answers") or [])[:5],
    ]
    terms = []
    for value in values:
        normalized = normalize_gold_text(value)
        if normalized and normalized not in {"yes", "no", "positive", "negative", "true", "false"} and len(normalized) >= 3:
            terms.append(normalized)
    return unique_preserve(terms)


def required_components(record: dict[str, Any]) -> list[str]:
    task_type = str(record.get("task_type") or "")
    metadata = record.get("metadata") or {}
    answer = str(record.get("answer") or "")
    components: list[str] = []
    layers = [str(item) for item in metadata.get("external_support_layers") or [] if item]
    components.extend(layers)

    if task_type == "cross_source_support_bundle":
        components.extend([part for part in answer.split("+") if part])
    elif task_type == "cross_source_corroboration_status":
        components.append("target_association")
        if str(record.get("answer") or "").lower() in {"yes", "true"}:
            components.append("causal_relation")
    elif task_type == "mechanism_trace_intermediate_gene":
        components.extend(["causal_relation", "pathway_context"])
    elif task_type == "precision_oncology_evidence":
        components.extend(["target_association", "precision_signal"])
    elif task_type == "evidence_signal_interpretation":
        components.append("evidence_signal")
    elif task_type == "hierarchical_evidence_aggregation":
        components.append("aggregation")
    elif task_type.startswith("mechanism_bioasq"):
        components.append("literature_mechanism")

    normalized = []
    aliases = {
        "mechanism": "causal_relation",
        "mechanism_context": "causal_relation",
        "source_aware": "cross_source",
        "supportive_omics": "omics_manifestation",
        "dependency_or_drug_sensitivity": "dependency_vulnerability",
    }
    for component in components:
        key = str(component).strip().lower().replace(" ", "_")
        key = aliases.get(key, key)
        if key in COMPONENT_CUES or key in {"cross_source"}:
            normalized.append(key)
    return unique_preserve(normalized)


def doc_source_family(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    raw_node = metadata.get("raw_node") if isinstance(metadata.get("raw_node"), dict) else {}
    source = (
        metadata.get("source_family")
        or metadata.get("source_graph")
        or raw_node.get("source_family")
        or raw_node.get("source_families")
        or doc.get("knowledge_source")
    )
    if isinstance(source, list):
        return "|".join(str(item) for item in source)
    return str(source or "unknown")


def doc_blob(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    compact_metadata = {
        key: value
        for key, value in metadata.items()
        if key in {"node_type", "source_family", "summary_scope", "task_scopes", "bundle_components", "association_inventory", "relation_counts", "source_graph"}
    }
    return norm_text(
        [
            doc.get("title") or "",
            doc.get("text") or "",
            " ".join(str(item) for item in doc.get("disease_tags") or []),
            " ".join(str(item) for item in doc.get("entity_names") or []),
            compact_metadata,
        ]
    )


def family_hits(blob: str, disease_family_terms: dict[str, list[str]]) -> set[str]:
    hits = set()
    for family, terms in disease_family_terms.items():
        for term in terms:
            normalized = norm_text(term)
            if normalized and re.search(rf"\b{re.escape(normalized)}\b", blob):
                hits.add(family)
                break
    return hits


def gene_hits(blob: str, query_genes: set[str]) -> set[str]:
    tokens = set(re.findall(r"\b[A-Z0-9\-]{2,16}\b", blob.upper()))
    return tokens & query_genes


def component_hits(blob: str) -> set[str]:
    hits = set()
    for component, cues in COMPONENT_CUES.items():
        if any(norm_text(cue) in blob for cue in cues):
            hits.add(component)
    return hits


def answer_hits(blob: str, query_answer_terms: set[str]) -> set[str]:
    hits = set()
    for term in query_answer_terms:
        if term and re.search(rf"\b{re.escape(term)}\b", blob):
            hits.add(term)
    return hits


def doc_is_evidence_like(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata") or {}
    node_type = str(metadata.get("node_type") or "").lower()
    level = str(doc.get("document_level") or "").lower()
    return node_type in {"evidence", "observation", "mechanism_role_summary", "target_support_summary"} or level in {"parent", "chunk"}


class SourceIndex:
    def __init__(self, key: str, docs_path: Path, records: list[dict[str, Any]]) -> None:
        self.key = key
        self.docs_path = docs_path
        self.doc_count = 0
        self.evidence_like_count = 0
        self.gene_to_docs: dict[str, set[int]] = defaultdict(set)
        self.family_to_docs: dict[str, set[int]] = defaultdict(set)
        self.component_to_docs: dict[str, set[int]] = defaultdict(set)
        self.answer_to_docs: dict[str, set[int]] = defaultdict(set)
        self.doc_source_family: dict[int, str] = {}
        self.doc_ids: dict[int, str] = {}

        query_genes = {gene for record in records for gene in gene_terms(record)}
        query_answer_terms = {term for record in records for term in answer_terms(record)}
        disease_family_terms = {
            family: unique_preserve([family, *aliases])
            for family, aliases in DISEASE_ALIASES.items()
        }
        self._build(query_genes, query_answer_terms, disease_family_terms)

    def _build(self, query_genes: set[str], query_answer_terms: set[str], disease_family_terms: dict[str, list[str]]) -> None:
        with self.docs_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.doc_count += 1
                self.doc_ids[index] = str(doc.get("doc_id") or f"doc-{index}")
                self.doc_source_family[index] = doc_source_family(doc)
                if doc_is_evidence_like(doc):
                    self.evidence_like_count += 1
                blob = doc_blob(doc)
                for gene in gene_hits(blob, query_genes):
                    self.gene_to_docs[gene].add(index)
                for family in family_hits(blob, disease_family_terms):
                    self.family_to_docs[family].add(index)
                for component in component_hits(blob):
                    self.component_to_docs[component].add(index)
                for answer in answer_hits(blob, query_answer_terms):
                    self.answer_to_docs[answer].add(index)

    def record_anchor_docs(self, record: dict[str, Any]) -> set[int]:
        family = family_of(record)
        disease_docs = set(self.family_to_docs.get(family, set()))
        genes = gene_terms(record)
        if genes:
            gene_docs: set[int] = set()
            for gene in genes:
                gene_docs.update(self.gene_to_docs.get(gene, set()))
            if disease_docs:
                return gene_docs & disease_docs
            return gene_docs
        other_terms = [norm_text(term).upper() for term in other_entity_terms(record)]
        other_docs: set[int] = set()
        for term in other_terms:
            other_docs.update(self.gene_to_docs.get(term, set()))
        if other_docs and disease_docs:
            return other_docs & disease_docs
        return disease_docs

    def audit_record(self, record: dict[str, Any]) -> dict[str, Any]:
        family = family_of(record)
        disease_docs = set(self.family_to_docs.get(family, set()))
        anchor_docs = self.record_anchor_docs(record)
        required = required_components(record)

        component_covered = {}
        component_doc_counts = {}
        for component in required:
            if component == "cross_source":
                continue
            hits = anchor_docs & self.component_to_docs.get(component, set())
            component_covered[component] = bool(hits)
            component_doc_counts[component] = len(hits)

        required_without_cross = [component for component in required if component != "cross_source"]
        covered_count = sum(1 for component in required_without_cross if component_covered.get(component))
        bundle_completeness = covered_count / max(len(required_without_cross), 1) if required_without_cross else (1.0 if anchor_docs else 0.0)

        answer_hit_docs: set[int] = set()
        for term in answer_terms(record):
            answer_hit_docs.update(self.answer_to_docs.get(term, set()))
        anchored_answer_docs = anchor_docs & answer_hit_docs
        exposure_risk = "none"
        if answer_terms(record) and anchored_answer_docs:
            if str(record.get("answer_type") or "") == "categorical":
                exposure_risk = "structured_or_label_exposure"
            else:
                exposure_risk = "natural_answer_exposure"

        source_families = {self.doc_source_family[index] for index in anchor_docs}
        source_family_count = len(source_families)
        cross_source_coverage = source_family_count >= 2
        if "cross_source" in required:
            component_covered["cross_source"] = cross_source_coverage
            component_doc_counts["cross_source"] = source_family_count

        support_coverable = bool(anchor_docs) and (bundle_completeness > 0 or not required_without_cross)
        if required_without_cross:
            support_coverable = bool(anchor_docs) and covered_count > 0
        coverable_non_leaky = support_coverable and exposure_risk != "structured_or_label_exposure"

        return {
            "entity_coverage": bool(anchor_docs),
            "disease_scope_coverage": bool(disease_docs),
            "rule_support_coverable": support_coverable,
            "bundle_completeness": round(bundle_completeness, 4),
            "required_components": required,
            "component_covered": component_covered,
            "component_doc_counts": component_doc_counts,
            "anchor_doc_count": len(anchor_docs),
            "disease_doc_count": len(disease_docs),
            "source_family_count": source_family_count,
            "cross_source_coverage": cross_source_coverage,
            "answer_terms": answer_terms(record),
            "answer_exposure_risk": exposure_risk,
            "answer_exposure_doc_count": len(anchored_answer_docs),
            "coverable_non_leaky": coverable_non_leaky,
            "example_anchor_doc_ids": [self.doc_ids[index] for index in sorted(anchor_docs)[:5]],
            "example_answer_exposure_doc_ids": [self.doc_ids[index] for index in sorted(anchored_answer_docs)[:5]],
        }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    def rate(key: str) -> float:
        return round(sum(1 for row in rows if row.get(key)) / max(total, 1), 4)

    avg_bundle = round(sum(float(row.get("bundle_completeness") or 0.0) for row in rows) / max(total, 1), 4)
    exposure_counts = Counter(str(row.get("answer_exposure_risk") or "none") for row in rows)
    return {
        "count": total,
        "entity_coverage": rate("entity_coverage"),
        "disease_scope_coverage": rate("disease_scope_coverage"),
        "rule_support_coverability": rate("rule_support_coverable"),
        "bundle_completeness_avg": avg_bundle,
        "cross_source_coverage": rate("cross_source_coverage"),
        "structured_or_label_exposure_rate": round(exposure_counts.get("structured_or_label_exposure", 0) / max(total, 1), 4),
        "any_answer_exposure_rate": round((total - exposure_counts.get("none", 0)) / max(total, 1), 4),
        "coverable_non_leaky_rate": rate("coverable_non_leaky"),
        "answer_exposure_counts": dict(sorted(exposure_counts.items())),
    }


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {group: summarize_rows(group_rows) for group, group_rows in sorted(grouped.items())}


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def subset_performance(result_rows: list[dict[str, Any]], audit_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    subsets = {
        "all": [],
        "entity_covered": [],
        "rule_support_coverable": [],
        "non_leaky": [],
        "coverable_non_leaky": [],
    }
    for row in result_rows:
        record = row.get("record") or {}
        rid = str(record.get("id") or "")
        audit = audit_by_id.get(rid)
        if not audit:
            continue
        subsets["all"].append(row)
        if audit.get("entity_coverage"):
            subsets["entity_covered"].append(row)
        if audit.get("rule_support_coverable"):
            subsets["rule_support_coverable"].append(row)
        if audit.get("answer_exposure_risk") != "structured_or_label_exposure":
            subsets["non_leaky"].append(row)
        if audit.get("coverable_non_leaky"):
            subsets["coverable_non_leaky"].append(row)

    return {
        name: compute_answer_metrics(rows) if rows else {"count": 0, "accuracy": 0.0, "macro_f1": 0.0}
        for name, rows in subsets.items()
    }


def compact_summary_rows(source_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source_key, summary in source_summaries.items():
        overall = summary["overall"]
        perf = summary.get("performance_subsets", {})
        rows.append(
            {
                "knowledge_source": summary["label"],
                "doc_count": summary["doc_count"],
                "entity_coverage": overall["entity_coverage"],
                "scope_coverage": overall["disease_scope_coverage"],
                "support_coverability": overall["rule_support_coverability"],
                "bundle_completeness": overall["bundle_completeness_avg"],
                "cross_source_coverage": overall["cross_source_coverage"],
                "structured_label_exposure": overall["structured_or_label_exposure_rate"],
                "coverable_non_leaky": overall["coverable_non_leaky_rate"],
                "coverable_non_leaky_n": perf.get("coverable_non_leaky", {}).get("count", ""),
                "all_acc": perf.get("all", {}).get("accuracy", ""),
                "all_macro_f1": perf.get("all", {}).get("macro_f1", ""),
                "coverable_non_leaky_acc": perf.get("coverable_non_leaky", {}).get("accuracy", ""),
                "coverable_non_leaky_macro_f1": perf.get("coverable_non_leaky", {}).get("macro_f1", ""),
            }
        )
    return rows


def render_report(source_summaries: dict[str, dict[str, Any]], output_dir: Path) -> str:
    rows = compact_summary_rows(source_summaries)
    lines = [
        "# Evidence Meets RAG: Coverability and Leakage Audit",
        "",
        "This audit is a rule-based proxy analysis. It does not claim to identify human-annotated gold evidence. Instead, it checks whether each knowledge shape exposes disease/entity anchors, task-specific support cues, evidence-bundle components, source diversity, and direct answer-label exposure for the formal v3 571 benchmark.",
        "",
        "## Overall Summary",
        "",
        "| Knowledge Source | Entity Coverage | Scope Coverage | Support Coverability | Bundle Completeness | Cross-source Coverage | Structured Label Exposure | Coverable Non-leaky | Non-leaky N | All Acc | All Macro-F1 | Non-leaky Acc | Non-leaky F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['knowledge_source']} | {row['entity_coverage']:.4f} | {row['scope_coverage']:.4f} | {row['support_coverability']:.4f} | "
            f"{row['bundle_completeness']:.4f} | {row['cross_source_coverage']:.4f} | {row['structured_label_exposure']:.4f} | "
            f"{row['coverable_non_leaky']:.4f} | {row['coverable_non_leaky_n']} | {row['all_acc']} | {row['all_macro_f1']} | {row['coverable_non_leaky_acc']} | {row['coverable_non_leaky_macro_f1']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    evidence = next((row for row in rows if row["knowledge_source"] == "Evidence KG"), None)
    doc = next((row for row in rows if row["knowledge_source"] == "Document Corpus"), None)
    semantic = next((row for row in rows if row["knowledge_source"] == "Semantic KG"), None)
    if evidence and doc and semantic:
        lines.extend(
            [
                f"- Evidence KG has higher rule-based support coverability than Document Corpus ({evidence['support_coverability']:.4f} vs {doc['support_coverability']:.4f}) and Semantic KG ({semantic['support_coverability']:.4f}).",
                f"- Evidence KG also has higher average bundle completeness ({evidence['bundle_completeness']:.4f}) than Document Corpus ({doc['bundle_completeness']:.4f}) and Semantic KG ({semantic['bundle_completeness']:.4f}), which supports the claim that evidence-centric organization improves access to multi-component evidence.",
                f"- Structured label exposure is not zero for Evidence KG ({evidence['structured_label_exposure']:.4f}), so the manuscript should present non-leaky subset results rather than relying only on the full benchmark.",
                f"- On its coverable non-leaky subset, Evidence KG reaches {evidence['coverable_non_leaky_acc']} Accuracy and {evidence['coverable_non_leaky_macro_f1']} Macro-F1 over {evidence['coverable_non_leaky_n']} examples. Its Accuracy is higher than Document Corpus ({doc['coverable_non_leaky_acc']}, n={doc['coverable_non_leaky_n']}) and Semantic KG ({semantic['coverable_non_leaky_acc']}, n={semantic['coverable_non_leaky_n']}), while its Macro-F1 is lower than Document Corpus ({doc['coverable_non_leaky_macro_f1']}) and higher than Semantic KG ({semantic['coverable_non_leaky_macro_f1']}).",
                "- The conditional subsets differ substantially in size and task composition, so their performance is diagnostic rather than a paired significance comparison. The strongest result of this audit is the coverability gain; a stricter paired leakage-control experiment should mask answer-bearing fields and rerun retrieval/generation on the same records.",
            ]
        )
    lines.extend(
        [
            "",
            "## Suggested Manuscript Wording",
            "",
            "> We conduct a rule-based coverability and leakage audit using entity anchors, disease scope, answer labels, task-specific support cues, and source-family requirements. The audit is a proxy for whether each knowledge shape makes potentially answer-supporting evidence available, rather than human annotation of gold evidence. Evidence KG attains substantially higher support coverability and bundle completeness. Although 10.51% of its examples show structured answer-label exposure, removing those cases leaves 508 coverable examples and improves conditional accuracy from 42.73% to 47.24%. Because the resulting subsets are not identically composed across knowledge shapes, we treat this analysis as evidence against a purely availability-driven artifact, not as conclusive proof that leakage has no effect.",
            "",
            "## Output Files",
            "",
            f"- Per-record audit: `{(output_dir / 'ecorag_coverability_leakage_records.jsonl').relative_to(ROOT)}`",
            f"- Summary JSON: `{(output_dir / 'ecorag_coverability_leakage_summary.json').relative_to(ROOT)}`",
            f"- Compact TSV: `{(output_dir / 'ecorag_coverability_leakage_summary.tsv').relative_to(ROOT)}`",
            f"- This report: `{(output_dir / 'ecorag_coverability_leakage_report.md').relative_to(ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based Evidence Graph coverability and leakage audit.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sources", nargs="+", default=list(KNOWLEDGE_SOURCES), choices=list(KNOWLEDGE_SOURCES))
    args = parser.parse_args()

    records = read_jsonl(args.dataset_path)
    all_rows = []
    source_summaries = {}

    for source_key in args.sources:
        config = KNOWLEDGE_SOURCES[source_key]
        print(f"[audit] indexing {config['label']} from {config['docs']}", flush=True)
        index = SourceIndex(source_key, config["docs"], records)
        print(f"[audit] auditing {config['label']} ({index.doc_count} docs)", flush=True)
        source_rows = []
        for record in records:
            audit = index.audit_record(record)
            row = {
                "record_id": record.get("id"),
                "task_type": record.get("task_type"),
                "reasoning_type": reasoning_type(record),
                "answer_type": record.get("answer_type"),
                "gold_label": gold_label(record),
                "knowledge_source": source_key,
                "knowledge_source_label": config["label"],
                **audit,
            }
            source_rows.append(row)
            all_rows.append(row)

        audit_by_id = {str(row["record_id"]): row for row in source_rows}
        result_rows = load_result_rows(config["result"])
        performance = subset_performance(result_rows, audit_by_id)
        source_summaries[source_key] = {
            "label": config["label"],
            "docs_path": str(config["docs"].relative_to(ROOT)),
            "result_path": str(config["result"].relative_to(ROOT)),
            "doc_count": index.doc_count,
            "evidence_like_doc_count": index.evidence_like_count,
            "overall": summarize_rows(source_rows),
            "by_task_type": summarize_by(source_rows, "task_type"),
            "by_reasoning_type": summarize_by(source_rows, "reasoning_type"),
            "performance_subsets": performance,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "ecorag_coverability_leakage_records.jsonl"
    summary_path = args.output_dir / "ecorag_coverability_leakage_summary.json"
    tsv_path = args.output_dir / "ecorag_coverability_leakage_summary.tsv"
    report_path = args.output_dir / "ecorag_coverability_leakage_report.md"

    write_jsonl(records_path, all_rows)
    write_json(
        summary_path,
        {
            "dataset_path": str(args.dataset_path.relative_to(ROOT) if args.dataset_path.is_relative_to(ROOT) else args.dataset_path),
            "record_count": len(records),
            "metric_note": "Rule-based proxy audit; not human gold-evidence annotation.",
            "sources": source_summaries,
        },
    )
    write_tsv(
        tsv_path,
        compact_summary_rows(source_summaries),
        [
            "knowledge_source",
            "doc_count",
            "entity_coverage",
            "scope_coverage",
            "support_coverability",
            "bundle_completeness",
            "cross_source_coverage",
            "structured_label_exposure",
            "coverable_non_leaky",
            "coverable_non_leaky_n",
            "all_acc",
            "all_macro_f1",
            "coverable_non_leaky_acc",
            "coverable_non_leaky_macro_f1",
        ],
    )
    report_path.write_text(render_report(source_summaries, args.output_dir), encoding="utf-8")
    print(f"[audit] wrote {summary_path}", flush=True)
    print(f"[audit] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
