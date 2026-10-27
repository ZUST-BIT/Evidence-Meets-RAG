from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DIR = ROOT / "kg" / "aligned"
NORMALIZATION_DIR = ROOT / "kg" / "normalization"
RAW_NORMALIZATION_DIR = NORMALIZATION_DIR / "raw"

from knowledge_build_paths import DEPMAP_ROOT, HGNC_PATH, TARKG_NODES_PATH
PORTAL_COMPOUNDS_PATH = DEPMAP_ROOT / "model-condition-mapping" / "PortalCompounds.csv"

INPUT_GRAPHS = {
    "pubmed": ALIGNED_DIR / "pubmed_aligned_latest.json",
    "tcga": ALIGNED_DIR / "tcga_atomic_compact_latest.json",
    "depmap": ALIGNED_DIR / "depmap_atomic_compact_latest.json",
}

OUTPUT_GRAPHS = {
    "pubmed": ALIGNED_DIR / "pubmed_aligned_normalized_latest.json",
    "tcga": ALIGNED_DIR / "tcga_atomic_compact_normalized_latest.json",
    "depmap": ALIGNED_DIR / "depmap_atomic_compact_normalized_latest.json",
}

MAPPING_JSON_PATH = ALIGNED_DIR / "multisource_entity_normalization_latest.json"
MAPPING_TSV_PATH = ALIGNED_DIR / "multisource_entity_normalization_latest.tsv"
LINKS_JSON_PATH = ALIGNED_DIR / "multisource_entity_links_latest.json"
STATS_JSON_PATH = ALIGNED_DIR / "entity_normalization_latest_stats.json"

ENTITY_GROUPS = {"Gene": "Gene", "Protein": "Gene"}

NAMESPACE_PRIORITY = {
    "Gene": {
        "HGNC": 0,
        "UniProtKB": 1,
        "NCBIGene": 2,
        "TarKG": 3,
        "Symbol": 4,
        "custom": 5,
    },
    "Disease": {
        "DOID": 0,
        "MONDO": 1,
        "NCIT": 2,
        "MESH": 3,
        "UMLS": 4,
        "CTD": 5,
        "CANCER_FAMILY": 6,
        "TarKG": 7,
        "custom": 8,
    },
    "Drug": {
        "ChEMBL": 0,
        "DrugBank": 1,
        "PubChemCID": 2,
        "InChIKey": 3,
        "TarKG": 4,
        "DepMapCompound": 5,
        "custom": 6,
    },
    "Pathway": {
        "Reactome": 0,
        "PathwayCommons": 1,
        "KEGG": 2,
        "GO": 3,
        "TarKG": 4,
        "custom": 5,
    },
    "Phenotype": {
        "HPO": 0,
        "MESH": 1,
        "custom": 2,
    },
}

DISEASE_FAMILY_MAP = {
    "AML": {
        "normalized_id": "DOID:9119",
        "normalized_label": "acute myeloid leukemia",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "BRCA": {
        "normalized_id": "DOID:3459",
        "normalized_label": "breast carcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "CRC": {
        "normalized_id": "DOID:9256",
        "normalized_label": "colorectal cancer",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "GLIOMA": {
        "normalized_id": "MONDO:0021042",
        "normalized_label": "glioma",
        "normalized_db_source": "MONDO",
        "method": "manual_family_to_mondo",
    },
    "GASTROESOPH": {
        "normalized_id": "DOID:0080375",
        "normalized_label": "gastroesophageal adenocarcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "HCC": {
        "normalized_id": "DOID:684",
        "normalized_label": "hepatocellular carcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "HNSC": {
        "normalized_id": "DOID:5520",
        "normalized_label": "head and neck squamous cell carcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "MELANOMA": {
        "normalized_id": "DOID:1909",
        "normalized_label": "melanoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "MESO": {
        "normalized_id": "DOID:1790",
        "normalized_label": "malignant mesothelioma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "NSCLC": {
        "normalized_id": "CANCER_FAMILY:NSCLC",
        "normalized_label": "non-small cell lung cancer",
        "normalized_db_source": "CANCER_FAMILY",
        "method": "stable_family_namespace",
    },
    "OVARIAN": {
        "normalized_id": "DOID:4001",
        "normalized_label": "ovarian carcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "PAAD": {
        "normalized_id": "DOID:4074",
        "normalized_label": "pancreatic adenocarcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "PRAD": {
        "normalized_id": "DOID:2526",
        "normalized_label": "prostate adenocarcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
    "RCC": {
        "normalized_id": "DOID:4450",
        "normalized_label": "renal cell carcinoma",
        "normalized_db_source": "DOID",
        "method": "manual_family_to_doid",
    },
}


@dataclass
class Candidate:
    namespace: str
    value: str
    label: str | None = None
    match_key: str | None = None


@dataclass
class NormalizationRecord:
    source_graph: str
    entity_id: str
    entity_name: str
    entity_type: str
    entity_group: str
    original_db_source: str | None
    candidate_ids: list[Candidate] = field(default_factory=list)
    match_keys: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)
    original_aliases: list[str] = field(default_factory=list)


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length].upper()


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def namespace_key(namespace: str, value: str) -> str:
    return f"{namespace}:{value}"


def group_entity_type(entity_type: str) -> str:
    return ENTITY_GROUPS.get(entity_type, entity_type)


def make_candidate(namespace: str, value: str | None, label: str | None = None) -> Candidate | None:
    clean_value = (value or "").strip()
    if not clean_value:
        return None
    return Candidate(namespace=namespace, value=clean_value, label=label, match_key=namespace_key(namespace, clean_value))


def candidate_id(candidate: Candidate) -> str:
    if candidate.namespace == "custom":
        return candidate.value
    return f"{candidate.namespace}:{candidate.value}"


def candidate_priority(entity_group: str, candidate: Candidate) -> tuple[int, str]:
    priorities = NAMESPACE_PRIORITY.get(entity_group, {})
    return (priorities.get(candidate.namespace, 99), candidate_id(candidate))


def load_tarkg_nodes() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    nodes: dict[str, dict[str, str]] = {}
    compound_by_name: dict[str, dict[str, str]] = {}
    with TARKG_NODES_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            nodes[row["unify_id"]] = row
            if row.get("kind") == "Compound":
                normalized_name = normalize_text(row.get("name"))
                if normalized_name:
                    compound_by_name.setdefault(normalized_name, row)
    return nodes, compound_by_name


def load_hgnc() -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    symbol_lookup: dict[str, dict[str, str]] = {}
    entrez_lookup: dict[str, dict[str, str]] = {}
    uniprot_lookup: dict[str, dict[str, str]] = {}
    name_lookup: dict[str, dict[str, str]] = {}
    with HGNC_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "Approved":
                continue
            symbol = row.get("symbol", "").strip()
            if symbol:
                keys = [symbol]
                keys.extend(x.strip() for x in row.get("alias_symbol", "").split("|") if x.strip())
                keys.extend(x.strip() for x in row.get("prev_symbol", "").split("|") if x.strip())
                for key in keys:
                    symbol_lookup.setdefault(key, row)
            entrez_id = row.get("entrez_id", "").strip()
            if entrez_id:
                entrez_lookup.setdefault(entrez_id, row)
            for uniprot_id in (x.strip() for x in row.get("uniprot_ids", "").split(",") if x.strip()):
                uniprot_lookup.setdefault(uniprot_id, row)
            name_keys = [row.get("name", "")]
            name_keys.extend(x.strip() for x in row.get("alias_name", "").split("|") if x.strip())
            name_keys.extend(x.strip() for x in row.get("prev_name", "").split("|") if x.strip())
            for key in name_keys:
                normalized_name = normalize_text(key)
                if normalized_name:
                    name_lookup.setdefault(normalized_name, row)
    return symbol_lookup, entrez_lookup, uniprot_lookup, name_lookup


def load_portal_compounds() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_dpc: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}
    with PORTAL_COMPOUNDS_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            compound_id = row["CompoundID"].strip()
            if compound_id:
                by_dpc[compound_id] = row
            name = normalize_text(row.get("CompoundName"))
            if name:
                by_name.setdefault(name, row)
            for synonym in (x.strip() for x in row.get("Synonyms", "").split(";") if x.strip()):
                by_name.setdefault(normalize_text(synonym), row)
    return by_dpc, by_name


def parse_gene_symbol(name: str) -> str:
    return (name or "").split("|", 1)[0].strip()


def hgnc_candidates(row: dict[str, str], *, label_override: str | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    label = label_override or row.get("symbol") or row.get("name")
    for candidate in [
        make_candidate("HGNC", row.get("hgnc_id", "").replace("HGNC:", ""), label),
        make_candidate("UniProtKB", next((x.strip() for x in row.get("uniprot_ids", "").split(",") if x.strip()), ""), label),
        make_candidate("NCBIGene", row.get("entrez_id"), label),
        make_candidate("Symbol", row.get("symbol"), label),
    ]:
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def tarkg_external_candidates(row: dict[str, str], *, label_override: str | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    label = label_override or row.get("name")
    db_source = (row.get("db_source") or "").strip()
    dbid = (row.get("dbid") or row.get("unify_id") or "").strip()
    if db_source == "UniProt":
        candidate = make_candidate("UniProtKB", dbid, label)
    elif db_source == "NCBI" and row.get("kind") == "Gene":
        candidate = make_candidate("NCBIGene", dbid, label)
    elif db_source == "DO":
        candidate = make_candidate("DOID", dbid.replace("DOID:", ""), label)
    elif db_source == "MONDO":
        candidate = make_candidate("MONDO", dbid.replace("MONDO:", ""), label)
    elif db_source == "HPO":
        candidate = make_candidate("HPO", dbid.replace("HP:", ""), label)
    elif db_source == "MESH":
        candidate = make_candidate("MESH", dbid.replace("MESH:", ""), label)
    elif db_source == "UMLS":
        value = dbid.replace("UMLS:", "").replace("CTD:", "")
        candidate = make_candidate("UMLS", value, label)
    elif db_source == "DrugBank":
        candidate = make_candidate("DrugBank", dbid, label)
    elif db_source == "CHEMBL":
        candidate = make_candidate("ChEMBL", dbid, label)
    elif db_source == "PUBCHEM":
        candidate = make_candidate("PubChemCID", dbid, label)
    elif db_source == "Pathway Commons":
        candidate = make_candidate("PathwayCommons", dbid, label)
    elif db_source == "REACTOME":
        candidate = make_candidate("Reactome", dbid, label)
    else:
        candidate = make_candidate("TarKG", row.get("unify_id"), label)
    if candidate is not None:
        candidates.append(candidate)
    return candidates


def extract_pubmed_record(
    entity: dict[str, Any],
    *,
    source_graph: str,
    tarkg_lookup: dict[str, dict[str, str]],
    hgnc_by_entrez: dict[str, dict[str, str]],
    hgnc_by_uniprot: dict[str, dict[str, str]],
) -> NormalizationRecord:
    entity_group = group_entity_type(entity["entity_type"])
    record = NormalizationRecord(
        source_graph=source_graph,
        entity_id=entity["id"],
        entity_name=entity["name"],
        entity_type=entity["entity_type"],
        entity_group=entity_group,
        original_db_source=entity.get("db_source"),
        original_aliases=entity.get("aliases", []),
    )
    record.methods.add("pubmed_existing_entity")
    tarkg_row = tarkg_lookup.get(entity["id"])
    if tarkg_row:
        record.methods.add("tarkg_lookup")
        if entity_group == "Gene":
            hgnc_row = None
            if tarkg_row.get("db_source") == "UniProt":
                hgnc_row = hgnc_by_uniprot.get(tarkg_row.get("dbid", ""))
            elif tarkg_row.get("db_source") == "NCBI":
                hgnc_row = hgnc_by_entrez.get(tarkg_row.get("dbid", ""))
            if hgnc_row:
                record.methods.add("tarkg_gene_to_hgnc")
                record.candidate_ids.extend(hgnc_candidates(hgnc_row))
            else:
                record.candidate_ids.extend(tarkg_external_candidates(tarkg_row, label_override=entity["name"]))
        else:
            record.candidate_ids.extend(tarkg_external_candidates(tarkg_row, label_override=entity["name"]))
    else:
        record.methods.add("preserved_existing_id")
        record.candidate_ids.append(Candidate(namespace="TarKG", value=entity["id"], label=entity["name"]))
    record.match_keys.add(f"NAME:{normalize_text(entity['name'])}")
    for alias in entity.get("aliases", []):
        alias_key = normalize_text(alias)
        if alias_key:
            record.match_keys.add(f"NAME:{alias_key}")
    for candidate in record.candidate_ids:
        if candidate.match_key:
            record.match_keys.add(candidate.match_key)
    return record


def extract_tcga_depmap_record(
    entity: dict[str, Any],
    *,
    source_graph: str,
    hgnc_by_symbol: dict[str, dict[str, str]],
    portal_compounds_by_dpc: dict[str, dict[str, str]],
    portal_compounds_by_name: dict[str, dict[str, str]],
    tarkg_lookup: dict[str, dict[str, str]],
    tarkg_compound_by_name: dict[str, dict[str, str]],
) -> NormalizationRecord:
    entity_group = group_entity_type(entity["entity_type"])
    record = NormalizationRecord(
        source_graph=source_graph,
        entity_id=entity["id"],
        entity_name=entity["name"],
        entity_type=entity["entity_type"],
        entity_group=entity_group,
        original_db_source=entity.get("db_source"),
        original_aliases=entity.get("aliases", []),
    )
    record.match_keys.add(f"NAME:{normalize_text(entity['name'])}")
    for alias in entity.get("aliases", []):
        alias_key = normalize_text(alias)
        if alias_key:
            record.match_keys.add(f"NAME:{alias_key}")

    if entity_group == "Gene":
        symbol = parse_gene_symbol(entity["name"])
        hgnc_row = hgnc_by_symbol.get(symbol)
        if hgnc_row:
            record.methods.add("hgnc_symbol_lookup")
            record.candidate_ids.extend(hgnc_candidates(hgnc_row, label_override=hgnc_row.get("symbol") or symbol))
        else:
            record.methods.add("preserved_gene_symbol")
            record.candidate_ids.append(Candidate(namespace="custom", value=f"GENE_SYMBOL:{symbol}", label=symbol))
    elif entity_group == "Disease":
        family_key = entity["name"].strip().upper()
        mapping = DISEASE_FAMILY_MAP.get(family_key)
        if mapping:
            record.methods.add(mapping["method"])
            namespace, value = mapping["normalized_id"].split(":", 1)
            record.candidate_ids.append(Candidate(namespace=namespace, value=value, label=mapping["normalized_label"]))
        else:
            record.methods.add("stable_family_namespace")
            record.candidate_ids.append(
                Candidate(
                    namespace="CANCER_FAMILY",
                    value=family_key,
                    label=entity["name"],
                )
            )
    elif entity_group == "Drug":
        compound_row = None
        dpc_id = entity["id"].split(":", 1)[1] if entity["id"].startswith("DRUG:") else ""
        if dpc_id:
            compound_row = portal_compounds_by_dpc.get(dpc_id)
        if compound_row is None:
            compound_row = portal_compounds_by_name.get(normalize_text(entity["name"]))
        if compound_row:
            chembl_id = (compound_row.get("ChEMBLID") or "").strip()
            pubchem_cid = (compound_row.get("PubChemCID") or "").strip()
            inchi_key = (compound_row.get("InChIKey") or "").strip()
            if chembl_id:
                record.methods.add("portal_compounds_chembl")
                record.candidate_ids.append(Candidate(namespace="ChEMBL", value=chembl_id, label=compound_row.get("CompoundName")))
            if pubchem_cid:
                record.methods.add("portal_compounds_pubchem")
                record.candidate_ids.append(Candidate(namespace="PubChemCID", value=pubchem_cid, label=compound_row.get("CompoundName")))
            if inchi_key:
                record.methods.add("portal_compounds_inchikey")
                record.candidate_ids.append(Candidate(namespace="InChIKey", value=inchi_key, label=compound_row.get("CompoundName")))
            for synonym in (x.strip() for x in compound_row.get("Synonyms", "").split(";") if x.strip()):
                record.match_keys.add(f"NAME:{normalize_text(synonym)}")
            record.match_keys.add(f"NAME:{normalize_text(compound_row.get('CompoundName'))}")
        else:
            record.methods.add("preserved_depmap_drug")

        tarkg_row = tarkg_lookup.get(entity["id"])
        if tarkg_row:
            record.methods.add("tarkg_lookup")
            record.candidate_ids.extend(tarkg_external_candidates(tarkg_row, label_override=entity["name"]))
        else:
            name_keys = [normalize_text(entity["name"])]
            name_keys.extend(normalize_text(alias) for alias in entity.get("aliases", []))
            matched_tarkg = next((tarkg_compound_by_name[key] for key in name_keys if key and key in tarkg_compound_by_name), None)
            if matched_tarkg:
                record.methods.add("tarkg_compound_name_match")
                record.candidate_ids.extend(tarkg_external_candidates(matched_tarkg, label_override=matched_tarkg.get("name") or entity["name"]))
        if not record.candidate_ids:
            if dpc_id:
                record.candidate_ids.append(Candidate(namespace="DepMapCompound", value=dpc_id, label=entity["name"]))
            else:
                record.candidate_ids.append(Candidate(namespace="custom", value=f"DRUG_NAME:{normalize_text(entity['name'])}", label=entity["name"]))
    elif entity_group == "Phenotype":
        record.methods.add("stable_context_namespace")
        record.candidate_ids.append(
            Candidate(
                namespace="custom",
                value=f"COGENTCTX:{stable_hash(entity['name'])}",
                label=entity["name"],
            )
        )
    else:
        record.methods.add("preserved_existing_id")
        record.candidate_ids.append(Candidate(namespace="custom", value=entity["id"], label=entity["name"]))

    for candidate in record.candidate_ids:
        if candidate.match_key:
            record.match_keys.add(candidate.match_key)
    return record


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def pick_group_candidate(entity_group: str, candidates: list[Candidate], fallback_label: str, fallback_value: str) -> Candidate:
    if candidates:
        unique_candidates = {(candidate.namespace, candidate.value, candidate.label or fallback_label): candidate for candidate in candidates}
        selected = sorted(unique_candidates.values(), key=lambda candidate: candidate_priority(entity_group, candidate))[0]
        return selected
    return Candidate(namespace="custom", value=fallback_value, label=fallback_label)


def build_normalization_groups(records: list[NormalizationRecord]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    grouped_records: dict[str, list[tuple[int, NormalizationRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_records[record.entity_group].append((index, record))

    record_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    link_groups: list[dict[str, Any]] = []

    for entity_group, items in grouped_records.items():
        union_find = UnionFind([index for index, _ in items])
        key_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, record in items:
            for key in record.match_keys:
                key_to_indices[key].append(index)
        for indices in key_to_indices.values():
            first = indices[0]
            for other in indices[1:]:
                union_find.union(first, other)

        clusters: dict[int, list[NormalizationRecord]] = defaultdict(list)
        for index, record in items:
            clusters[union_find.find(index)].append(record)

        for members in clusters.values():
            candidates: list[Candidate] = []
            methods: set[str] = set()
            source_graphs: set[str] = set()
            names: Counter[str] = Counter()
            member_rows: list[dict[str, Any]] = []
            for member in members:
                candidates.extend(member.candidate_ids)
                methods.update(member.methods)
                source_graphs.add(member.source_graph)
                names.update([member.entity_name])
                member_rows.append(
                    {
                        "source_graph": member.source_graph,
                        "entity_id": member.entity_id,
                        "entity_type": member.entity_type,
                        "entity_name": member.entity_name,
                    }
                )
            fallback_name = names.most_common(1)[0][0]
            fallback_value = f"{entity_group.upper()}:{stable_hash('|'.join(sorted(row['entity_id'] for row in member_rows)))}"
            selected = pick_group_candidate(entity_group, candidates, fallback_name, fallback_value)
            normalized_id = candidate_id(selected)
            normalized_db_source = selected.namespace
            normalized_label = selected.label or fallback_name
            standardized = normalized_db_source not in {"custom", "TarKG", "DepMapCompound", "CANCER_FAMILY"}
            if normalized_db_source == "CANCER_FAMILY":
                normalization_status = "custom_family"
            elif normalized_db_source in {"custom", "TarKG", "DepMapCompound"}:
                normalization_status = "custom_or_preserved"
            else:
                normalization_status = "standardized"

            link_payload = {
                "normalized_id": normalized_id,
                "normalized_label": normalized_label,
                "normalized_entity_type": entity_group,
                "normalized_db_source": normalized_db_source,
                "normalization_status": normalization_status,
                "methods": sorted(methods),
                "source_graphs": sorted(source_graphs),
                "members": sorted(member_rows, key=lambda row: (row["source_graph"], row["entity_id"])),
                "member_count": len(member_rows),
                "cross_source": len(source_graphs) > 1,
                "supporting_candidate_ids": sorted({candidate_id(candidate) for candidate in candidates}),
            }
            link_groups.append(link_payload)

            for member in members:
                record_map[(member.source_graph, member.entity_id, member.entity_type)] = {
                    "normalized_id": normalized_id,
                    "normalized_label": normalized_label,
                    "normalized_entity_type": entity_group,
                    "normalized_db_source": normalized_db_source,
                    "normalization_status": normalization_status,
                    "normalization_method": sorted(member.methods),
                    "source_graph": member.source_graph,
                    "original_entity_type": member.entity_type,
                    "original_db_source": member.original_db_source,
                    "match_keys": sorted(member.match_keys),
                    "supporting_candidate_ids": sorted({candidate_id(candidate) for candidate in member.candidate_ids}),
                    "cross_source_group": len(source_graphs) > 1,
                    "cross_source_group_size": len(source_graphs),
                    "group_member_count": len(member_rows),
                    "standardized": standardized,
                }
    return record_map, sorted(link_groups, key=lambda row: (-row["cross_source"], row["normalized_entity_type"], row["normalized_id"]))


def attach_normalization_to_graph(
    graph: dict[str, Any],
    *,
    source_graph: str,
    normalization_map: dict[tuple[str, str, str], dict[str, Any]],
    hgnc_by_symbol: dict[str, dict[str, str]],
    hgnc_by_name: dict[str, dict[str, str]],
    portal_compounds_by_name: dict[str, dict[str, str]],
    tarkg_compound_by_name: dict[str, dict[str, str]],
) -> dict[str, Any]:
    augmented = deepcopy(graph)
    entity_lookup: dict[str, dict[str, Any]] = {}
    entity_name_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for node in augmented["nodes"]:
        if node.get("node_type") != "entity":
            continue
        info = normalization_map.get((source_graph, node["id"], node["entity_type"]))
        if not info:
            continue
        node.update(
            {
                "normalized_id": info["normalized_id"],
                "normalized_label": info["normalized_label"],
                "normalized_entity_type": info["normalized_entity_type"],
                "normalized_db_source": info["normalized_db_source"],
                "normalization_status": info["normalization_status"],
                "normalization_method": info["normalization_method"],
            }
        )
        entity_lookup[node["id"]] = info
        name_keys = [node.get("name", "")]
        name_keys.extend(node.get("aliases", []))
        name_keys.extend(x for x in node.get("normalized_from", []) if isinstance(x, str))
        for key in name_keys:
            normalized_name = normalize_text(key)
            if normalized_name:
                entity_name_lookup.setdefault((group_entity_type(node["entity_type"]), normalized_name), info)

    def resolve_ref_info(value: dict[str, Any]) -> dict[str, Any] | None:
        entity_id = value.get("id")
        if entity_id in entity_lookup:
            return entity_lookup[entity_id]
        entity_group = group_entity_type(value.get("entity_type", ""))
        label = value.get("label") or value.get("name") or entity_id or ""
        normalized_name = normalize_text(label)
        if normalized_name:
            match = entity_name_lookup.get((entity_group, normalized_name))
            if match:
                return match
        if entity_group == "Gene":
            hgnc_row = hgnc_by_symbol.get(parse_gene_symbol(label)) or hgnc_by_name.get(normalized_name)
            if hgnc_row:
                selected = pick_group_candidate(
                    "Gene",
                    hgnc_candidates(hgnc_row),
                    hgnc_row.get("symbol") or label,
                    f"GENE:{stable_hash(label)}",
                )
                return {
                    "normalized_id": candidate_id(selected),
                    "normalized_label": selected.label or label,
                    "normalized_entity_type": "Gene",
                    "normalized_db_source": selected.namespace,
                    "normalization_status": "standardized" if selected.namespace != "custom" else "custom_or_preserved",
                }
        if entity_group == "Drug":
            compound_row = portal_compounds_by_name.get(normalized_name)
            drug_candidates: list[Candidate] = []
            if compound_row:
                if compound_row.get("ChEMBLID"):
                    drug_candidates.append(Candidate(namespace="ChEMBL", value=compound_row["ChEMBLID"].strip(), label=compound_row.get("CompoundName") or label))
                if compound_row.get("PubChemCID"):
                    drug_candidates.append(Candidate(namespace="PubChemCID", value=compound_row["PubChemCID"].strip(), label=compound_row.get("CompoundName") or label))
                if compound_row.get("InChIKey"):
                    drug_candidates.append(Candidate(namespace="InChIKey", value=compound_row["InChIKey"].strip(), label=compound_row.get("CompoundName") or label))
            tarkg_row = tarkg_compound_by_name.get(normalized_name)
            if tarkg_row:
                drug_candidates.extend(tarkg_external_candidates(tarkg_row, label_override=tarkg_row.get("name") or label))
            if drug_candidates:
                selected = pick_group_candidate("Drug", drug_candidates, label, f"DRUG:{stable_hash(label)}")
                return {
                    "normalized_id": candidate_id(selected),
                    "normalized_label": selected.label or label,
                    "normalized_entity_type": "Drug",
                    "normalized_db_source": selected.namespace,
                    "normalization_status": "standardized" if selected.namespace not in {"custom", "TarKG", "DepMapCompound"} else "custom_or_preserved",
                }
        if entity_group == "Disease":
            family_key = (entity_id or value.get("label") or "").strip().upper()
            family = DISEASE_FAMILY_MAP.get(family_key)
            if family:
                normalized_id = family["normalized_id"]
                normalized_db_source, _ = normalized_id.split(":", 1)
                return {
                    "normalized_id": normalized_id,
                    "normalized_label": family["normalized_label"],
                    "normalized_entity_type": "Disease",
                    "normalized_db_source": normalized_db_source,
                    "normalization_status": "standardized" if normalized_db_source != "CANCER_FAMILY" else "custom_family",
                }
            for mapping in DISEASE_FAMILY_MAP.values():
                normalized_family_name = normalize_text(mapping["normalized_label"])
                if normalized_family_name and normalized_family_name in normalized_name:
                    normalized_id = mapping["normalized_id"]
                    normalized_db_source, _ = normalized_id.split(":", 1)
                    return {
                        "normalized_id": normalized_id,
                        "normalized_label": mapping["normalized_label"],
                        "normalized_entity_type": "Disease",
                        "normalized_db_source": normalized_db_source,
                        "normalization_status": "standardized" if normalized_db_source != "CANCER_FAMILY" else "custom_family",
                    }
        return None

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            if {"id", "label", "entity_type"}.issubset(value.keys()):
                info = resolve_ref_info(value)
                if info:
                    value = dict(value)
                    value.update(
                        {
                            "normalized_id": info["normalized_id"],
                            "normalized_label": info["normalized_label"],
                            "normalized_entity_type": info["normalized_entity_type"],
                            "normalized_db_source": info["normalized_db_source"],
                            "normalization_status": info["normalization_status"],
                        }
                    )
                    return value
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    augmented["nodes"] = [visit(node) for node in augmented["nodes"]]
    augmented["edges"] = [visit(edge) for edge in augmented.get("edges", [])]
    augmented.setdefault("metadata", {})
    augmented["metadata"]["entity_normalization"] = {
        "generated_at": now_iso(),
        "mapping_table": str(MAPPING_JSON_PATH.relative_to(ROOT)),
        "link_index": str(LINKS_JSON_PATH.relative_to(ROOT)),
        "canonical_rule": {
            "gene_protein": "HGNC preferred, with UniProtKB / NCBIGene support keys",
            "disease": "DOID or MONDO when available; stable family namespace for unmatched family-level diseases",
            "drug": "ChEMBL preferred, then DrugBank, PubChemCID, InChIKey, then stable fallback",
        },
    }
    return augmented


def build_stats(records: list[dict[str, Any]], links: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"generated_at": now_iso(), "sources": {}, "cross_source_links": {}}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["source_graph"]].append(row)
    for source_graph, rows in grouped.items():
        source_stats = {
            "entity_count": len(rows),
            "standardized_count": sum(1 for row in rows if row["standardized"]),
            "cross_source_group_count": sum(1 for row in rows if row["cross_source_group"]),
            "by_entity_type": {},
        }
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_type[row["original_entity_type"]].append(row)
        for entity_type, entity_rows in by_type.items():
            source_stats["by_entity_type"][entity_type] = {
                "entity_count": len(entity_rows),
                "standardized_count": sum(1 for row in entity_rows if row["standardized"]),
                "cross_source_group_count": sum(1 for row in entity_rows if row["cross_source_group"]),
                "by_db_source": dict(Counter(row["normalized_db_source"] for row in entity_rows)),
            }
        stats["sources"][source_graph] = source_stats

    cross_source_links = [row for row in links if row["cross_source"]]
    source_combo_counter = Counter(tuple(row["source_graphs"]) for row in cross_source_links)
    stats["cross_source_links"] = {
        "group_count": len(cross_source_links),
        "source_combinations": {"|".join(combo): count for combo, count in sorted(source_combo_counter.items())},
        "by_entity_type": dict(Counter(row["normalized_entity_type"] for row in cross_source_links)),
    }
    return stats


def main() -> None:
    tarkg_lookup, tarkg_compound_by_name = load_tarkg_nodes()
    hgnc_by_symbol, hgnc_by_entrez, hgnc_by_uniprot, hgnc_by_name = load_hgnc()
    portal_compounds_by_dpc, portal_compounds_by_name = load_portal_compounds()

    graphs = {name: read_json(path) for name, path in INPUT_GRAPHS.items()}
    records: list[NormalizationRecord] = []

    for source_graph, graph in graphs.items():
        for entity in (node for node in graph["nodes"] if node.get("node_type") == "entity"):
            if source_graph == "pubmed":
                record = extract_pubmed_record(
                    entity,
                    source_graph=source_graph,
                    tarkg_lookup=tarkg_lookup,
                    hgnc_by_entrez=hgnc_by_entrez,
                    hgnc_by_uniprot=hgnc_by_uniprot,
                )
            else:
                record = extract_tcga_depmap_record(
                    entity,
                    source_graph=source_graph,
                    hgnc_by_symbol=hgnc_by_symbol,
                    portal_compounds_by_dpc=portal_compounds_by_dpc,
                    portal_compounds_by_name=portal_compounds_by_name,
                    tarkg_lookup=tarkg_lookup,
                    tarkg_compound_by_name=tarkg_compound_by_name,
                )
            records.append(record)

    normalization_map, link_groups = build_normalization_groups(records)
    mapping_rows: list[dict[str, Any]] = []
    for record in records:
        info = normalization_map[(record.source_graph, record.entity_id, record.entity_type)]
        mapping_rows.append(
            {
                "source_graph": record.source_graph,
                "entity_id": record.entity_id,
                "entity_name": record.entity_name,
                "original_entity_type": record.entity_type,
                "normalized_id": info["normalized_id"],
                "normalized_label": info["normalized_label"],
                "normalized_entity_type": info["normalized_entity_type"],
                "normalized_db_source": info["normalized_db_source"],
                "normalization_status": info["normalization_status"],
                "normalization_method": info["normalization_method"],
                "supporting_candidate_ids": info["supporting_candidate_ids"],
                "cross_source_group": info["cross_source_group"],
                "cross_source_group_size": info["cross_source_group_size"],
                "group_member_count": info["group_member_count"],
                "standardized": info["standardized"],
            }
        )

    for source_graph, graph in graphs.items():
        write_json(
            OUTPUT_GRAPHS[source_graph],
            attach_normalization_to_graph(
                graph,
                source_graph=source_graph,
                normalization_map=normalization_map,
                hgnc_by_symbol=hgnc_by_symbol,
                hgnc_by_name=hgnc_by_name,
                portal_compounds_by_name=portal_compounds_by_name,
                tarkg_compound_by_name=tarkg_compound_by_name,
            ),
        )

    write_json(MAPPING_JSON_PATH, {"generated_at": now_iso(), "records": mapping_rows})
    write_json(LINKS_JSON_PATH, {"generated_at": now_iso(), "groups": link_groups})
    write_json(STATS_JSON_PATH, build_stats(mapping_rows, link_groups))

    with MAPPING_TSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_graph",
            "entity_id",
            "entity_name",
            "original_entity_type",
            "normalized_id",
            "normalized_label",
            "normalized_entity_type",
            "normalized_db_source",
            "normalization_status",
            "normalization_method",
            "supporting_candidate_ids",
            "cross_source_group",
            "cross_source_group_size",
            "group_member_count",
            "standardized",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in mapping_rows:
            output_row = dict(row)
            output_row["normalization_method"] = "|".join(output_row["normalization_method"])
            output_row["supporting_candidate_ids"] = "|".join(output_row["supporting_candidate_ids"])
            writer.writerow(output_row)


if __name__ == "__main__":
    main()
