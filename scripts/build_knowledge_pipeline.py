"""Run the paper's source adapters and three knowledge projections in order.

No source data are downloaded and no API is used. With no action flag this prints
a plan. --check verifies external inputs; --run executes only after preflight.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_NAMES = ["literature", "tcga", "depmap", "normalize", "source_relations", "supporting", "integrate", "raw", "complete", "semantic", "project", "controlled_forms", "single_source"]
COHORTS = ["LIHC", "COADREAD", "BRCA", "LUAD", "LUSC", "GBM", "OV", "KIRC", "LAML", "PAAD", "HNSC", "STAD"]


@dataclass
class Stage:
    name: str
    script: str
    inputs: list[Path]
    outputs: list[Path]
    arguments: tuple[str, ...] = ()

    def command(self) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / self.script), *self.arguments]


def make_stages(benchmark: Path) -> list[Stage]:
    import knowledge_build_paths as paths
    aligned = ROOT / "kg" / "aligned"
    raw = ROOT / "kg" / "raw-multisource"
    schema = ROOT / "kg" / "schema" / "multisource_evidence_kg_schema_v2.json"
    literature = [paths.PUBMED_ROOT / cancer / name for cancer in ("hcc", "crc") for name in ("evidence_nodes.json", "evidence_graph.json")]
    # Pin all paper source layers, including optional SCNA/RPPA adapter inputs.
    native_manifest = json.loads((ROOT / "configs" / "native_source_files.json").read_text())
    tcga_inputs = [paths.LINKEDOMICS_ROOT / filename for filename in native_manifest["tcga_relative_files"]]
    depmap_inputs = [paths.DEPMAP_ROOT / filename for filename in (
        "model-condition-mapping/Model.csv", "crispr-screen/CRISPRGeneEffect.csv", "crispr-screen/CRISPRGeneDependency.csv",
        "expression/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", "copy-number/OmicsCNGeneWGS.csv",
        "mutations/OmicsSomaticMutationsMatrixHotspot.csv", "mutations/OmicsSomaticMutationsMatrixDamaging.csv",
        "global-genomic-features/OmicsGlobalSignatures.csv", "global-genomic-features/OmicsInferredMolecularSubtypes.csv",
        "model-condition-mapping/PortalCompounds.csv", "drug-response/GDSC1_fitted_dose_response_24Jul22.csv",
        "drug-response/GDSC2_fitted_dose_response_24Jul22.csv", "drug-response/screened_compounds_rel_8.4.csv",
    )]
    atomic = [aligned / name for name in ("pubmed_aligned_latest.json", "tcga_atomic_compact_latest.json", "depmap_atomic_compact_latest.json")]
    normalized = [aligned / name for name in ("pubmed_aligned_normalized_latest.json", "tcga_atomic_compact_normalized_latest.json", "depmap_atomic_compact_normalized_latest.json")]
    supporting = [aligned / name for name in ("disgenet_aligned_normalized_latest.json", "cellline_annotation_aligned_normalized_latest.json")]
    integrated = aligned / "multisource_evidence_graph_latest.json"
    complete = aligned / "multisource_evidence_graph_complete.json"
    semantic = aligned / "multisource_semantic_kg.json"
    raw_files = [raw / name for name in ("raw_multisource_docs.jsonl", "raw_multisource_edges.jsonl", "raw_multisource_manifest.jsonl")]
    return [
        Stage("literature", "align_pubmed_to_unified_schema.py", literature + [schema], [atomic[0]]),
        Stage("tcga", "extract_tcga_atomic_evidence_v2.py", tcga_inputs + [atomic[0]], [atomic[1]]),
        Stage("depmap", "extract_depmap_dependency_evidence.py", depmap_inputs + [atomic[0]], [atomic[2]]),
        Stage("normalize", "build_entity_normalization_layer.py", atomic + [paths.HGNC_PATH, paths.TARKG_NODES_PATH, depmap_inputs[9]], normalized),
        Stage("source_relations", "build_tcga_depmap_evidence_relations.py", atomic[1:] + normalized[1:], normalized[1:]),
        Stage("supporting", "build_supporting_alignment_graphs.py", [paths.HGNC_PATH, paths.PROCESSED_DIR / "disgenet.jsonl", paths.PROCESSED_DIR / "tdc_cell_line.jsonl"], supporting),
        Stage("integrate", "build_multisource_integrated_evidence_graph.py", normalized + supporting + [schema], [integrated]),
        Stage("raw", "build_raw_multisource_corpus.py", tcga_inputs + depmap_inputs + [normalized[0], paths.PAPERS_ROOT / "hcc_papers" / "*.pdf", paths.PAPERS_ROOT / "crc_papers" / "*.pdf", ROOT / "configs" / "raw_parent_id_map.json"], raw_files),
        Stage("complete", "build_multisource_complete_graph.py", [integrated, raw_files[0], raw_files[2], schema, Path(os.environ.get("EVIDENCEMEETSRAG_LINEAGE_PREFERENCES", str(ROOT / "configs/publication_compatibility/observation_lineage.json.gz"))), Path(os.environ.get("EVIDENCEMEETSRAG_LINEAGE_PREFERENCES", str(ROOT / "configs/publication_compatibility/observation_lineage.json.gz"))).with_name("manifest.json")], [complete]),
        Stage("semantic", "build_multisource_semantic_kg.py", [complete], [semantic]),
        Stage("project", "prepare_rag_knowledge_sources.py", [complete, semantic, *raw_files[:2], benchmark], [ROOT / "dataset" / "rag" / "knowledge_sources" / "manifest_latest.json"],
              ("--views", "multi_cancer", "--sources", "raw_observation_view", "semantic_kg", "evidence_graph_complete", "--benchmark", str(benchmark))),
        Stage("controlled_forms", "build_multisource_controlled_sources.py", [integrated], [ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / f"{name}_{kind}.jsonl" for name in ("controlled_linearized_text", "controlled_triple_kg", "controlled_evidence_graph") for kind in ("docs", "edges")]),
        Stage("single_source", "build_single_source_knowledge_forms.py", [ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / f"{name}_docs.jsonl" for name in ("controlled_linearized_text", "controlled_triple_kg", "controlled_evidence_graph")] + [ROOT / "dataset" / "rag" / "knowledge_sources" / "multi_cancer" / "controlled_evidence_graph_edges.jsonl"], [ROOT / "dataset" / "rag" / "knowledge_sources" / "single_source_forms" / "single_source_forms_report.json"]),
    ]


def present(path: Path) -> bool:
    if any(char in path.name for char in "*?["):
        return path.parent.is_dir() and any(item.is_file() for item in path.parent.glob(path.name))
    return path.is_file()


def preflight(stages: list[Stage]) -> dict[str, object]:
    produced: set[Path] = set()
    required: dict[Path, list[str]] = {}
    for stage in stages:
        for path in stage.inputs:
            if path not in produced:
                required.setdefault(path, []).append(stage.name)
        produced.update(stage.outputs)
    missing = [{"path": str(path), "needed_by": names} for path, names in required.items() if not present(path)]
    report = {"passed": not missing, "external_input_count": len(required), "missing_input_count": len(missing), "missing": missing}
    if any(stage.name == "raw" for stage in stages):
        import knowledge_build_paths as paths
        from paper_fulltext_sources import MANIFEST, verify_manifest
        # The raw adapter reads every PDF in these two folders. Checking the
        # complete set prevents extra/missing articles from changing its input.
        paper_root = Path(os.environ.get("EVIDENCEMEETSRAG_PAPERS_ROOT", str(paths.PAPERS_ROOT)))
        if MANIFEST.is_file():
            report["paper_fulltexts"] = verify_manifest(json.loads(MANIFEST.read_text()), paper_root)
            report["passed"] = report["passed"] and report["paper_fulltexts"]["passed"]
        else:
            report["paper_fulltexts"] = {"passed": False, "error": "Missing configs/paper_fulltext_sources.json"}
            report["passed"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="Print build commands and missing inputs without changing data (default).")
    action.add_argument("--check", action="store_true", help="Only check selected stages' required inputs; exit 2 when any are missing.")
    action.add_argument("--run", action="store_true", help="Execute the selected stages after a successful preflight.")
    parser.add_argument("--stages", nargs="+", choices=STAGE_NAMES, help="Select stages, executed in the canonical order.")
    parser.add_argument("--start-at", choices=STAGE_NAMES, help="Run a suffix of the pipeline from supplied intermediate snapshots.")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl")
    for option in ("raw-root", "linkedomics-root", "depmap-root", "pubmed-root", "papers-root", "processed-root", "hgnc-path", "tarkg-nodes-path"):
        parser.add_argument("--" + option, type=Path, help="Override EVIDENCEMEETSRAG_" + option.replace("-", "_").upper())
    args = parser.parse_args()
    if args.stages and args.start_at:
        parser.error("Use either --stages or --start-at")
    for option in ("raw_root", "linkedomics_root", "depmap_root", "pubmed_root", "papers_root", "processed_root", "hgnc_path", "tarkg_nodes_path"):
        value = getattr(args, option)
        if value is not None:
            os.environ["EVIDENCEMEETSRAG_" + option.upper()] = str(value)
    benchmark = args.benchmark if args.benchmark.is_absolute() else ROOT / args.benchmark
    stages = make_stages(benchmark)
    if args.stages:
        stages = [stage for stage in stages if stage.name in args.stages]
    elif args.start_at:
        stages = stages[STAGE_NAMES.index(args.start_at):]
    report = {"repository": str(ROOT), "stages": [{"name": stage.name, "command": stage.command(), "outputs": [str(path) for path in stage.outputs]} for stage in stages], "preflight": preflight(stages)}
    print(json.dumps(report, indent=2), flush=True)
    if args.check or args.run:
        if not report["preflight"]["passed"]:
            raise SystemExit(2)
    if args.run:
        for stage in stages:
            missing = [str(path) for path in stage.inputs if not present(path)]
            if missing:
                raise SystemExit(f"Stage {stage.name} missing inputs: {missing}")
            print(f"[pipeline] {stage.name}", flush=True)
            subprocess.run(stage.command(), cwd=ROOT, check=True)
            missing_outputs = [str(path) for path in stage.outputs if not present(path)]
            if missing_outputs:
                raise SystemExit(f"Stage {stage.name} did not produce: {missing_outputs}")


if __name__ == "__main__":
    main()
