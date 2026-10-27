"""Offline integrity and behavior checks for the public reproduction package."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_artifacts import extract_verified, safe_destination
from rag_eval_lib import PreparedCorpus, PreparedDoc, RETRIEVERS, compute_answer_metrics, set_rag_profile, set_stage_config
from run_significance_analysis import exact_mcnemar_pvalue
from summarize_faithfulness_results import summarize_rows


class BenchmarkIntegrity(unittest.TestCase):
    def test_published_benchmark_is_complete_and_unchanged(self):
        data = (ROOT / "dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), "191ce330e81366cf893c327d3a7d69b30e74c769bd77e9f443f4155b73ccc405")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        self.assertEqual(len(rows), 571)
        self.assertEqual(len({row["id"] for row in rows}), 571)
        self.assertEqual(Counter(row["task_type"] for row in rows), {
            "cross_source_corroboration_status": 100,
            "cross_source_support_bundle": 81,
            "evidence_signal_interpretation": 85,
            "hierarchical_evidence_aggregation": 86,
            "mechanism_bioasq_factoid": 7,
            "mechanism_bioasq_list": 7,
            "mechanism_bioasq_yesno": 5,
            "mechanism_trace_intermediate_gene": 100,
            "precision_oncology_evidence": 100,
        })


class Metrics(unittest.TestCase):
    def test_accuracy_macro_f1_and_group_alignment(self):
        rows = []
        for i, (gold, predicted) in enumerate(zip(["a", "a", "b", "b"], ["a", "b", "b", "b"])):
            record = {"id": str(i), "task_type": "synthetic", "source": "synthetic", "oncology_focus": "hcc",
                      "metadata": {"priority_cancer_family": "hcc"},
                      "gold_standard": {"answer_format": "categorical", "class_label": gold}}
            rows.append({"record": record, "normalized_prediction": predicted})
        metrics = compute_answer_metrics(rows)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["macro_f1"], 0.7333)
        self.assertEqual(metrics["by_task_type"]["synthetic"]["count"], 4)

    def test_faithful_correctness_uses_correctness_and_support(self):
        rows = [
            {"correct": 1, "retrieval_sufficient": 0, "answer_supported": 1, "answer_contradicted": 0},
            {"correct": 1, "retrieval_sufficient": 0, "answer_supported": 0, "answer_contradicted": 0},
            {"correct": 0, "retrieval_sufficient": 0, "answer_supported": 1, "answer_contradicted": 0},
        ]
        result = summarize_rows(rows)
        self.assertEqual(result["fcr"], 0.3333)
        self.assertEqual(result["ucr"], 0.3333)
        self.assertEqual(result["accuracy"], 0.6667)
        self.assertEqual(result["es"], 0.0)

    def test_exact_paired_test(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_pvalue(6, 0), 0.03125)
        self.assertEqual(exact_mcnemar_pvalue(0, 6), 0.03125)


class Retrieval(unittest.TestCase):
    def test_frameworks_run_without_provider_calls(self):
        def doc(identifier, title, gene, disease):
            return PreparedDoc(identifier, "evidence_graph_complete", "evidence", None,
                               title, f"{gene} evidence in {disease}: expression change measured in a patient cohort.",
                               [disease], [gene], [], {"node_type": "evidence", "source_family": "tcga_linkedomics"})
        corpus = PreparedCorpus([doc("A", "TP53 evidence", "TP53", "HCC"),
                                 doc("B", "KRAS evidence", "KRAS", "CRC"),
                                 doc("C", "TP53 cohort support", "TP53", "HCC")],
                                [{"source_doc_id": "A", "target_doc_id": "C", "relation": "SUPPORTS", "weight": 1.0}])
        record = {"id": "synthetic", "question": "What evidence supports TP53 in HCC?", "task_type": "synthetic",
                  "entities": {"gene_symbol": "TP53", "disease": "HCC"}, "oncology_focus": "HCC", "metadata": {}}
        set_rag_profile("tuned")
        set_stage_config("full", "structured", "grounded")
        for name, retrieve in RETRIEVERS.items():
            if name == "dense_rag":
                continue  # Dense smoke requires the separately installed frozen encoder.
            with self.subTest(framework=name):
                results = retrieve(corpus, record, 2)
                self.assertTrue(results)
                self.assertTrue(any("TP53" in item.title or "TP53" in item.text for item, _ in results))


class DownloadIntegrity(unittest.TestCase):
    def test_archive_paths_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["../secret", "/tmp/secret", "nested/../../secret", "..\\secret"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_destination(root, name)

    def test_verified_extraction_and_corrupt_payload(self):
        data = b"publication input\n"
        entry = {"path": "dataset/example.jsonl", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "inputs.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo(entry["path"])
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
            extract_verified(archive, {"files": [entry]}, root / "valid")
            self.assertEqual((root / "valid" / entry["path"]).read_bytes(), data)
            bad = {**entry, "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                extract_verified(archive, {"files": [bad]}, root / "invalid")
            self.assertFalse((root / "invalid" / entry["path"]).exists())


if __name__ == "__main__":
    unittest.main()
