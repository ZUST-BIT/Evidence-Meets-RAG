# Evidence Meets RAG

Reproduction code for **Evidence Meets RAG: How Multi-Form Knowledge Representations Shape Evidence Access in Cancer Question Answering**.

The repository contains the complete **571-question benchmark**, source-to-graph construction code, three knowledge representations, seven RAG framework adapters, evaluation, ablations, and the analyses used in the paper. Large source files and generated artifacts are managed separately from Git.

## Setup

Python 3.11 or 3.12 is recommended. Run commands from the repository root.

```bash
git clone https://github.com/ZUST-BIT/Evidence-Meets-RAG.git
cd Evidence-Meets-RAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/benchmark/validate_benchmark.py
python scripts/reproduce_paper.py smoke
python scripts/smoke_knowledge_pipeline.py
```

To run all seven adapters, also install `python -m pip install -r requirements-dense.txt`. Dense RAG uses `sentence-transformers/all-MiniLM-L6-v2`; generation from saved contexts does not require the encoder. The unpinned compatibility ranges are in `requirements.txt`.

## Data and construction

[Data documentation](docs/data.md) gives the source versions, fixed URLs, expected paths, licenses, checksums, and complete construction commands. Small benchmark-construction inputs are bundled under `data_sources/benchmark/`. EvidenceNet HCC/CRC records are available from a pinned public Figshare release.

```bash
# Rebuild the benchmark from its retained construction inputs.
python scripts/benchmark/prepare_sources.py --only benchmark --download
python scripts/benchmark/build_reasoning_benchmark_formal_v2.py
python scripts/benchmark/validate_benchmark.py \
  --benchmark dataset/evaluated/rebuilt/external_reasoning_benchmark_formal_v3_571.jsonl \
  --reference dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl

# Acquire the literature inputs; obtain native omics inputs as documented.
python scripts/benchmark/fetch_evidencenet.py
python scripts/build_knowledge_pipeline.py --plan
python scripts/build_knowledge_pipeline.py --check
python scripts/build_knowledge_pipeline.py --run
python scripts/verify_knowledge_views.py
```

The build pipeline extracts and normalizes source evidence, constructs the Evidence Graph, adds observation-level lineage, and projects Document Corpus, Semantic KG, and Evidence Graph retrieval units. It also builds the controlled and source-separated representations. CLI arguments and `EVIDENCEMEETSRAG_*` variables support data stored outside the checkout.

Download GDSC from its fixed official release and build its derived evidence locally. Mixed-source graph/context archives are not redistributed. Archived-input manifests retain the paper snapshot checksums; `scripts/fetch_artifacts.py --offline` can verify compatible local archives.

## Experiments

Copy `.env.example` to `.env` and configure the required model endpoints locally. The default generator is GPT-5 mini; the additional model and judge roles are listed in [the experiment guide](docs/experiments.md).

```bash
cp .env.example .env
python scripts/reproduce_paper.py plan
python scripts/reproduce_paper.py main --input-mode retrieve --workers 4
python scripts/reproduce_paper.py ablation --workers 4
python scripts/reproduce_paper.py stages --input-mode retrieve --workers 4
python scripts/reproduce_paper.py grounding --phase prepare
python scripts/reproduce_paper.py grounding --workers 4
python scripts/reproduce_paper.py analyze
```

The seven adapters are Vanilla, Dense, Hybrid, Graph RAG, CRAG, RAPTOR, and KG2RAG. Main evaluation, fixed-pack ablations, and stage experiments use the same 571 IDs. Vanilla Full and Graph Full are each shared with their corresponding main and ablation results. The guide maps **Tables 1–10, Figures 1–7, and all appendix analyses** to commands, inputs, and outputs, including masking, matched budgets, hyperparameter controls, multiple generators, error analysis, source comparisons, and significance tests. It also identifies the source-specific secondary subsets retained by the paper.

The default retrieves from locally constructed corpora. Optional `--input-mode frozen` repeats generation on compatible restored local packs. Model calls incur the configured provider's charges; offline commands are identified in the guide. Fresh API outputs and machine-dependent timing measurements can vary.

## Layout

| Directory | Contents |
| --- | --- |
| `scripts/` | Construction, framework adapters, experiment entry points, evaluation, and analysis |
| `dataset/evaluated/` | Full benchmark, construction seeds, schema, and anonymous expert-review inputs |
| `data_sources/` | Compact source caches, acquisition manifests, and attribution notices |
| `configs/` | Source files, stable-ID mappings, frozen-input hashes, and experiment plans |
| `kg/schema/` | Evidence Graph and Semantic KG schemas |
| `figures/` | Editable conceptual diagrams and their rendering helper |
| `docs/` | Data acquisition, paper-to-code mapping, and figure instructions |

Generated graphs, indices, predictions, downloads, and local credentials are ignored by Git. Offline checks run with `python -m unittest discover -s tests -v` and in GitHub Actions. See [release validation](docs/validation.md) for the checked execution paths.

## License

Project code is MIT licensed. Source datasets and source-derived inputs retain their individual terms and attribution, documented in `data_sources/` and [docs/data.md](docs/data.md). Please cite the paper and the upstream resources used in your experiments.
