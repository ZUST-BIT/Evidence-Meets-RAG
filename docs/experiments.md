# Reproducing the experiments

Run commands from the repository root after installing the dependencies and obtaining the documented source files or release input assets. `python scripts/reproduce_paper.py plan` prints the protocol without contacting an API. `python scripts/reproduce_paper.py smoke` exercises retrieval, normalization, scoring, judge parsing, and paired inference offline. Neither command requires credentials or downloads a model.

The publication cohort is `dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl`: exactly 571 distinct records. The original task types, answer contracts, source identifiers, entity/disease anchors, and six-family mapping in `metadata.reasoning_type` are retained. Default main runs use seven framework adapters and three knowledge views. Figure 3 shows the six published framework families and excludes the additional Hybrid RAG adapter.

## Inputs and API configuration

The main comparisons use `raw_observation_view`, `semantic_kg`, and `evidence_graph_complete`. Code for reconstructing the knowledge views is exposed through `scripts/build_knowledge_pipeline.py`; see the data documentation for source acquisition and construction. Full-corpus retrieval requires the corresponding prepared `*_docs.jsonl` and `*_edges.jsonl` files under `dataset/rag/knowledge_sources/multi_cancer/`. Dense retrieval additionally requires the optional dense dependencies and downloads `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` on its first actual retrieval run. The original cache's runtime-file hashes and official revision metadata are recorded in [`configs/dense_encoder.json`](../configs/dense_encoder.json). `DENSE_RAG_MODEL` and `DENSE_RAG_REVISION` explicitly override the encoder; an alternative model does not inherit the MiniLM revision. Dense index manifests record the revision and reject a different or unrecorded revision. Frozen-context generation does not load that encoder.

The paper used several archived input snapshots. They must not be silently substituted for one another:

| Input group | Destination | Use |
| --- | --- | --- |
| Frozen main evidence packs | `dataset/frozen_contexts/main/` | Main generation, fixed-pack ablation, grounding, masking, and generator comparisons |
| Frozen R0/R1 stage contexts | `dataset/frozen_contexts/stage/` | Eight pipeline configurations with identical retrieval within each R level |
| Frozen top-100 contexts | `dataset/frozen_contexts/long_context/` | Long-context control |
| Frozen single-source and dual-source contexts | `dataset/frozen_contexts/source_forms/`, `dual_source/` | Source-separated comparisons and the 13-record exploratory check |
| Archived full retrieval-control corpora | `dataset/frozen_corpora/retrieval/multi_cancer/` | Top-k/BM25 controls; these can contain linked-summary text absent from the frozen main packs |
| Fixed matched corpora | `dataset/rag/analysis/ecorag_budget_control/*_matched_18251_docs.jsonl` | Size-matched retrieval control |

The input-asset manifest is `configs/input_snapshots.json`. The public workflow downloads native sources and constructs corpora locally. Mixed-source frozen archives are not redistributed; their manifests support optional verification of existing local copies. Frozen contexts contain questions and retrieved evidence, not predictions, normalized predictions, judge labels, or experiment scores. Descriptive source-path strings inside archived evidence are preserved as input literals; runners never open those strings as local files. `scripts/export_paper_contexts.py` can export new contexts from a completed local run.

Set credentials and endpoint URLs locally as described in `.env.example`. No credentials are bundled. The model roles are:

| Environment variable | Paper model ID | Endpoint configuration |
| --- | --- | --- |
| `RAG_EVAL_MODEL` | `gpt-5-mini` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| `RAG_EVAL_MODEL2` | `gemini-3.8-flash` | Shared OpenAI-compatible endpoint, or both `GEMINI_API_KEY` and `GEMINI_BASE_URL` |
| `RAG_EVAL_MODEL3` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` |
| `RAG_EVAL_MODEL4` | `gpt-5.4-mini` | Shared OpenAI-compatible endpoint |
| `RAG_EVAL_MODEL6` | `qwen3.5-flash` | `QWEN_API_KEY`, `QWEN_BASE_URL` |
| `JUDGE_MODEL` | `gpt-5.4-mini` | Shared OpenAI-compatible endpoint |

The runner defaults to these IDs while honoring explicit environment settings. Provider availability and stochastic API output can change. A fresh run reproduces the protocol and recomputes its own scores; it is not guaranteed to reproduce every historical prediction. This release does not ship historical model outputs.

## Main results and shared Full baselines

These commands make generation API calls and resume completed work:

```bash
python scripts/reproduce_paper.py main --workers 4
python scripts/reproduce_paper.py ablation --workers 4
python scripts/reproduce_paper.py stages --workers 4
python scripts/reproduce_paper.py analyze
```

`main` defaults to retrieval from locally constructed corpora. It covers Vanilla (`naive_rag`), Dense, Hybrid, Graph, CRAG (`crag_rag`), RAPTOR, and KG2RAG across all three views. Select a subset with `--frameworks naive_rag graph_rag --sources raw_observation_view semantic_kg`. Use `--input-mode frozen` only after restoring compatible local context archives. Record the selected input mode when comparing runs.

Vanilla Evidence Graph Full is written to `dataset/rag/results/fixed_context_component_ablation_571_vanilla__full.jsonl`. Fixed-pack component ablations, masking, grounding, the GPT-5 mini model panel, and the Vanilla main result use that exact output. The component ablation changes structure, evidence text, cross-source links, aggregation, or relation types while retaining each selected pack as the source.

Graph Evidence Graph Full is `dataset/rag/results/unified_571/stage/R1C1G1.jsonl`. `main` generates this through the stage runner rather than making a second independent Full run. The remaining stage configurations resume it. The offline analysis copies it to the canonical alias `unified_571/canonical/evidence_graph_rag.jsonl`.

The stage study uses the same 571 records in all eight R×C×G cells: base/full retrieval, flat/structured context, and plain/grounded generation. Distinct configurations with an identical direct generation input can reuse the same output after hash checks. Each conditional effect changes one stage while holding the other two fixed. It is not a comparison among the three knowledge views. `stages --phase retrieve`, `--phase generate`, and `--phase summary` expose individual phases. `--input-mode retrieve` reruns graph retrieval from the prepared corpus instead of restoring R0/R1 packs. The lower-level stage runner also accepts `--knowledge-dir` for an explicitly restored archived graph corpus.

`analyze` requires complete main and component-ablation outputs. It writes `dataset/rag/analysis/unified_571/main_metrics.json`, `main_and_ablation.json`, `core_significance/`, `template_cluster_bootstrap/`, `cross_cancer/`, and `significance/`. It verifies 571 unique matching benchmark records before paired analyses. Accuracy uses normalized exact match; macro-F1 uses the union of observed gold and predicted classes. The original paired bootstrap and exact McNemar implementations, iteration counts, and seed schedules are retained. `configs/significance_plan_571_main.json` gives all fourteen main Evidence-vs-baseline comparisons. The Vanilla/Graph core comparisons use the original seed schedule 20260628/20260635/20260642/20260649, and the same Vanilla authority is reused in the GPT-5 mini model panel. Unrounded exact p values are retained; a rounded display zero is not evidence of p=0.

## Grounding, errors, masking, and model robustness

```bash
python scripts/reproduce_paper.py grounding --phase prepare
python scripts/reproduce_paper.py grounding --workers 4
python scripts/reproduce_paper.py errors
python scripts/reproduce_paper.py masking --workers 4
python scripts/reproduce_paper.py models --workers 4
python scripts/reproduce_paper.py models --phase summary
```

`prepare` is offline; the other generation/judging commands call the configured API. Grounding has 571 judgments per view. Four additional generators each answer the three fixed Vanilla context sets; GPT-5 mini reuses the canonical main predictions. Model summaries live in `dataset/rag/results/unified_571/models/summary.json`; grounding summaries live in `faithfulness/summary.json`.

The grounding judge jointly assigns sufficiency S, answer support A, and contradiction X. ES=mean(S), and FCR=mean(normalized-correctness×A). There is no hard sufficiency/support nesting check. The parser excludes A=X=1. All criteria use the same compact judge input: up to six top-ranked units, text excerpts capped at 520 characters each, total context capped at 3,600 characters. Retrieval metrics use all eight retrieved units. The criteria share one judge response and are assessed separately.

The error judge classifies current wrong answers into the original taxonomy. `errors` produces all three sets under `unified_571/error_taxonomy/` and recomputes `residual_errors.json`. Labels are paired to the current predictions rather than imported from historical error results.

Masking uses the same 571 Evidence Graph packs and removes the original answer-bearing fields from context without changing retrieval IDs, order, or scores. Its output is `unified_571/masking/masked.jsonl`; `analysis/unified_571/masking_571.json` contains masked-minus-Full accuracy/F1 differences, 10,000 paired percentile bootstrap samples, and the exact McNemar test. Benchmark anchors in queries and prompts, and the original ranking, are retained during this context intervention. Run `masking --phase prepare` for offline input preparation, or `--phase analyze` to recompute statistics from completed predictions.

The source implementation retains the paper's metadata-assisted query construction and ranking. In particular, the mechanistic-tracing intermediate can occur in the supplied entity anchors even when it is absent from the natural-language question. The shared metadata-assisted protocol is retained across the released comparisons.

## Retrieval controls and reference settings

```bash
python scripts/reproduce_paper.py retrieval-controls --stage budget
python scripts/reproduce_paper.py retrieval-controls --stage hyperparam
python scripts/reproduce_paper.py long-context --workers 4
python scripts/reproduce_paper.py oracle --workers 4
```

The first two commands are offline. The matched-budget condition uses exactly 18,251 units per view, top eight, and an 8,000-character delivered-context budget. It compares Evidence Graph with Document Corpus for the Table 5 headline. The fixed corpus IDs/order are in `configs/matched_corpus_ids.json`. If the matched-corpus asset is absent, the runner selects those exact IDs from the prepared sources. Missing IDs cause an error; it never silently resamples from the 571 questions. The other condition budgets full-corpus main packs. Paired effects against both baselines are saved in `analysis/unified_571/budget/budget_control_summary.json`.

Hyperparameter controls use all 571 questions for each of 24 source/setting combinations: k∈{2,4,8,16}, plus the original k1/b settings in `run_retrieval_hyperparameter_stability.py`. The compact BM25 index preserves the original formula, document lengths, vocabulary frequencies, token repetition, and tie order. Full results and paired metric differences are written under `analysis/unified_571/hyperparam/`. If the archived retrieval-control directory exists, the public entry uses it explicitly. Its k=8 reference need not equal Table 4, whose diagnostics are attached to frozen main packs.

The last two commands call generation APIs. `long-context` compares two top-100 runs against the canonical top-eight outputs; its statistics use the original 10,000 paired resamples. `--input-mode retrieve` repeats top-100 retrieval from prepared sources. The gold-context oracle intentionally supplies record reference evidence as an upper reference condition and must not be presented as ordinary retrieval.

## Source-separated experiments and human validation

```bash
python scripts/reproduce_paper.py source-forms --workers 4
python scripts/reproduce_paper.py dual-source --workers 4
python scripts/reproduce_paper.py human-review
python scripts/reproduce_paper.py corpus-audit
```

Source-form generation compares Document/Semantic/Evidence representations with TCGA-only knowledge on all 571 questions, and PubMed-only knowledge on the original 127 HCC/CRC questions. These are the publication's explicit secondary denominators. The default retrieves from the single-source corpora produced by `build_single_source_knowledge_forms.py`; `--input-mode frozen` can restore compatible local contexts. Results and analysis use `single_source_form_generation.jsonl` and `analysis/evidence_meets_rag_single_source_forms/`.

The dual-source check uses the original 30-record mixed-task panel with three variants (90 input rows) and a source-role-aware prompt. Its analysis separately reports the 13-record precision-oncology subset discussed in the paper. Frozen inputs restore the same records and variants; `--input-mode retrieve` runs `build_tcga_pubmed_complementarity.py` first. The exploratory check is reported with its own denominator and is not generalized to the full benchmark.

Human review is offline analysis of two retained anonymous 120-record review forms, their frozen panel, and `HUMAN_REVIEW_GUIDE.md` under `dataset/evaluated/benchmark_validity_audit/`. `analyze_benchmark_human_review.py` recomputes acceptance, raw agreement, Cohen's kappa, and uncertainty from those ratings.

`corpus-audit` runs the deterministic, task-aware pre-retrieval lexical availability audit and combines main retrieval diagnostics with the completed grounding judgments. The audit needs full prepared corpora and is CPU/memory intensive, but does not use a model. Entity scope, any support, component coverage, natural-language support, and structured-label exposure are proxies defined in `audit_ecorag_coverability_leakage.py`. These metrics are lexical availability proxies defined by the audit rules. All 571 records are used.

## Figures, cases, and cost measurements

```bash
python scripts/reproduce_paper.py figures
python scripts/reproduce_paper.py profile
python scripts/reproduce_paper.py case
```

`figures` requires complete main analysis, error labels, and the 571-question oracle summary; it writes Figures 3–5 and 7 to `outputs/figures/`. The oracle ceiling is read from that summary. Figure 6 is generated when the completed stage study is summarized under `results/unified_571/stage/`. The stage figure renders the same twelve conditional effects using Matplotlib, so no local TeX installation or private Codex runtime is needed. Original conceptual figures are documented separately as figure sources.

The ABL1/AML case is a trace through the same frozen record, source evidence, retrieved packs, and predictions used by the main evaluation. `case` writes `abl1_aml_case_trace.json`, including predictions, retrieved units, deterministic diagnostics, and completed grounding labels if present. Its record ID is `cross-source-support-bundle-v2-aml-ABL1`. The gold component contract, lexical presence of a component, and substantive contextual support are distinct checks; a term hit alone does not prove evidential support. This case is different from the ABI1→ABL1→ABL2 mechanistic-tracing record.

`profile` makes no API calls. For each view it performs separate single-process naive/top-eight runs with limits 1 and 101, reports initialization and `(T101−T1)/100`, and counts units, edges, and on-disk bytes. Context tokens are the paper's approximation: average `sum(len(title)+len(text))/4` over 571 main Vanilla packs. The archived contexts give rounded estimates 3149/2982/2996 for Document/Semantic/Evidence. `configs/paper_cost_profile_reference.json` retains the original small hardware-specific measurement record. Fresh timings depend on hardware, OS cache, and corpus snapshot and are not expected to equal the original seconds.

## Paper-to-code map

| Paper item | Inputs and entry points | Generated output |
| --- | --- | --- |
| Table 1; §4.1 | Frozen benchmark; `analyze`; benchmark audit entry in data documentation | `benchmark_composition.json`, automatic benchmark audit |
| Table 2 | `main`, `corpus-audit`, `profile` | Availability audit; `cost_profile.json` |
| Table 3 | `main`, `ablation`, `stages`, `analyze` | `main_metrics.json`, `core_significance/`, `significance/` |
| Table 4 | `grounding`, `analyze` | `faithfulness/summary.json`, `retrieval_571.json` |
| Table 5 | `retrieval-controls`, `masking`, `models`, `long-context`, `oracle` | Corresponding budget, hyperparameter, masking, model, long-context, and oracle summaries |
| Table 6; B.2 | `human-review` | `human_review_metrics.json` and disagreement table |
| Table 7; B.3 | `corpus-audit`, main predictions | Coverability/label-exposure audit and answer metrics |
| Table 8; C.3 | `models`, then `models --phase summary` after core analysis | `models/summary.json` with canonical GPT-5 mini CI authority |
| Table 9; C.4 | `source-forms`, `dual-source` | Source-separated metrics/paired comparisons; exploratory dual-source report |
| Table 10; C.5 | `profile` | Units, edges, disk size, load/query times and raw timing trials |
| Figures 1–2 | Conceptual/method figure sources, not model outputs | See figure-source documentation |
| Figure 3 | `analyze`, `figures`; six framework task profiles | `figure3_radar_metrics.json`, radar plots |
| Figure 4 | `analyze`, `errors`, `figures` | Retrieval access and current residual-error plots |
| Figure 5; C.2 | `ablation`, `analyze`, `figures` | Canonical-Full component effects and dumbbell plot |
| Figure 6; C.2 | `stages` | `figure6_metrics.json`, twelve effect cells, plots |
| Figure 7; §5.4 | `analyze`, `figures` | Cancer-level paired performance and accuracy matrix |
| Appendix A | Knowledge construction and schema code | Source-backed graph, aliases, lineage, and three views |
| B.1 | Frozen benchmark + automatic validity audit | Source/task/answer/template checks |
| B.4 | `analyze_template_cluster_bootstrap.py`, invoked by `analyze` | 129-template clustered bootstrap and largest-template exclusion |
| C.1 | Main frozen ABL1/AML record, evidence, and canonical outputs | Direct case trace using the same inputs and predictions |

For a complete run, finish the generation commands, then run `analyze`, refresh model summaries, run the corpus audit, and generate figures. Outputs are ignored by Git. Configuration and input manifests are versioned; generated predictions, caches, models, and credentials are not.
