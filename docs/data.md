# Data and source provenance

## The released 571-question benchmark

`dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl` is the complete paper benchmark, copied without changing its records or bytes. It contains 571 unique questions, six reasoning families and 11 cancer families. Its size is 1,917,087 bytes and its SHA-256 is `191ce330e81366cf893c327d3a7d69b30e74c769bd77e9f443f4155b73ccc405`. `benchmark_manifest.json`, `benchmark.schema.json` and `checksums.sha256` describe and validate the release.

All 14 original top-level fields are retained: `id`, `source`, `task_type`, `question`, `answer`, `answer_type`, `oncology_focus`, `evidence_text`, `entities`, `provenance`, `metadata`, `gold_standard`, `is_machine_evaluable`, and `benchmark_annotation`. Nested entity identifiers, aliases, source records, answer contracts, evidence requirements and reasoning annotations are retained.

| Paper reasoning family | Questions |
| --- | ---: |
| Multi-hop evidence reasoning | 100 |
| Cross-source corroboration | 100 |
| Precision oncology evidence judgment | 100 |
| Multi-evidence logical synthesis | 81 |
| Evidence signal interpretation | 97 |
| Hierarchical evidence aggregation | 93 |
| Total | 571 |

The stored `task_type` field has nine values. BioASQ's seven factoid and five yes/no questions contribute to evidence signal interpretation, and its seven list questions contribute to hierarchical aggregation. Consequently, the six paper family counts differ from the nine storage task counts. The manifest records both, together with all 11 cancer counts.

The benchmark fields `answer`, `gold_standard` and `evidence_text` describe reference annotations; source-backed retrieval corpora are constructed separately. The released evaluation follows the paper's metadata-assisted input setting: benchmark entity/cancer anchors can enter query construction, ranking and prompts. All 100 intermediate-gene mechanism records retain their original intermediate-gene metadata anchors. Source-derived context generation and the masking control retain the same documented input settings.

From the repository root, validate the complete frozen benchmark without network access or external packages:

```bash
python scripts/benchmark/validate_benchmark.py
```

## Reconstructing the benchmark

The source inputs for construction are separate from experimental predictions. `data_sources/benchmark/manifest.json` pins 14 compact gzip snapshots and one public Pathway Commons download by SHA-256. The compressed snapshots total 19,436,247 bytes. They contain CIViC records, four BioASQ 12B golden batches, SIGNOR interactions, Open Targets slices, two IntOGen archives, and the ClinVar somatic subset used during construction. No model outputs or judge scores are included.

Restore the bundled inputs and obtain the fixed Pathway Commons file:

```bash
python scripts/benchmark/prepare_sources.py --only benchmark --download
python scripts/benchmark/prepare_sources.py --only benchmark --check
python scripts/benchmark/build_reasoning_benchmark_formal_v2.py
python scripts/benchmark/validate_benchmark.py \
  --benchmark dataset/evaluated/rebuilt/external_reasoning_benchmark_formal_v3_571.jsonl \
  --reference dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl
```

The historical constructor filename is retained even though its final output has 571 questions. The builder uses the 73 retained construction seeds in `construction/formal_seed_73.jsonl`; the discarded historical 499-record dataset is needed only as a task histogram, supplied in `historical_seed_task_counts.json`. An explicit `record_overrides.json` retains the archived support-layer list and matching evidence wording for one GLIOMA mechanism chain. This preserves its original construction annotation where the raw-source reconstruction exposes one additional IntOGen support layer. It changes no question, answer, entity or scoring contract. Historical grouped CIViC IDs used a process-randomized Python hash; the builder preserves released IDs for matching question/answer pairs, with a stable SHA-256 fallback for new groups. It writes to `evaluated/rebuilt/` and does not overwrite the paper benchmark.

Construction requires Python 3.11+, pandas, pyarrow and requests. Live acquisition is disabled when an expected cache is missing unless `--allow-downloads` is explicitly selected. Fresh upstream data can produce a new benchmark; use the pinned inputs for the paper version. `--raw-dir`, `--seed`, `--reference`, `--overrides`, `--output` and `--summary` support alternate directories without user-specific absolute paths.

`python scripts/audit_benchmark_validity.py` writes the automatic validity report and a newly sampled review packet under `dataset/evaluated/rebuilt/benchmark_validity_audit/`. It preserves the retained human-review inputs in the separate `benchmark_validity_audit/` directory.

| Source | Fixed version or archived input | Official access and preparation |
| --- | --- | --- |
| CIViC | Evidence/assertion cache acquired 2026-06-09 | [CIViC](https://civicdb.org/), [releases](https://civicdb.org/releases/main). The included JSONL preserves evidence/assertion identifiers; construction filters oncology evidence, normalizes categorical labels and groups related assertions. |
| BioASQ | 12B golden test batches 1–4, 2024 | [Official datasets](https://participants-area.bioasq.org/datasets/). Upstream access may require registration. The included four golden JSON files are restored to `dataset/raw/bioasq/test/`; construction selects 19 oncology questions and normalizes answer contracts. |
| SIGNOR | Archived `signor_latest.tsv`, identified by checksum | [Official release endpoint](https://signor.uniroma2.it/releases/getLatestRelease.php). Construction selects supported protein interactions and directed two-edge mechanism chains, retaining relation IDs and PMIDs. A live “latest” download is not substituted for the archived file. |
| Open Targets | Platform 25.03 | [Official 25.03 output](https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.03/output/). Included disease/target/association Parquet inputs preserve the original supported-cancer slices. Construction joins disease families, target symbols and direct association support. |
| IntOGen | Drivers/Cohorts 20200201 | [Drivers](https://www.intogen.org/download?file=IntOGen-Drivers-20200201.zip), [cohorts](https://www.intogen.org/download?file=IntOGen-Cohorts-20200201.zip). Both archives are included as exact compressed inputs; construction maps cohort cancer families and driver support. |
| Pathway Commons | PC2 v14, HGNC GMT | [Fixed GMT download](https://download.baderlab.org/PathwayCommons/PC2/v14/pc-hgnc.gmt.gz). Fetch using `prepare_sources.py --download`; SHA-256 is checked. Construction uses cancer-related pathway classes and their memberships. |
| ClinVar | Archived cancer-filtered somatic submission subset, acquired 2026-06-15 | [NCBI submission summary](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz). The precise monthly release was not recorded; the included filtered JSONL is the checksum-pinned paper input. Construction groups variation submissions and normalizes somatic/oncogenic interpretations. |

These sources retain their individual terms. In particular, BioASQ attribution, SIGNOR attribution, ClinVar provenance and Pathway Commons source-specific terms are not replaced by the project code license. See [benchmark input notices](../data_sources/benchmark/NOTICE.md).

## EvidenceNet literature inputs

The HCC/CRC structured literature inputs are publicly available as [EvidenceNet dataset version 10](https://doi.org/10.6084/m9.figshare.31888399.v10). `data_sources/evidencenet/manifest.json` contains fixed public file IDs, URLs, byte counts, MD5 and SHA-256. All four local paper inputs were matched byte-for-byte to that public release by MD5. The record files are named `evidence_nodes.json` locally for compatibility with the adapter.

| Local target | Public v10 name | File ID | Bytes |
| --- | --- | ---: | ---: |
| `kg/pubmed-evidence-kg/hcc/evidence_graph.json` | `evidence_graph_hcc.json` | [64670229](https://ndownloader.figshare.com/files/64670229) | 27,687,460 |
| `kg/pubmed-evidence-kg/hcc/evidence_nodes.json` | `evidence_records_hcc.json` | [64670247](https://ndownloader.figshare.com/files/64670247) | 47,420,159 |
| `kg/pubmed-evidence-kg/crc/evidence_graph.json` | `evidence_graph_crc.json` | [64670223](https://ndownloader.figshare.com/files/64670223) | 23,507,508 |
| `kg/pubmed-evidence-kg/crc/evidence_nodes.json` | `evidence_records_crc.json` | [64670241](https://ndownloader.figshare.com/files/64670241) | 41,081,656 |

```bash
python scripts/benchmark/fetch_evidencenet.py
python scripts/benchmark/prepare_sources.py --only evidencenet --check
```

The HCC export contains 7,872 evidence records, 10,328 graph nodes and 49,756 edges; CRC contains 6,622 records, 8,795 nodes and 39,361 edges. These exports cover HCC and CRC, while the benchmark covers 11 cancer families. AML benchmark reference annotations are constructed from the separately documented reference resources. Figshare marks the structured dataset MIT; article text and other upstream material retain their own rights. No original article PDFs are bundled. See [EvidenceNet notice](../data_sources/evidencenet/NOTICE.md) and the [upstream extraction code](https://github.com/ZUST-BIT/EvidenceNet-code).

## Knowledge-source build inputs

Restore the compact supporting inputs and exact frozen HGNC complete set before building the supporting graphs:

```bash
python scripts/benchmark/prepare_sources.py --only supporting
python scripts/benchmark/prepare_sources.py --only supporting --check
python scripts/build_supporting_alignment_graphs.py --sources disgenet
python scripts/build_supporting_alignment_graphs.py --sources cellline_annotation
```

`data_sources/supporting/manifest.json` pins the lossless gzip copies of `dataset/processed/disgenet.jsonl` (18,543 records), `dataset/processed/tdc_cell_line.jsonl` (723 records), and `kg/normalization/raw/hgnc_complete_set.txt`. These inputs avoid invoking the legacy general-purpose benchmark-data collector. Cell-line records contain names, identifiers and disease classifications, with no GDSC drug-sensitivity measurements; the complete field inventory is in `field_audit.json`. Their HCC/CRC outputs reproduce 1,958 nodes/2,452 edges for DisGeNET and 182 nodes/176 edges for cell-line annotation. See [supporting-source attribution and terms](../data_sources/supporting/NOTICE.md).

`scripts/build_knowledge_pipeline.py --plan` prints the full source-to-graph pipeline. `--check` reports missing inputs before a build. Source-native extraction uses the following directories; CLI flags and corresponding `EVIDENCEMEETSRAG_*` environment variables can override them. `configs/native_source_files.json` lists all 81 exact LinkedOmics files used by the paper, including layers that an adapter otherwise treats as optional.

| Input | Expected location | Acquisition/version and preparation |
| --- | --- | --- |
| TCGA/LinkedOmics | `dataset/raw/LinkedOmics/` | All 81 exact files in `configs/native_source_files.json`, stamped `Firehose01_28_2016`, are linked from the [official cohort download pages](https://linkedomics.org/data_download/TCGA-BRCA/) and have verified direct URLs in `data_sources/native/manifest.json`. All 81 official objects returned HTTP 200; their combined size is 341,253,272 bytes. Use `fetch_native_sources.py --source linkedomics --download`. |
| Broad DepMap | `dataset/raw/DepMap/` | Select **DepMap Public 26Q1** under [All Data](https://depmap.org/portal/data_page/?tab=allData), using the exact filenames and target paths below and in `data_sources/native/depmap_required_files.json`. [Official 26Q1 release notes](https://forum.depmap.org/t/announcing-the-26q1-release/4606) establish the release. Preserve file-specific metadata and terms. |
| Sanger GDSC | `dataset/raw/DepMap/drug-response/` | Actual native inputs are `GDSC1_fitted_dose_response_24Jul22.csv`, `GDSC2_fitted_dose_response_24Jul22.csv`, and `screened_compounds_rel_8.4.csv`. [Official release 8.4 directory](https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/), [data policy](https://depmap.sanger.ac.uk/documentation/data-usage-policy/). These are Sanger source tables, distinct from TDC's packaged GDSC training dataset. |
| HGNC | `kg/normalization/raw/hgnc_complete_set.txt` | The exact frozen complete-set TSV is included in `data_sources/supporting/` and restored by `prepare_sources.py --only supporting`. [Official complete-set TSV](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt), [downloads](https://hgnc.genenames.org/download/). The normalizer uses HGNC identifiers/symbols and aliases. |
| TarKG nodes | `kg/TarKG/TarKG_nodes.csv` | [Official node CSV](https://tarkg.ddtmlab.org/download/KG_Data/TarKG_nodes.csv), [source README](https://tarkg.ddtmlab.org/download/README.md), [publication](https://doi.org/10.1093/bioinformatics/btae598). The public site's Download component explicitly links this 65,205,492-byte file. Native normalization needs its `unify_id`, `kind`, `name`, `db_source` and `dbid` columns, rather than the 1.7 GB relation table. The fixed paper-input SHA-256 is recorded in the native manifest. |
| DisGeNET/TDC | `dataset/processed/disgenet.jsonl` | Restore the exact processed input with `prepare_sources.py --only supporting`. Original raw source: [Harvard file 4168282](https://dataverse.harvard.edu/api/access/datafile/4168282), [official TDC dataset/license](https://tdcommons.ai/multi_pred_tasks/gdi/), with 52,476 pairs, 7,399 genes and 7,095 diseases. The included processed cache preserves 18,543 oncology-filtered records; the supporting-source adapter aligns its HCC/CRC records. |
| Cell-line annotation | `dataset/processed/tdc_cell_line.jsonl` | Restore all 723 exact normalized annotations with `prepare_sources.py --only supporting`. Their source mapping combines names with DepMap `Model.csv` and [Cellosaurus](https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt), preserving identifiers and provenance. No response matrix is needed to rebuild this supporting graph. |

The actual GDSC source CSV URLs in `data_sources/native/manifest.json` were checked directly against the official release server. The downloader uses the fixed object sizes and publisher ETags, verifies MD5 against the single-part ETag, checks a pinned SHA-256 where available, and records local SHA-256 values after a complete download. It resumes `.part` files on rerun; a missing large-file SHA-256 is not presented as a completed historical-byte verification.

```bash
python scripts/benchmark/fetch_native_sources.py
python scripts/benchmark/fetch_native_sources.py --download
python scripts/benchmark/fetch_native_sources.py --check
```

The extraction code labels the Broad inputs `26Q1`; the original source-native directory and download log are no longer available for byte comparison. The archived normalized-input hashes establish the paper snapshot, while the code's release label alone does not establish an upstream file hash.

For LinkedOmics, the manifest resolves every exact paper filename through its official cohort page and records the current object's size, ETag and last-modified time. No filename or compression conversion is required. Historical whole-file checksums were not recorded; a complete acquisition generates a local SHA-256 receipt. The downloader's default TLS validation applies:

```bash
python scripts/benchmark/fetch_native_sources.py --source linkedomics --download
python scripts/benchmark/fetch_native_sources.py --source linkedomics --check
```

The non-GDSC Broad files required by the native adapter are:

| Exact file in DepMap Public 26Q1 | Target below `dataset/raw/DepMap/` |
| --- | --- |
| `Model.csv` | `model-condition-mapping/Model.csv` |
| `CRISPRGeneEffect.csv` | `crispr-screen/CRISPRGeneEffect.csv` |
| `CRISPRGeneDependency.csv` | `crispr-screen/CRISPRGeneDependency.csv` |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | `expression/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` |
| `OmicsCNGeneWGS.csv` | `copy-number/OmicsCNGeneWGS.csv` |
| `OmicsSomaticMutationsMatrixHotspot.csv` | `mutations/OmicsSomaticMutationsMatrixHotspot.csv` |
| `OmicsSomaticMutationsMatrixDamaging.csv` | `mutations/OmicsSomaticMutationsMatrixDamaging.csv` |
| `OmicsGlobalSignatures.csv` | `global-genomic-features/OmicsGlobalSignatures.csv` |
| `OmicsInferredMolecularSubtypes.csv` | `global-genomic-features/OmicsInferredMolecularSubtypes.csv` |
| `PortalCompounds.csv` | `model-condition-mapping/PortalCompounds.csv` |

The portal may require interactive browser verification for download. Save the selected release metadata, publisher checksum and file-specific terms with the acquired files; short-lived signed URLs should be obtained at download time. The [official no-CAPTCHA metadata endpoint](https://depmap.org/portal/api/no-captcha/download/files) is documented by [Broad staff](https://forum.depmap.org/t/provide-an-open-endpoint-for-latest-version-retrieval/4652), but returned an HTML verification page during this audit. No publisher hashes or fixed file URLs were invented from that response. `depmap_required_files.json` makes the ten unresolved publisher URLs/hash bindings explicit.

The official, downloadable [DepMap 24Q4 release](https://doi.org/10.25452/figshare.plus.27993248.v1) is a different quarter, with some different file schemas. The [Public 26Q1 Chronos parameter archive](https://doi.org/10.6084/m9.figshare.31660582.v1) includes `gene_effect.csv`, whose equivalence to the required corrected `CRISPRGeneEffect.csv` has not been established. Neither is substituted for the listed paper inputs. EvidenceNet's public v10 deposit contains literature exports and validation files; it does not contain exact TCGA/DepMap normalized snapshots or the mixed raw corpus.

For TarKG, use a manually obtained file or the fixed public URL:

```bash
python scripts/benchmark/fetch_native_sources.py --source tarkg \
  --local-file PATH_TO_TarKG_nodes.csv
python scripts/benchmark/fetch_native_sources.py --source tarkg --download
```

The official TarKG certificate was expired during this audit. Default TLS verification remains enabled. The optional `--allow-legacy-tarkg-tls` flag permits a credential-free download only from the fixed TarKG URL, rejects redirects, and requires the known full-file SHA-256 before installing the result. It does not change global TLS settings or apply to other sources:

```bash
python scripts/benchmark/fetch_native_sources.py --source tarkg --download \
  --allow-legacy-tarkg-tls
```

The expected SHA-256 is `9e0e3743507c562c7e97d676a7f5227202ed18e1ae8de960cdec61edfea6f9d6`; MD5 is `aa772a5e41c3f6a1f837a8017c30bcaa`. The official source README does not state an independent blanket dataset license, so upstream source attribution and terms remain applicable. The article's open-access license is not treated as a blanket relicense of all integrated source data. Frozen normalized build inputs bypass this normalization step when that route is selected.

The local frozen build-input route is specified by `configs/build_snapshot_inputs.json`: five normalized source graphs and three original raw-corpus inputs are restored before the integration, complete-graph, semantic projection, controlled-form and single-source stages. The exact prepared-context and corpus groups are listed separately in `configs/input_snapshots.json`. These inputs contain context/source data, not model predictions or judge labels. Their separate manifests identify the archived text versions. Mixed-source archives are not publicly redistributed; the public workflow downloads sources under their individual terms and reconstructs the graph locally. Existing local snapshots can be verified against the manifests.

The supplied [publication compatibility metadata](../configs/publication_compatibility/README.md) preserves historical equal-score lineage choices, corpus row order, and guarded retrieval-provenance timestamps. These small assets contain IDs, permutations, timestamps, and hashes rather than source text or response measurements. Historical runtime values are restored only if the whole non-runtime retrieval record matches its fingerprint; native graph files keep their current build metadata. Changed inputs receive deterministic fallback ordering and retain fresh provenance where fingerprints differ.

Full-text source acquisition for the paper-sourced document view is described in [paper_fulltext.md](paper_fulltext.md).

After acquiring the declared native sources, check and run the full local pipeline:

```bash
python scripts/build_knowledge_pipeline.py --check
python scripts/build_knowledge_pipeline.py --run
```

If the intermediate build inputs already exist locally and have been verified, run only the downstream stages:

```bash
python scripts/build_knowledge_pipeline.py \
  --stages integrate complete semantic project controlled_forms single_source --run
```

Data permission follows the actual source. Broad's [2026 acquisition guidance](https://forum.depmap.org/t/provide-an-open-endpoint-for-latest-version-retrieval/4652) states that newly generated DepMap data use source-specific research terms, with separate commercial licensing; the older general CC BY 4.0 guidance is not assumed to cover 26Q1. Sanger's GDSC policy is separate and is not relabeled as Broad CC BY or TDC's packaged-data license. GDSC source tables are obtained officially and rebuilt locally; mixed GDSC-derived snapshots are not supplied as publication assets. The historical DisGeNET/TDC subset and its derived content retain CC BY-NC-SA 4.0, documented in [its notice and manifest](../data_sources/disgenet/NOTICE.md). MIT applies to project code, not as a blanket data license. Release acquisition manifests state which components are supplied as snapshots and which are fetched under their source terms.

## Human review inputs

Table 6 uses the retained anonymous human annotations in `dataset/evaluated/benchmark_validity_audit/`: `blinded_review_panel_120.jsonl`, `blinded_review_form_120_1.csv`, `blinded_review_form_120_2.csv`, and `HUMAN_REVIEW_GUIDE.md`. Each reviewer form has 120 rows. The panel and guide preserve the review material and rubric; these are actual human ratings rather than synthetic API outputs. Recompute their summary with `scripts/analyze_benchmark_human_review.py`. No precomputed human-review metrics are required as inputs.
