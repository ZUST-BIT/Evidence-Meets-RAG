# Release validation

The package was checked in a fresh Python 3.11 environment on 2026-10-27, independently of the development project's installed dependencies. The dependency check, offline unit tests, benchmark validator, experiment smoke check, and synthetic construction smoke check passed.

| Check | Result |
| --- | --- |
| Benchmark reconstruction | All 571 records, field values, order, and output bytes match the released benchmark; SHA-256 `191ce330e81366cf893c327d3a7d69b30e74c769bd77e9f443f4155b73ccc405` |
| Native PDF reconstruction | With the pinned `pypdf==6.13.2`, all 66,244 ordered chunk texts from 997 nonempty PDFs match their archived SHA-256 values; the other three of 1,000 input PDFs reproduce the original empty-text skip |
| Supporting-source reconstruction | DisGeNET: 1,958 nodes / 2,452 edges; cell-line annotation: 182 nodes / 176 edges. Complete content matches the original source graphs apart from timestamps |
| Full-scale Evidence Graph construction | 295,063 nodes / 1,771,770 edges; node types, relation counts, relation-family counts and observation-source counts match the original build |
| Three retrieval views | All six reconstructed document/edge files match the paper snapshot byte-for-byte by SHA-256. Document Corpus: 18,251 units; Semantic KG: 51,980 units / 223,418 edges; Evidence Graph: 351,783 units |
| Framework adapters | 10,278 synthetic-corpus retrievals across six non-Dense adapters and three views; actual offline MiniLM encoding, FAISS indexing and index reload also passed |
| Mocked generation and judging | All 21 main systems, six component conditions, eight stage cells, 1,713 grounding rows, 571 masking rows and 8,565 model-panel rows completed with networking disabled |
| Run integrity | Shared Vanilla/Graph Full, exact 571-ID alignment, changed-input cache rejection, and resume without repeated model calls verified |
| Appendix experiments | Source forms, dual-source controls, oracle, error classification, long context, matched budgets with 18,251 units per view, and all 24 hyperparameter cells executed with mocked provider responses |
| Analysis and plotting | Availability/retrieval audits, cost profiling and case tracing completed on controlled inputs; Figures 3–7 rendered to PDF and were visually checked |
| Publication contents | Credential-value/pattern scan and file-size checks passed; generated graphs, predictions, indices, downloaded matrices and credentials are excluded from Git |

The mocked runs exercise request construction and downstream computations without purchasing model calls. Their synthetic scores are not paper results. They use fewer bootstrap replicates for execution checks; the released paper defaults remain 10,000/5,000. Fresh provider responses and machine-dependent runtime measurements must be obtained using the experiment commands.

Full-scale construction verification starts from the original normalized source graphs and raw-corpus inputs, identified in the snapshot manifests. The source-native adapters were additionally checked with controlled fixtures. Large omics matrices and article PDFs are acquired separately through the documented source resources; this release does not claim a completed fresh download of every upstream input.

After a full local construction, `python scripts/verify_knowledge_views.py` verifies the six retrieval files against the same original checksums. A different upstream snapshot is reported explicitly.

To repeat the local checks:

```bash
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
python scripts/benchmark/validate_benchmark.py
python scripts/reproduce_paper.py smoke
python scripts/smoke_knowledge_pipeline.py
```
