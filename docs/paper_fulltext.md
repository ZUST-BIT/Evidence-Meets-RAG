# Native full-text literature inputs

The native raw-document adapter reads **every `*.pdf`** in `kg/papers-full/hcc_papers/` and `kg/papers-full/crc_papers/`. The paper input is exactly 500 HCC and 500 CRC files (2,189,672,042 bytes), producing 66,244 full-text chunks with the published extraction settings. The family is the original input folder assignment, not a new classification inferred from the article title.

[`configs/paper_fulltext_sources.json`](../configs/paper_fulltext_sources.json) records all 1,000 expected filenames, byte sizes, SHA-256, MD5, available article identifiers, and source entry URLs. Each `target_path` is relative to the repository. The PDF bytes are not redistributed. DOI/PMC/PubMed links identify an article and its official acquisition routes; the original download log was not retained, so `original_download_url` is explicitly unset. Article identifiers and local possession do not establish a redistribution license. Follow the license and access terms shown by the publisher or archive.

Place lawfully obtained PDFs at their exact `target_path`, then verify the full collection:

```bash
python scripts/paper_fulltext_sources.py --check
# For an existing input directory outside this repository:
python scripts/paper_fulltext_sources.py --check --paper-root /path/to/papers-full
```

Verification requires all 1,000 files, their original sizes and SHA-256, and no extra PDF in either input directory. It exits with status 2 and lists missing, mismatched or extra files. The `raw` stage of `scripts/build_knowledge_pipeline.py` performs the same check before execution. Renaming an article or substituting a different PDF version is not an exact reconstruction of the paper input.

## Optional official open-access acquisition

The helper defaults to a local manifest summary. Its metadata mode resolves article identifiers using the [official PMC ID Converter API](https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/) and reads [current PMC Cloud metadata](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/):

```bash
python scripts/fetch_paper_fulltexts.py
python scripts/fetch_paper_fulltexts.py --resolve --limit 10
python scripts/fetch_paper_fulltexts.py --resolve --path 'kg/papers-full/crc_papers/000525695.pdf'
# Request downloads only when an official cloud record explicitly offers a CC PDF:
python scripts/fetch_paper_fulltexts.py --download --limit 10
```

The machine-readable plan is written to `outputs/paper_fulltext_oa_plan.json`; `--report` overrides it. `--path` can be repeated, and `--limit` limits the selected records. No publisher-page scraping, login, or access-control bypass is implemented. The helper uses the current `pmc-oa-opendata` bucket, whose metadata schema and license fields are documented in the [official bucket README](https://pmc-oa-opendata.s3.amazonaws.com/README.txt). Legacy PMC OA service endpoints are not used.

The helper accepts only PDF URLs explicitly returned by official PMC metadata with `is_pmc_openaccess` and a recognized Creative Commons license. It checks the official MD5 and original SHA-256/size. Exact historical files go to their manifest paths. A lawful file whose bytes differ is saved under `outputs/fulltext_candidates/`, outside the native adapter's PDF input glob. It does not overwrite a different existing source PDF. Articles without an eligible cloud PDF require the official publisher/PubMed route listed in the manifest; their availability and exact historical bytes are not guaranteed by a landing URL.

On 2026-10-27, metadata-only validation found an exact original-MD5 PDF for `10.3389/fimmu.2024.1358602` (PMC11165049, version 1, CC BY). DOI `10.1159/000525695` returned no unique PMC record and no PDF URL was fabricated. The validation did not download article full text.

The historical parser version is unrecorded; after a 24-version comparison, pinned `pypdf==6.13.2` reproduced all 66,244 archived chunks from 997 nonempty PDFs and the original empty-text skips for the remaining 3 of 1,000 inputs (see the [validation receipt](../configs/paper_fulltext_validation.json)).

## Manifest maintenance and native build

The inventory builder requires `pypdf` and the two EvidenceNet `evidence_nodes.json` exports. It scans PDF metadata and a bounded number of opening pages locally, storing candidate identifiers but no article body text. Official identifier corrections are retained on rebuild only when the local SHA-256 is unchanged.

```bash
python scripts/paper_fulltext_sources.py --build
python scripts/paper_fulltext_sources.py --unresolved
python scripts/build_knowledge_pipeline.py --stages raw --check
python scripts/build_knowledge_pipeline.py --stages raw --run
```

Metadata identity in this manifest describes the **actual PDF**, which can differ from an upstream EvidenceNet source label. Such corrections preserve the upstream values separately and do not change EvidenceNet records, graph construction or the PDF bytes. The native build requires the other raw adapter inputs described in [data.md](data.md); full-text verification alone does not replace those inputs. Frozen experimental context packs are documented separately in [experiments.md](experiments.md).
