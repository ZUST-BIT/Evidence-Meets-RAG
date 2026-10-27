# EvidenceNet source data

The four inputs in `manifest.json` are published as EvidenceNet v10:

Chang Zong (2026). *EvidenceNet: source-grounded biomedical evidence records from full-text disease-focused literature*. Figshare. https://doi.org/10.6084/m9.figshare.31888399.v10

The public Figshare metadata identifies the dataset license as MIT. Preserve the source attribution, record-level article provenance and source identifiers. This does not assign a new license to the original articles. Original literature PDFs are not redistributed here.

All four local construction inputs were checked against the public file metadata: their sizes and MD5 values match exactly. The manifest also records SHA-256 values for verification after download. Public `evidence_records_hcc.json` and `evidence_records_crc.json` are installed under the local names `evidence_nodes.json`; graph files retain `evidence_graph.json`.

Public metadata: https://api.figshare.com/v2/articles/31888399

Upstream extraction code: https://github.com/ZUST-BIT/EvidenceNet-code
