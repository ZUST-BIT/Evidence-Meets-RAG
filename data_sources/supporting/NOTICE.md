# Supporting records and HGNC lookup

These gzip snapshots restore the paper's exact processed source records and HGNC lookup. They are inputs to graph construction, not experimental predictions. `manifest.json` records original and compressed SHA-256 values. `field_audit.json` records the complete observed field inventory.

## DisGeNET through TDC

`disgenet.jsonl.gz` contains 18,543 oncology-filtered association records derived from the curated DisGeNET subset distributed by Therapeutics Data Commons, Harvard Dataverse file 4168282. The fields include gene/disease identifiers, disease descriptions, association scores and protein-sequence lengths. The HCC/CRC supporting adapter selects 1,226 records from this retained input.

This component and its derived content retain **CC BY-NC-SA 4.0**, with attribution to DisGeNET and TDC. The project changes consist of filtering, normalized source records and graph alignment. See the [source binding, attribution and license notice](../disgenet/NOTICE.md). It is not covered by a blanket MIT data license.

## Cell-line disease annotations

`tdc_cell_line.jsonl.gz` contains 723 source-derived annotation records: 693 resolved through Broad DepMap model metadata and 30 through Cellosaurus fallback. It retains cell-line names, Model IDs, RRIDs, cancer labels and OncoTree categories. The original helper selected the cell-line-name universe from TDC inputs and resolved those names against DepMap/Cellosaurus; it did not carry drug-sensitivity measurements into these annotation records.

A complete field/text audit found no IC50, LN_IC50, AUC, Z_SCORE, RMSE, dose, viability or drug-response fields or values. Numeric OncoTree codes are classification labels. No GDSC response matrix is included. The HCC/CRC graph adapter selects 88 cell-line annotations.

Retain attribution to **DepMap, Broad Institute**, [DepMap Portal](https://depmap.org/portal/), and to **Cellosaurus, SIB Swiss Institute of Bioinformatics**. Cite Bairoch A. *The Cellosaurus, a cell-line knowledge resource*. J. Biomol. Tech. 29:25–38 (2018), [doi:10.7171/jbt.18-2902-002](https://doi.org/10.7171/jbt.18-2902-002). The original Cellosaurus input was release 55.0, dated 31 March 2026. Cellosaurus data are [CC BY 4.0](https://www.cellosaurus.org/description.html); changes here consist of selected disease annotation and identifier mapping. Broad metadata retains its source-specific terms. Broad's [2026 acquisition guidance](https://forum.depmap.org/t/provide-an-open-endpoint-for-latest-version-retrieval/4652) describes research-use terms for newly generated data; its older general CC BY 4.0 guidance is not assumed to license the 26Q1 source. No independent blanket MIT license is assigned to the mixed annotation records.

## HGNC complete set

`hgnc_complete_set.txt.gz` is the complete frozen HGNC TSV used for gene-ID, symbol and alias resolution. Its exact bytes are pinned rather than replaced by a current live download. HGNC releases its data under **CC0 1.0** according to its [official license page](https://www.genenames.org/about/license/). The TSV is unmodified; only lossless compression is applied. Please credit the HUGO Gene Nomenclature Committee, [genenames.org](https://www.genenames.org/), RRID:SCR_002827, and Seal et al., *Genenames.org: the HGNC resources in 2023*. Nucleic Acids Research 51(D1):D1003–D1009 (2023), [doi:10.1093/nar/gkac888](https://doi.org/10.1093/nar/gkac888).
