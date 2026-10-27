# DisGeNET via Therapeutics Data Commons

The project uses the curated DisGeNET subset distributed by Therapeutics Data Commons (TDC), obtained from Harvard Dataverse file **4168282**. TDC's [official gene–disease association dataset page](https://tdcommons.ai/multi_pred_tasks/gdi/) identifies this dataset as **CC BY-NC-SA 4.0**, and its [official metadata source](https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py) maps `disgenet` to that exact file ID. The archived CSV has 52,476 associations, 7,399 genes and 7,095 diseases, matching TDC's published counts. Exact hashes and the project acquisition record are in `manifest.json`.

Attribution: Piñero, J. et al. *The DisGeNET knowledge platform for disease genomics: 2019 update*. Nucleic Acids Research 48(D1), D845–D855 (2020), [doi:10.1093/nar/gkz1021](https://doi.org/10.1093/nar/gkz1021); and Therapeutics Data Commons for its curated subset and MedGen/UniProt mappings.

Project changes consist of oncology filtering, disease-label parsing, normalized association records, HCC/CRC alignment, and conversion to integrated graph and retrieval representations. The frozen `disgenet_aligned_normalized_latest.json` and DisGeNET-derived content in mixed graph/corpus snapshots retain **CC BY-NC-SA 4.0** terms, including attribution, noncommercial use and share-alike requirements. They are not relicensed under the project code's MIT license. Other source components retain their respective terms.

The [license text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) applies to this historical TDC-distributed dataset; this notice does not substitute a newer DISGENET product or assign its current access terms to the archived TDC subset.
