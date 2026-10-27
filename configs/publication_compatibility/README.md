# Publication compatibility metadata

These compact files preserve the historical selection and serialization
used for the paper's `multi_cancer` retrieval inputs. They contain identifiers,
integer permutations, provenance timestamps, counts, and checksums; no source full text, expression or
drug-response measurements, prompts, model outputs, or evaluation results.
The separately documented source inputs are still required to rebuild content.

`observation_lineage.json.gz` maps 105,190 evidence identifiers to their chosen
observation identifiers, in their original order (114,685 `observation_trace`
`DERIVED_FROM` edges). The complete-graph builder uses these as tie preferences
only among candidates with equal original scores. All candidate filters, score
formulas, the `best_score - 2` threshold, and link limits remain unchanged.
Unlisted candidates use observation ID order; a historical preference never
overrides a different score or introduces a candidate absent from current input.

The four `*.order.u32.gz` files are little-endian unsigned 32-bit permutations
over lexically sorted, unique document IDs or edge triples. They cover 18,251
raw-view documents, 51,980 semantic documents, 351,783 complete evidence-graph
documents, and 1,943,076 complete evidence-graph edges. The manifest defines the
key encoding and stores the original key-set hash, permutation hash, compressed
file hash, and historical corpus hash.

`scripts/publication_compatibility.py` validates checksums, counts, uniqueness,
and permutation bijectivity before restoring the historical order. A changed
input key set uses stable lexical order and emits a warning, instead of applying
an unrelated permutation. Ordering applies only to the four registered
`multi_cancer` outputs.

`retrieval_runtime.json.gz` contains the historical `created_at` and
`extracted_at` values for all 256,101 original retrieval records carrying either
field (171,786 evidence and 84,315 observation records), their matching
occurrences in rendered provenance, and a canonical SHA256 fingerprint for each
whole record after normalizing only these timestamp paths and occurrences.
The serializer restores these timestamps only when that fingerprint matches;
changed non-runtime content keeps fresh provenance and is counted in a warning.
This restores the historical retrieval snapshot, including its timestamp tokens.
Native graph files retain the actual current build metadata and are not edited
by this serialization step.
