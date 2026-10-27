# 571 Benchmark Independent Human Review Guide

## Objective

Assess whether the benchmark questions are scientifically meaningful, answerable, source-supported, and neutral to the proposed Evidence Graph representation. This review evaluates benchmark validity, not RAG model performance.

## Reviewers

- Use two reviewers who did not construct the benchmark or inspect system predictions.
- Prefer researchers with cancer genomics, bioinformatics, precision oncology, or biomedical knowledge-resource experience.
- Reviewers work independently before adjudication.
- Record background, highest degree, relevant research area, and years of experience in a separate reviewer metadata note. Do not include identifying information in the anonymous manuscript.

## Files

- Give each reviewer `reviewer_packet_blinded_120.csv` and a separate copy of `blinded_review_form_120.csv`.
- Do not give reviewers `blinded_review_panel_120.jsonl` during Phase 1 because it contains adjudicator-only gold and provenance fields.
- The project administrator keeps `blinded_review_panel_120.jsonl` for Phase 2 and adjudication.

## Phase 1: Question-Only Blind Review

For every item, score the following fields without seeing the gold answer or source evidence.

1. `scientifically_well_formed_0_1`
   - 1: the question defines a coherent cancer/genomics claim or inference target.
   - 0: the question is nonsensical, materially underspecified, or uses invalid biological concepts.
2. `answer_identifiable_0_1`
   - 1: the requested answer type and scope are sufficiently clear to permit a unique answer under the stated source setting.
   - 0: multiple incompatible answers would be equally valid or the scope is ambiguous.
3. `requires_evidence_integration_0_1`
   - 1: answering requires interpreting, combining, tracing, or aggregating evidence rather than merely repeating an entity named in the question.
   - 0: the answer is a trivial lexical lookup or can be inferred from wording alone.
4. `representation_neutral_0_1`
   - 1: the question can in principle be answered from equivalent evidence represented as text, a conventional KG, or an Evidence Graph.
   - 0: the wording directly assumes an Evidence Graph-specific field, node type, or label unavailable to other faithful representations.
5. `leakage_risk_0_1`
   - 1: the wording or answer options reveal the gold answer or make it nearly deterministic without evidence.
   - 0: no obvious answer cue is present.
6. `confidence_1_5`
   - 1: guess or outside expertise; 3: reasonably confident; 5: highly confident.
7. `notes`
   - Briefly explain every 0 judgment and every leakage-risk=1 judgment.

The field `gold_supported_0_1` must remain blank in Phase 1.

## Phase 2: Gold and Provenance Review

After both reviewers finish and lock Phase 1, the administrator supplies the gold answer, source evidence, and source-record identifier from `blinded_review_panel_120.jsonl`.

For every item, score `gold_supported_0_1`:

- 1: the cited external record supports the normalized gold answer under the benchmark rule.
- 0: the source contradicts the answer, does not support the transformation, cannot be traced, or leaves the answer materially ambiguous.

Reviewers may revise confidence and add Phase 2 notes, but must not revise Phase 1 scores after seeing the gold answer.

## Adjudication

1. Merge the two completed forms by `review_id`.
2. Calculate per-criterion acceptance rate, raw agreement, and Cohen's kappa.
3. Send every disagreement and every item with any validity score of 0 to an adjudication meeting.
4. Record the final decision and a one-sentence reason.
5. Correct or exclude invalid benchmark records before rerunning final paper statistics; preserve both the original and revised benchmark hashes.

## Recommended Acceptance Criteria

These are prospective targets, not results.

- Scientifically well formed: at least 95% after adjudication.
- Gold supported by external source: at least 95%.
- Answer identifiable: at least 90%.
- Representation neutral: at least 90%.
- Requires evidence integration: at least 80%.
- Leakage-risk prevalence: at most 5%.
- Raw inter-reviewer agreement: at least 85% for each core validity criterion.
- Cohen's kappa: at least 0.60; at least 0.70 is preferable.

If a criterion misses its target, report the observed value and revise or exclude affected items. Do not tune the threshold after seeing the results.

## Paper Reporting Template

> Two domain reviewers independently assessed a deterministic, task-stratified sample of 120/571 benchmark items. Reviewers first evaluated scientific well-formedness, answer identifiability, evidence-integration demand, representation neutrality, and leakage risk without seeing gold answers. They then inspected external source records and judged gold support. Disagreements were adjudicated. The final acceptance rates were XX% for scientific validity, XX% for gold support, XX% for answer identifiability, and XX% for representation neutrality; leakage risk was XX%. Raw agreement was XX--XX%, with Cohen's kappa of XX--XX across criteria.

Replace all placeholders with observed results. If reviewers are not independent domain experts, describe their actual status rather than using this wording.
