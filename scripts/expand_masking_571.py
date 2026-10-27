"""Extend the existing fixed-retrieval masking protocol to the frozen 571 IDs.

Prepare is offline. Run explicitly calls the project's configured generation API.
All outputs live in unified_571; historical experiment files are left intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from rag_eval_lib import (
    OpenAICompatibleClient, build_prompt, compute_answer_metrics,
    set_rag_profile, set_stage_config,
)
from run_fixed_context_component_ablation_571 import load_env_file
from run_fixed_retrieval_label_masking import mask_retrieved_doc
from run_rag_evaluation import build_context_blocks, generate_prediction, json_ready
from analyze_core_paired_significance import compare
from analyze_retrieval_evidence_quality import audit_retrieval
from analyze_answer_label_masking_experiment import correct, binomial_two_sided

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'dataset/rag/results'
OUT = RESULTS / 'unified_571/masking'
ANALYSIS = ROOT / 'dataset/rag/analysis/unified_571'
BENCHMARK = ROOT / 'dataset/evaluated/external_reasoning_benchmark_formal_v3_571.jsonl'
FULL = RESULTS / 'fixed_context_component_ablation_571_vanilla__full.jsonl'
DOC = RESULTS / 'multi_cancer__raw_observation_view__naive_rag__formalv3_571_classic4_raw_vs_graph.jsonl'
SEM = RESULTS / 'multi_cancer__semantic_kg__naive_rag__formalv3_571_classic4_semantic_kg.jsonl'
OLD_MASKS = [
    RESULTS / 'multi_cancer__evidence_graph_complete_label_masked__naive_rag__labelmask120.jsonl',
    RESULTS / 'multi_cancer__evidence_graph_complete_label_masked__naive_rag__highgain181.jsonl',
]


def read(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rid(row):
    return str((row.get('record') or row).get('id') or '')


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_row(row):
    result = deepcopy(row)
    result['retrieved_docs'] = [mask_retrieved_doc(doc) for doc in result['retrieved_docs']]
    for key in ('prediction', 'normalized_prediction', 'inferred_evidence_state', 'prediction_mode'):
        result[key] = None
    return result


def prompt_hash(row):
    return hashlib.sha256(build_prompt(row['record'], build_context_blocks(row)).encode()).hexdigest()


def prepare():
    set_rag_profile('tuned')
    set_stage_config('full', 'structured', 'grounded')
    records = read(BENCHMARK)
    ids = [row['id'] for row in records]
    assert len(ids) == len(set(ids)) == 571
    canonical = {rid(row): row for row in read(FULL)}
    assert set(canonical) == set(ids) and len(read(FULL)) == 571
    ordered = [canonical[key] for key in ids]
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / 'original.jsonl', ordered)
    canonical_dir = RESULTS / 'unified_571/canonical'
    write_rows(canonical_dir / 'evidence_vanilla.jsonl', ordered)
    sources = {'doc_corpus': DOC, 'semantic_kg': SEM, 'evidence_kg': FULL}
    retrieval = {}
    for name, path in sources.items():
        source_rows = {rid(row): row for row in read(path)}
        assert set(source_rows) == set(ids)
        if name != 'evidence_kg':
            write_rows(canonical_dir / f'{name}.jsonl', [source_rows[key] for key in ids])
        vals = [audit_retrieval(source_rows[key]['record'], source_rows[key]['retrieved_docs']) for key in ids]
        retrieval[name] = {
            'count': len(vals),
            **{key: sum(row[key] for row in vals) / len(vals) for key in
               ('component_recall', 'complete_bundle', 'entity_hit', 'joint_scope_precision')},
        }
    dump(ANALYSIS / 'retrieval_571.json', {
        'scope': 'same existing metadata-assisted configuration; no leakage correction',
        'count': 571, 'sources': retrieval,
    })
    completed = {rid(row): row for row in read(OUT / 'masked.jsonl')}
    assert set(completed) <= set(ids)
    imported = 0
    rejected = 0
    for path in OLD_MASKS:
        for old in read(path):
            key = rid(old)
            if key in completed or key not in canonical:
                continue
            target = masked_row(canonical[key])
            # Exact generation prompt equality is required to reuse a prediction.
            if old.get('prediction_mode') == 'direct_prompt' and old.get('normalized_prediction') is not None and prompt_hash(old) == prompt_hash(target):
                target.update({k: old.get(k) for k in ('prediction', 'normalized_prediction', 'inferred_evidence_state', 'prediction_mode')})
                target['reuse_source'] = str(path.relative_to(ROOT))
                target['inference_prompt_sha256'] = prompt_hash(target)
                completed[key] = target
                imported += 1
            else:
                rejected += 1
    write_rows(OUT / 'masked.jsonl', [completed[key] for key in ids if key in completed])
    manifest = {
        'benchmark': str(BENCHMARK.relative_to(ROOT)), 'benchmark_sha256': digest(BENCHMARK),
        'count': 571,
        'canonical_full': str(FULL.relative_to(ROOT)), 'canonical_full_sha256': digest(FULL),
        'full_metrics': compute_answer_metrics(ordered),
        'configuration': {'model': 'gpt-5-mini', 'rag_profile': 'tuned', 'retrieval': 'full', 'context': 'structured', 'generation': 'grounded', 'mask': 'fixed_retrieval_global_label_mask_v1'},
        'metadata_policy': 'unchanged; known answer-bearing anchor issue not corrected',
        'masking': {'completed': len(completed), 'imported_prompt_identical': imported,
                    'rejected_cache_rows': rejected, 'pending': 571-len(completed)},
        'semantic_note': 'The former audit-panel and high-gain runs use the same masking intervention. On all 571 IDs they become one full-benchmark result; subgroup statistics keep their actual denominators.',
    }
    dump(ANALYSIS / 'canonical_manifest.json', manifest)
    print(json.dumps({'count': 571, 'full_accuracy': manifest['full_metrics']['accuracy'],
                      'full_macro_f1': manifest['full_metrics']['macro_f1'], **manifest['masking']}, indent=2), flush=True)
    return ids, canonical, completed


def run(workers):
    load_env_file(ROOT / '.env')
    ids, canonical, completed = prepare()
    client = OpenAICompatibleClient(model='gpt-5-mini')
    if not client.is_configured():
        raise RuntimeError('Generation credential is not configured')
    pending = [key for key in ids if key not in completed]
    errors = []
    lock = threading.Lock()
    def one(key):
        row = masked_row(canonical[key])
        prediction, normalized, state, mode = generate_prediction(client, row)
        row.update(prediction=prediction, normalized_prediction=normalized,
                   inferred_evidence_state=state, prediction_mode=mode,
                   inference_prompt_sha256=prompt_hash(row))
        return json_ready(row)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                # Credentials and provider responses must not be printed.
                errors.append({'id': key, 'error_type': type(exc).__name__})
                continue
            with lock:
                with (OUT / 'masked.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            completed[key] = row
            if len(completed) % 25 == 0 or len(completed) == 571:
                print(f'[masking] {len(completed)}/571', flush=True)
    dump(OUT / 'errors.json', errors)
    if len(completed) != 571:
        raise RuntimeError(f'Masking incomplete: {len(completed)}/571; resume run after checking errors.json')
    write_rows(OUT / 'masked.jsonl', [completed[key] for key in ids])


def analyze():
    original = read(OUT / 'original.jsonl')
    masked = read(OUT / 'masked.jsonl')
    assert len(original) == len(masked) == 571
    assert {rid(row) for row in original} == {rid(row) for row in masked}
    # Existing core significance implementation avoids the old quadratic
    # recomputation of normalized labels inside each bootstrap replicate.
    effect = compare(OUT / 'masked.jsonl', OUT / 'original.jsonl', None, 10000, 20260627)
    dump(ANALYSIS / 'masking_571.json', {'count': 571, 'original': compute_answer_metrics(original),
        'masked': compute_answer_metrics(masked), 'masked_minus_original': effect,
        'inference': 'paired record percentile bootstrap, 10000; two-sided exact McNemar'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run', 'analyze'], default='prepare')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        run(args.workers)
        analyze()
    else:
        analyze()


if __name__ == '__main__':
    main()
