"""Single isolated, no-API timing trial for the paper cost profile."""
import argparse
import json
import time
from paper_paths import ROOT, BENCHMARK, read_rows
from rag_eval_lib import load_prepared_corpus, set_rag_profile, set_stage_config
from run_rag_evaluation import retrieve_docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    set_rag_profile("tuned"); set_stage_config("full", "structured", "grounded")
    started = time.perf_counter()
    corpus = load_prepared_corpus(ROOT/"dataset/rag/knowledge_sources", "multi_cancer", args.source, use_cache=False)
    loaded = time.perf_counter()
    for record in read_rows(BENCHMARK)[:args.limit]: retrieve_docs(corpus, "naive_rag", record, 8)
    print(json.dumps({"limit": args.limit, "load_seconds": loaded-started,
        "total_seconds": time.perf_counter()-started, "units": len(corpus.docs), "edges": len(corpus.edges)}))
