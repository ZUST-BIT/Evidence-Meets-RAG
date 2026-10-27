"""Verify locally reconstructed retrieval views against the paper snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-dir", type=Path,
                        default=ROOT / "dataset/rag/knowledge_sources/multi_cancer")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/release_artifacts.json").read_text())
    expected = next(item for item in manifest["artifacts"] if item["group"] == "retrieval")
    results = []
    for entry in expected["files"]:
        path = args.knowledge_dir / Path(entry["path"]).name
        result = {"file": path.name, "matches_paper_snapshot": False}
        if not path.is_file():
            result["status"] = "missing"
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            result.update(size_bytes=path.stat().st_size, sha256=digest.hexdigest())
            result["matches_paper_snapshot"] = (
                result["size_bytes"] == entry["size_bytes"]
                and result["sha256"] == entry["sha256"]
            )
            result["status"] = "verified" if result["matches_paper_snapshot"] else "different_source_snapshot"
        results.append(result)
    passed = all(item["matches_paper_snapshot"] for item in results)
    print(json.dumps({"passed": passed, "files": results}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
