"""Restore bundled construction inputs and fetch checksum-pinned public files."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes the selected data root")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["benchmark", "evidencenet", "supporting", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--download", action="store_true", help="Fetch missing public files with fixed URLs and checksums.")
    parser.add_argument("--check", action="store_true", help="Read-only validation; do not restore or download.")
    args = parser.parse_args()
    groups = ["benchmark", "evidencenet", "supporting"] if args.only == "all" else [args.only]
    missing = []
    verified = []
    for group in groups:
        manifest = json.loads((ROOT / "data_sources" / group / "manifest.json").read_text())
        for item in manifest["entries"]:
            target = safe_path(args.data_root, item["target"])
            if target.exists():
                if sha256(target) != item["sha256"]:
                    raise SystemExit("Checksum mismatch: " + item["target"])
                verified.append(item["target"])
                continue
            if args.check:
                missing.append(item["target"])
                continue
            snapshot = safe_path(ROOT, item["snapshot"]) if item.get("snapshot") else None
            if snapshot and snapshot.exists():
                if sha256(snapshot) != item["snapshot_sha256"]:
                    raise SystemExit("Snapshot checksum mismatch: " + item["snapshot"])
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                with gzip.open(snapshot, "rb") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            elif item.get("url") and args.download:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                request = urllib.request.Request(item["url"], headers={"User-Agent": "EvidenceMeetsRAG/1.0"})
                with urllib.request.urlopen(request, timeout=300) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                missing.append(item["target"])
                continue
            if sha256(temporary) != item["sha256"]:
                temporary.unlink()
                raise SystemExit("Downloaded/restored checksum mismatch: " + item["target"])
            temporary.replace(target)
            verified.append(item["target"])
            print("verified " + item["target"], flush=True)
    print(json.dumps({"verified": len(verified), "missing": missing}, indent=2))
    if missing:
        raise SystemExit("Missing inputs; use --download for the pinned public resources. See docs/data.md.")


if __name__ == "__main__":
    main()
