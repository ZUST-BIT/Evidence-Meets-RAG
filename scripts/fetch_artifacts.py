"""Download and verify the frozen inputs used by the paper.

The manifest pins both each archive and every extracted file by SHA-256.
This script uses only the Python standard library and never extracts links.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "release_artifacts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_file(root: Path, entry: dict) -> bool:
    path = root / entry["path"]
    return (path.is_file() and path.stat().st_size == entry["size_bytes"]
            and sha256(path) == entry["sha256"])


def safe_destination(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    target = (root / str(relative)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes the output directory: {name}")
    return target


def extract_verified(archive: Path, artifact: dict, root: Path) -> None:
    expected = {entry["path"]: entry for entry in artifact["files"]}
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            target = safe_destination(root, member.name)
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            entry = expected[member.name]
            if member.size != entry["size_bytes"]:
                raise ValueError(f"Size mismatch in archive: {member.name}")
            seen.add(member.name)
            if valid_file(root, entry):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            try:
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                if sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {member.name}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing archive members: {sorted(missing)}")


def download(artifact: dict, path: Path) -> None:
    if not artifact.get("url"):
        raise SystemExit(
            f"{artifact['name']} is checksum-pinned but has no public download in this release. "
            "Use the source-download and local-build route in docs/data.md, or a verified local archive with --offline."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(artifact["url"], headers={"User-Agent": "EvidenceMeetsRAG/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, 4 * 1024 * 1024)
        if temporary.stat().st_size != artifact["size_bytes"] or sha256(temporary) != artifact["sha256"]:
            raise ValueError(f"Archive verification failed: {artifact['name']}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="all", help="Manifest group or 'all'; see --list.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Extraction root (default: repository root).")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "artifacts" / "downloads")
    parser.add_argument("--list", action="store_true", help="List pinned download groups without network access.")
    parser.add_argument("--verify", action="store_true", help="Verify extracted files without downloading.")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded archives only.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    if args.list:
        for artifact in artifacts:
            status = "public download" if artifact.get("url") else "local archive; see source-build instructions"
            print(f"{artifact['group']}: {artifact['name']} ({artifact['size_bytes'] / 1_000_000:.1f} MB; {status})")
        return
    selected = [a for a in artifacts if args.group in {"all", a["group"]}]
    if not selected:
        parser.error(f"Unknown group {args.group!r}; use --list.")
    for artifact in selected:
        missing = [entry for entry in artifact["files"] if not valid_file(args.root, entry)]
        if not missing:
            print(f"[verified] {artifact['group']}: {len(artifact['files'])} files")
            continue
        if args.verify:
            raise SystemExit("Missing or modified inputs:\n" + "\n".join(e["path"] for e in missing))
        archive = args.archive_dir / artifact["name"]
        archive_ok = (archive.is_file() and archive.stat().st_size == artifact["size_bytes"]
                      and sha256(archive) == artifact["sha256"])
        if not archive_ok:
            if args.offline:
                raise SystemExit(f"Missing verified archive: {archive}")
            print(f"[download] {artifact['url']}", flush=True)
            try:
                download(artifact, archive)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise SystemExit(f"Download failed: {exc}. You can download the release asset manually to {archive}.") from exc
        print(f"[extract] {artifact['name']}", flush=True)
        extract_verified(archive, artifact, args.root)
        print(f"[verified] {artifact['group']}: {len(artifact['files'])} files")


if __name__ == "__main__":
    main()
