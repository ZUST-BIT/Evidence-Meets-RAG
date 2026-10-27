"""Fetch fixed official GDSC, TarKG or LinkedOmics native source inputs.

Default is a read-only plan. Downloads require --download, support partial-file
resumption, and verify the publisher's size/ETag plus any pinned SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import ssl
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TARKG_URL = "https://tarkg.ddtmlab.org/download/KG_Data/TarKG_nodes.csv"
TARKG_SHA256 = "9e0e3743507c562c7e97d676a7f5227202ed18e1ae8de960cdec61edfea6f9d6"


class NoLegacyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("Redirect rejected for the fixed legacy TarKG URL: " + new_url)


def hashes(path: Path) -> dict:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
            size += len(block)
    return {"bytes": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def verify(path: Path, item: dict) -> dict:
    actual = hashes(path)
    if actual["bytes"] != item["bytes"]:
        raise ValueError("File size differs from the fixed publisher object: " + item["target"])
    for kind in ("md5", "sha256"):
        if item.get(kind) and actual[kind] != item[kind]:
            raise ValueError(kind + " mismatch: " + item["target"])
    etag = item.get("publisher_etag", "")
    if re.fullmatch(r"[0-9a-f]{32}", etag) and actual["md5"] != etag:
        raise ValueError("MD5 differs from the recorded single-part publisher ETag: " + item["target"])
    return actual


def fetch(target: Path, item: dict, timeout: float, allow_legacy_tarkg_tls: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["bytes"]:
        raise ValueError("Partial file exceeds expected size: " + str(partial))
    if offset < item["bytes"]:
        headers = {"User-Agent": "EvidenceMeetsRAG/1.0"}
        if offset:
            headers.update({"Range": f"bytes={offset}-", "If-Range": '"' + item["publisher_etag"] + '"'})
        request = urllib.request.Request(item["url"], headers=headers)
        if allow_legacy_tarkg_tls and item["url"] == TARKG_URL:
            if item.get("sha256") != TARKG_SHA256:
                raise ValueError("Legacy TarKG TLS requires the fixed paper input SHA-256")
            # Explicit exception for this single credential-free static URL.
            # No global SSL setting is changed and redirects are rejected.
            context = ssl._create_unverified_context()
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), NoLegacyRedirect())
            response_context = opener.open(request, timeout=timeout)
        else:
            response_context = urllib.request.urlopen(request, timeout=timeout)
        with response_context as response:
            append = offset > 0 and response.status == 206
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected partial response: " + content_range)
            received_etag = response.headers.get("ETag", "").strip('"')
            if received_etag and received_etag != item["publisher_etag"]:
                raise ValueError("The fixed publisher object has changed: " + item["url"])
            with partial.open("ab" if append else "wb") as output:
                while block := response.read(64 * 1024):
                    output.write(block)
    verify(partial, item)
    partial.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--source", choices=["gdsc", "tarkg", "linkedomics", "all"], default="gdsc")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--file", action="append", help="Select source basenames within --source.")
    parser.add_argument("--local-file", type=Path, help="Verify and install a manually obtained input; requires selection of exactly one file.")
    parser.add_argument("--allow-legacy-tarkg-tls", action="store_true", help="Opt in only for the fixed TarKG URL with its legacy certificate; redirects are rejected and the pinned SHA-256 must match before installation.")
    parser.add_argument("--timeout", type=float, default=180, help="Socket timeout; rerun to resume a partial download.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data_sources/native/manifest.json").read_text())
    selected = [item for item in manifest["entries"] if (args.source == "all" or item["source"] == args.source)
                and (not args.file or Path(item["target"]).name in args.file)]
    if args.file and len(selected) != len(set(args.file)):
        parser.error("Unknown or duplicate --file; inspect data_sources/native/manifest.json")
    if args.local_file and len(selected) != 1:
        parser.error("--local-file requires exactly one selected source file")
    if sum(bool(value) for value in (args.check, args.download, args.local_file)) > 1:
        parser.error("Select --check, --download or --local-file")
    if not args.check and not args.download and not args.local_file:
        terms = list(dict.fromkeys(item.get("source_description_url") or manifest["source_terms_url"] for item in selected))
        print(json.dumps({"source_notes": terms, "files": selected}, indent=2))
        return
    missing, receipts = [], []
    root = args.data_root.resolve()
    for item in selected:
        target = (root / item["target"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Source path escapes --data-root")
        if args.local_file:
            verify(args.local_file, item)
            if args.local_file.resolve() != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                shutil.copyfile(args.local_file, temporary)
                verify(temporary, item)
                temporary.replace(target)
        if not target.exists() and args.download:
            print("fetching " + item["target"], flush=True)
            fetch(target, item, args.timeout, args.allow_legacy_tarkg_tls)
        if not target.exists():
            missing.append(item["target"])
            continue
        actual = verify(target, item)
        receipts.append({"path": item["target"], "url": item["url"], **actual})
    print(json.dumps({"verified": receipts, "missing": missing}, indent=2))
    if missing:
        raise SystemExit(1)
    if args.download or args.local_file:
        receipt_path = root / "dataset/raw/acquisition_receipts/native_sources.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"files": []}
        by_path = {item["path"]: item for item in previous["files"]}
        by_path.update({item["path"]: item for item in receipts})
        receipt_path.write_text(json.dumps({"files": list(by_path.values())}, indent=2) + "\n")


if __name__ == "__main__":
    main()
