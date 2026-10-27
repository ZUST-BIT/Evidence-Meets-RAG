"""Plan lawful article retrieval or fetch PDFs explicitly offered by PMC Cloud.

Default: print local manifest coverage, without network access. --resolve fetches
only official identifier/JSON metadata. --download additionally retrieves only
Creative Commons PDFs explicitly linked by PMC's current cloud metadata. There
is no publisher-page scraper, authentication, or access-control bypass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from knowledge_build_paths import ROOT, PAPERS_ROOT
from paper_fulltext_sources import MANIFEST, normalize_doi

ID_API = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com"
CC_CODES = {"cc0", "ccby", "ccbysa", "ccbynd", "ccbync", "ccbyncsa", "ccbyncnd"}


def request_bytes(url: str, limit: int = 4 * 1024 * 1024) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "EvidenceMeetsRAG/1.0"}), timeout=35) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Metadata response exceeds its bounded size")
    return data


def identify(records: list[dict], email: str | None) -> dict:
    resolved = {}
    groups: dict[str, list[str]] = {"doi": [], "pmid": [], "pmcid": []}
    for row in records:
        for kind in ("doi", "pmid", "pmcid"):
            if row.get(kind):
                groups[kind].append(str(row[kind]))
                break
    for kind, values in groups.items():
        values = list(dict.fromkeys(values))
        for start in range(0, len(values), 100):
            batch = values[start:start + 100]
            query = {"ids": ",".join(batch), "idtype": kind, "format": "json", "tool": "EvidenceMeetsRAG"}
            if email:
                query["email"] = email
            response = json.loads(request_bytes(ID_API + "?" + urlencode(query)))
            for result in response.get("records", []):
                requested = str(result.get("requested-id") or "").lower()
                resolved[(kind, requested)] = {key: result.get(key) for key in ("doi", "pmid", "pmcid", "live", "release-date", "errmsg")}
            time.sleep(0.4)
    return resolved


def cloud_pdf_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "s3" and parsed.netloc == "pmc-oa-opendata":
        return CLOUD + parsed.path + ("?" + parsed.query if parsed.query else "")
    if parsed.scheme == "https" and parsed.netloc == "pmc-oa-opendata.s3.amazonaws.com":
        return value
    raise ValueError("PMC metadata did not return an allowed official-cloud URL")


def cc_license(value: object) -> bool:
    code = re.sub(r"[^a-z]", "", str(value or "").lower())
    return code in CC_CODES


def cloud_plan(row: dict, resolved: dict) -> dict:
    item = {"target_path": row["target_path"], "doi": row.get("doi"), "expected_sha256": row["sha256"], "expected_md5": row.get("md5"), "expected_size_bytes": row["size_bytes"]}
    identifier = next(((kind, str(row[kind]).lower()) for kind in ("doi", "pmid", "pmcid") if row.get(kind)), None)
    info = resolved.get(identifier, {}) if identifier else {}
    pmcid = info.get("pmcid") or row.get("pmcid")
    item["resolved_identifiers"] = info
    if not pmcid or not re.fullmatch(r"PMC\d+", str(pmcid)):
        return {**item, "status": "no_unique_pmc_record", "doi_landing_url": row.get("doi_landing_url")}
    if info.get("live") is False or str(info.get("live")).lower() == "false":
        return {**item, "status": "pmc_embargo_or_not_live"}
    query = urlencode({"list-type": "2", "prefix": "metadata/" + str(pmcid) + ".", "max-keys": 1000})
    root = ET.fromstring(request_bytes(CLOUD + "/?" + query))
    keys = [node.text for node in root.findall(".//{*}Contents/{*}Key") if node.text and node.text.endswith(".json")]
    options = []
    for key in keys:
        metadata_url = CLOUD + "/" + quote(key, safe="/")
        metadata = json.loads(request_bytes(metadata_url))
        if row.get("doi") and normalize_doi(metadata.get("doi")) != normalize_doi(row["doi"]):
            continue
        openaccess = str(metadata.get("is_pmc_openaccess") or "").lower() in {"true", "yes", "1"}
        if not openaccess or not cc_license(metadata.get("license_code")) or not metadata.get("pdf_url"):
            continue
        pdf_url = cloud_pdf_url(metadata["pdf_url"])
        md5 = (parse_qs(urlparse(pdf_url).query).get("md5") or [None])[0]
        options.append({"pmcid": str(pmcid), "version": metadata.get("version"), "pmid": metadata.get("pmid"), "doi": metadata.get("doi"), "title": metadata.get("title"), "is_manuscript": metadata.get("is_manuscript"), "license_code": metadata.get("license_code"), "metadata_url": metadata_url, "pdf_url": pdf_url, "cloud_md5": md5, "matches_paper_md5": bool(md5 and row.get("md5") and md5.lower() == row["md5"].lower())})
    if not options:
        return {**item, "status": "no_cc_pdf_in_official_cloud", "pmcid": str(pmcid)}
    # Prefer exact historical bytes. A version number alone does not identify the
    # publisher version; all alternatives remain visible for review.
    selected = next((option for option in options if option["matches_paper_md5"]), None)
    if selected is None:
        selected = next((option for option in options if str(option["is_manuscript"]).lower() in {"false", "no", "0"}), options[0])
    return {**item, "status": "exact_paper_md5_available" if selected["matches_paper_md5"] else "lawful_pdf_available_bytes_unconfirmed", "selected": selected, "alternatives": options}


def download(item: dict, paper_root: Path, candidates: Path) -> dict:
    selected = item.get("selected")
    if not selected:
        return item
    target_relative = Path(item["target_path"]).relative_to("kg/papers-full")
    target = paper_root / target_relative
    if target.is_file():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest == item["expected_sha256"]:
            return {**item, "download_status": "already_matches_paper"}
        raise ValueError("Refusing to overwrite a different local source PDF: " + str(target_relative))
    with tempfile.TemporaryDirectory(prefix="official-pmc-pdf-") as directory:
        temp = Path(directory) / "article.pdf"
        sha, md5 = hashlib.sha256(), hashlib.md5()
        size = 0
        with urlopen(Request(selected["pdf_url"], headers={"User-Agent": "EvidenceMeetsRAG/1.0"}), timeout=60) as response, temp.open("wb") as stream:
            if urlparse(response.url).netloc != "pmc-oa-opendata.s3.amazonaws.com":
                raise ValueError("Unexpected redirect outside the official PMC cloud")
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 128 * 1024 * 1024:
                    raise ValueError("Article exceeds the per-file size bound")
                sha.update(block)
                md5.update(block)
                stream.write(block)
        if not temp.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Official-cloud response is not a PDF")
        if selected.get("cloud_md5") and md5.hexdigest().lower() != selected["cloud_md5"].lower():
            raise ValueError("Official cloud MD5 verification failed")
        exact = sha.hexdigest() == item["expected_sha256"] and size == item["expected_size_bytes"]
        output = target if exact else candidates / target_relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temp, output)
    return {**item, "download_status": "matches_paper" if exact else "candidate_only_different_historical_bytes", "downloaded_sha256": sha.hexdigest(), "downloaded_size_bytes": size, "saved_path": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--resolve", action="store_true", help="Query only official identifier and cloud JSON metadata")
    parser.add_argument("--download", action="store_true", help="Additionally download CC PDFs explicitly offered by current official cloud metadata")
    parser.add_argument("--path", action="append", help="Restrict to this target_path; may be repeated")
    parser.add_argument("--limit", type=int, help="Restrict metadata queries/downloads to the first N selected records")
    parser.add_argument("--email", help="Optional maintainer email requested by the NCBI ID API")
    parser.add_argument("--paper-root", type=Path, default=PAPERS_ROOT)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "outputs/fulltext_candidates")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/paper_fulltext_oa_plan.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [row for row in manifest["records"] if not args.path or row["target_path"] in args.path]
    if args.limit is not None:
        records = records[:args.limit]
    if not args.resolve and not args.download:
        print(json.dumps({"selected": len(records), "network_used": False, "manifest_summary": manifest["summary"]}, indent=2))
        return
    resolved = identify(records, args.email)
    def prepare(row: dict) -> dict:
        try:
            item = cloud_plan(row, resolved)
            return download(item, args.paper_root, args.candidate_dir) if args.download else item
        except Exception as error:
            return {"target_path": row["target_path"], "status": "metadata_or_download_error", "error_type": type(error).__name__, "http_status": error.code if isinstance(error, HTTPError) else None}
    with ThreadPoolExecutor(max_workers=3) as pool:
        items = list(pool.map(prepare, records))
    report = {"metadata_only": not args.download, "official_sources": [ID_API, "https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/", CLOUD + "/README.txt"], "records": items}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(items), "report": str(args.report), "full_text_download_requested": args.download, "statuses": {status: sum(item["status"] == status for item in items) for status in sorted({item["status"] for item in items})}}, indent=2))


if __name__ == "__main__":
    main()
