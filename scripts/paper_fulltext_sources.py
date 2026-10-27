"""Inventory or verify the exact PDFs used by the native literature adapter.

The manifest contains identifiers and hashes, never article full text. Building
it reads PDF metadata and the first two pages (up to six for cover-page files)
locally; checking uses only stdlib.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import quote, unquote

from knowledge_build_paths import ROOT, PAPERS_ROOT, PUBMED_ROOT

MANIFEST = ROOT / "configs/paper_fulltext_sources.json"
FAMILIES = {"HCC": "hcc_papers", "CRC": "crc_papers"}
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_doi(value: object) -> str | None:
    if not value:
        return None
    match = DOI_PATTERN.search(unquote(str(value)))
    if not match:
        return None
    doi = match.group(0).rstrip(".,;:")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    return doi.lower()


def unique(values: list) -> list:
    return list(dict.fromkeys(value for value in values if value))


def filename(value: str) -> str:
    return unicodedata.normalize("NFC", Path(value).name)


def source_index(source_root: Path) -> tuple[dict, dict]:
    by_file: dict = defaultdict(list)
    by_doi: dict = defaultdict(list)
    for family in FAMILIES:
        data = json.loads((source_root / family.lower() / "evidence_nodes.json").read_text())
        seen = set()
        for node in data.values():
            source = node.get("source") or {}
            entry = {key: source.get(key) for key in ("pdf_path", "doi", "pmid", "pmcid", "title", "journal", "year", "url", "download_url")}
            key = json.dumps(entry, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            if entry["pdf_path"]:
                by_file[(family, filename(entry["pdf_path"]))].append(entry)
            doi = normalize_doi(entry["doi"])
            if doi:
                by_doi[doi].append(entry)
    return by_file, by_doi


def inspect_pdf(item: tuple[str, Path]) -> dict:
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    family, path = item
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    result = {"family": family, "target_path": str(Path("kg/papers-full") / FAMILIES[family] / path.name), "size_bytes": path.stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}
    try:
        reader = PdfReader(path, strict=False)
        info = reader.metadata or {}
        info_title = str(info.get("/Title") or "").strip()
        meta_text = " ".join(str(value) for value in info.values())
        selected_pages = list(reader.pages[:2])
        page_text = "\n".join(page.extract_text() or "" for page in selected_pages)
        uris = []
        for page in selected_pages:
            for reference in page.get("/Annots") or []:
                annotation = reference.get_object()
                action = annotation.get("/A") or {}
                if action.get("/URI"):
                    uris.append(str(action["/URI"]))
        doi_uris = unique([normalize_doi(uri) for uri in uris if "doi.org/" in uri.lower()])
        doi_text = unique([normalize_doi(match.group(0)) for match in DOI_PATTERN.finditer(page_text)])
        doi_metadata = unique([normalize_doi(match.group(0)) for match in DOI_PATTERN.finditer(meta_text)])
        if not doi_uris and not doi_text and not doi_metadata:
            extra_pages = list(reader.pages[2:6])
            selected_pages.extend(extra_pages)
            page_text += "\n" + "\n".join(page.extract_text() or "" for page in extra_pages)
            for page in extra_pages:
                for reference in page.get("/Annots") or []:
                    action = reference.get_object().get("/A") or {}
                    if action.get("/URI"):
                        uris.append(str(action["/URI"]))
            doi_uris = unique([normalize_doi(uri) for uri in uris if "doi.org/" in uri.lower()])
            doi_text = unique([normalize_doi(match.group(0)) for match in DOI_PATTERN.finditer(page_text)])
        all_identifiers = " ".join([meta_text, page_text, *uris])
        pmcids = unique(re.findall(r"\bPMC\d{4,10}\b", all_identifiers, re.IGNORECASE))
        pmids = unique(re.findall(r"\bPMID\s*:?\s*(\d{5,10})\b", all_identifiers, re.IGNORECASE))
        result.update({"page_count": len(reader.pages), "identifier_page_count": len(selected_pages), "pdf_metadata_title": info_title if len(info_title) > 8 and not re.match(r"^(untitled|PII:|Microsoft|Elsevier)", info_title, re.IGNORECASE) else None, "pdf_doi_uri_candidates": doi_uris, "pdf_doi_text_candidates": doi_text, "pdf_doi_metadata_candidates": doi_metadata, "pdf_pmcid_candidates": [value.upper() for value in pmcids], "pdf_pmid_candidates": pmids})
    except Exception as error:
        result["pdf_inspection_error"] = type(error).__name__ + ": " + str(error)[:160]
    return result


def annotate(result: dict, by_file: dict, by_doi: dict) -> dict:
    sources = by_file.get((result["family"], filename(result["target_path"])), [])
    exact = bool(sources)
    source_dois = unique([normalize_doi(item.get("doi")) for item in sources])
    candidates = unique(result.get("pdf_doi_uri_candidates", []) + result.get("pdf_doi_metadata_candidates", []) + result.get("pdf_doi_text_candidates", []))
    matched = unique([doi for doi in candidates if doi in by_doi])
    if len(source_dois) == 1:
        doi = source_dois[0]
        doi_basis = "evidencenet_exact_filename"
    elif len(matched) == 1:
        doi = matched[0]
        doi_basis = "pdf_identifier_matches_evidencenet"
        if not sources:
            sources = by_doi[doi]
    else:
        preferred = unique(result.get("pdf_doi_uri_candidates", []) or result.get("pdf_doi_metadata_candidates", []) or result.get("pdf_doi_text_candidates", []))
        doi = preferred[0] if len(preferred) == 1 else None
        doi_basis = "pdf_self_identifier" if doi else "unresolved_or_ambiguous"
    titles = unique([str(item["title"]).strip() for item in sources if item.get("title")])
    pmids = unique([str(item["pmid"]) for item in sources if item.get("pmid")]) or result.get("pdf_pmid_candidates", [])
    pmcids = unique([str(item["pmcid"]).upper() for item in sources if item.get("pmcid")]) or result.get("pdf_pmcid_candidates", [])
    result.update({"doi": doi, "doi_basis": doi_basis, "pmid": pmids[0] if len(pmids) == 1 else None, "pmcid": pmcids[0] if len(pmcids) == 1 else None, "title": titles[0] if len(titles) == 1 else result.get("pdf_metadata_title"), "title_basis": "evidencenet_metadata" if len(titles) == 1 else "pdf_metadata" if result.get("pdf_metadata_title") else "unresolved", "metadata_match": "exact_source_filename" if exact else "pdf_doi_matches_source" if sources else "none", "evidencenet_source_paths": unique([item["pdf_path"] for item in sources if item.get("pdf_path")]), "original_download_url": None, "doi_landing_url": "https://doi.org/" + quote(doi, safe="/():") if doi else None, "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/" + str(pmids[0]) + "/" if len(pmids) == 1 else None, "pmc_landing_url": "https://pmc.ncbi.nlm.nih.gov/articles/" + pmcids[0] + "/" if len(pmcids) == 1 else None, "oa_download_status": "not_checked", "redistribution_permission": "not_inferred_from_identifier_or_local_possession"})
    if len(source_dois) == 1 and candidates and source_dois[0] not in candidates:
        result["identifier_review_note"] = "Source-metadata DOI was not found among PDF first-two-page candidates; retain the candidates for review."
    if not doi:
        result["identifier_review_note"] = "No unique article DOI established from source metadata and local PDF identifiers."
    return result


def build_manifest(paper_root: Path, source_root: Path, workers: int) -> dict:
    by_file, by_doi = source_index(source_root)
    items = [(family, path) for family, directory in FAMILIES.items() for path in sorted((paper_root / directory).glob("*.pdf"))]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(inspect_pdf, items), 1):
            records.append(annotate(result, by_file, by_doi))
            if index % 100 == 0:
                print(f"[fulltext inventory] {index}/{len(items)}", file=sys.stderr, flush=True)
    summary = {"file_count": len(records), "total_size_bytes": sum(row["size_bytes"] for row in records), "families": dict(Counter(row["family"] for row in records)), "metadata_match": dict(Counter(row["metadata_match"] for row in records)), "with_doi": sum(bool(row["doi"]) for row in records), "with_pmid": sum(bool(row["pmid"]) for row in records), "with_pmcid": sum(bool(row["pmcid"]) for row in records), "with_title": sum(bool(row["title"]) for row in records), "pdf_inspection_errors": sum("pdf_inspection_error" in row for row in records), "identifier_review_count": sum("identifier_review_note" in row for row in records)}
    return {"format_version": 1, "scope": "Every *.pdf in the HCC and CRC directories used by build_raw_multisource_corpus.build_paper_corpus, in its sorted input order.", "source_metadata": ["kg/pubmed-evidence-kg/hcc/evidence_nodes.json", "kg/pubmed-evidence-kg/crc/evidence_nodes.json"], "provenance_note": "The original PDF download log was not retained. DOI/PMC/PubMed URLs identify lawful publisher/archive entry points, not a verified historic download URL or redistribution license. No full-text PDF is included in this manifest.", "summary": summary, "unresolved_identifier_paths": [row["target_path"] for row in records if not row["doi"] and not row["pmid"] and not row["pmcid"]], "records": records}


def verify_manifest(manifest: dict, paper_root: Path, *, hashes: bool = True) -> dict:
    missing, mismatched = [], []
    expected = set()
    for row in manifest["records"]:
        relative = Path(row["target_path"]).relative_to("kg/papers-full")
        expected.add(str(relative))
        path = paper_root / relative
        if not path.is_file():
            missing.append(row["target_path"])
        elif path.stat().st_size != row["size_bytes"] or (hashes and sha256(path) != row["sha256"]):
            mismatched.append(row["target_path"])
    actual = {str(path.relative_to(paper_root)) for directory in FAMILIES.values() for path in (paper_root / directory).glob("*.pdf")}
    unexpected = sorted(actual - expected)
    return {"passed": not missing and not mismatched and not unexpected, "expected_file_count": len(expected), "missing_count": len(missing), "mismatched_count": len(mismatched), "unexpected_count": len(unexpected), "missing": missing, "mismatched": mismatched, "unexpected": unexpected, "hashes_checked": hashes}


def refresh_summary(data: dict) -> None:
    records = data["records"]
    data["summary"].update({"with_doi": sum(bool(row.get("doi")) for row in records), "with_pmid": sum(bool(row.get("pmid")) for row in records), "with_pmcid": sum(bool(row.get("pmcid")) for row in records), "with_title": sum(bool(row.get("title")) for row in records), "with_source_url": sum(any(row.get(key) for key in ("doi_landing_url", "pubmed_url", "pmc_landing_url", "publisher_landing_url")) for row in records), "identifier_review_count": sum("identifier_review_note" in row for row in records)})
    data["unresolved_identifier_paths"] = [row["target_path"] for row in records if not any(row.get(key) for key in ("doi_landing_url", "pubmed_url", "pmc_landing_url", "publisher_landing_url"))]


def retain_verified_metadata(data: dict, previous: dict) -> None:
    """Keep official metadata enrichment only when the local bytes still match."""
    old = {row["target_path"]: row for row in previous["records"]}
    fields = ("doi", "doi_basis", "pmid", "pmcid", "title", "title_basis", "doi_landing_url", "pubmed_url", "pmc_landing_url", "publisher_landing_url", "identifier_verification", "official_identifier_resolution", "evidencenet_reported_identity")
    for row in data["records"]:
        prior = old.get(row["target_path"], {})
        if prior.get("sha256") != row["sha256"]:
            continue
        if not (prior.get("identifier_verification") or prior.get("official_identifier_resolution")):
            continue
        for field in fields:
            if prior.get(field) is not None:
                row[field] = prior[field]
            elif prior.get("identifier_verification") and field in ("pmid", "pmcid", "pubmed_url", "pmc_landing_url", "official_identifier_resolution"):
                row.pop(field, None)
        if prior.get("identifier_verification"):
            row.pop("identifier_review_note", None)
    refresh_summary(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build", action="store_true", help="Build the manifest from original local PDFs and EvidenceNet JSON; no download")
    action.add_argument("--check", action="store_true", help="Check all expected PDFs, sizes, SHA-256, and unexpected extra files")
    action.add_argument("--unresolved", action="store_true", help="List source paths still lacking a unique verified entry URL")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--paper-root", type=Path, default=PAPERS_ROOT)
    parser.add_argument("--source-root", type=Path, default=PUBMED_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.build:
        data = build_manifest(args.paper_root, args.source_root, args.workers)
        if args.manifest.is_file():
            retain_verified_metadata(data, json.loads(args.manifest.read_text()))
        refresh_summary(data)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(data["summary"], indent=2))
    else:
        data = json.loads(args.manifest.read_text())
        if args.unresolved:
            print(json.dumps(data["unresolved_identifier_paths"], ensure_ascii=False, indent=2))
        elif args.check:
            report = verify_manifest(data, args.paper_root)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["passed"]:
                raise SystemExit(2)
        else:
            print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
