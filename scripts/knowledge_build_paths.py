"""Portable input locations for the unchanged knowledge-construction adapters.

Relative environment values are resolved against this repository, not the shell's
working directory. Large source datasets are supplied separately by the user.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configured_path(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


RAW_ROOT = configured_path("EVIDENCEMEETSRAG_RAW_ROOT", ROOT / "dataset" / "raw")
LINKEDOMICS_ROOT = configured_path("EVIDENCEMEETSRAG_LINKEDOMICS_ROOT", RAW_ROOT / "LinkedOmics")
DEPMAP_ROOT = configured_path("EVIDENCEMEETSRAG_DEPMAP_ROOT", RAW_ROOT / "DepMap")
PUBMED_ROOT = configured_path("EVIDENCEMEETSRAG_PUBMED_ROOT", ROOT / "kg" / "pubmed-evidence-kg")
PAPERS_ROOT = configured_path("EVIDENCEMEETSRAG_PAPERS_ROOT", ROOT / "kg" / "papers-full")
HGNC_PATH = configured_path("EVIDENCEMEETSRAG_HGNC_PATH", ROOT / "kg" / "normalization" / "raw" / "hgnc_complete_set.txt")
TARKG_NODES_PATH = configured_path("EVIDENCEMEETSRAG_TARKG_NODES_PATH", ROOT / "kg" / "TarKG" / "TarKG_nodes.csv")
PROCESSED_DIR = configured_path("EVIDENCEMEETSRAG_PROCESSED_ROOT", ROOT / "dataset" / "processed")


def source_relpath(path: Path) -> str:
    """Use logical source paths in provenance and IDs, independent of a username.

    Raw document IDs historically hashed machine-specific absolute paths. Portable
    rebuilds use source-relative paths instead; no published snapshot is rewritten.
    """
    path = path.resolve()
    for directory, prefix in (
        (LINKEDOMICS_ROOT, Path("LinkedOmics")),
        (DEPMAP_ROOT, Path("DepMap")),
        (PAPERS_ROOT, Path("kg/papers-full")),
        (PUBMED_ROOT, Path("kg/pubmed-evidence-kg")),
    ):
        try:
            return str(prefix / path.relative_to(directory))
        except ValueError:
            pass
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name
