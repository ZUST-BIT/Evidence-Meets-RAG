"""Update Figure 1's existing editable objects from the 571-question metrics.

Uses only the standard library for PPTX patching. PDF export optionally uses
LibreOffice/soffice plus pypdf and preserves the original publication crop box.
The source PPTX is never overwritten. Figure 2 is an unchanged concept diagram.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from html import escape
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SLIDE = "ppt/slides/slide1.xml"
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
SHAPE = re.compile(r"<p:sp(?:\s[^>]*)?>.*?</p:sp>", re.DOTALL)


def resolved(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def display(value: float, places: int = 2) -> str:
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def shape_id(xml: str) -> str:
    match = re.search(r'<p:cNvPr\b[^>]*\bid="(\d+)"', xml)
    if not match:
        raise ValueError("Template shape has no native shape ID")
    return match.group(1)


def geometry(xml: str) -> tuple[int, int]:
    element = ET.fromstring(xml.replace("<p:sp>", '<p:sp xmlns:p="' + NS["p"] + '" xmlns:a="' + NS["a"] + '" xmlns:a16="http://schemas.microsoft.com/office/drawing/2014/main">', 1))
    transform = element.find("p:spPr/a:xfrm", NS)
    if transform is None:
        raise ValueError("Expected a directly editable shape transform")
    return int(transform.find("a:off", NS).get("x")), int(transform.find("a:ext", NS).get("cx"))


def replace_integer(xml: str, tag: str, attribute: str, value: int) -> str:
    result, count = re.subn(r'(<a:' + tag + r'\b[^>]*\b' + attribute + r'=")\d+(\")', lambda m: m.group(1) + str(value) + m.group(2), xml, count=1)
    if count != 1:
        raise ValueError(f"Cannot locate {tag}.{attribute} in the selected native object")
    return result


def replace_text(xml: str, value: str) -> str:
    if len(re.findall(r"<a:t(?:\s[^>]*)?>", xml)) != 1:
        raise ValueError("Numeric label is no longer a single editable text run")
    return re.sub(r"(<a:t(?:\s[^>]*)?>).*?(</a:t>)", lambda m: m.group(1) + escape(value) + m.group(2), xml, count=1, flags=re.DOTALL)


def update_pptx(template: Path, output: Path, config: dict) -> dict:
    if config["benchmark_count"] != 571:
        raise ValueError("This publication figure uses the complete 571-question benchmark")
    if template.resolve() == output.resolve():
        raise ValueError("Use a separate output file; the editable source is never overwritten")
    with ZipFile(template) as source:
        presentation = ET.fromstring(source.read("ppt/presentation.xml"))
        size = presentation.find("p:sldSz", NS)
        if size is None or [int(size.get("cx")), int(size.get("cy"))] != [12192000, 6858000]:
            raise ValueError("Use the original Figure 1 slide dimensions")
        xml = source.read(SLIDE).decode("utf-8")
        shapes = {shape_id(match.group(0)): match.group(0) for match in SHAPE.finditer(xml)}
        changes: dict[str, str] = {}
        labels = {}
        for metric in ("component_recall", "accuracy"):
            values = config["metrics"][metric]
            if any(not 0 <= float(values[name]) <= 1 for name in ("document", "semantic", "evidence")):
                raise ValueError("Metrics must be proportions between 0 and 1")
            scale = config["template_scale"][metric]
            units = Decimal(scale["reference_width_emu"]) / Decimal(str(scale["reference_value"]))
            for system in ("document", "semantic", "evidence"):
                reference = config["shape_map"][metric][system]
                bar = shapes[reference["bar"]]
                label = shapes[reference["label"]]
                x, old_width = geometry(bar)
                label_x, _ = geometry(label)
                gap = label_x - x - old_width
                width = max(1, int((units * Decimal(str(values[system]))).quantize(Decimal(1), rounding=ROUND_HALF_UP)))
                changes[reference["bar"]] = replace_integer(bar, "ext", "cx", width)
                changes[reference["label"]] = replace_text(replace_integer(label, "off", "x", x + width + gap), display(values[system]))
                labels[metric + "." + system] = display(values[system])
            baseline = Decimal(str(values[config["relative_gain_baseline"]]))
            if baseline <= 0:
                raise ValueError("The relative-gain baseline must be positive")
            gain = (Decimal(str(values["evidence"])) / baseline - 1) * 100
            gain_text = "+" + display(gain, 0) + "%"
            identifier = config["shape_map"][metric]["gain_label"]
            changes[identifier] = replace_text(shapes[identifier], gain_text)
            labels[metric + ".gain"] = gain_text
        updated = SHAPE.sub(lambda match: changes.get(shape_id(match.group(0)), match.group(0)), xml)
        parsed = ET.fromstring(updated)
        if len(parsed.findall(".//p:sp", NS)) != len(ET.fromstring(xml).findall(".//p:sp", NS)):
            raise ValueError("Unexpected shape-count change")
        for identifier, original in shapes.items():
            if identifier not in changes and original not in updated:
                raise ValueError("An unrelated template object changed")
        output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output, "w") as target:
            for item in source.infolist():
                if item.filename == "docProps/thumbnail.jpeg":
                    continue  # Do not retain an old numeric preview in the updated deck.
                data = source.read(item.filename)
                if item.filename == SLIDE:
                    data = updated.encode("utf-8")
                elif item.filename == "_rels/.rels":
                    data = re.sub(rb'<Relationship\b[^>]*Type="[^"]*/thumbnail"[^>]*/>', b"", data)
                target.writestr(item, data)
    with ZipFile(output) as check:
        if check.testzip() is not None:
            raise ValueError("Invalid output PPTX archive")
    return {"benchmark_count": config["benchmark_count"], "labels": labels, "modified_native_shapes": len(changes), "pptx": str(output)}


def export_pdf(pptx: Path, directory: Path, config: dict, executable: str | None) -> Path:
    converter = executable or shutil.which("soffice") or shutil.which("libreoffice")
    if not converter:
        raise RuntimeError("Install LibreOffice or pass --soffice to export PDF; PPTX updating needs no office application")
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="concept-figure-export-") as temp:
        work = Path(temp)
        subprocess.run([converter, "-env:UserInstallation=" + (work / "profile").as_uri(), "--headless", "--convert-to", "pdf", "--outdir", str(work), str(pptx)], check=True, timeout=120)
        source = work / (pptx.stem + ".pdf")
        if not source.is_file():
            raise RuntimeError("LibreOffice did not produce the requested PDF")
        reader = PdfReader(source)
        if len(reader.pages) != 1:
            raise ValueError("Figure 1 must remain a single slide")
        writer = PdfWriter()
        page = reader.pages[0]
        page.cropbox = RectangleObject(config["pdf_cropbox_points"])
        writer.add_page(page)
        output = directory / "intro-motivation.pdf"
        with output.open("wb") as stream:
            writer.write(stream)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path("figures/source/intro-motivation.pptx"))
    parser.add_argument("--config", type=Path, default=Path("figures/figure1_values.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/generated/intro-motivation.pptx"))
    parser.add_argument("--metrics-dir", type=Path, help="Read fresh retrieval_571.json and main_and_ablation.json from this directory instead of the pinned metrics")
    parser.add_argument("--export-pdf", action="store_true")
    parser.add_argument("--pdf-dir", type=Path, default=Path("figures/generated"))
    parser.add_argument("--soffice", help="Optional path to the LibreOffice/soffice executable")
    args = parser.parse_args()
    config = json.loads(resolved(args.config).read_text())
    if args.metrics_dir:
        directory = resolved(args.metrics_dir)
        retrieval_payload = json.loads((directory / "retrieval_571.json").read_text())
        retrieval = retrieval_payload["sources"]
        main_results = json.loads((directory / "main_and_ablation.json").read_text())["vanilla"]
        if retrieval_payload["count"] != 571 or any(retrieval[name]["count"] != 571 for name in ("doc_corpus", "semantic_kg", "evidence_kg")):
            raise ValueError("Figure 1 retrieval diagnostics must cover all 571 questions")
        if any(main_results[name]["count"] != 571 for name in ("document", "semantic", "evidence")):
            raise ValueError("Figure 1 requires the same complete 571-question evaluation")
        config["metrics"] = {"component_recall": {name: retrieval[source]["component_recall"] for name, source in (("document", "doc_corpus"), ("semantic", "semantic_kg"), ("evidence", "evidence_kg"))}, "accuracy": {name: main_results[name]["accuracy"] for name in ("document", "semantic", "evidence")}}
    report = update_pptx(resolved(args.template), resolved(args.output), config)
    if args.export_pdf:
        report["pdf"] = str(export_pdf(resolved(args.output), resolved(args.pdf_dir), config, args.soffice))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
