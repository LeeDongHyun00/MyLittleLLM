"""Audit pinned D01 PDF extraction and the public V12 retrieval path offline.

PDFs, page text, images and reconstructed indexes stay under a NEW outputs path.
This checks lexical preservation, not semantic truth, human gold or latest policy.
The historical Jeju and observed Dongseo bytes are separate source versions.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mylittlellm.retrieval import build_index, read_jsonl, retrieve

MANIFEST = ROOT / "data/development/d01/source_manifest.json"
FIXTURES = ROOT / "data/development/d01/clauses.json"
MAX_BYTES = 40 * 1024 * 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", text)


def anchors_present(text, anchors):
    normalized = compact(text)
    return all(compact(anchor) in normalized for anchor in anchors)


def validate_pdf(path, version):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("PDF exceeds 40 MiB")
    body = path.read_bytes()
    if not body.startswith(b"%PDF-"):
        raise ValueError("Input is not a PDF")
    if hashlib.sha256(body).hexdigest() != version["sha256"]:
        raise ValueError("Source SHA-256 mismatch; do not relabel another version")
    if len(body) != version["bytes"]:
        raise ValueError("Source byte count mismatch")


def validate_output(path):
    path = Path(path).resolve()
    outputs = (ROOT / "outputs").resolve()
    if not path.is_relative_to(outputs) or path == outputs:
        raise ValueError("Choose a new directory underneath repository outputs/")
    if path.exists():
        raise ValueError("Output already exists; choose a new run path")
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit(pdf, source_version, output, render=False):
    import pypdf

    started = time.monotonic()
    manifest = json.loads(MANIFEST.read_text())
    version = manifest["versions"][source_version]
    fixtures = json.loads(FIXTURES.read_text())
    validate_pdf(pdf, version)
    output = validate_output(output)
    reader = pypdf.PdfReader(pdf)
    if len(reader.pages) != version["pdf_pages"]:
        raise ValueError("PDF page count mismatch")
    # mkdir without exist_ok protects the run from races and overwrites.
    output.mkdir(parents=True)
    root = output / "reconstruction"
    for folder in ("data/raw", "data/extracted", "config"):
        (root / folder).mkdir(parents=True)
    raw = Path("data/raw") / (version["sha256"] + ".pdf")
    extracted = Path("data/extracted") / (version["sha256"] + ".jsonl")
    shutil.copyfile(pdf, root / raw)
    sid = version["source_id"]
    # Only policy pages + the FAQ boundary; do not need the full 123-page text.
    pages = [dict(source_id=sid, sha256=version["sha256"], pdf_page=n,
                  text=reader.pages[n-1].extract_text() or "") for n in range(1, 29)]
    (root / extracted).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                        for row in pages), encoding="utf-8")
    source = dict(source_id=sid, url=version["download_url"],
                  landing_url=version["landing_url"], **version["scope"])
    (root / "data/sources.jsonl").write_text(json.dumps(source, ensure_ascii=False) + "\n")
    snapshot = dict(source_id=sid, sha256=version["sha256"], status="retrieved",
                    raw_path=raw.as_posix(), extracted_path=extracted.as_posix(),
                    pdf_pages=len(reader.pages))
    (root / "data/retrieval_snapshot.jsonl").write_text(json.dumps(snapshot) + "\n")
    write_json(root / "config/corpus.json", {"sources": {sid: {
        "task": "housing", "institution": "한국장학재단", "page_ranges": [[1, 25]]}}})
    # Use production extraction-to-index and ranking code without modifying it.
    chunks = build_index(root)
    question = next(row for row in read_jsonl(ROOT / "data/evaluation/grounding_v2_scenarios.jsonl")
                    if row["id"] == "V12")
    query_scope = {key: question[key] for key in ("year", "task", "institution", "semester")}
    ranked = retrieve(chunks, question["question"], **query_scope, k=len(chunks))
    historical = next(row for row in read_json(ROOT / "reports/grounding_v2_input_audit.json")
                      if row["id"] == "V12")
    comparisons, page_results = [], []
    for n in (10, 11):
        plain = pages[n-1]["text"]
        layout = reader.pages[n-1].extract_text(extraction_mode="layout") or ""
        plain_path, layout_path = output / f"p{n}-plain.txt", output / f"p{n}-layout.txt"
        plain_path.write_text(plain, encoding="utf-8")
        layout_path.write_text(layout, encoding="utf-8")
        page_results.append(dict(pdf_page=n, plain_sha256=sha(plain_path), layout_sha256=sha(layout_path),
                                 plain_characters=len(plain), layout_characters=len(layout)))
        for clause in fixtures["clauses"]:
            if clause["pdf_page"] != n:
                continue
            comparisons.append(dict(
                clause_id=clause["id"], pdf_page=n,
                plain_anchors_present=anchors_present(plain, clause["anchors"]),
                layout_anchors_present=anchors_present(layout, clause["anchors"]),
                complete_anchor_chunks=[c["chunk_id"] for c in chunks
                                        if c["pdf_page"] == n and anchors_present(c["text"], clause["anchors"])]))
    rendered = []
    render_version = None
    if render:
        import fitz
        render_version = fitz.VersionBind
        with fitz.open(pdf) as document:
            for n in (10, 11, 26):
                image_path = output / f"p{n}.png"
                document[n-1].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(image_path)
                rendered.append(dict(pdf_page=n, sha256=sha(image_path), file=image_path.name))

    cap = next(c for c in comparisons if c["clause_id"] == "cap")
    cap_chunks = set(cap["complete_anchor_chunks"])
    cap_rank = next((i for i, c in enumerate(ranked, 1) if c["chunk_id"] in cap_chunks), None)
    # Pairing is stronger than finding the cap phrase alone, but still lexical.
    paired_anchors = [a for c in fixtures["clauses"] if c["id"] in ("cap", "first_month", "cost_scope")
                      for a in c["anchors"]]
    complete_context = [c["chunk_id"] for c in chunks if c["pdf_page"] == 10
                        and anchors_present(c["text"], paired_anchors)]
    relative_files = ["scripts/audit_d01_source.py", "mylittlellm/retrieval.py",
                      "data/development/d01/source_manifest.json", "data/development/d01/clauses.json",
                      "data/evaluation/grounding_v2_scenarios.jsonl", "reports/grounding_v2_input_audit.json"]
    summary = dict(
        schema_version=1, task_id="D01", cohort="public_regression", source_version=source_version,
        source_id=sid, source_sha256=version["sha256"], source_bytes=version["bytes"],
        fixture_source_sha256=fixtures["source_sha256"],
        fixture_bound_to_same_bytes=fixtures["source_sha256"] == version["sha256"],
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dependency_hashes={name: sha(ROOT / name) for name in relative_files},
        run_path=output.relative_to(ROOT).as_posix(),
        python_version=platform.python_version(), pypdf_version=pypdf.__version__,
        optional_pymupdf_render_version=render_version,
        pdf_pages=len(reader.pages), extracted_pages=28,
        indexed_chunks=len(chunks), indexed_pages=sorted({c["pdf_page"] for c in chunks}),
        source_sections=dict(legacy_allowed_pdf_pages=[1, 25], faq_boundary_pdf_page=26,
                             faq_marker_present="FAQ" in pages[25]["text"],
                             faq_pages_in_index=any(c["pdf_page"] >= 26 for c in chunks)),
        comparisons=comparisons, page_text_hashes=page_results,
        local_rendered_pages=rendered,
        lexical_context_chunks=complete_context,
        retrieval=dict(query_id="V12", scope=query_scope, k=2,
                       selected=[dict(chunk_id=c["chunk_id"], pdf_page=c["pdf_page"], score=c["score"])
                                 for c in ranked[:2]],
                       cap_page_rank=cap_rank, cap_in_top2=any(c["chunk_id"] in cap_chunks for c in ranked[:2]),
                       cap_present_in_extraction=cap["plain_anchors_present"],
                       cap_present_in_index=bool(cap_chunks),
                       ranked_page_metadata=[dict(rank=i, pdf_page=c["pdf_page"],
                                                  chunk_id=c["chunk_id"], score=c["score"])
                                             for i, c in enumerate(ranked, 1)]),
        historical_comparison=dict(
            source_sha256=manifest["versions"]["historical_jeju"]["sha256"],
            selected_pages=[s["pdf_page"] for s in historical["sources"]],
            same_source_bytes=version["sha256"] == manifest["versions"]["historical_jeju"]["sha256"],
            historical_full_index_reproduced=False,
            note="Only this source/scope reconstructed; do not assert original full-index equality."),
        stage_diagnosis=("cap_survives_extraction_and_index_but_misses_top2"
                         if cap["plain_anchors_present"] and cap_chunks and cap_rank and cap_rank > 2
                         else "inspect_comparisons"),
        elapsed_seconds=round(time.monotonic()-started, 4),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        model_calls=0, human_gold_approved=False,
        limitations=["Lexical checks do not validate table cells, footnote semantics or eligibility.",
                     "Visual confirmation is a separate AI review, not performed by this script.",
                     "Cross-version fixture probes need fresh visual review; coordinates are not transferable.",
                     "Distinct source versions cannot substitute for the missing historical PDF.",
                     "No generation, candidate retrieval change or accuracy measurement."])
    write_json(output / "audit.json", summary)
    print(json.dumps({k: summary[k] for k in ("source_version", "indexed_chunks", "stage_diagnosis",
                                             "elapsed_seconds", "peak_rss_kib")}, ensure_ascii=False))
    return summary


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--source-version", choices=("historical_jeju", "dongseo_republication"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true", help="Optional PyMuPDF page rendering; kept locally")
    args = parser.parse_args()
    try:
        audit(args.pdf, args.source_version, args.output, args.render)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(1, f"D01 audit failed: {exc}\n")


if __name__ == "__main__":
    main()
