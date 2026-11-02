"""Download the curated public sources; preserve content hashes and PDF pages.

Raw copyrighted documents stay local. Successful retrieval is not label review.
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 40 * 1024 * 1024


def collect(source, root):
    result = {"source_id": source["source_id"], "requested_url": source["url"],
              "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "MyLittleLLM-research/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(MAX_BYTES + 1)
            result.update(final_url=response.url, content_type=response.headers.get("Content-Type"))
        if len(body) > MAX_BYTES:
            raise ValueError("Download exceeds 40 MiB limit")
        if source["format"] == "pdf" and not body.startswith(b"%PDF-"):
            raise ValueError("Expected PDF but received another format")
        if source["format"] == "hwpx":
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                if "Contents/content.hpf" not in archive.namelist():
                    raise ValueError("Expected HWPX contents")
        digest = hashlib.sha256(body).hexdigest()
        raw = root / "data/raw" / (digest + "." + source["format"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(body)
        result.update(sha256=digest, bytes=len(body), raw_path=str(raw.relative_to(root)))
        if source["format"] == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(raw)
            output = root / "data/extracted" / (digest + ".jsonl")
            output.parent.mkdir(parents=True, exist_ok=True)
            low_text_pages = []
            with output.open("w", encoding="utf-8") as stream:
                for number, page in enumerate(reader.pages, 1):
                    extracted = page.extract_text() or ""
                    if len(extracted.strip()) < 40:
                        low_text_pages.append(number)
                    row = {"source_id": source["source_id"], "sha256": digest,
                           "pdf_page": number, "text": extracted}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            result.update(pdf_pages=len(reader.pages), extracted_path=str(output.relative_to(root)),
                          low_text_pages=low_text_pages,
                          text_review_required=True)
        result["status"] = "retrieved"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/sources.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument("--ids", nargs="*", help="Optional source IDs")
    args = parser.parse_args()
    sources = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.ids:
        unknown = set(args.ids) - {s["source_id"] for s in sources}
        if unknown:
            parser.error(f"Unknown source IDs: {sorted(unknown)}")
        sources = [s for s in sources if s["source_id"] in args.ids]
    for source in sources:
        if not re.fullmatch(r"[a-z0-9_-]+", source["source_id"]):
            parser.error("Unsafe source ID")
        if source["format"] not in {"pdf", "html", "hwpx"}:
            parser.error("Unsupported format")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool, args.output.open("w") as stream:
        for row in pool.map(lambda s: collect(s, ROOT), sources):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(row["source_id"], row["status"], row.get("pdf_pages", ""), row.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
