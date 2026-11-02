"""Small CPU-only BM25 retriever with Korean character n-grams and strict scope."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def ambiguous_comparison(text):
    """Flat PDF extraction loses column-year associations; quarantine these pages."""
    years = set(re.findall(r"20\d{2}\s*학년도", text))
    return len(years) > 1 and bool(re.search(r"변경|전년\s*대비", text))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def terms(text):
    words = re.findall(r"[가-힣]+|[a-zA-Z]+|\d+(?:\.\d+)?%?", text.lower())
    result = list(words)
    for word in words:
        if re.fullmatch(r"[가-힣]+", word):
            result.extend(word[i:i+n] for n in [2, 3] for i in range(len(word)-n+1))
    return result


def build_index(root=ROOT):
    manifest = {s["source_id"]: s for s in read_jsonl(root / "data/sources.jsonl")}
    snapshots = {s["source_id"]: s for s in read_jsonl(root / "data/retrieval_snapshot.jsonl")}
    config = json.loads((root / "config/corpus.json").read_text())["sources"]
    rows, quarantined = [], []
    for source_id, policy in config.items():
        source, snapshot = manifest[source_id], snapshots[source_id]
        if snapshot["status"] != "retrieved":
            raise ValueError(f"Missing source: {source_id}")
        path = root / snapshot["extracted_path"]
        # Re-extraction is necessary if the original document changes.
        if hashlib.sha256((root / snapshot["raw_path"]).read_bytes()).hexdigest() != snapshot["sha256"]:
            raise ValueError(f"Original document hash mismatch: {source_id}")
        for page in read_jsonl(path):
            if page["source_id"] != source_id or page["sha256"] != snapshot["sha256"]:
                raise ValueError(f"Extraction provenance mismatch: {source_id}")
            ranges = policy.get("page_ranges", [[1, snapshot["pdf_pages"]]])
            if not any(a <= page["pdf_page"] <= b for a, b in ranges):
                continue
            text = re.sub(r"\s+", " ", page["text"]).strip()
            if ambiguous_comparison(text):
                quarantined.append({"source_id": source_id, "pdf_page": page["pdf_page"],
                                    "reason": "multi_year_comparison_requires_layout_review"})
                continue
            if len(text) < 40:
                continue
            for start in range(0, len(text), 740):
                content = text[start:start+900]
                if len(content) < 40:
                    continue
                rows.append({"chunk_id": f"{source_id}:p{page['pdf_page']}:c{start}",
                             "source_id": source_id, "sha256": snapshot["sha256"],
                             "pdf_page": page["pdf_page"], "year": source["academic_year"],
                             "semester": source["semester"], "task": policy["task"],
                             "institution": policy["institution"], "url": source["landing_url"],
                             "text": content})
    output = root / "outputs/index.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (root / "outputs/index_review.json").write_text(json.dumps({"quarantined_pages": quarantined},
                                                             ensure_ascii=False, indent=2) + "\n")
    return rows


def retrieve(rows, question, *, year, task, institution=None, semester=None, k=3):
    if task == "admissions" and not institution:
        raise ValueError("Admissions queries require an institution")
    if task == "housing" and semester is None:
        raise ValueError("Housing queries require a semester")
    scoped = [r for r in rows if r["year"] == year and r["task"] == task
              and (institution is None or r["institution"] == institution)
              and (semester is None or r["semester"] == semester)]
    if not scoped:
        return []
    counters = [Counter(terms(r["text"])) for r in scoped]
    df = Counter(t for counter in counters for t in counter)
    avg = sum(sum(c.values()) for c in counters) / len(counters)
    query = set(terms(question))
    ranked = []
    for row, counter in zip(scoped, counters):
        size = sum(counter.values())
        score = sum(math.log(1 + (len(scoped)-df[t]+.5)/(df[t]+.5)) *
                    counter[t] * 2.5 / (counter[t] + 1.5 * (.25 + .75 * size/avg))
                    for t in query if counter[t])
        if score > 0:
            ranked.append(dict(row, score=round(score, 6)))
    selected, seen = [], set()
    for row in sorted(ranked, key=lambda r: (-r["score"], r["chunk_id"])):
        page = row["source_id"], row["pdf_page"]
        if page in seen:
            continue
        selected.append(row); seen.add(page)
        if len(selected) == k:
            break
    return selected
