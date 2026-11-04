"""Opt-in public-development retrieval candidates; runtime defaults stay unchanged.

No source IDs, expected pages, answer values or evaluation fixtures are used here.
"""
from collections import Counter
import math
import re

from mylittlellm.retrieval import retrieve, terms


def clause_window_spans(text, max_chars=900, stride=740):
    """Start windows at generic policy bullets; cover long blocks with overlap.

    Keep adjacent clauses/footnotes within the same character budget. This does
    not infer semantic clause relationships and cannot promise complete tables.
    """
    if max_chars < 40 or not 0 < stride < max_chars:
        raise ValueError("Require max_chars >= 40 and 0 < stride < max_chars")
    if not text:
        return []
    boundaries = sorted({0, *(match.start() for match in re.finditer(r"[□◦]", text))})
    starts = set(boundaries)
    for start, end in zip(boundaries, boundaries[1:] + [len(text)]):
        if end - start > max_chars:
            starts.update(range(start + stride, end, stride))
    spans = [(start, min(start + max_chars, len(text))) for start in sorted(starts)]
    return [(start, end) for start, end in spans
            if end - start >= 40 or not any(a < start and b >= end for a, b in spans)]


def build_clause_windows(rows, pages, max_chars=900, stride=740):
    """Rechunk exactly the already allowed source/pages; no labels are consulted."""
    pages_by_key = {(page["source_id"], page["pdf_page"]): page for page in pages}
    originals = {}
    for row in rows:
        originals.setdefault((row["source_id"], row["pdf_page"]), []).append(row)
    result = []
    for key, original_rows in originals.items():
        page = pages_by_key.get(key)
        if page is None:
            raise ValueError("Allowed index page is missing from extraction")
        text = re.sub(r"\s+", " ", page["text"]).strip()
        if any(row["sha256"] != page["sha256"] or row["text"] not in text
               for row in original_rows):
            raise ValueError("Extraction and original index provenance/text mismatch")
        metadata = {key: value for key, value in original_rows[0].items()
                    if key not in ("chunk_id", "text", "score")}
        for start, end in clause_window_spans(text, max_chars, stride):
            result.append(dict(metadata,
                               chunk_id=f"{page['source_id']}:p{page['pdf_page']}:clause{start}-{end}",
                               text=text[start:end]))
    return result


def sentence_focus_weights(question, context_weight=0.5):
    """Keep every query term, emphasizing the last explicit question sentence.

    This is a lexical experiment, not semantic condition extraction. In particular,
    negative and exception words in context remain present, but may receive less
    weight. A single sentence (or no final '?') keeps baseline weights unchanged.
    """
    if not 0 < context_weight <= 1:
        raise ValueError("context_weight must be in (0, 1]; dropping terms is not allowed")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", question.strip()) if s.strip()]
    if len(sentences) < 2 or not sentences[-1].endswith("?"):
        return {term: 1.0 for term in terms(question)}
    weights = {term: context_weight for term in terms(question)}
    weights.update({term: 1.0 for term in terms(sentences[-1])})
    return weights


def retrieve_sentence_focus(rows, question, *, year, task, institution=None,
                            semester=None, k=3, context_weight=0.5):
    """BM25 with one query-weight change; filters, chunks and page dedup are fixed."""
    weights = sentence_focus_weights(question, context_weight)
    # Production validation also guards tasks that require a scope field.
    if all(value == 1 for value in weights.values()):
        return retrieve(rows, question, year=year, task=task, institution=institution,
                        semester=semester, k=k)
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
    df = Counter(term for counter in counters for term in counter)
    avg = sum(sum(counter.values()) for counter in counters) / len(counters)
    ranked = []
    for row, counter in zip(scoped, counters):
        size = sum(counter.values())
        score = sum(weights[term] * math.log(1 + (len(scoped) - df[term] + .5) / (df[term] + .5))
                    * counter[term] * 2.5
                    / (counter[term] + 1.5 * (.25 + .75 * size / avg))
                    for term in weights if counter[term])
        if score > 0:
            ranked.append(dict(row, score=round(score, 6)))
    selected, seen = [], set()
    for row in sorted(ranked, key=lambda r: (-r["score"], r["chunk_id"])):
        page = row["source_id"], row["pdf_page"]
        if page in seen:
            continue
        selected.append(row)
        seen.add(page)
        if len(selected) == k:
            break
    return selected
