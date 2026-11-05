"""Compare opt-in retrieval against BM25 on public D01 development fixtures.

Full local evidence stays in a new outputs directory. The shareable report has
only query text, ranks, provenance, scores and lexical checks, never page text.
No generation, answer values, annotated pages or query IDs enter the retrievers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mylittlellm.retrieval import read_jsonl, retrieve
from mylittlellm.retrieval_experiments import build_clause_windows, retrieve_sentence_focus, sentence_focus_weights
from scripts.audit_d01_source import anchors_present, validate_output, validate_pdf, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clause_hit(row, clause, fixture):
    return (row["source_id"] == fixture["source_id"]
            and row["sha256"] == fixture["source_sha256"]
            and row["pdf_page"] == clause["pdf_page"]
            and anchors_present(row["text"], clause["anchors"]))


def score_retrieval(ranked, case, fixture, k):
    """Annotated pages/anchors are evaluated only after retrieval has finished."""
    selected = ranked[:k]
    required = case["required_clauses"]
    clauses = {clause["id"]: clause for clause in fixture["clauses"]}
    matched = [cid for cid in required if any(clause_hit(row, clauses[cid], fixture)
                                             for row in selected)]
    first_ranks = {cid: next((rank for rank, row in enumerate(ranked, 1)
                             if clause_hit(row, clauses[cid], fixture)), None)
                   for cid in required}
    primary = case.get("primary_clause")
    return dict(required_clauses=required, matched_clauses=matched,
                missing_clauses=sorted(set(required) - set(matched)),
                lexical_recall_at_k=len(matched) / len(required) if required else None,
                complete_bundle_at_k=len(matched) == len(required) if required else None,
                primary_hit_at_k=primary in matched if primary else None,
                primary_rank=first_ranks.get(primary), clause_first_ranks=first_ranks,
                expected_empty=case.get("expected_empty", False),
                scope_empty_pass=not ranked if case.get("expected_empty") else None)


def evidence_metadata(row):
    keys = ("chunk_id", "source_id", "sha256", "pdf_page", "year", "semester", "task",
            "institution", "url", "score")
    return {**{key: row[key] for key in keys},
            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "text_characters": len(row["text"])}


def evaluate(reconstruction, output, plan_path):
    started = time.monotonic()
    plan = json.loads(plan_path.read_text())
    queries_path = ROOT / "data/development/r01/queries.json"
    fixture_path = ROOT / "data/development/d01/clauses.json"
    queries, fixture = json.loads(queries_path.read_text()), json.loads(fixture_path.read_text())
    manifest = json.loads((ROOT / "data/development/d01/source_manifest.json").read_text())
    version = manifest["versions"][plan["source_version"]]
    if plan["source_sha256"] != version["sha256"] or fixture["source_sha256"] != version["sha256"]:
        raise ValueError("Source version/fixture mismatch")
    snapshots = read_jsonl(reconstruction / "data/retrieval_snapshot.jsonl")
    if len(snapshots) != 1 or snapshots[0]["source_id"] != plan["source_id"]:
        raise ValueError("Expected isolated D01 source reconstruction")
    snapshot = snapshots[0]
    raw = reconstruction / snapshot["raw_path"]
    validate_pdf(raw, version)
    index = reconstruction / "outputs/index.jsonl"
    rows = read_jsonl(index)
    scope = queries["scope"]
    if not rows or any(row["source_id"] != plan["source_id"]
                       or row["sha256"] != plan["source_sha256"]
                       or not 1 <= row["pdf_page"] <= 25
                       or any(row[key] != value for key, value in scope.items())
                       for row in rows):
        raise ValueError("Index provenance/scope mismatch or FAQ page in legacy policy scope")
    candidate_id = plan["candidate"]["id"]
    if candidate_id not in {"sentence_focus", "clause_windows"}:
        raise ValueError("Unsupported experimental candidate")
    if plan["baseline"]["k"] != 2 or plan["candidate"]["k"] != 2:
        raise ValueError("This comparison requires equal k=2 evidence budgets")
    candidate_rows = rows
    if candidate_id == "clause_windows":
        candidate_rows = build_clause_windows(
            rows, read_jsonl(reconstruction / snapshot["extracted_path"]),
            max_chars=plan["candidate"]["max_chars"], stride=plan["candidate"]["stride"])
    output = validate_output(output)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(output / "queries.json", queries)
    weight = plan["candidate"].get("context_weight")
    runs, local = [], []
    arm_totals = {}
    candidate_retriever = retrieve_sentence_focus if candidate_id == "sentence_focus" else retrieve
    for name, retriever, index_rows in (("bm25", retrieve, rows),
                                        (candidate_id, candidate_retriever, candidate_rows)):
        arm_rows = []
        for case in queries["cases"]:
            query_scope = {**scope, **case.get("scope_override", {})}
            kwargs = dict(context_weight=weight) if name == "sentence_focus" else {}
            query_started = time.monotonic()
            # Extra ranks are diagnostic only; actual evidence budget stays top two.
            ranked = retriever(index_rows, case["question"], **query_scope, k=len(index_rows), **kwargs)
            elapsed = time.monotonic() - query_started
            metric = score_retrieval(ranked, case, fixture, plan["baseline"]["k"])
            result = dict(arm=name, query_id=case["id"], group=case["group"],
                          question=case["question"], scope=query_scope,
                          evidence_budget_k=plan["baseline"]["k"],
                          elapsed_seconds=round(elapsed, 6), metrics=metric,
                          ranked=[dict(rank=rank, **evidence_metadata(row))
                                  for rank, row in enumerate(ranked, 1)])
            if name == "sentence_focus":
                result["query_weights"] = sentence_focus_weights(case["question"], weight)
            runs.append(result)
            arm_rows.append(result)
            local.append(dict(arm=name, query_id=case["id"], question=case["question"],
                              scope=query_scope, selected=ranked[:plan["baseline"]["k"]],
                              ranked=ranked))
        answerable = [row["metrics"] for row in arm_rows if row["metrics"]["required_clauses"]]
        scopes = [row["metrics"] for row in arm_rows if row["metrics"]["expected_empty"]]
        hits = sum(len(row["matched_clauses"]) for row in answerable)
        required = sum(len(row["required_clauses"]) for row in answerable)
        arm_totals[name] = dict(
            retrieval_queries=len(arm_rows), answerable_queries=len(answerable),
            primary_hits_at_2=sum(row["primary_hit_at_k"] for row in answerable),
            required_clause_hits_at_2=hits, required_clause_total=required,
            lexical_micro_recall_at_2=hits / required,
            complete_bundles_at_2=sum(row["complete_bundle_at_k"] for row in answerable),
            scope_empty_cases=len(scopes), scope_empty_passes=sum(row["scope_empty_pass"] for row in scopes))
    by_query = {case["id"]: {row["arm"]: row for row in runs if row["query_id"] == case["id"]}
                for case in queries["cases"]}
    gains = [qid for qid, arms in by_query.items()
             if arms[candidate_id]["metrics"]["complete_bundle_at_k"] is True
             and arms["bm25"]["metrics"]["complete_bundle_at_k"] is False]
    regressions = [qid for qid, arms in by_query.items()
                   if arms[candidate_id]["metrics"]["complete_bundle_at_k"] is False
                   and arms["bm25"]["metrics"]["complete_bundle_at_k"] is True]
    local_path = output / "retrieval.jsonl"
    with local_path.open("x", encoding="utf-8") as stream:
        for row in local:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    candidate_index_path = output / "candidate_index.jsonl"
    with candidate_index_path.open("x", encoding="utf-8") as stream:
        for row in candidate_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dependency_paths = ["mylittlellm/retrieval.py", "mylittlellm/retrieval_experiments.py",
                        "scripts/evaluate_r01_retrieval.py", "scripts/audit_d01_source.py",
                        "data/development/r01/queries.json", "data/development/d01/clauses.json",
                        "data/development/d01/source_manifest.json"]
    report = dict(
        schema_version=1, task_id="R01", run_id=output.name,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), cohort="public_regression",
        review_status=queries["review_status"], independent_human_approved=False,
        plan=plan, plan_sha256=sha(plan_path),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dependency_hashes={name: sha(ROOT / name) for name in dependency_paths},
        source_id=plan["source_id"], source_sha256=sha(raw), source_bytes=raw.stat().st_size,
        extracted_sha256=sha(reconstruction / snapshot["extracted_path"]),
        index_sha256=sha(index), indexed_chunks=len(rows),
        candidate_index_sha256=sha(candidate_index_path), candidate_indexed_chunks=len(candidate_rows),
        indexed_pages=sorted({row["pdf_page"] for row in rows}),
        source_limits="Separate Dongseo bytes. Historical Jeju original and latest official corrected version remain unverified.",
        python_version=platform.python_version(), models_called=0,
        model_responses_reused=0, generated_answers=0,
        lexical_retrieval_judgments=len(runs),
        input_question_count=len(queries["cases"]),
        independence_note="Public variants share four development rule/topic groups; rows are not independent samples or human-approved gold.",
        local_evidence_path=local_path.relative_to(ROOT).as_posix(),
        local_evidence_sha256=sha(local_path), summary=arm_totals,
        complete_bundle_gains=gains, complete_bundle_regressions=regressions,
        results=runs, elapsed_seconds=round(time.monotonic() - started, 6),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("run_id", "summary", "complete_bundle_gains",
                                                  "complete_bundle_regressions", "models_called",
                                                  "elapsed_seconds", "peak_rss_kib")}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "data/development/r01/experiment_plan.json")
    args = parser.parse_args()
    try:
        evaluate(args.reconstruction.resolve(), args.output, args.plan.resolve())
    except (ValueError, OSError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
