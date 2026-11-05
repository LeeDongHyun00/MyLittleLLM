"""Check repository memory metadata and frozen evidence without model dependencies.

This is a purpose-built consistency checker, not a general JSON Schema validator
or a factual/semantic evaluation of admissions answers. It does not access the network.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_file(relative, *, parent=ROOT):
    require(isinstance(relative, str) and relative, "Expected a nonempty path")
    path = (parent / relative).resolve()
    require(path.is_relative_to(ROOT), f"Path outside repository: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def read_json(relative):
    return json.loads(local_file(relative).read_text(encoding="utf-8"))


def check_links(relative):
    path = local_file(relative)
    text = path.read_text(encoding="utf-8")
    # Repository documentation uses ordinary inline links; anchors and remote
    # availability are intentionally not validated by this offline check.
    for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", text):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        local_file(unquote(parsed.path), parent=path.parent)


def check_handoff(relative, manifest):
    handoff = read_json(relative)
    require(isinstance(handoff, dict), "Handoff must be an object")
    required = {"schema_version", "contract_version", "date", "session_role", "task_id",
                "branch", "code_baseline", "status", "summary", "completed", "verification",
                "known_limits", "next_tasks", "artifacts", "open_questions"}
    require(not required - handoff.keys(), f"Missing handoff fields: {sorted(required - handoff.keys())}")
    require(handoff["schema_version"] == 1, "Unsupported handoff schema_version")
    require(handoff["contract_version"] == manifest["contract_version"], "Handoff contract version mismatch")
    require(handoff["session_role"] in manifest["roles"], "Unknown handoff role")
    require(handoff["status"] in {"completed", "partial", "blocked"}, "Invalid handoff status")
    require(re.fullmatch(r"[0-9a-f]{40}", handoff["code_baseline"]) is not None, "Invalid baseline commit")
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", handoff["date"]) is not None, "Invalid handoff date")
    for key in ("task_id", "branch", "summary"):
        require(isinstance(handoff[key], str) and handoff[key].strip(), f"Empty handoff {key}")
    for key in ("completed", "verification", "known_limits", "next_tasks", "artifacts", "open_questions"):
        require(isinstance(handoff[key], list), f"Handoff {key} must be a list")
    require(handoff["verification"], "At least one verification record is required")
    for check in handoff["verification"]:
        require(isinstance(check, dict), "Invalid verification record")
        require(all(isinstance(check.get(k), str) and check[k].strip()
                    for k in ("command", "outcome", "status")), "Incomplete verification record")
        require(check["status"] in {"passed", "failed", "not_run", "observed"}, "Invalid verification status")
    for artifact in handoff["artifacts"]:
        local_file(artifact)
    require(all(isinstance(task, str) and task in manifest["task_ids"]
                for task in handoff["next_tasks"]), "Unknown next task")


def check_project(handoff_override=None):
    manifest = read_json("contracts/project.json")
    require(manifest["schema_version"] == 1, "Unsupported project schema_version")
    version = manifest["contract_version"]
    require(re.fullmatch(r"\d+\.\d+\.\d+", version) is not None, "Invalid contract version")
    for relative in manifest["required_paths"]:
        local_file(relative)
    for relative in manifest["markdown_files"]:
        check_links(relative)
    for relative in ("PROJECT_CONTRACT.md", "docs/project/STATE.md"):
        require(version in local_file(relative).read_text(encoding="utf-8"), f"Version missing from {relative}")
    tasks = local_file("docs/project/TASKS.md").read_text(encoding="utf-8")
    documented_tasks = re.findall(r"^## ([A-Z][0-9]{2}) —", tasks, flags=re.MULTILINE)
    require(sorted(documented_tasks) == sorted(manifest["task_ids"]), "Task registry mismatch")
    for role, task_ids in manifest["roles"].items():
        require(set(task_ids) <= set(manifest["task_ids"]), f"Unregistered task for {role}")

    rules = read_json("config/grounding_rules.json")["rules"]
    require(sorted(rule["id"] for rule in rules) == sorted(manifest["expected_rule_ids"]),
            "Rule IDs changed: update current state and contract manifest intentionally")
    for relative, expected in manifest["frozen_artifacts"].items():
        actual = hashlib.sha256(local_file(relative).read_bytes()).hexdigest()
        require(actual == expected, f"Frozen evidence changed: {relative}; preserve it and add a new run/correction")
    for observation in manifest["observations"]:
        value = read_json(observation["path"])
        for key in observation["keys"]:
            value = value[key]
        require(value == observation["expected"], f"Evidence ledger mismatch: {observation}")

    research = read_json("data/research/reference_projects_snapshot.json")
    projects = research["selected"]
    require(len(projects) == 10, "Exactly 10 selected reference projects are required")
    require([p["rank"] for p in projects] == list(range(1, 11)), "Reference ranks must be 1..10")
    require(len({p["repository"] for p in projects}) == 10, "Duplicate reference project")
    for project in projects:
        require(re.fullmatch(r"[0-9a-f]{40}", project["head_commit"]) is not None, "Invalid reference commit")
        require(project["head_commit"] in project["sources"]["readme"], "README reference is not commit-pinned")
        require(project["limitations"] and project["sources"]["inspected_files"], "Missing reference evidence/limits")
    check_handoff(handoff_override or manifest["latest_handoff"], manifest)
    print(f"PASS: contract {version}; {len(projects)} references; {len(rules)} rules; "
          f"{len(manifest['frozen_artifacts'])} frozen files; {len(manifest['markdown_files'])} linked documents")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", help="Validate a candidate handoff inside this repository")
    args = parser.parse_args()
    try:
        check_project(args.handoff)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
