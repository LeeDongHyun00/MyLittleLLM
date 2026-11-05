"""Validate declared work sessions and render a GitHub Actions handoff summary.

Offline metadata check only: it neither launches agents nor verifies H1 review,
semantic accuracy, remote PR state, or whether a conversation is still running.
Historical handoffs retain the contract version on their original task branch.
"""
import argparse
from pathlib import Path
import re
import sys

try:
    from . import check_project_contract as contract
except ImportError:
    import check_project_contract as contract


def validate_registry(registry, manifest, load_handoff, validate_handoff, branch=None):
    require = contract.require
    require(registry["schema_version"] == 1, "Unsupported session schema")
    require(registry["project_id"] == manifest["project_id"], "Session project mismatch")
    require(registry["event_transport"] == "github_actions", "Unsupported event transport")
    require(registry["external_webhook_configured"] is False,
            "External webhook setup needs a separately verified implementation")
    versions = {manifest["contract_version"], *registry["historical_contract_versions"]}
    require(all(isinstance(v, str) and re.fullmatch(r"\d+\.\d+\.\d+", v) for v in versions),
            "Invalid historical contract version")
    sessions = registry["sessions"]
    require(isinstance(sessions, list) and sessions, "No registered sessions")
    seen_ids, seen_branches = set(), set()
    for session in sessions:
        sid, role, task = session["id"], session["role"], session["task_id"]
        require(isinstance(sid, str) and re.fullmatch(r"[a-z0-9-]+", sid), "Invalid session ID")
        require(sid not in seen_ids, f"Duplicate session ID: {sid}")
        seen_ids.add(sid)
        require(role in manifest["roles"] and task in manifest["roles"][role],
                f"Role/task mismatch: {role}/{task}")
        ref = session["branch"]
        require(isinstance(ref, str) and re.fullmatch(
            rf"{role.lower()}/{task.lower()}(?:-[a-z0-9][a-z0-9-]*)?", ref),
            f"Branch does not match role/task: {ref}")
        require(ref not in seen_branches, f"Shared session branch: {ref}")
        seen_branches.add(ref)
        require(re.fullmatch(r"[0-9a-f]{40}", session["code_baseline"]), "Invalid session baseline")
        require(session["runtime"] in {"conversation_root", "conversation_subagent"},
                "Unknown session runtime")
        version = session["contract_version"]
        require(version in versions, f"Undeclared historical contract version: {version}")
        status = session["status"]
        require(status in {"running", "completed", "partial", "blocked"}, "Invalid session status")
        if session.get("result_commit") is not None:
            require(re.fullmatch(r"[0-9a-f]{40}", session["result_commit"]), "Invalid result commit")
        if session["pull_request"] is not None:
            require(re.fullmatch(r"https://github.com/LeeDongHyun00/MyLittleLLM/pull/[1-9]\d*",
                                 session["pull_request"]), "Invalid session PR URL")
        handoff_path = session["handoff"]
        require(status == "running" or handoff_path, f"Finished session has no handoff: {sid}")
        if handoff_path:
            require(isinstance(handoff_path, str) and re.fullmatch(
                r"docs/handoffs/[a-z0-9-]+\.json", handoff_path), "Invalid handoff path")
            handoff = load_handoff(handoff_path)
            for key, expected in {"session_role": role, "task_id": task, "branch": ref,
                                  "code_baseline": session["code_baseline"],
                                  "contract_version": version, "status": status}.items():
                require(handoff[key] == expected, f"{sid}: handoff {key} mismatch")
            # Validate archived files against their declared contract snapshot,
            # without rewriting history to the current integration version.
            validate_handoff(handoff_path, {**manifest, "contract_version": version})
    require(registry["integration_branch"] in seen_branches, "Integration branch is not a session")
    if branch and re.match(r"s[0-5]/", branch):
        require(branch in seen_branches, f"Role branch is not registered: {branch}")
    return sessions


def render_summary(registry, sessions, branch):
    lines = ["## Session collaboration", "",
             f"Checked branch: `{branch or '(local)'}`", "",
             "| Session | Role / task | State | Branch | Handoff | PR |",
             "|---|---|---|---|---|---|"]
    for session in sessions:
        handoff = f"`{session['handoff']}`" if session["handoff"] else "pending"
        pr = f"[PR]({session['pull_request']})" if session["pull_request"] else "pending"
        lines.append(f"| {session['id']} | {session['role']} / {session['task_id']} | "
                     f"{session['status']} | `{session['branch']}` | {handoff} | {pr} |")
    lines += ["", f"Coordination: {registry['coordination_issue']}", "",
              "This is a committed status snapshot, not a live session heartbeat.",
              "Native GitHub push/PR events validate metadata; no external webhook or agent launcher is installed.",
              "Passing checks do not establish H1 review, semantic correctness, or final evaluation readiness.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", help="PR head branch or push ref name (no shell evaluation)")
    parser.add_argument("--summary", type=Path, help="Append Markdown to GITHUB_STEP_SUMMARY")
    args = parser.parse_args()
    try:
        registry = contract.read_json("contracts/sessions.json")
        manifest = contract.read_json("contracts/project.json")
        sessions = validate_registry(registry, manifest, contract.read_json,
                                     contract.check_handoff, args.branch)
        summary = render_summary(registry, sessions, args.branch)
        if args.summary:
            with args.summary.open("a", encoding="utf-8") as stream:
                stream.write(summary)
        print(summary)
        print(f"PASS: {len(sessions)} sessions; role/task/branch/handoff consistency")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
