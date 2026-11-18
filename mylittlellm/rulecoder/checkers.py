"""Checker rules used by the repair loop: selected probe families turned into targeted hints."""
import json

from .probes import run_probes
from .rules import BANK

PROBLEM_TEXT = {"exception": "raised", "timeout": "did not finish within a few seconds"}


def active_families(path=BANK):
    """Probe families that survived selection on development data (see rule_bank.json)."""
    bank = json.loads(path.read_text())
    return {c["family"] for c in bank.get("checkers", []) if c.get("status") == "active"}


def hint(finding):
    if finding["problem"] == "timeout":
        return (f"The call {finding['call']} did not finish within a few seconds. "
                "Hidden tests may use inputs like this; make the function efficient enough.")
    return (f"The call {finding['call']} raised {finding['detail']}. "
            "Hidden tests may use inputs like this; return a sensible result instead of raising.")


def make_checkers(families):
    def run_checkers(task, code, asserts, check):
        # Probing only makes sense for a candidate that already satisfies the visible example.
        if not check["ok"] or not families:
            return []
        return [hint(f) for f in run_probes(code, task["entry_point"], asserts, families)]
    return run_checkers
