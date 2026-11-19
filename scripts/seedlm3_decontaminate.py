"""Drop teacher-authored tasks that resemble an evaluation task (counts only are reported).

Teachers were told to invent tasks and never saw benchmark data, but they have read the internet: a
task they "invent" can be one they remember. A teacher task is dropped when its description, without
the opening boilerplate ("Write a python function to"), shares a run of six words with, or has a
content-word Jaccard similarity of at least 0.6 to, the description of any task used for
evaluation: MBPP+ validation and test, HumanEval+, and the extra pool's held-out fifth (the
development check). Nothing about the evaluation tasks is printed. (A first run that kept the
boilerplate dropped a third of the teacher tasks for sharing "write a python function to count the".)

  python scripts/seedlm3_decontaminate.py                           writes data/seedlm/teacher_tasks_dropped.json
  python scripts/seedlm3_decontaminate.py --prefix variant_tasks    the same rules for SeedLM-5's seeded variants
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylittlellm.rulecoder import humaneval as he
from mylittlellm.rulecoder.tasks import ROOT, load_split, parse_prompt, read_tasks
from mylittlellm.seedlm import language
from seedlm2_real import held_out

STOP = set("a an the of to in for and or is are be given write function python that which with from by on "
           "return returns list number numbers string strings check whether if".split())
DATA = ROOT / "data/seedlm"


OPENER = re.compile(r"^(?:(?:write|create|implement|define|make)\s+(?:a\s+|an\s+)?(?:python\s+)?"
                    r"(?:function|program|method)(?:\s+(?:to|that|which|for))?|given)\b")


def body(text):
    """The description without its opening boilerplate ("Write a python function to ..."), as words."""
    return language.words(OPENER.sub("", text.strip().lower(), count=1))


def shingles(words, n=6):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def content(words):
    return {w for w in words if w not in STOP}


def evaluation_descriptions():
    texts = []
    tasks, split = read_tasks(), load_split()
    for part in ("val", "test"):
        texts += [parse_prompt(tasks[t]["prompt"])[0] for t in split[part]]
    texts += [t["description"] for t in he.read_tasks().values()]
    bank = json.loads((ROOT / "outputs/rule_pilot/teacher/round3/all_tasks.json").read_text())
    texts += [i["description"] for i in bank if i["source"] == "extra" and held_out(i["task_id"])]
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="teacher_tasks", help="authored task files data/seedlm/<prefix>_*.json")
    args = parser.parse_args()
    references = [body(t) for t in evaluation_descriptions()]
    reference_shingles = set().union(*(shingles(w) for w in references))
    reference_content = [content(w) for w in references]
    dropped, total = [], 0
    for path in sorted(DATA.glob(f"{args.prefix}_*.json")):
        if path.name.endswith("_dropped.json"):
            continue
        for task in json.loads(path.read_text()):
            total += 1
            words = body(task["description"])
            mine = content(words)
            overlap = bool(shingles(words) & reference_shingles)
            similar = any(mine and len(mine & other) / len(mine | other) >= 0.6 for other in reference_content)
            if overlap or similar:
                dropped.append({"id": task["id"], "reason": "six-word run" if overlap else "content Jaccard >= 0.6"})
    (DATA / f"{args.prefix}_dropped.json").write_text(json.dumps(dropped, indent=1) + "\n")
    print(json.dumps({args.prefix: total, "dropped": len(dropped),
                      "by_reason": {r: sum(d["reason"] == r for d in dropped) for r in {d["reason"] for d in dropped}},
                      "evaluation_descriptions": len(references)}))


if __name__ == "__main__":
    main()
