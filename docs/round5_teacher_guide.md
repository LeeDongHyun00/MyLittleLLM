# 5차 라운드 시범: 연산 교사 지침 (원문)

연산 교사 서브에이전트(Opus 5.5)에게 그대로 준 영어 지침이다. [4차 지침](round4_teacher_guide.md)을 5차 시범의 은행과 파일 위치에 맞게 고쳤다. 은행은 다른 교사가 지어낸 연습문제 가운데 엔진이 못 푸는 것이고, 연산 교사는 실제 과제를 보지 않는다. 연산은 교사가 본 적 없는 실제 학습 과제 434개로 평가한다(`reports/seedlm/round5_pilot_protocol.json`).

---

## Teacher guide — round 5 pilot operator pack

## What this is

`mylittlellm/rulecoder/backinfer.py` is a rule engine that writes small Python functions with no
model at all. A task gives a description and ONE visible example, `assert f(args) == expected`.
The engine enumerates compositions of small operators (at most two operators deep) until one
reproduces `expected` from `args`, then emits that composition as the function body.
The operator library is the rule base: about 780 operators from earlier rounds
(`backinfer.py` and the packs `mylittlellm/rulecoder/oppacks/r3_*.py`).

You are the teacher for the round-5 pilot. You get practice exercises that another teacher invented and
that the engine fails on. Your job: add **general, executable operators** so the engine solves more of
them. The exercises are only a sample of what programmers ask for: your pack will be judged on real tasks
that neither you nor anyone in this round has seen, so an operator is only worth adding if it is a library
entry that other people's tasks of the same kind would need.

## Files

- Your failures: `outputs/rule_round5/bank_imagined.json`. Each item:
  - `task_id`, `source` (`imagined`: an invented exercise judged by the asserts in `asserts`), `category`;
  - `description`, `entry_point`, `visible_assert` (the only example the engine sees);
  - `reference_code` — a correct solution; read it to learn the intended behaviour and edge cases;
  - `baseline.status` — `none` (nothing reproduces the example) or `wrong` (a coincidental
    program was found; `baseline.expr` shows it, `baseline.confident` whether the engine trusted it).
- Engine source, read-only: `mylittlellm/rulecoder/backinfer.py` (operators defined from line ~110 to
  ~770; `search` and `synthesize` near the end) and the round-3 packs `mylittlellm/rulecoder/oppacks/r3_*.py`.
- Exercises the engine already solves (for the regression check): `outputs/rule_round5/regression.json`.
- Your pack (create it): `mylittlellm/rulecoder/oppacks/r5_pilot.py`.
- Your report (create it): `outputs/rule_round5/report_pilot.json`.
- Scratch work: only under `outputs/rule_round5/scratch_pilot/` (create it).

## Hard rules

1. **Real tasks are off limits.** Do not open, list or print anything under `data/`, `reports/`, or
   `outputs/` other than `outputs/rule_round5/bank_imagined.json`, `outputs/rule_round5/regression.json` and
   your own files; in particular never look at `outputs/rule_round4/` or `outputs/rule_pilot/` (they hold the
   real tasks your pack is judged on) nor at the round-4 packs `oppacks/r4_*.py`. Do not look for MBPP, MBPP+
   or HumanEval data anywhere, and do not write operators from memory of benchmark tasks. Use only the
   check tool to run things.
2. **Touch only your files** (pack, report, your scratch folder). No other edits, no git commands, no
   servers, no long jobs (each check run takes a few minutes; run one at a time).
3. **Operators must be general.** An operator is a reusable transformation or a standard
   definition. Not allowed: anything that only makes sense for one task — entry-point names,
   constants copied from one example, special-casing particular inputs, a "composite" that is
   just one task's whole solution. Ask: would this be a sensible entry in a library that will be
   used on other people's tasks? A named mathematical definition (e.g. "Smith number") is fine
   as a gated operator even if only one bank task needs it; an ad-hoc recipe is not.
4. **Every operator that returns a bool must be gated.** One True/False example is reproduced
   by half of all predicates, so an ungated predicate would "solve" anything by accident.
5. **Deterministic and safe.** No randomness, no I/O, no dependence on set iteration order
   (sort, or use `dict.fromkeys`), never mutate an argument. Bound the work: an operator is
   called thousands of times per task, so guard sizes (`lib.small(n, limit)`, length checks)
   and keep each call well under a millisecond on small inputs.
6. Code style: English comments that say WHY (not what), lines at most 118 characters.

## How the engine uses an operator

- **Search shape.** Level 1 applies unary operators to each argument, binary operators to every
  ordered pair of different arguments, ternary operators to ordered triples. Level 2 applies
  unary operators to level-1 values and binary operators to (level-1 value, original argument)
  in both orders. It never combines two level-1 values and never goes three deep. If a common
  idea needs three steps, one operator that does two natural steps at once ("sum of the squares
  of a list") brings it within reach — only when that operator is a natural concept on its own.
- **Types must match exactly.** `[1, 2]` is not `(1, 2)`, `3` is not `3.0`, `True` is not `1`.
  Use `lib.like(source, items)` to keep the input's container type. The visible assert may also
  be `assert set(f(..)) == set(..)` (compared as sets) or `assert math.isclose(f(..), x, ..)`.
- **Gating.** `gated=True` means the operator is tried only when the description contains its
  keywords. Keyword syntax: `"a&b|c"` = (a and b) or c, matched as substrings of the lower-cased
  description. Use word stems (`"frequen"`, `"multipl"`). Too-broad keywords on a gated operator
  defeat the gate; too-narrow ones never fire. The bank descriptions are one teacher's wording and the
  real tasks are phrased by other people, so cover the common synonyms of a concept, not just the bank's words.
- **Ranking.** Among programs that reproduce the example, those whose operators are named by
  the description come first, then smaller programs. So keywords matter on ungated operators
  too: a well-named operator outranks an accidental one (this is how you fix `wrong` items).
- **Ungated operators are expensive for everyone.** Each one adds accidental matches to every
  task. Keep ungated only what is cheap and broadly meaningful; gate named concepts, formulas,
  definitions and all predicates.
- Values whose `repr` is longer than 600 characters are dropped during the search.

## Writing an operator

A pack is a module with one function. Minimal example:

```python
"""Round 5 pilot operators: <one line on what kind of rule this pack adds>."""


def register(lib):
    need, is_int, numseq, like = lib.need, lib.is_int, lib.numseq, lib.like

    # Divisor lists come up under several names; the keyword gate keeps them out of other tasks.
    lib.unary("divisors_r5", lambda n: (need(is_int(n) and 0 < n <= 10 ** 5),
                                        [d for d in range(1, n + 1) if n % d == 0])[1],
              "[d for d in range(1, {0} + 1) if {0} % d == 0]", "divisor|factors of", gated=True, kind="knowledge")
```

- Registration: `lib.unary(name, fn, template, keywords, **options)`, `lib.binary(...)`,
  `lib.ternary(...)`. Options: `gated=True`, `kind="knowledge"` (use for definitions/formulas/
  predicates), `cost=0.5` (only for pure type conversions), `helper=<source>`.
  `lib.know(arity, name, fn, template, keywords, helper=...)` is shorthand for gated knowledge.
- `fn` runs during the search; `template` is the Python **expression** emitted as code, with
  `{0}`, `{1}`, `{2}` for the arguments (literal braces must be doubled: `{{k: v ...}}`).
  **They must compute the same thing**, including on empty lists, negatives, floats, duplicates.
- `fn` must reject inputs it does not apply to by raising — use `need(condition)`. Type guards
  available on `lib`: `is_num, is_int, is_seq, numseq, intseq, seqseq, strseq, small, like`.
- Emitted code may use builtins and these modules without importing them (imports are added
  automatically): math, re, itertools, collections, heapq, functools, string, bisect, operator,
  statistics, fractions, decimal, datetime, copy.
- Multi-line logic: write a helper function as source text, register with
  `fn, text = lib.helper(source)`; the function name must start with `_`; pass `helper=text`
  and call it in the template (`"_name({0})"`). See `_SEQUENCES` in backinfer.py for examples.
- An existing operator has the right behaviour but the description uses other words?
  `lib.also_named("sum/1", "aggregate|accumulat")` adds keywords to it.
- Names must not collide with existing operator names (the loader raises). Look before adding:
  `grep -n '^unary("\|^binary("\|^ternary("' mylittlellm/rulecoder/backinfer.py` and
  `grep -n 'lib\.\(unary\|binary\|ternary\|know\)(' mylittlellm/rulecoder/oppacks/r3_*.py`.

## Check tool

```bash
.venv/bin/python scripts/teacher_check.py --round 5 --pack r5_pilot --bank imagined            # your failures
.venv/bin/python scripts/teacher_check.py --round 5 --pack r5_pilot --bank imagined --regress  # + exercises solved now
.venv/bin/python scripts/teacher_check.py --round 5 --pack r5_pilot --bank imagined --only Imag/1-042 --show
```

The pack is loaded on top of the shipped library (`--pack r5_pilot` adds it to the active packs; the
check tool takes care of that). It prints every task whose result changed (`SOLVED` / `WRONG` / `NONE`),
a summary, and for each of your operators how often it ended up in a correct or a wrong final program.
`BROKEN (was solved)` must be 0 before you finish: if your operator displaces a program that was
correct, narrow its keywords, gate it, or drop it.

## Workflow

1. Read your whole bank once. Group failures by the operator they would need. Priorities:
   (a) needs that recur across items (one operator fixing five tasks beats five operators fixing one
   each); (b) the rest.
2. Write operators, run the check, look at `WRONG` and `NONE`, iterate.
3. Many items cannot be solved by a two-step composition (stateful loops, dynamic programming,
   ambiguous descriptions, constants that appear only in the tests). Leave them; record a one-line
   reason. Do not force a task.
4. Run with `--regress`; get `BROKEN` to 0.
5. Guideline size: 20–60 operators. Stop when new operators stop fixing bank items.

## Report

Write `outputs/rule_round5/report_pilot.json`:

```json
{"operators": [{"name": "divisors_r5/1", "gated": true, "targets": ["Imag/1-045"], "why": "one line"}],
 "keyword_extensions": [{"operator": "sum/1", "added": "aggregate", "why": "one line"}],
 "not_addressed": [{"task_id": "Imag/1-077", "reason": "needs a stateful loop"}],
 "final_check": {"newly_solved": 0, "broken": 0},
 "memory": [],
 "notes": "risky operators, ideas that would need an engine change, anything the coordinator should know"}
```

List under `memory` any operator you wrote from memory of a benchmark task rather than for a bank item
(those are removed before anything is measured). Finish with a short plain-text summary of the same numbers.
