# 3차 라운드 교사 지침 (원문)

교사 서브에이전트 다섯에게 그대로 준 영어 지침이다. 범주별로 다른 부분은 각 에이전트에게 따로 전한 짧은 메모뿐이다. 측정 결과는 [3차 라운드 문서](rule_round3.md)에 있다.

---

## Teacher guide — round 3 operator packs

## What this is

`mylittlellm/rulecoder/backinfer.py` is a rule engine that writes small Python functions with no
model at all. A task gives a description and ONE visible example, `assert f(args) == expected`.
The engine enumerates compositions of small operators (at most two operators deep) until one
reproduces `expected` from `args`, then emits that composition as the function body.
The operator library is the rule base. In earlier rounds one teacher wrote 328 operators.

You are a teacher for round 3. You get the tasks the engine still fails on in ONE category.
Your job: add **general, executable operators** so the engine solves more of them — and solves
tasks of the same kind that neither you nor the engine has seen (a locked test split exists;
nobody may look at it, and your operators will be judged on it).

## Files

- Your failures: `outputs/rule_pilot/teacher/round3/bank_<category>.json`. Each item:
  - `task_id`, `source` — `train` (MBPP+ train split, judged by hidden tests with edge cases) or
    `extra` (other MBPP tasks, judged by the three asserts in `asserts`);
  - `description`, `entry_point`, `visible_assert` (the only example the engine sees);
  - `reference_code` — a correct solution; read it to learn the intended behaviour and edge cases;
  - `baseline.status` — `none` (nothing reproduces the example) or `wrong` (a coincidental
    program was found; `baseline.expr` shows it);
  - `pipeline_1p5b_solved` (train only) — whether the full system (a 1.5B model plus this engine)
    already solves the task. `false` means fixing it raises the headline number: top priority.
- Engine source, read-only: `mylittlellm/rulecoder/backinfer.py` (operators are defined from
  line ~110 to ~770; `search` and `synthesize` near the end).
- Your pack (create it): `mylittlellm/rulecoder/oppacks/r3_<category>.py`.
- Your report (create it): `outputs/rule_pilot/teacher/round3/report_<category>.json`.

## Hard rules

1. **Held-out data is off limits.** Do not open or print `data/raw/`, `data/rule_pilot/split.json`,
   `outputs/rule_pilot/gen`, `outputs/rule_pilot/scored`, `reports/rule_pilot/scored`, or any
   bank file other than your own and `regression.json`. Use only the check tool to run things.
2. **Touch only your two files** (pack and report). No other edits, no git commands, no servers,
   no long jobs. Other teachers are working in this directory at the same time.
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
  defeat the gate; too-narrow ones never fire. Read the bank descriptions for the real wording.
- **Ranking.** Among programs that reproduce the example, those whose operators are named by
  the description come first, then smaller programs. So keywords matter on ungated operators
  too: a well-named operator outranks an accidental one (this is how you fix `wrong` items).
- **Ungated operators are expensive for everyone.** Each one adds accidental matches to every
  task. Keep ungated only what is cheap and broadly meaningful (like `sum`, `sorted`); gate
  named concepts, formulas, definitions and all predicates.
- Values whose `repr` is longer than 600 characters are dropped during the search.

## Writing an operator

A pack is a module with one function. Minimal example:

```python
"""Round 3 operators for <category>: <one line on what kind of rule this pack adds>."""


def register(lib):
    need, is_int, numseq, like = lib.need, lib.is_int, lib.numseq, lib.like

    # Divisor lists come up under several names; the keyword gate keeps them out of other tasks.
    lib.unary("divisors", lambda n: (need(is_int(n) and 0 < n <= 10 ** 5),
                                     [d for d in range(1, n + 1) if n % d == 0])[1],
              "[d for d in range(1, {0} + 1) if {0} % d == 0]", "divisor|factors of", gated=True, kind="knowledge")

    lib.binary("count_greater", lambda x, n: (need(numseq(x) and lib.is_num(n)), sum(v > n for v in x))[1],
               "sum(v > {1} for v in {0})", "count&greater|number of&greater|count&larger")
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
  many operators exist already (`grep -n '^unary("\|^binary("\|^ternary("' backinfer.py`).

## Check tool

```bash
.venv/bin/python scripts/teacher_check.py --pack r3_<category> --bank <category>            # your failures
.venv/bin/python scripts/teacher_check.py --pack r3_<category> --bank <category> --regress  # + everything solved before
.venv/bin/python scripts/teacher_check.py --pack r3_<category> --bank <category> --only Mbpp/104 MbppX/321 --show
```

It prints every task whose result changed (`SOLVED` / `WRONG` / `NONE`), a summary, and for each
of your operators how often it ended up in a correct or a wrong final program.
`BROKEN (was solved)` must be 0 before you finish: if your operator displaces a program that
was correct, narrow its keywords, gate it, or drop it.

## Workflow

1. Read the whole bank once. Group failures by the operator they would need. Priorities:
   (a) train items with `pipeline_1p5b_solved: false`; (b) needs that recur across items
   (one operator fixing five tasks beats five operators fixing one each); (c) the rest.
2. Write operators, run the check, look at `WRONG` and `NONE`, iterate.
3. Many items cannot be solved by a two-step composition (stateful loops, dynamic programming,
   ambiguous descriptions). Leave them; record a one-line reason. Do not force a task.
4. Run with `--regress`; get `BROKEN` to 0.
5. Guideline size: 25–70 operators. Stop when new operators stop fixing bank items.

## Report

Write `outputs/rule_pilot/teacher/round3/report_<category>.json`:

```json
{"category": "...",
 "operators": [{"name": "divisors/1", "gated": true, "targets": ["Mbpp/123", "MbppX/45"], "why": "one line"}],
 "keyword_extensions": [{"operator": "sum/1", "added": "aggregate", "why": "one line"}],
 "not_addressed": [{"task_id": "Mbpp/77", "reason": "needs a stateful loop"}],
 "final_check": {"train_newly_solved": 0, "extra_newly_solved": 0, "broken": 0},
 "notes": "risky operators, ideas that would need an engine change, anything the coordinator should know"}
```

Finish with a short plain-text summary of the same numbers.
