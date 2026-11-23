# 5차 라운드 시범: 과제 교사 지침 (원문)

과제 교사 서브에이전트(Opus 5.5)에게 그대로 준 영어 지침이다. 왜 이 데이터를 만드는지는 [SeedLM-36~39](seed_lm36.md) 5b절에 있다. 엔진의 남은 실패는 지금 연산의 조합이 아니라 라이브러리에 없는 지식을 요구한다. 교사가 실패 사례를 보지 않고 과제를 새로 쓰고, 엔진이 못 푸는 것에만 연산을 쓰게 해, 그 연산이 교사가 본 적 없는 실제 과제로 옮겨 가는지 잰다.

---

## Teacher guide — invent practice exercises

A rule engine in this project writes small Python functions from a one-sentence task description and ONE
example call. We want to find out which kinds of everyday programming exercises it cannot do yet. Your job:
invent new, realistic practice exercises of the kind found in introductory and intermediate Python practice
sets, each with a reference solution and three test asserts.

## Format

Write `outputs/rule_round5/tasks_part<K>.json`, a JSON list of objects:

    {"task_id": "Imag/<K>-<nnn>", "category": "numbers|sequences|strings|nested|predicates",
     "description": "Write a function to ...",
     "entry_point": "snake_case_name",
     "asserts": ["assert name(<args>) == <expected>", "assert ...", "assert ..."],
     "reference_code": "def name(...):\n    ..."}

- `description`: one sentence, as a person setting an exercise would write it ("Write a function to ...",
  "Write a python function to ..."). State every constant the task needs (numbers, separators, words such as
  'Yes'/'No'), so that the description and the FIRST assert determine the intended behaviour.
- `asserts`: exactly three, plain `assert f(...) == value` with literal arguments; the first is the one the
  engine sees. They must give at least two different outputs, and together they should pin the behaviour
  down (include an edge case when natural). Keep values small (short lists and strings).
- `reference_code`: a correct, self-contained solution (any Python; loops are fine; standard library only).
- `category`: numbers (arithmetic, number theory, formulas), sequences (lists, tuples, sorting, searching),
  strings (text processing, formatting, regex), nested (lists of lists/tuples, dictionaries), predicates
  (True/False checks).

## What to write

300 exercises, about 60 per category, covering a broad range of topics: definitions from number theory and
geometry, unit conversions and formulas, text conventions (case, words, vowels, palindromes, ciphers),
list and tuple manipulation, counting and grouping, dictionaries, matrices, bit manipulation, simple
recursion or dynamic programming, dates and times, regular expressions, and anything else that commonly
appears in such practice sets. Vary the wording, the data types and the difficulty; avoid near-duplicates.
Invent: do not reproduce exercises from any benchmark or dataset you remember (MBPP, HumanEval and the like).
If an exercise you write turns out to be one you remember from such a source, drop it.

## Checker (run from /home/user/MyLittleLLM; repeat until 0 problems)

    .venv/bin/python scripts/round5_task_check.py --part <K>

It runs every reference solution on its asserts in a sandbox and checks the format; it prints each problem
and a count line.

## Hard rules

1. Do not open, list, grep or print anything under `data/`, `reports/`, `outputs/` other than your own output
   file and scratch folder, nor anything under `mylittlellm/rulecoder/` (the engine: knowing what it can do would
   bias the exercises). Do not look for MBPP, MBPP+ or HumanEval data.
2. Touch only `outputs/rule_round5/tasks_part<K>.json` and `outputs/rule_round5/scratch_part<K>/`. No git
   commands. Use `.venv/bin/python`. No network.
3. Finish with a short plain-text summary: tasks written, the checker's final count line, any topic you
   avoided and why.
