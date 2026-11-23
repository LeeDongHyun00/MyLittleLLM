# 5차 라운드 2차 시범: 과제 교사 지침 (원문)

2차 시범의 과제 교사에게 그대로 준 영어 지침이다. [1차 지침](round5_task_guide.md)과 다른 곳은 "What to write" 절뿐이다. 1차 교사는 벤치마크에서 본 기억이 나는 주제를 통째로 피해 실제 과제와 겹칠 곳을 비웠고, 지어낸 문제가 실제 과제보다 어려웠다([1차 결과](rule_round5_pilot.md)). 그래서 고전 개념은 일반 지식으로 허용하고, 벤치마크 항목의 문장·예시·상수만 금지했다. 난이도는 입문 위주로 맞췄다.

---

## Teacher guide — invent practice exercises (second pilot)

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

300 exercises, about 60 per category, mostly at the introductory level: the kind a beginner solves in three
to ten lines (about a fifth may be intermediate). Cover a broad range of topics: definitions from number
theory and geometry, unit conversions and formulas, text conventions (case, words, vowels, palindromes,
ciphers), list and tuple manipulation, counting and grouping, dictionaries, matrices, bit manipulation, simple
recursion, dates and times, regular expressions, and anything else that commonly appears in such practice sets.
Vary the wording, the data types and the difficulty; avoid near-duplicates.

**Classic exercises are welcome.** Textbook staples (Roman numerals, palindromes, run-length encoding, coin
change, GCD and LCM, prime checks, Caesar shifts, case conversions, digit sums, vowel counts, and so on) are
general programming knowledge: write them, and do not skip a topic just because some benchmark also uses it.
What you must not do is reproduce a specific benchmark item from memory (MBPP, HumanEval and the like): its
exact wording, its example values or its particular constants. Write every exercise in your own words with
your own examples. If you notice you are recalling a specific item's wording or examples, rewrite it or drop it.

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
