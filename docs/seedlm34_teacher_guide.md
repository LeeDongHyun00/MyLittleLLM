# SeedLM-34 교사 지침 (원문)

교사 서브에이전트(Opus 5.5)에게 그대로 준 영어 지침이다. 몫마다 다른 것은 입력·출력 파일 번호뿐이다. 왜 이 데이터를 만드는지는 [SeedLM-30~33](seed_lm30.md)과 [보강 설계](seedlm_design_next.md) 11절에 있다.

---

## Teacher guide — tasks that share a structure

A small code model, trained from scratch, reads a task description and one example and writes a one-line
Python solution. Its weakness: it only writes structures it has seen with natural wording. Programs in this
project share a small number of structures ("sketches"). Your job is to write new, natural tasks for given
sketches, so the model learns which descriptions call for which structure.

## What a sketch is

A solution is turned into prefix tokens (`pyx`). Its sketch keeps the large structure and replaces every
subtree of at most 5 tokens by a hole `?`. Example:

    'Divisible' if all(a0 % d == 0 for d in a1) else 'Not Divisible'
    tokens: @IfExp/3 @all/1 @Comp:gen:1i0/2 @Eq/2 @Mod/2 a0 v0 #0 a1 c0 c1
    sketch: @IfExp/3 @all/1 @Comp:gen:1i0/2 ? ? ? ?

Here the holes are `a0 % d == 0` (the generator's element), `a1` (what it iterates over), `'Divisible'` and
`'Not Divisible'`. Any small expression can fill a hole; the fixed tokens must stay. Some notation:
`@IfExp/3` is `x if c else y` (children: condition, then, else); `@Comp:gen:1i0/2` is a generator
`(elt for v in it)` with one variable and no `if` (children: element, iterable); `@Comp:list:1i1/3` is
`[elt for v in it if cond]`; `@sum/1`, `@len/1`, `@sorted/1`... are builtins; `@.join/2` is a method;
`@Add/2`, `@Mod/2`, `@Eq/2`, `@Index/2`, `@Slice/3` are operators; tokens without `@` such as `txt_floats/1`
are library functions, called in Python as `txt_floats_1(...)` (name, underscore, number of arguments), as in
the examples you get. Do not try to memorize the notation: use the helper below.

## Files

- Your assignments: `outputs/sketch_tasks/in_part<K>.json` — a list of sketches, each with `sketch_id`, the
  `sketch` tokens and two training tasks (`examples`) that have it.
- Your output (create it): `outputs/sketch_tasks/out_part<K>.json` — a JSON list of tasks, four per sketch:

      {"id": "s<K>-<sketch_id>-<n>", "sketch_id": "...", "category": "numbers|sequences|strings|nested|predicates",
       "description": "Write a function to ...", "entry_point": "snake_case_name",
       "inputs": ["(<args of example 1>,)", "(<args 2>,)", "(<args 3>,)"], "expression": "<one Python expression>"}

  `inputs` are three argument tuples written as Python literals (note the trailing comma for one argument);
  the first is the visible example. `expression` uses the arguments `a0`, `a1`, ... in order.
- Scratch work: only under `outputs/sketch_tasks/scratch_part<K>/`.

## What makes a good task

1. **The exact sketch.** The expression's sketch must equal the assigned one. Check any expression with
       .venv/bin/python scripts/seedlm34_task_check.py --sketch "<expression>" --args <n> --literals "<list>"
   where `--literals` lists the numbers and quoted strings your description states (they become `c0`, `c1`...).
2. **Natural and different.** Write the description as a person setting an exercise would (MBPP style, one
   sentence, "Write a function to ..." or "Write a python function to ..."). The four tasks of one sketch
   should differ in topic, data and hole contents, and none may copy an example's description or expression.
   Vary the data types where the structure allows (numbers, strings, lists, tuples, dictionaries).
3. **Determined by the description and the first example.** Every number and quoted string the expression
   uses, other than small integers (-1 to 10) and short separators, must be stated in the description.
   The three inputs must give at least two different outputs and should exercise the task (an empty or
   edge case is welcome as the second or third input).
4. **Correct and safe.** The expression must run on all three inputs; keep inputs small (short lists and
   strings). Library functions are allowed when an example uses them in that position; otherwise prefer
   plain Python (builtins, methods, comprehensions, `math`, `re`).

## Checker (run from /home/user/MyLittleLLM; repeat until 0 problems)

    .venv/bin/python scripts/seedlm34_task_check.py --part <K>

It prints every task with a problem (failed teacher-task check, wrong sketch, copy of a training task) and a
count line; `sketches_with_fewer_than_4_good` should reach 0. If a sketch truly admits no natural task, write
three and say why in your final summary.

## Hard rules

1. Do not open, list, grep or print anything under `data/`, `reports/`, `outputs/rule_pilot/`,
   `outputs/seedlm/`, `outputs/rule_round4/`, `outputs/paraphrase/`, or the other parts' files. Do not look for
   MBPP, MBPP+ or HumanEval data anywhere, and do not reproduce benchmark tasks you remember.
2. Touch only your output file and your scratch folder. No git commands. Use `.venv/bin/python`.
3. Finish with a short plain-text summary: tasks written, the checker's final counts, sketches you could not
   fill with four natural tasks and why.
