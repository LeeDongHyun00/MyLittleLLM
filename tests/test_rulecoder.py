import unittest
from unittest import mock

from mylittlellm.rulecoder import backinfer, rules
from mylittlellm.rulecoder.checkers import hint, make_checkers
from mylittlellm.rulecoder.codemods import add_missing_imports, alias_entry_point, autofix
from mylittlellm.rulecoder.extract import extract_code, keep_definitions
from mylittlellm.rulecoder.pipeline import agrees, back_infer, choose, public_check, solve, solve_branching
from mylittlellm.rulecoder.probes import example_args, probe_inputs, run_probes
from mylittlellm.rulecoder.sandbox import run_call, run_public_asserts, syntax_error
from mylittlellm.rulecoder.stats import mcnemar_exact, wilson
from mylittlellm.rulecoder.tasks import make_split, parse_prompt

TASK = {"task_id": "Mbpp/0", "entry_point": "double",
        "prompt": '"""\nWrite a function to double a number.\nassert double(2) == 4\n"""\n'}


def reply(text):
    return {"text": text, "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 5,
            "prompt_tokens_processed": 10, "prompt_ms": 1.0, "predicted_ms": 1.0, "elapsed_seconds": 0.0}


class ScriptedStudent:
    """Stands in for the model so the loop logic is tested without a server."""

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def __call__(self, messages, max_tokens):
        self.calls.append([dict(m) for m in messages])
        return reply(self.texts[len(self.calls) - 1])


class TaskTests(unittest.TestCase):
    def test_prompt_is_split_into_description_and_visible_assert(self):
        description, asserts = parse_prompt(TASK["prompt"])
        self.assertEqual(description, "Write a function to double a number.")
        self.assertEqual(asserts, "assert double(2) == 4")

    def test_split_is_deterministic_disjoint_and_keeps_proportions(self):
        ids = [f"Mbpp/{i}" for i in range(100)]
        first, second = make_split(ids), make_split(reversed(ids))
        self.assertEqual(first, second)
        self.assertEqual({k: len(v) for k, v in first.items()}, {"test": 40, "val": 12, "train": 48})
        self.assertEqual(len(set(first["test"]) | set(first["val"]) | set(first["train"])), 100)


class ExtractTests(unittest.TestCase):
    def test_block_defining_the_entry_point_wins_over_an_earlier_block(self):
        text = "Usage:\n```python\nprint(double(2))\n```\nCode:\n```python\ndef double(x):\n    return 2 * x\n```"
        self.assertEqual(extract_code(text, "double"), "def double(x):\n    return 2 * x\n")

    def test_unterminated_fence_and_plain_text_are_still_extracted(self):
        self.assertEqual(extract_code("```python\ndef f():\n    return 1", "f"), "def f():\n    return 1\n")
        self.assertEqual(extract_code("def f():\n    return 1", "f"), "def f():\n    return 1\n")

    def test_demo_statements_are_dropped_but_needed_constants_survive(self):
        code = "import math\nK = 2\nUNUSED = 5\ndef double(x):\n    return K * x\nprint(double(1))\nassert double(1) == 2\n"
        kept = keep_definitions(code)
        self.assertIn("K = 2", kept)
        self.assertNotIn("UNUSED", kept)
        self.assertNotIn("print", kept)
        self.assertNotIn("assert", kept)

    def test_unparseable_code_is_returned_unchanged(self):
        self.assertEqual(keep_definitions("def f(:\n"), "def f(:\n")


class SandboxTests(unittest.TestCase):
    def test_visible_assert_outcomes(self):
        self.assertEqual(run_public_asserts("def double(x):\n    return 2 * x\n", "assert double(2) == 4")["status"], "pass")
        failed = run_public_asserts("def double(x):\n    return x\n", "assert double(2) == 4")
        self.assertEqual((failed["status"], failed["got"], failed["expected"]), ("fail", "2", "4"))
        error = run_public_asserts("def double(x):\n    return y\n", "assert double(2) == 4")
        self.assertEqual(error["status"], "error")
        self.assertIn("NameError", error["detail"])
        self.assertIn("solution line 2", error["detail"])

    def test_infinite_loop_times_out(self):
        result = run_public_asserts("def double(x):\n    while True:\n        pass\n", "assert double(2) == 4", timeout=2)
        self.assertEqual(result["status"], "timeout")

    def test_harness_math_does_not_hide_a_missing_import_in_the_solution(self):
        asserts = "assert math.isclose(area(1), 3.14159, rel_tol=0.001)"
        self.assertEqual(run_public_asserts("import math\ndef area(r):\n    return math.pi * r * r\n", asserts)["status"], "pass")
        self.assertEqual(run_public_asserts("def area(r):\n    return math.pi * r * r\n", asserts)["status"], "error")

    def test_single_call_reports_value_exception_and_missing_function(self):
        self.assertEqual(run_call("def f(x):\n    return [x] * 2\n", "f", (3,))["repr"], "[3, 3]")
        self.assertEqual(run_call("def f(x):\n    return x[0]\n", "f", ([],))["status"], "exception")
        self.assertEqual(run_call("def g(x):\n    return x\n", "f", (1,))["stage"], "entry_point")

    def test_syntax_error_is_reported_without_executing(self):
        self.assertIn("SyntaxError", syntax_error("def f(:\n    pass"))
        self.assertIsNone(syntax_error("def f():\n    pass"))


class PipelineTests(unittest.TestCase):
    good, bad = "```python\ndef double(x):\n    return 2 * x\n```", "```python\ndef double(x):\n    return x\n```"

    def test_without_verification_the_first_answer_is_final(self):
        student = ScriptedStudent(self.bad)
        result = solve(TASK, client=student)
        self.assertEqual(len(student.calls), 1)
        self.assertEqual(result["final_code"], "def double(x):\n    return x\n")

    def test_failed_visible_example_triggers_repair_with_concrete_feedback(self):
        student = ScriptedStudent(self.bad, self.good)
        result = solve(TASK, verify=True, client=student)
        self.assertEqual(len(student.calls), 2)
        feedback = student.calls[1][-1]["content"]
        self.assertIn("assert double(2) == 4", feedback)
        self.assertIn("evaluated to: 2", feedback)
        self.assertEqual(result["final_code"], "def double(x):\n    return 2 * x\n")

    def test_repairs_stop_at_the_budget(self):
        student = ScriptedStudent(self.bad, self.bad, self.bad)
        result = solve(TASK, verify=True, max_repairs=2, client=student)
        self.assertEqual(len(student.calls), 3)
        self.assertFalse(result["attempts"][-1]["check"]["ok"])

    def test_rule_block_is_appended_to_the_system_prompt_only(self):
        student = ScriptedStudent(self.good)
        solve(TASK, rule_block="Follow these rules:\n1. Return, do not print.", client=student)
        self.assertIn("1. Return, do not print.", student.calls[0][0]["content"])
        self.assertNotIn("Follow these rules", student.calls[0][1]["content"])

    def test_checker_hint_on_a_passing_candidate_asks_for_one_more_attempt(self):
        student = ScriptedStudent(self.good, self.good)
        hints = iter([["Handle negative numbers."], []])
        result = solve(TASK, verify=True, checkers=lambda *a: next(hints), client=student)
        self.assertEqual(len(student.calls), 2)
        self.assertIn("Handle negative numbers.", student.calls[1][-1]["content"])
        self.assertEqual(result["attempts"][1]["hints"], [])

    def test_choice_prefers_passing_then_fewest_flags_then_earliest(self):
        ok, no = {"ok": True}, {"ok": False}
        attempts = [{"round": 0, "check": ok, "hints": ["a"], "code": "A"},
                    {"round": 1, "check": ok, "hints": ["a"], "code": "B"},
                    {"round": 2, "check": no, "hints": [], "code": "C"}]
        self.assertEqual(choose(attempts)["code"], "A")
        attempts[1]["hints"] = []
        self.assertEqual(choose(attempts)["code"], "B")
        self.assertEqual(choose([{"round": 0, "check": no, "hints": [], "code": "A"},
                                 {"round": 1, "check": no, "hints": [], "code": "B"}])["code"], "B")

    def test_syntax_gate_runs_before_execution(self):
        check = public_check("def double(x:\n    return 1\n", "assert double(2) == 4")
        self.assertEqual((check["ok"], check["kind"]), (False, "syntax"))


class RuleBankTests(unittest.TestCase):
    def test_lasso_recovers_helpful_rule_and_zeroes_the_noise_rule(self):
        # Outcome change depends only on rule 0; rule 1 is irrelevant by construction.
        rows = [[float(i % 2), float((i // 2) % 2)] for i in range(40)]
        targets = [0.5 * row[0] for row in rows]
        intercept, coef = rules.lasso(rows, targets, 0.0)
        self.assertAlmostEqual(coef[0], 0.5, places=6)
        self.assertAlmostEqual(coef[1], 0.0, places=6)
        self.assertAlmostEqual(intercept, 0.0, places=6)
        _, sparse = rules.lasso(rows, targets, 0.2)
        self.assertEqual(sparse, [0.0, 0.0])

    def test_ucb_reward_favours_untested_then_accurate_rules(self):
        self.assertEqual(rules.ucb_reward(0, 0, 10), float("inf"))
        self.assertGreater(rules.ucb_reward(8, 10, 100), rules.ucb_reward(4, 10, 100))
        self.assertGreater(rules.ucb_reward(1, 2, 100), rules.ucb_reward(50, 100, 100))

    def test_random_subsets_are_reproducible_per_key(self):
        ids = [f"R{i}" for i in range(12)]
        self.assertEqual(rules.random_subset(ids, "run:Mbpp/1:0"), rules.random_subset(ids, "run:Mbpp/1:0"))
        self.assertNotEqual(rules.random_subset(ids, "run:Mbpp/1:0"), rules.random_subset(ids, "run:Mbpp/2:0"))

    def test_rendered_block_is_numbered(self):
        self.assertEqual(rules.render(["a", "b"], header="H"), "H\n1. a\n2. b")

    def test_importance_vanishes_for_always_or_never_used_rules(self):
        self.assertEqual(rules.rule_importance(0.3, 1.0), 0.0)
        self.assertAlmostEqual(rules.rule_importance(-0.3, 0.5), 0.15)


class TriggerTests(unittest.TestCase):
    def test_rule_fires_only_when_its_pattern_matches_the_prompt(self):
        rule = {"id": "R1", "text": "t", "trigger": {"type": "prompt_regex", "pattern": r"double\(\d"}}
        self.assertTrue(rules.applies(rule, TASK))
        self.assertFalse(rules.applies(dict(rule, trigger={"type": "prompt_regex", "pattern": "tuple"}), TASK))
        self.assertTrue(rules.applies({"id": "R2", "text": "t"}, TASK))
        with self.assertRaises(ValueError):
            rules.applies({"trigger": {"type": "llm_judge"}}, TASK)


class ProbeTests(unittest.TestCase):
    def test_example_arguments_are_read_from_the_visible_assert(self):
        self.assertEqual(example_args("assert set(f((3, 4), [5])) == {4}", "f"), [(3, 4), [5]])
        self.assertIsNone(example_args("assert f(g(1)) == 2", "f"))
        self.assertIsNone(example_args("assert other(1) == 2", "f"))

    def test_each_probe_changes_one_argument_and_keeps_its_type(self):
        probes = list(probe_inputs([[3, 1], 2]))
        self.assertIn(("empty", [[], 2]), probes)
        self.assertIn(("negative", [[3, 1], -2]), probes)
        self.assertIn(("zero", [[0, 0], 2]), probes)
        self.assertTrue(all(isinstance(args[0], list) for _, args in probes))
        self.assertFalse(any(family == "negative" for family, _ in probe_inputs([True])))

    def test_probe_reports_crashes_but_never_judges_values(self):
        crash = run_probes("def first(xs):\n    return xs[0]\n", "first", "assert first([3, 1, 2]) == 3")
        self.assertEqual([f["family"] for f in crash], ["empty"])
        self.assertIn("IndexError", crash[0]["detail"])
        wrong_but_total = run_probes("def first(xs):\n    return None\n", "first", "assert first([3, 1, 2]) == 3")
        self.assertEqual(wrong_but_total, [])

    def test_only_selected_families_are_run(self):
        code = "def first(xs):\n    return xs[0]\n"
        self.assertEqual(run_probes(code, "first", "assert first([3, 1, 2]) == 3", families={"negative"}), [])


class CheckerTests(unittest.TestCase):
    def test_checkers_stay_silent_until_the_visible_example_passes(self):
        run = make_checkers({"empty"})
        code = "def first(xs):\n    return xs[0]\n"
        task = {"entry_point": "first"}
        self.assertEqual(run(task, code, "assert first([3]) == 3", {"ok": False}), [])
        hints = run(task, code, "assert first([3]) == 3", {"ok": True})
        self.assertEqual(len(hints), 1)
        self.assertIn("first([])", hints[0])
        self.assertEqual(make_checkers(set())(task, code, "assert first([3]) == 3", {"ok": True}), [])

    def test_hint_names_the_failing_call(self):
        text = hint({"family": "large", "call": "f(107)", "problem": "timeout", "detail": "no result"})
        self.assertIn("f(107)", text)
        self.assertIn("efficient", text)


class CodemodTests(unittest.TestCase):
    def test_used_but_unimported_stdlib_names_get_an_import(self):
        fixed = add_missing_imports("def f(x):\n    return math.sqrt(x) + len(Counter(str(x)))\n")
        self.assertIn("import math\n", fixed)
        self.assertIn("from collections import Counter\n", fixed)
        self.assertEqual(run_call(fixed, "f", (4,))["repr"], "3.0")

    def test_names_the_answer_binds_itself_are_left_alone(self):
        code = "def sqrt(x):\n    return x\ndef f(math):\n    return sqrt(math)\n"
        self.assertEqual(add_missing_imports(code), code)
        imported = "import math\ndef f(x):\n    return math.pi * x\n"
        self.assertEqual(add_missing_imports(imported), imported)

    def test_single_function_is_aliased_to_the_name_the_example_calls(self):
        fixed = alias_entry_point("def sum_common_divisors(a, b):\n    return a + b\n", "sum")
        self.assertEqual(run_call(fixed, "sum", (1, 2))["repr"], "3")

    def test_with_helpers_only_a_spelling_variant_is_aliased(self):
        code = "def helper(x):\n    return x\ndef check_occurrences(x):\n    return helper(x)\n"
        self.assertIn("check_occurences = check_occurrences", alias_entry_point(code, "check_occurences"))
        self.assertEqual(alias_entry_point(code, "solve"), code)

    def test_defined_entry_point_and_broken_code_are_untouched(self):
        good = "def f(x):\n    return x\n"
        self.assertEqual(autofix(good, "f"), good)
        self.assertEqual(autofix("def f(:\n", "f"), "def f(:\n")

    def test_pipeline_applies_codemods_before_checking(self):
        student = ScriptedStudent("```python\ndef twice(x):\n    return 2 * x\n```")
        result = solve(TASK, verify=True, autofix=autofix, client=student)
        self.assertEqual(len(student.calls), 1)
        self.assertTrue(result["attempts"][0]["autofixed"])
        self.assertTrue(result["attempts"][0]["check"]["ok"])
        self.assertNotIn("double =", result["attempts"][0]["raw_code"])
        self.assertIn("double = twice", result["final_code"])


class BackInferenceTests(unittest.TestCase):
    def infer(self, description, asserts, entry_point):
        return backinfer.synthesize(description, asserts, entry_point, weights={})

    def test_example_is_parsed_in_every_assert_shape(self):
        eq = backinfer.parse_example("assert f((1, 2), 3) == [1]", "f")
        self.assertEqual((eq["args"], eq["expected"], eq["mode"]), ([(1, 2), 3], [1], "eq"))
        as_set = backinfer.parse_example("assert set(f([1, 2])) == set((2, 1))", "f")
        self.assertEqual((as_set["expected"], as_set["mode"]), ({1, 2}, "set"))
        close = backinfer.parse_example("assert math.isclose(f(2), 12.56, rel_tol=0.001)", "f")
        self.assertEqual((close["expected"], close["mode"]), (12.56, "close"))
        bare = backinfer.parse_example('assert f("annie")', "f")
        self.assertEqual((bare["args"], bare["expected"], bare["mode"]), (["annie"], True, "eq"))
        negated = backinfer.parse_example("assert not f([1, 2], 3)", "f")
        self.assertEqual((negated["args"], negated["expected"]), ([[1, 2], 3], False))
        self.assertIsNone(backinfer.parse_example("assert f(x)", "f"))        # not a literal argument
        self.assertIsNone(backinfer.parse_example("assert g(3) == 1", "f"))

    def test_most_common_breaks_ties_by_first_occurrence_in_the_search_and_in_the_emitted_code(self):
        op = backinfer.OPERATORS["most_common/1"]
        tied = ["pear", "fig", "kiwi", "fig", "pear"]
        self.assertEqual(op.fn(tied), "pear")
        self.assertEqual(eval(op.template.format("a0"), {"a0": tied}), "pear")
        # String hashing is seeded per process, so only a fresh interpreter shows an order dependence.
        code = "def f(a0):\n    return " + op.template.format("a0") + "\n"
        self.assertEqual({run_call(code, "f", [tied])["repr"] for _ in range(4)}, {"'pear'"})

    def test_found_program_reproduces_the_example_and_generalises(self):
        found = self.infer("Write a function to perform the mathematical bitwise and operation across the tuples.",
                           "assert and_tuples((10, 4, 6, 9), (5, 2, 3, 3)) == (0, 0, 2, 1)", "and_tuples")
        self.assertTrue(found["confident"])
        self.assertEqual(run_public_asserts(found["code"], "assert and_tuples((1, 2), (3, 3)) == (1, 2)")["status"],
                         "pass")

    def test_description_decides_between_programs_that_fit_the_same_example(self):
        asserts = "assert pick([2, 7, 7]) == 7"
        largest = self.infer("Write a function to find the largest value in a list.", asserts, "pick")
        last = self.infer("Write a function to return the last element of a list.", asserts, "pick")
        self.assertEqual((largest["expr"], last["expr"]), ("max(a0)", "a0[-1]"))

    def test_knowledge_operator_needs_both_the_name_and_the_example(self):
        named = self.infer("Write a function to find the nth hexagonal number.", "assert hexagonal_num(10) == 190",
                           "hexagonal_num")
        self.assertTrue(named["knowledge"])
        self.assertEqual(named["expr"], "a0 * (2 * a0 - 1)")
        unnamed = self.infer("Write a function to compute the special value of n.", "assert special(10) == 190",
                             "special")
        self.assertFalse(unnamed and unnamed["knowledge"])
        wrong_example = self.infer("Write a function to find the nth hexagonal number.",
                                   "assert hexagonal_num(10) == 191", "hexagonal_num")
        self.assertFalse(wrong_example and wrong_example["knowledge"])

    def test_helper_source_is_shipped_with_the_answer(self):
        found = self.infer("Write a function to find the n'th lucas number.", "assert find_lucas(9) == 76", "find_lucas")
        self.assertIn("def _lucas(n):", found["code"])
        self.assertEqual(run_public_asserts(found["code"], "assert find_lucas(4) == 7")["status"], "pass")

    def test_tiny_expected_value_without_named_steps_is_not_trusted(self):
        found = self.infer("Write a function to compute the answer.", "assert answer([5, 1, 3]) == 3", "answer")
        self.assertTrue(found is None or not found["confident"])

    def test_boolean_examples_are_left_to_the_model(self):
        self.assertIsNone(self.infer("Write a function to check for an even number.", "assert is_even(4) == True",
                                     "is_even"))

    def test_pipeline_uses_inference_before_spending_a_repair_call(self):
        task = {"task_id": "Mbpp/0", "entry_point": "and_tuples",
                "prompt": '"""\nWrite a function to extract the elementwise and tuples from the given two tuples.\n'
                          'assert and_tuples((10, 4, 6, 9), (5, 2, 3, 3)) == (0, 0, 2, 1)\n"""\n'}
        student = ScriptedStudent("```python\ndef and_tuples(a, b):\n    return a\n```")
        result = solve(task, verify=True, infer=True, client=student)
        self.assertEqual(len(student.calls), 1)
        self.assertEqual(result["attempts"][-1]["source"], "backinfer")
        self.assertEqual(run_public_asserts(result["final_code"], "assert and_tuples((6, 5), (3, 4)) == (2, 4)")["status"],
                         "pass")

    def test_pipeline_falls_back_to_repair_when_nothing_is_inferred(self):
        student = ScriptedStudent("```python\ndef double(x):\n    return True\n```",
                                  "```python\ndef double(x):\n    return 2 * x\n```")
        task = dict(TASK, prompt='"""\nWrite a function.\nassert double(2) == True\n"""\n')
        result = solve(task, verify=True, infer=True, client=student)
        self.assertEqual(len(student.calls), 1)   # the first answer already satisfies this example
        task = dict(TASK, prompt='"""\nWrite a function to frobnicate.\nassert double("ab") == None\n"""\n')
        student = ScriptedStudent("```python\ndef double(x):\n    return 1\n```",
                                  "```python\ndef double(x):\n    return None\n```")
        result = solve(task, verify=True, infer=True, client=student)
        self.assertEqual(len(student.calls), 2)
        self.assertTrue(result["attempts"][-1]["check"]["ok"])


class BranchingTests(unittest.TestCase):
    good, bad = PipelineTests.good, PipelineTests.bad
    rule = "Follow these rules when writing the solution:\n1. Doubling means multiplying by two."

    def test_rule_answer_that_passes_the_example_ends_the_search(self):
        student = ScriptedStudent(self.good)
        result = solve_branching(TASK, rule_block=self.rule, client=student)
        self.assertEqual((len(student.calls), result["chosen"]), (1, "rules"))
        self.assertIn("Doubling", student.calls[0][0]["content"])

    def test_plain_answer_replaces_a_rule_answer_that_fails_the_example(self):
        student = ScriptedStudent(self.bad, self.good)
        result = solve_branching(TASK, rule_block=self.rule, client=student)
        self.assertEqual((len(student.calls), result["chosen"]), (2, "plain"))
        self.assertNotIn("Doubling", student.calls[1][0]["content"])
        self.assertEqual(result["final_code"], "def double(x):\n    return 2 * x\n")

    def test_inference_is_the_last_branch_and_costs_no_call(self):
        task = {"task_id": "Mbpp/0", "entry_point": "and_tuples",
                "prompt": '"""\nWrite a function to extract the elementwise and tuples from the given two tuples.\n'
                          'assert and_tuples((10, 4, 6, 9), (5, 2, 3, 3)) == (0, 0, 2, 1)\n"""\n'}
        student = ScriptedStudent("```python\ndef and_tuples(a, b):\n    return a\n```")
        result = solve_branching(task, client=student)
        self.assertEqual((len(student.calls), result["chosen"]), (1, "backinfer"))

    def test_when_every_branch_fails_the_first_answer_is_returned(self):
        task = dict(TASK, prompt='"""\nWrite a function to frobnicate.\nassert double("ab") == None\n"""\n')
        student = ScriptedStudent("```python\ndef double(x):\n    return 1\n```",
                                  "```python\ndef double(x):\n    return 2\n```")
        result = solve_branching(task, rule_block=self.rule, client=student)
        self.assertEqual(len(student.calls), 2)
        self.assertEqual(result["final_code"], "def double(x):\n    return 1\n")

    def test_more_public_examples_pick_the_candidate_that_fits_all_of_them(self):
        # One example cannot tell "drop the first item" from "sort, then drop the largest".
        task = {"task_id": "Mbpp/0", "entry_point": "rest",
                "prompt": '"""\nWrite a function to return the remaining items.\nassert rest([3, 1, 2]) == [1, 2]\n"""\n'}
        asserts = "assert rest([3, 1, 2]) == [1, 2]"
        single = back_infer(task, asserts, 0)
        self.assertEqual(run_public_asserts(single["code"], asserts)["status"], "pass")
        examples = [([[3, 1, 2]], [1, 2]), ([[5, 4, 9]], [4, 5])]
        checked = []

        def accept(code):
            checked.append(code)
            return True

        several = back_infer(task, asserts, 0, accept=accept, examples=examples)
        self.assertEqual(len(checked), 1)                    # only a candidate that fits every example is checked
        self.assertEqual(run_public_asserts(several["code"], "assert rest([9, 7, 8]) == [7, 8]")["status"], "pass")
        self.assertTrue(several["check"]["ok"])
        self.assertIsNone(back_infer(task, asserts, 0, accept=lambda code: False, examples=examples))

    def test_agreement_filter_rejects_wrong_results_errors_and_runaway_code(self):
        examples = [([2], 4), ([3], 6)]
        self.assertTrue(agrees("def f(a0):\n    return 2 * a0\n", "f", examples))
        self.assertFalse(agrees("def f(a0):\n    return a0 + 2\n", "f", examples))
        self.assertFalse(agrees("def f(a0):\n    return a0[0]\n", "f", examples))
        self.assertFalse(agrees("def f(a0):\n    while True:\n        pass\n", "f", examples, seconds=0.05))
        self.assertTrue(agrees("def f(a0):\n    return a0 / 3\n", "f", [([1], 0.3333333333)]))
        unordered = [([[2, 1, 2]], [1, 2])]
        self.assertFalse(agrees("def f(a0):\n    return list(dict.fromkeys(a0))\n", "f", unordered))
        self.assertTrue(agrees("def f(a0):\n    return list(dict.fromkeys(a0))\n", "f", unordered, mode="set"))

    small_sum = {"task_id": "Mbpp/0", "entry_point": "small_sum",
                 "prompt": '"""\nWrite a function to add up the n smallest items.\n'
                           'assert small_sum([5, 1, 4, 2], 2) == 3\n"""\n'}
    wrong_sum = "```python\ndef small_sum(a, n):\n    return 0\n```"

    def test_a_proposer_is_the_last_branch_after_the_search_finds_nothing(self):
        student, asked = ScriptedStudent(self.wrong_sum), []

        def proposer(args, expected, named):
            asked.append((args, expected))
            return [{"program": ["sum/1", "take/2", "sorted/1", "a0", "a1"], "log_prob": -0.5}]

        with mock.patch("mylittlellm.rulecoder.pipeline.back_infer", return_value=None):
            result = solve_branching(self.small_sum, client=student, proposer=proposer)
        self.assertEqual(asked, [([[5, 1, 4, 2], 2], 3)])
        self.assertEqual((len(student.calls), result["chosen"]), (1, "seedlm"))
        self.assertEqual(run_public_asserts(result["final_code"], "assert small_sum([9, 7, 8], 1) == 7")["status"],
                         "pass")

    def test_a_proposer_is_not_asked_when_an_earlier_branch_reproduces_the_example(self):
        def proposer(args, expected, named):
            raise AssertionError("an answer that already passes must not be replaced")

        result = solve_branching(TASK, client=ScriptedStudent(self.good), proposer=proposer)
        self.assertEqual(result["chosen"], "plain")

    def test_a_proposal_that_does_not_reproduce_the_example_is_discarded(self):
        def proposer(args, expected, named):
            return [{"program": ["sum/1", "a0"], "log_prob": -0.1}]      # 12, not 3

        with mock.patch("mylittlellm.rulecoder.pipeline.back_infer", return_value=None):
            result = solve_branching(self.small_sum, client=ScriptedStudent(self.wrong_sum), proposer=proposer)
        self.assertEqual(result["chosen"], "plain")
        self.assertEqual(result["final_code"], "def small_sum(a, n):\n    return 0\n")

    def test_boolean_examples_are_never_sent_to_the_proposer(self):
        def proposer(args, expected, named):
            raise AssertionError("the proposer was not trained on boolean results")

        task = dict(TASK, prompt='"""\nWrite a function to frobnicate.\nassert double(2) == True\n"""\n')
        student = ScriptedStudent("```python\ndef double(x):\n    return 0\n```")     # fails: 0 != True
        with mock.patch("mylittlellm.rulecoder.pipeline.back_infer", return_value=None):
            result = solve_branching(task, client=student, proposer=proposer)
        self.assertEqual(result["chosen"], "plain")
        self.assertFalse(result["attempts"][0]["check"]["ok"])


class StatsTests(unittest.TestCase):
    def test_wilson_interval_brackets_the_estimate(self):
        low, high = wilson(60, 100)
        self.assertTrue(low < 60 < high)
        self.assertEqual(wilson(0, 0), (0.0, 0.0))

    def test_mcnemar_exact_matches_hand_computed_values(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
