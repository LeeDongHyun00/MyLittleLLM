import unittest

from mylittlellm.rulecoder import consensus as cs


class ProbeGrammarTests(unittest.TestCase):
    def test_probes_are_deterministic_distinct_and_keep_types(self):
        args = [[3, 1, 2], (4, 5), "hello world", 7, {"a": 1}]
        first, second = cs.probe_inputs(args), cs.probe_inputs(args)
        self.assertEqual(first, second)
        self.assertEqual(len({cs.canonical(p) for p in first}), len(first))
        self.assertNotIn(args, first)
        for probe in first:
            self.assertEqual([type(v) for v in probe], [type(v) for v in args])

    def test_records_inside_containers_keep_their_length(self):
        for probe in cs.probe_inputs([[(1, "a"), (2, "b"), (3, "c")]], count=30):
            self.assertTrue(all(isinstance(r, tuple) and len(r) == 2 for r in probe[0]))

    def test_an_integer_that_was_a_length_follows_the_new_length(self):
        for values, n in cs.probe_inputs([[5, 6, 7, 8], 4], count=30):
            if len(values) != 4:
                self.assertEqual(n, len(values))

    def test_boundary_probes_come_first_only_when_asked(self):
        args = [[3, 1, 2], 3]
        self.assertEqual(cs.probe_inputs(args, edges=True)[:3], [[[], 0], [[3], 1], [[3, 1], 2]])
        self.assertNotIn([[], 0], cs.probe_inputs(args))


class AgreementTests(unittest.TestCase):
    probes = [[[3, 1, 2], 2], [[5, 4, 9], 1], [[1, 2], 2], [[], 0]]

    def run_engine(self, body):
        return cs.run_engine_code("def f(a0, a1):\n    return " + body + "\n", "f", self.probes)

    def test_agreement_conflict_and_too_little_evidence(self):
        model = cs.run_model_code("def f(a, n):\n    return sorted(a)[:n]\n", "f", self.probes)
        self.assertEqual(cs.compare(model, self.run_engine("sorted(a0)[:a1]")), "agree")
        self.assertEqual(cs.compare(model, self.run_engine("a0[:a1]")), "conflict")
        self.assertEqual(cs.compare(model, self.run_engine("1 // 0")), "unknown")

    def test_an_exception_counts_against_a_value_only_on_strict_boundary_probes(self):
        boundary_first = [[[], 0]] + self.probes[:-1]
        raising = cs.run_engine_code("def f(a0, a1):\n    return sorted(a0)[:a1] if a0 else [][0]\n", "f",
                                     boundary_first)
        model = cs.run_model_code("def f(a, n):\n    return sorted(a)[:n]\n", "f", boundary_first)
        self.assertEqual(cs.compare(raising, model), "agree")
        self.assertEqual(cs.compare(raising, model, strict=1), "conflict")

    def test_floats_compare_to_nine_digits_and_sets_ignore_order(self):
        a = cs.run_engine_code("def f(x):\n    return (0.1 + 0.2, {3, 1, 2})\n", "f", [[0]] * 3)
        b = cs.run_engine_code("def f(x):\n    return (0.3, {1, 2, 3})\n", "f", [[0]] * 3)
        self.assertEqual(cs.compare(a, b), "agree")

    def test_model_code_cannot_hang_or_break_the_comparison(self):
        hanging = cs.run_model_code("def f(a, n):\n    while True:\n        pass\n", "f", self.probes[:2], per_call=0.1)
        self.assertEqual([status for status, _ in hanging], ["exc", "exc"])
        broken = cs.run_model_code("def g(:\n", "f", self.probes[:2])
        self.assertEqual([status for status, _ in broken], ["exc", "exc"])


class SelectionTests(unittest.TestCase):
    def test_mbr_prefers_the_best_supported_behaviour_and_keeps_the_first_on_ties(self):
        same = [("ok", "1"), ("ok", "2"), ("ok", "3")]
        other = [("ok", "1"), ("ok", "9"), ("ok", "3")]
        pool = [{"results": other, "weight": 1}, {"results": same, "weight": 1}, {"results": same, "weight": 1}]
        self.assertEqual(cs.mbr_select(pool)[0], 1)
        tie = [{"results": other, "weight": 1}, {"results": same, "weight": 1}]
        self.assertEqual(cs.mbr_select(tie)[0], 0)


if __name__ == "__main__":
    unittest.main()
