import unittest

from mylittlellm.rulecoder import backinfer as bi, oplint


class OperatorLibraryTests(unittest.TestCase):
    def test_search_function_and_emitted_code_agree_for_every_shipped_operator(self):
        problems = [p for p in oplint.lint(trials=600) if not p[1].startswith("UNTESTED")]
        self.assertEqual(problems, [])

    def test_lint_catches_a_template_that_differs_from_its_function(self):
        broken = bi.Op("bad_sum", 1, lambda x: (bi.need(bi.numseq(x)), sum(x))[1], "max({0})", "sum")
        found = oplint.lint({"bad_sum/1": broken})
        self.assertEqual(len(found), 1)
        self.assertIn("emitted code returns", found[0][1])

    def test_lint_rejects_an_ungated_predicate_and_a_mutating_operator(self):
        predicate = bi.Op("is_even", 1, lambda x: (bi.need(bi.is_int(x)), x % 2 == 0)[1], "{0} % 2 == 0", "even")
        mutator = bi.Op("sort_in_place", 1, lambda x: (bi.need(isinstance(x, list)), x.sort(), x)[2],
                        "sorted({0})", "sort")
        found = dict((key, problem) for key, problem, _ in oplint.lint({"is_even/1": predicate,
                                                                         "sort_in_place/1": mutator}))
        self.assertIn("not gated", found["is_even/1"])
        self.assertIn("modifies its arguments", found["sort_in_place/1"])

    def test_pack_operators_are_labelled_and_names_are_unique(self):
        keys = [op.name + "/" + str(op.arity) for op in bi.U + bi.B + bi.T]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(op.pack == "core" or op.pack in bi.ACTIVE_PACKS for op in bi.OPERATORS.values()))


if __name__ == "__main__":
    unittest.main()
