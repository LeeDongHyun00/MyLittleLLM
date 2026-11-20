import random
import unittest

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.seedlm import pysynth, pyx

HOSTS = [
    {"id": "numbers", "cases": [[[3, 1, 4, 1, 5]], [[9, 2, 6]], [[5, 3, 5, 8, 9, 7]]], "literals": []},
    {"id": "words", "cases": [["hello world", 2], ["a quick fox", 1], ["python", 3]], "literals": []},
    {"id": "pairs", "cases": [[[1, 2, 3], [4, 5, 6]], [[0, 7], [1, 1]], [[2, 2, 2, 2], [3, 1, 3, 1]]], "literals": [10]},
]


class PysynthTests(unittest.TestCase):
    def setUp(self):
        self.rows = pysynth.generate(HOSTS, 120, seed=3)

    def test_programs_run_to_their_stored_outputs(self):
        self.assertGreater(len(self.rows), 60)
        hosts = {h["id"]: h for h in HOSTS}
        for row in self.rows:
            host = hosts[row["host"]]
            for case, output in zip(host["cases"], row["outputs"]):
                self.assertEqual(repr(pyx.execute(list(row["tokens"]), case, host["literals"])), output, row["text"])

    def test_programs_use_an_argument_and_depend_on_the_input(self):
        for row in self.rows:
            self.assertTrue(any(t.startswith("a") and t[1:].isdigit() for t in row["tokens"]), row["text"])
            self.assertGreater(len(set(row["outputs"])), 1, row["text"])

    def test_python_nodes_and_library_operators_meet(self):
        mixed = [r for r in self.rows if any(t.startswith("@") for t in r["tokens"])
                 and any(t in bi.OPERATORS for t in r["tokens"])]
        self.assertTrue(mixed)

    def test_same_seed_same_programs(self):
        again = pysynth.generate(HOSTS, 120, seed=3)
        self.assertEqual([r["tokens"] for r in again], [r["tokens"] for r in self.rows])

    def test_library_modes(self):
        first, second = sorted(bi.OPERATORS)[:2]
        programs = [[first, "a0"], [first, "a1"], [second, "a0", "a1"], ["@len/1", "a0"]]
        self.assertIsNone(pysynth.library_from(programs, "all"))
        operators, weights = pysynth.library_from(programs, "training")
        self.assertEqual([k for k, _ in operators], [first, second])
        self.assertIsNone(weights)
        _, weights = pysynth.library_from(programs, "sqrt")
        self.assertAlmostEqual(weights[0] ** 2, 2.0)

    def test_steps_never_return_their_operand_unchanged(self):
        maker = pysynth.Generator(random.Random(0))
        ctx = maker._context(HOSTS[0]["cases"], [])
        for _ in range(200):
            made = maker._apply(ctx, ctx.leaves[0], pysynth.FAMILIES, depth=0)
            if made is not None:
                self.assertNotEqual(made.values, ctx.leaves[0].values, made.text)


if __name__ == "__main__":
    unittest.main()
