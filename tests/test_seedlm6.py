import importlib.util
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mylittlellm.seedlm import language, pyx, pyx_words

TORCH = importlib.util.find_spec("torch") is not None


class PyxWordsTests(unittest.TestCase):
    def test_python_nodes_get_task_words_in_reading_order(self):
        tokens = pyx.serialize("sum(v * v for v in a0 if v % 2 == 1)", 1)
        words = pyx_words.describe(random.Random(0), tokens, (), [[1, 2, 3]])
        text = " ".join(words)
        # Outer pieces first: the aggregate is named before the element computation.
        self.assertTrue(any(w in words for w in ("sum", "total", "add")))
        self.assertIn("list", words)
        first_sum = min(i for i, w in enumerate(words) if w in ("sum", "total", "add"))
        self.assertLess(first_sum, max(i for i, w in enumerate(words) if w in ("multiply", "product", "times")),
                        text)

    def test_every_node_head_in_training_has_words(self):
        for head in list(pyx_words.BINARY) + list(pyx_words.BUILTIN):
            token = f"@{head}/2"
            self.assertTrue(pyx_words.node_words(token), token)
        self.assertEqual(pyx_words.node_words("a0"), None)
        self.assertIn("split", pyx_words.node_words("@.split/2"))

    def test_all_words_are_words(self):
        words = pyx_words.all_words()
        self.assertIn("product", words)
        self.assertTrue(all(language.words(w) == [w] for w in words))


@unittest.skipUnless(TORCH, "the graft generator imports the training scripts")
class GraftTests(unittest.TestCase):
    def test_variables_must_be_used(self):
        from seedlm6_grafts import uses_its_variables
        self.assertFalse(uses_its_variables(pyx.serialize("[a0 for v in a0]", 1)))
        self.assertFalse(uses_its_variables(pyx.serialize("[v for v in a0 if 1]", 1)))
        self.assertTrue(uses_its_variables(pyx.serialize("sum(1 for v in a0 if v > 0)", 1)))
        self.assertTrue(uses_its_variables(pyx.serialize("sorted(a0, key=lambda w: len(w))", 1)))
        self.assertFalse(uses_its_variables(pyx.serialize("sorted(a0, key=lambda w: len(a0))", 1)))

    def test_repeated_operands_are_flagged_but_not_squares_or_slices(self):
        from seedlm6_grafts import repeats_operand
        self.assertTrue(repeats_operand(pyx.serialize("a0 >> a0", 1)))
        self.assertFalse(repeats_operand(pyx.serialize("a0 * a0", 1)))
        self.assertFalse(repeats_operand(pyx.serialize("a0[::-1]", 1)))

    def test_nodes_cover_every_position_with_its_parent(self):
        from seedlm6_grafts import nodes
        tokens = pyx.serialize("[v * 2 for v in a0 if v > 1]", 1)
        found = nodes(tokens)
        self.assertEqual(sorted(s for s, _, _, _, _ in found), list(range(len(tokens))))
        root = [n for n in found if n[3] is None]
        self.assertEqual(len(root), 1)
        self.assertEqual((root[0][0], root[0][1]), (0, len(tokens)))

    def test_grafts_join_at_a_new_link_and_keep_values_as_text(self):
        from seedlm6_grafts import grafts, known_edges, parsed, programs
        items = programs()[:300]
        found, _ = grafts(20, seed=1, items=items, tries_per_graft=200)
        self.assertTrue(found)
        edges = known_edges(items)
        for graft in found:
            self.assertNotIn(tuple(graft["junction"]), edges)
            self.assertTrue(all(isinstance(text, str) for text in graft["cases"] + graft["outputs"]))
            values = parsed(graft)
            for args, expected in zip(values["cases"], values["outputs"]):
                self.assertEqual(repr(pyx.execute(graft["tokens"], args, graft["literals"])), repr(expected))


@unittest.skipUnless(TORCH, "the novelty check imports the training scripts")
class NoveltyTests(unittest.TestCase):
    def test_a_known_program_is_not_novel_and_a_new_join_is(self):
        from seedlm6_novelty import verdict
        seen = {"edges": {("@sum/1", 0, "ARG")}, "twigs": {("@sum/1", "ARG")}}
        plain = {"id": "a", "description": "Write a function to find the sum of a list.", "entry_point": "f",
                 "inputs": ["([1, 2],)", "([3],)", "([],)"], "expression": "sum(a0)"}
        joined = dict(plain, id="b", expression="sum(x * x for x in a0)",
                      description="Write a function to find the sum of the squares of a list.")
        self.assertEqual(verdict(plain, seen)["novel"], False)
        self.assertEqual(verdict(joined, seen)["novel"], True)
        self.assertFalse(verdict({"id": "c"}, seen)["ok"])


if __name__ == "__main__":
    unittest.main()
