import importlib.util
import random
import unittest

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.seedlm import data2, language

TORCH = importlib.util.find_spec("torch") is not None


class LanguageAndDataTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = data2.Vocabulary(language.build_vocabulary(["find the sum of a list"], phrases={}),
                                           list(bi.OPERATORS))
        self.describer = language.Describer(bi.OPERATORS, phrases={"sum/1": ["add up the numbers in the list"]})

    def test_words_are_lowercase_alphanumeric_runs(self):
        self.assertEqual(language.words("Find the 2nd-largest item, e.g. 'X'!"), ["find", "the", "2", "nd", "largest",
                                                                               "item", "e", "g", "x"])

    def test_generated_examples_reexecute_and_encode_within_limits(self):
        rows = data2.build_synthetic(300, 3, self.vocabulary, self.describer, list(bi.OPERATORS.values()))
        for tokens, start, example in rows:
            self.assertLessEqual(len(tokens), data2.MAX_LENGTH)
            self.assertEqual(tokens[start - 1], data2.PROG)
            self.assertEqual(data2.execute(example["program"], example["args"], example["literals"]), example["expected"])
            self.assertEqual([self.vocabulary.symbol(t) for t in tokens[start:-1]], example["program"])
        self.assertTrue(any(isinstance(ex["expected"], bool) for _, _, ex in rows))
        self.assertTrue(any("c0" in ex["program"] for _, _, ex in rows))

    def test_a_literal_is_inlined_into_the_emitted_expression(self):
        expr, _ = data2.to_expression(["remove_sub/2", "a0", "c0"], ["std"])
        self.assertIn("'std'", expr)
        self.assertEqual(eval(expr, {"a0": "a std b"}), "a  b")

    def test_descriptions_mention_the_teacher_phrase_and_the_literal(self):
        text = self.describer.describe(random.Random(1), ["sum/1", "a0"])
        self.assertIn("numbers", text)
        text = self.describer.describe(random.Random(2), ["remove_sub/2", "a0", "c0"], ["std"])
        self.assertIn("std", text)


@unittest.skipUnless(TORCH, "PyTorch is only needed for the from-scratch model")
class BeamTests(unittest.TestCase):
    def test_beam_search_respects_arguments_literals_and_arity(self):
        import torch
        from mylittlellm.seedlm import model, propose2
        torch.manual_seed(0)
        vocabulary = data2.Vocabulary(["sum"], list(bi.OPERATORS))
        net = model.SeedLM(model.Config(len(vocabulary), 32, 1, 2, data2.MAX_LENGTH))
        prefix = vocabulary.condition(["sum"], [7], [[1, 2]], 3, [])
        for program, _ in propose2.beam_search(net, vocabulary, prefix, 1, 1, width=6):
            self.assertNotIn("a1", program)
            self.assertNotIn("c1", program)
            pending = 1
            for token in program:
                pending += (bi.OPERATORS[token].arity if "/" in token else 0) - 1
            self.assertEqual(pending, 0)


if __name__ == "__main__":
    unittest.main()
