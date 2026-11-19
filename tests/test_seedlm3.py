import importlib.util
import unittest

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.seedlm import data2, language

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "torch not installed")
class ReviewerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from mylittlellm.seedlm import model as seedmodel
        torch.manual_seed(0)
        self.vocabulary = data2.Vocabulary(language.build_vocabulary(["find the sum of a list"], phrases={}),
                                           list(bi.OPERATORS))
        config = seedmodel.Config(len(self.vocabulary), 32, 1, 2, data2.MAX_LENGTH)
        self.net = seedmodel.SeedLM(config).eval()

    def test_only_the_description_is_trained_on(self):
        from mylittlellm.seedlm import review
        words = ["find", "the", "sum"]
        tokens, start = review.encode(self.vocabulary, ["sum/1", "a0"], [], [[1, 2]], 3, words)
        self.assertEqual(tokens[start - 1], data2.DESC)
        self.assertEqual(tokens[1], data2.PROG)
        self.assertEqual(tokens[start:], [self.vocabulary.word_index[w] for w in words] + [data2.EOS])
        _, labels = data2.collate([(tokens, start, {})], __import__("torch"))
        self.assertEqual(int((labels != -100).sum()), len(words) + 1)

    def test_scores_match_one_at_a_time_and_skip_unknown_operators(self):
        from mylittlellm.seedlm import review
        programs = [["sum/1", "a0"], ["len/1", "a0"], ["no_such_op/1", "a0"], ["max/1", "reverse/1", "a0"]]
        together = review.scores(self.net, self.vocabulary, programs, [], [[1, 2]], 3, ["find", "the", "sum"])
        self.assertIsNone(together[2])
        for program, value in zip(programs, together):
            if value is None:
                continue
            alone = review.scores(self.net, self.vocabulary, [program], [], [[1, 2]], 3, ["find", "the", "sum"])[0]
            self.assertAlmostEqual(value, alone, places=2)
            self.assertLess(value, 0.0)


if __name__ == "__main__":
    unittest.main()
