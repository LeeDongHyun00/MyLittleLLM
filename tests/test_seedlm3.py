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


class PyxTests(unittest.TestCase):
    def test_round_trip_keeps_meaning_and_binds_variables_by_depth(self):
        from mylittlellm.seedlm import pyx
        source = "sum(1 for x in a0 if x % 3 == 0) + max(len(w) for w in a1)"
        tokens = pyx.serialize(source, 2)
        self.assertEqual(tokens.count("v0"), 2)          # both comprehensions bind v0 at depth 0
        self.assertNotIn("v1", tokens)
        args = [[3, 4, 6, 9], ["ab", "abcd"]]
        self.assertEqual(pyx.execute(tokens, args), 3 + 4)
        nested = pyx.serialize("[[x * y for y in a0] for x in a0]", 1)
        self.assertIn("v1", nested)
        self.assertEqual(pyx.execute(nested, [[1, 2]]), [[1, 2], [2, 4]])

    def test_constants_become_literal_output_and_small_integer_tokens(self):
        from mylittlellm.seedlm import pyx
        tokens = pyx.serialize("'Equal' if a0.count('xy') > 2 else 'Not Equal'", 1, ["xy"], "Equal")
        self.assertEqual(tokens, ["@IfExp/3", "@Gt/2", "@.count/2", "a0", "c0", "#2", "o", "o!"])
        self.assertEqual(pyx.execute(tokens, ["xyxyxy"], ["xy"], "Equal"), "Equal")
        self.assertEqual(pyx.execute(tokens, ["xy"], ["xy"], "Equal"), "Not Equal")
        with self.assertRaises(pyx.Unsupported):
            pyx.serialize("a0 + 12345", 1)

    def test_map_filter_and_key_functions_are_rewritten(self):
        from mylittlellm.seedlm import pyx
        tokens = pyx.serialize("sorted(map(str, a0), key=len, reverse=True)", 1)
        self.assertEqual(tokens[0], "@sorted:key:reverse/2")
        self.assertEqual(pyx.execute(tokens, [[5, 100, 23]]), ["100", "23", "5"])
        tokens = pyx.serialize("list(filter(None, a0))", 1)
        self.assertEqual(pyx.execute(tokens, [[0, 1, "", "a"]]), [1, "a"])

    def test_emitted_code_is_standalone(self):
        from mylittlellm.seedlm import pyx
        tokens = pyx.serialize("len_1(shortest_1(a0)) * 2", 1)
        code = pyx.to_code("f", 1, tokens)
        scope = {}
        exec(code, scope)
        self.assertEqual(scope["f"]([[1, 2], [3]]), 2)


@unittest.skipUnless(TORCH, "torch not installed")
class Coder3Tests(unittest.TestCase):
    def test_beam_search_only_returns_well_formed_programs_with_bound_variables(self):
        import torch
        from mylittlellm.seedlm import coder3, model as seedmodel, pyx
        torch.manual_seed(1)
        extra = pyx.serialize("sum(1 for x in a0 if x > 2)", 1) + pyx.serialize("sorted(a0, key=lambda t: -t)", 1)
        vocabulary = coder3.Vocabulary(language.build_vocabulary(["count"], phrases={}), coder3.program_tokens(extra))
        for symbol in extra:
            if symbol not in ("a0",):
                self.assertEqual(vocabulary.symbol(vocabulary.token(symbol)), symbol)
        grammar = coder3.Grammar(vocabulary)
        net = seedmodel.SeedLM(seedmodel.Config(len(vocabulary), 32, 1, 2, coder3.MAX_LENGTH)).eval()
        prefix = vocabulary.condition(["count"], [], [[1, 5]], 1, [])
        found = coder3.beam_search(net, vocabulary, grammar, prefix, 1, 0, False, width=8, max_tokens=10)
        self.assertTrue(found)
        for program, _ in found:
            pyx.deserialize(program)                     # raises if malformed
            depth_ok = all(not s.startswith("v") or "@Comp" in " ".join(program) or ":key" in " ".join(program)
                           for s in program)
            self.assertTrue(depth_ok, program)
            self.assertNotIn("o", program)               # the expected output is not a string

    def test_child_depths_follow_comprehension_scoping(self):
        from mylittlellm.seedlm import coder3
        self.assertEqual(coder3.child_depths("@Comp:gen:1i1/3", 0), [1, 0, 1])
        self.assertEqual(coder3.child_depths("@Comp:list:1i0,1i0/3", 1), [3, 1, 2])
        self.assertEqual(coder3.child_depths("@sorted:key/2", 0), [1, 0])
        self.assertEqual(coder3.child_depths("len/1", 2), [2])



@unittest.skipUnless(TORCH, "torch not installed")
class CacheTests(unittest.TestCase):
    def test_cached_decoding_matches_full_recomputation(self):
        import torch
        from mylittlellm.seedlm import model as seedmodel
        torch.manual_seed(0)
        net = seedmodel.SeedLM(seedmodel.Config(50, 32, 2, 4, 64)).eval()
        # Zero-initialised residual projections would make every position identical; perturb them.
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        tokens = torch.randint(0, 50, (3, 12))
        with torch.no_grad():
            full = net(tokens)
            logits, cache = net.prefill(tokens[:, :8])
            torch.testing.assert_close(logits, full[:, 7], rtol=1e-4, atol=1e-4)
            for position in range(8, 12):
                logits, cache = net.step(tokens[:, position], cache)
                torch.testing.assert_close(logits, full[:, position], rtol=1e-4, atol=1e-4)


class GuardTests(unittest.TestCase):
    def test_exploding_operations_fail_fast_instead_of_hanging(self):
        import time
        from mylittlellm.seedlm import pyx
        started = time.monotonic()
        for source in ("a0 ** a0 ** a0", "'ab' * 10 ** 8", "1 << 10 ** 9", "math.factorial(10 ** 6)"):
            with self.assertRaises(OverflowError):
                pyx.execute(pyx.serialize(source, 1), [99])
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(pyx.execute(pyx.serialize("a0 ** 3 * 'ab'.count('a')", 1), [4]), 64)

if __name__ == "__main__":
    unittest.main()
