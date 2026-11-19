import importlib.util
import unittest

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.seedlm import data2, language, post, pyx

TORCH = importlib.util.find_spec("torch") is not None

PROGRAMS = [
    ("sum(a0)", 1),
    ("[v * 2 for v in a0 if v > 1]", 1),
    ("[[y for y in x] for x in a0]", 1),
    ("[y for x in a0 for y in x if y]", 1),
    ("max(a0, key=lambda r: r[1])[0]", 1),
    ("sorted(a0, key=lambda w: (len(w), w), reverse=True)", 1),
    ("{k: v for k, v in zip(a0, a1)}", 2),
    ("max(a0, a1, key=abs)", 2),
    ("min(a0, key=len, default=a1)", 2),
    ("any(str(a0).count(d) > 1 for d in str(a0))", 1),
]


class PostfixTests(unittest.TestCase):
    def test_round_trip_through_postfix_keeps_every_program(self):
        for source, n_args in PROGRAMS:
            tokens = pyx.serialize(source, n_args)
            postfix = post.to_postfix(tokens)
            self.assertEqual(post.to_prefix(postfix), tokens, source)
            self.assertEqual(len(postfix), len(tokens) + sum(t.startswith(post.FOR) for t in postfix))

    def test_operands_come_before_operators_and_scopes_open_on_the_iterable(self):
        postfix = post.to_postfix(pyx.serialize("[v * 2 for v in a0 if v > 1]", 1))
        self.assertEqual(postfix, ["a0", "@For:1", "v0", "#1", "@Gt/2", "v0", "#2", "@Mult/2", "@Comp:list:1i1/3"])

    def test_closing_tokens_must_match_the_open_scopes(self):
        frames = post.start()
        for token in ["a0", "@For:1", "v0", "#1", "@Gt/2", "v0"]:
            frames = post.step(frames, token)
        with self.assertRaises(ValueError):
            post.step(frames, "@Comp:list:1i0/2")       # two items in the scope: one condition, one element
        with self.assertRaises(ValueError):
            post.step(frames, "@sorted:key/2")           # a key function's scope holds exactly its body
        closed = post.step(frames, "@Comp:list:1i1/3")
        self.assertTrue(post.finished(closed))

    def test_annotations_report_values_relations_and_vectors(self):
        postfix = post.to_postfix(pyx.serialize("[v * 2 for v in a0 if v > 1]", 1))
        notes = dict(zip(range(len(postfix)), post.annotated(postfix, [[1, 2, 3]], [], [4, 6])))
        self.assertEqual(notes[0][1], ["K:list", "R:type", "N:3"])
        self.assertEqual(notes[1][1][0], "K:vint")                   # @For:1 shows the elements
        self.assertEqual(notes[4][1], ["K:vbool", "R:none", "N:3"])  # v0 > 1 over the bindings
        self.assertEqual(notes[7][1][0], "K:vint")                   # v0 * 2 over the bindings
        self.assertEqual(notes[8][1], ["K:list", "R:eq", "N:2"])
        sums = post.annotated(post.to_postfix(pyx.serialize("len(a0) + 1", 1)), [[5, 5, 5]], [], 4)
        self.assertEqual(sums[1][1], ["K:int", "R:add1", "N:3"])

    def test_relations(self):
        self.assertEqual(post.relation([3, 1, 2], [1, 2, 3]), "perm")
        self.assertEqual(post.relation("abc", "cba"), "rev")
        self.assertEqual(post.relation([1, 2, 3], 3), "len")
        self.assertEqual(post.relation(40, 10), "div4")
        self.assertEqual(post.relation(6, 3), "sub3")      # small differences are reported first
        self.assertNotEqual(post.relation(1, True), "eqv")
        self.assertEqual(post.relation("Hello", "hello"), "case")
        self.assertEqual(post.vector_relation([True, False, True], 2), "count")
        self.assertEqual(post.vector_relation(["a", "b"], True), "none")   # strings are not verdicts

    def test_errors_and_slow_items_are_annotated_not_raised(self):
        notes = post.annotated(post.to_postfix(pyx.serialize("a0 // 0", 1)), [5], [], 1)
        self.assertEqual(notes[-1][1], ["K:err", "R:none", "N:-"])


@unittest.skipUnless(TORCH, "torch not installed")
class Coder4Tests(unittest.TestCase):
    def setUp(self):
        import torch
        from mylittlellm.seedlm import coder4, model as seedmodel
        torch.manual_seed(0)
        extra = {t for source, n in PROGRAMS for t in pyx.serialize(source, n)}
        self.vocabulary = coder4.Vocabulary(language.build_vocabulary(["double the large values"], phrases={}),
                                            coder4.program_tokens(extra))
        config = seedmodel.Config(len(self.vocabulary), 32, 1, 2, coder4.MAX_LENGTH, n_notes=len(coder4.NOTE_INDEX))
        self.net = seedmodel.SeedLM(config).eval()
        self.grammar = coder4.Grammar(self.vocabulary)

    def test_only_program_tokens_are_trained_on_and_notes_ride_along(self):
        from mylittlellm.seedlm import coder4
        postfix = post.to_postfix(pyx.serialize("[v * 2 for v in a0 if v > 1]", 1))
        prefix = self.vocabulary.condition(["double"], [], [[1, 2, 3]], [4, 6], [])
        tokens, side = coder4.encode(self.vocabulary, prefix, post.annotated(postfix, [[1, 2, 3]], [], [4, 6]))
        self.assertEqual(len(tokens), len(prefix) + len(postfix) + 1)
        inputs, notes, labels = coder4.collate([(tokens, len(prefix), {"notes": side})])
        self.assertEqual(int((labels != -100).sum()), len(postfix) + 1)
        self.assertEqual(int((notes[0, :len(prefix)] != 0).sum()), 0)
        self.assertTrue(bool((notes[0, len(prefix):len(prefix) + len(postfix)] != 0).all()))

    def test_no_note_adds_nothing(self):
        import torch
        tokens = torch.tensor([[1, 2, 3]])
        plain = self.net(tokens)
        with_zero_notes = self.net(tokens, notes=torch.zeros((1, 3, 3), dtype=torch.long))
        self.assertTrue(torch.allclose(plain, with_zero_notes))

    def test_grammar_allows_the_gold_program_at_every_step(self):
        for source, n_args in PROGRAMS:
            postfix = post.to_postfix(pyx.serialize(source, n_args))
            base = self.grammar.base_mask(n_args, 0, False)
            frames = post.start()
            for position, token in enumerate(postfix):
                mask = self.grammar.allowed(base, frames, post.MAX_POST - position - 1)
                self.assertTrue(bool(mask[self.vocabulary.token(token)]), (source, token))
                frames = post.step(frames, token)
            self.assertTrue(post.finished(frames))

    def test_beam_search_returns_well_formed_programs(self):
        from mylittlellm.seedlm import coder4
        prefix = self.vocabulary.condition(["double"], [], [[1, 2, 3]], [4, 6], [])
        found = coder4.beam_search(self.net, self.vocabulary, self.grammar, prefix, [[1, 2, 3]], [], [4, 6],
                                   width=4, max_tokens=8)
        self.assertTrue(found)
        for postfix, score in found:
            prefix_tokens = post.to_prefix(postfix)
            pyx.deserialize(prefix_tokens)
            self.assertLessEqual(len(postfix), 8)

    def test_log_prob_matches_the_beam_score(self):
        from mylittlellm.seedlm import coder4
        prefix = self.vocabulary.condition(["double"], [], [[1, 2, 3]], [4, 6], [])
        found = coder4.beam_search(self.net, self.vocabulary, self.grammar, prefix, [[1, 2, 3]], [], [4, 6],
                                   width=4, max_tokens=6)
        postfix, score = found[0]
        again = coder4.program_log_prob(self.net, self.vocabulary, prefix, postfix, [[1, 2, 3]], [], [4, 6])
        # The beam's score is over grammar-allowed tokens without renormalising, as is the full pass.
        self.assertAlmostEqual(score, again, delta=0.15)


@unittest.skipUnless(TORCH, "torch not installed")
class ExtendTests(unittest.TestCase):
    def test_extending_by_several_tokens_matches_one_at_a_time_and_a_full_pass(self):
        import torch
        from mylittlellm.seedlm import model as seedmodel
        torch.manual_seed(0)
        net = seedmodel.SeedLM(seedmodel.Config(50, 32, 2, 2, 64)).eval()
        for parameter in net.parameters():      # the identity start would hide attention mistakes
            torch.nn.init.normal_(parameter, std=0.2)
        tokens = torch.randint(0, 50, (3, 12))
        with torch.no_grad():
            full = net(tokens)[:, -1]
            _, cache = net.prefill(tokens[:, :8])
            together, _ = net.extend(tokens[:, 8:], cache)
            _, cache = net.prefill(tokens[:, :8])
            for column in range(8, 12):
                single, cache = net.step(tokens[:, column], cache)
        self.assertTrue(torch.allclose(together, full, atol=1e-4))
        self.assertTrue(torch.allclose(single, full, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
