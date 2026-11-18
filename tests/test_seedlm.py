import ast
import importlib.util
import random
import unittest

from mylittlellm.rulecoder import backinfer as bi
from mylittlellm.rulecoder.sandbox import run_call
from mylittlellm.seedlm import data, synth

TORCH = importlib.util.find_spec("torch") is not None


class SyntheticDataTests(unittest.TestCase):
    def examples(self, count=200, seed=3):
        rng, found = random.Random(seed), []
        while len(found) < count:
            example = synth.sample(rng)
            if example:
                found.append(example)
        return found

    def test_every_generated_program_reproduces_its_own_output(self):
        for example in self.examples():
            self.assertEqual(synth.execute(example["program"], example["args"]), example["expected"])
            self.assertLessEqual(len(example["program"]), synth.MAX_PROGRAM)
            self.assertTrue(any(token.startswith("a") and "/" not in token for token in example["program"]))

    def test_rendered_expression_is_the_same_program_as_the_token_form(self):
        checked = 0
        for example in self.examples(60, seed=5):
            expr, helpers, _ = synth.to_expression(example["program"])
            code = bi.to_code("f", len(example["args"]), {"expr": expr, "helpers": helpers})
            outcome = run_call(code, "f", example["args"])
            if outcome["status"] == "ok" and len(repr(example["expected"])) < 250:
                self.assertEqual(ast.literal_eval(outcome["repr"]), example["expected"])
                checked += 1
        self.assertGreater(checked, 40)

    def test_malformed_programs_are_rejected(self):
        with self.assertRaises(Exception):
            synth.execute(["sum/1"], [[1, 2]])
        with self.assertRaises(ValueError):
            synth.execute(["sum/1", "a0", "a0"], [[1, 2]])


class TokenLayoutTests(unittest.TestCase):
    def test_sequence_has_condition_then_program_and_only_the_program_is_a_target(self):
        vocabulary = data.Vocabulary()
        tokens, start = vocabulary.encode([[3, 1]], 4, ["sum/1"], ["sum/1", "a0"])
        self.assertEqual(tokens[0], data.BOS)
        self.assertEqual(tokens[start - 1], data.PROG)
        self.assertEqual(tokens[-1], data.EOS)
        self.assertEqual([vocabulary.symbol(t) for t in tokens[start:-1]], ["sum/1", "a0"])
        self.assertEqual(bytes(tokens[1:tokens.index(data.OUT)]).decode(), "[3, 1]")

    def test_new_operators_are_appended_without_renumbering_old_ones(self):
        old = data.Vocabulary(["sum/1", "max/1"])
        new = data.Vocabulary(["sum/1", "max/1", "min/1"])
        self.assertEqual(old.token("max/1"), new.token("max/1"))
        self.assertEqual(len(new), len(old) + 1)


@unittest.skipUnless(TORCH, "PyTorch is only needed for the from-scratch model")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from mylittlellm.seedlm import model
        self.torch, self.model = torch, model
        torch.manual_seed(0)
        self.vocabulary = data.Vocabulary()
        self.net = model.SeedLM(model.Config(len(self.vocabulary), d_model=40, n_layer=2, n_head=2,
                                             context=data.MAX_LENGTH),
                                slice(data.FIRST_OP, len(self.vocabulary)))

    def test_rotary_embedding_is_an_orthogonal_map(self):
        x = self.torch.randn(2, 2, 7, 20)
        rotated = self.model.rotary(x, self.net.cos[:7], self.net.sin[:7])
        self.assertTrue(self.torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5))

    def test_every_block_starts_as_the_identity(self):
        x = self.torch.randn(1, 5, 40)
        out = self.net.blocks[0](x, self.net.cos[:5], self.net.sin[:5])
        self.assertTrue(self.torch.allclose(out, x))

    def test_operator_rows_start_orthogonal_up_to_the_width(self):
        rows = self.net.embed.weight[data.FIRST_OP:].detach()
        gram = rows.T @ rows                      # more rules than dimensions: columns are orthonormal
        self.assertTrue(self.torch.allclose(gram, self.torch.eye(40), atol=1e-4))

    def test_loss_ignores_everything_before_the_program(self):
        rows = data.build(8, 1, self.vocabulary)
        inputs, labels = data.collate(rows, self.torch)
        for (tokens, start, _), label_row in zip(rows, labels):
            self.assertTrue((label_row[:start - 1] == -100).all())
            self.assertEqual(label_row[start - 1].item(), tokens[start])
        self.assertTrue(self.torch.isfinite(self.net(inputs, labels)))

    def test_beam_search_only_returns_well_formed_programs(self):
        from mylittlellm.seedlm import propose
        prefix = self.vocabulary.condition([[3, 1, 2]], 6, [])
        programs = propose.beam_search(self.net, self.vocabulary, prefix, n_args=1, width=6)
        self.assertTrue(programs)
        for program, _ in programs:
            self.assertLessEqual(len(program), synth.MAX_PROGRAM)
            self.assertNotIn("a1", program)                 # only arguments that exist
            pending = 1
            for token in program:
                pending += (bi.OPERATORS[token].arity if "/" in token else 0) - 1
            self.assertEqual(pending, 0)


@unittest.skipUnless(TORCH, "PyTorch is only needed for the from-scratch model")
class MuonTests(unittest.TestCase):
    def setUp(self):
        import torch
        from mylittlellm.seedlm import muon
        self.torch, self.muon = torch, muon
        torch.manual_seed(0)

    def test_update_is_close_to_semi_orthogonal_for_wide_and_tall_matrices(self):
        for shape in ((48, 12), (12, 48)):
            update = self.muon.orthogonalize(self.torch.randn(shape))
            values = self.torch.linalg.svdvals(update)
            self.assertEqual(tuple(update.shape), shape)
            self.assertGreater(values.min().item(), 0.4)
            self.assertLess(values.max().item(), 1.5)

    def test_every_direction_moves_by_a_similar_amount_even_if_the_gradient_is_lopsided(self):
        left, _ = self.torch.linalg.qr(self.torch.randn(12, 12))
        right, _ = self.torch.linalg.qr(self.torch.randn(48, 12))
        lopsided = left @ self.torch.diag(self.torch.logspace(0, -3, 12)) @ right.T
        values = self.torch.linalg.svdvals(self.muon.orthogonalize(lopsided))
        self.assertLess((values.max() / values.min()).item(), 20)      # the gradient itself spans 1000x

    def test_step_reduces_a_quadratic_and_rejects_non_matrices(self):
        weight = self.torch.nn.Parameter(self.torch.randn(6, 6))
        target = self.torch.randn(6, 6)
        optimizer = self.muon.Muon([weight], lr=0.05)
        first = ((weight - target) ** 2).sum().item()
        for _ in range(40):
            optimizer.zero_grad()
            ((weight - target) ** 2).sum().backward()
            optimizer.step()
        self.assertLess(((weight - target) ** 2).sum().item(), first)
        vector = self.torch.nn.Parameter(self.torch.randn(6))
        vector.grad = self.torch.ones(6)
        with self.assertRaises(ValueError):
            self.muon.Muon([vector]).step()


if __name__ == "__main__":
    unittest.main()
