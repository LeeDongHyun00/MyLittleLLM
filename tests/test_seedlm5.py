import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "the training scripts import torch")
class VariantSplitTests(unittest.TestCase):
    def test_all_variants_of_a_seed_fall_on_the_same_side(self):
        from seedlm4_train import variant_held
        seeds = [f"MbppX/{n}" for n in range(200)]
        for seed in seeds:
            sides = {variant_held({"id": f"variant-p1-{seed}-{k}", "seed": seed}) for k in range(8)}
            self.assertEqual(len(sides), 1)
        # Roughly a tenth of the seeds are held out, so the split is not degenerate.
        held = sum(variant_held({"id": "x", "seed": seed}) for seed in seeds)
        self.assertTrue(5 <= held <= 40)

    def test_a_task_without_a_seed_is_split_by_its_own_id(self):
        from seedlm4_train import variant_held
        self.assertEqual(variant_held({"id": "teacher-7"}), variant_held({"id": "other", "seed": "teacher-7"}))

    def test_teacher_rows_follow_the_given_split(self):
        from mylittlellm.seedlm import coder4, data2
        from seedlm3_train import teacher_rows
        vocabulary = coder4.Vocabulary(["sum", "of", "the", "list"], coder4.program_tokens(["@sum/1"]))
        tasks = [{"id": f"t{k}", "seed": "s1" if k < 2 else "s2", "description": "the sum of the list",
                  "inputs": ["([1, 2],)"], "outputs": ["3"], "expression": "sum(a0)", "tokens": ["@sum/1", "a0"]}
                 for k in range(4)]
        on_seed = lambda task: task["seed"] == "s1"
        held = teacher_rows(vocabulary, tasks, held=True, held_out=on_seed)
        kept = teacher_rows(vocabulary, tasks, held=False, held_out=on_seed)
        self.assertEqual(sorted(r[2]["task_id"] for r in held), ["t0", "t1"])
        self.assertEqual(sorted(r[2]["task_id"] for r in kept), ["t2", "t3"])
        self.assertTrue(all(r[0][-1] == data2.EOS for r in held + kept))


@unittest.skipUnless(TORCH, "the near-miss diagnostic imports torch")
class NearMissTests(unittest.TestCase):
    def test_edit_distance(self):
        from seedlm5_nearmiss import edit_distance
        self.assertEqual(edit_distance(list("kitten"), list("sitting")), 3)
        self.assertEqual(edit_distance(["@sum/1", "a0"], ["@sum/1", "a0"]), 0)
        self.assertEqual(edit_distance(["@sum/1", "a0"], ["@max/1", "a0"]), 1)
        self.assertEqual(edit_distance([], ["a0"]), 1)

    def test_buckets_cover_every_distance(self):
        from seedlm5_nearmiss import BUCKETS, bucket
        self.assertEqual([bucket(d) for d in (0, 1, 2, 3, 4, 5, 6, 40)],
                         ["0", "1", "2", "3", "4-5", "4-5", "6+", "6+"])
        self.assertTrue(all(bucket(d) in BUCKETS for d in range(50)))


@unittest.skipUnless(TORCH, "the coverage diagnostic imports the training scripts")
class CoverageTests(unittest.TestCase):
    def test_a_program_passes_only_if_every_example_matches(self):
        from mylittlellm.rulecoder import backinfer as bi
        from mylittlellm.seedlm import pyx
        from seedlm5_coverage import passes
        tokens = pyx.serialize("sum(a0)", 1, [], None)
        good = [bi.parse_example(line, "f") for line in ("assert f([1, 2]) == 3", "assert f([]) == 0")]
        bad = good + [bi.parse_example("assert f([1]) == 2", "f")]
        self.assertTrue(passes(tokens, good, [], None))
        self.assertFalse(passes(tokens, bad, [], None))


if __name__ == "__main__":
    unittest.main()
