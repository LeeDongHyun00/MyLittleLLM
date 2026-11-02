import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("split_dataset", Path(__file__).resolve().parents[1] / "scripts/split_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SplitTests(unittest.TestCase):
    def rows(self):
        return [{"id": f"{g}-{y}", "question": f"{g} rule {y}", "answer": "fixture",
                 "evidence": [{"source_id": "fixture"}], "review_status": "approved",
                 "leakage_group": str(g)} for g in range(25) for y in [2025, 2026]]

    def test_years_and_paraphrases_stay_together_and_reordering_is_stable(self):
        rows = self.rows()
        result = module.split(rows)
        self.assertEqual(result, module.split(list(reversed(rows))))
        groups = [{r["leakage_group"] for r in result[k]} for k in ["train", "dev", "test"]]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual([len(result[k]) for k in ["train", "dev", "test"]], [24, 6, 20])
        self.assertEqual(sum(map(len, result.values())), len(rows))

    def test_rejects_unreviewed_and_cross_group_duplicates(self):
        rows = self.rows()
        rows[0]["review_status"] = "pending"
        with self.assertRaises(ValueError): module.split(rows)
        rows = self.rows()
        rows[2]["question"] = rows[0]["question"]
        with self.assertRaises(ValueError): module.split(rows)


if __name__ == "__main__": unittest.main()
