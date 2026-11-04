"""Retrieval-only guards with synthetic text; no source PDF or model required."""
import copy
import unittest

from mylittlellm.retrieval import retrieve, terms
from mylittlellm.retrieval_experiments import (build_clause_windows, clause_window_spans,
                                             retrieve_sentence_focus, sentence_focus_weights)
from scripts.evaluate_r01_retrieval import score_retrieval


def row(identifier="ok", **overrides):
    result = dict(chunk_id=identifier, source_id="synthetic", sha256="fixture", pdf_page=1,
                  year=2026, task="housing", institution="A", semester=1, url="fixture",
                  text="방학은 미지원. 단 계절학기 수강은 예외. 허구의 지원 규칙 설명입니다.")
    result.update(overrides)
    return result


class R01RetrievalTests(unittest.TestCase):
    def test_negative_and_exception_context_terms_are_never_dropped(self):
        for question in ("계절학기를 수강하지 않습니다. 방학 지원이 되나요?",
                         "보증금 이자는 중복 청구할 수 없습니다. 월세 대출 이자는 예외인가요?"):
            weights = sentence_focus_weights(question)
            self.assertEqual(set(weights), set(terms(question)))
            self.assertTrue(all(value > 0 for value in weights.values()))
        with self.assertRaises(ValueError):
            sentence_focus_weights("문맥입니다. 질문인가요?", context_weight=0)

    def test_single_sentence_and_unfocused_context_keep_baseline(self):
        rows = [row(), row("other", pdf_page=2, text="생활 공간과 공부에 대한 허구의 안내")]
        scope = dict(year=2026, task="housing", institution="A", semester=1, k=2)
        for question in ("계절학기 예외인가요?", "문맥입니다. 마지막은 서술문입니다."):
            self.assertEqual(retrieve_sentence_focus(rows, question, **scope),
                             retrieve(rows, question, **scope))
        question = "저는 학생입니다. 방학 미지원 예외인가요?"
        self.assertEqual(retrieve_sentence_focus(rows, question, context_weight=1, **scope),
                         retrieve(rows, question, **scope))

    def test_multisentence_retrieval_filters_each_scope_field(self):
        rows = [row(), row("year", year=2025), row("institution", institution="B"),
                row("semester", semester=2), row("task", task="admissions")]
        question = "저는 학생입니다. 방학 예외인가요?"
        selected = retrieve_sentence_focus(rows, question, year=2026, task="housing",
                                           institution="A", semester=1, k=2)
        self.assertEqual([item["chunk_id"] for item in selected], ["ok"])
        self.assertEqual(retrieve_sentence_focus(rows, question, year=2027, task="housing",
                                                institution="A", semester=1), [])

    def test_required_scope_fields_still_fail(self):
        for scope in (dict(year=2026, task="housing"), dict(year=2026, task="admissions")):
            with self.assertRaises(ValueError):
                retrieve_sentence_focus([row()], "문맥입니다. 규칙 질문인가요?", **scope)

    def test_selected_evidence_preserves_negation_exception_and_input(self):
        rows = [row()]
        original = copy.deepcopy(rows)
        selected = retrieve_sentence_focus(rows, "문맥입니다. 방학 예외인가요?",
                                           year=2026, task="housing", semester=1, k=2)
        self.assertEqual(rows, original)
        self.assertEqual(selected[0]["text"], original[0]["text"])
        self.assertIn("미지원", selected[0]["text"])
        self.assertIn("예외", selected[0]["text"])

    def test_amount_fragment_and_wrong_page_do_not_satisfy_exception_bundle(self):
        fixture = dict(source_id="synthetic", source_sha256="fixture", clauses=[
            dict(id="amount", pdf_page=1, anchors=["fictional_cap"]),
            dict(id="exception", pdf_page=1, anchors=["fictional_exception"])])
        case = dict(required_clauses=["amount", "exception"], primary_clause="amount")
        scored = score_retrieval([row(text="fictional_cap only"),
                                  row("wrong", pdf_page=2, text="fictional_exception")],
                                 case, fixture, k=2)
        self.assertEqual(scored["lexical_recall_at_k"], .5)
        self.assertFalse(scored["complete_bundle_at_k"])

    def test_scope_empty_recall_is_not_scored_as_successful_answer(self):
        result = score_retrieval([], dict(expected_empty=True, required_clauses=[]),
                                 dict(clauses=[]), k=2)
        self.assertTrue(result["scope_empty_pass"])
        self.assertIsNone(result["lexical_recall_at_k"])
        self.assertIsNone(result["complete_bundle_at_k"])

    def test_clause_boundaries_preserve_full_page_coverage_within_budget(self):
        for text in ("x" * 930 + "□짧은 예외", "□" + "가" * 2000 + "◦예외" + "나" * 73,
                     "문서 " * 75, "□" + "가" * 899 + "◦미지원 예외"):
            spans = clause_window_spans(text)
            covered = {position for start, end in spans for position in range(start, end)}
            self.assertEqual(covered, set(range(len(text))))
            self.assertTrue(all(end - start <= 900 for start, end in spans))
        self.assertEqual(clause_window_spans(""), [])

    def test_clause_index_keeps_allowed_scope_and_negative_footnotes(self):
        text = "□가상 지원 규정 " + "일반 안내 " * 15 + "◦방학은 미지원. 단 계절학기는 예외입니다."
        originals = [row(text=text)]
        pages = [dict(source_id="synthetic", sha256="fixture", pdf_page=1, text=text),
                 dict(source_id="synthetic", sha256="fixture", pdf_page=26,
                      text="FAQ: 허용되지 않은 정답 문구"),
                 dict(source_id="other-year", sha256="other", pdf_page=1,
                      text="다른 학년도 원문")]
        saved = copy.deepcopy(originals)
        candidate = build_clause_windows(originals, pages)
        self.assertEqual(originals, saved)
        self.assertTrue(candidate)
        self.assertEqual({(item["source_id"], item["pdf_page"]) for item in candidate},
                         {("synthetic", 1)})
        for item in candidate:
            for key in ("sha256", "year", "semester", "institution", "task"):
                self.assertEqual(item[key], saved[0][key])
        self.assertTrue(any("미지원" in item["text"] and "예외" in item["text"]
                            for item in candidate))

    def test_clause_index_rejects_missing_or_mismatched_source_extraction(self):
        originals = [row()]
        for pages in ([], [dict(source_id="synthetic", sha256="wrong", pdf_page=1,
                               text=originals[0]["text"])],
                      [dict(source_id="synthetic", sha256="fixture", pdf_page=1,
                            text="다른 추출 텍스트")]):
            with self.assertRaises(ValueError):
                build_clause_windows(originals, pages)


if __name__ == "__main__":
    unittest.main()
