import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mylittlellm.preprocessing import PreprocessedRetriever, clause_blocks, expanded_table_rows, focused_query, scope_compatible
from mylittlellm.preprocessed_assistant import ask_preprocessed, calculation_evidence, make_request, validate_calculated_amount


class CharacterTokenizer:
    def encode(self,text,**kwargs):return SimpleNamespace(ids=list(text))


class PreprocessingTests(unittest.TestCase):
    def test_exception_stays_with_clause_even_when_oversize(self):
        text='가) 지원 자격을 확인합니다.\n※ 단, 논술전형은 제출하지 않습니다.\n나) 다음 규정입니다.'
        blocks=clause_blocks(text,CharacterTokenizer(),12)
        self.assertIn('※ 단, 논술전형은 제출하지 않습니다.',blocks[0])
        self.assertIn('가) 지원 자격',blocks[0])
        self.assertNotIn('나)',blocks[0])

    def test_merged_cell_geometry_does_not_fill_unrelated_blank(self):
        cells=[(0,0,10,20),(10,0,20,10),(10,10,20,20),(20,0,30,10)]
        values={cells[0]:'1학기 신청',cells[1]:'1차',cells[2]:'2차',cells[3]:'비고'}
        class Page:
            def crop(self,box):return SimpleNamespace(extract_text=lambda:values[box])
        rows=expanded_table_rows(SimpleNamespace(cells=cells),Page())
        self.assertEqual(rows[1][0]['text'],'1학기 신청')
        self.assertEqual(rows[1][1]['text'],'2차')
        self.assertEqual(rows[1][2]['text'],'')
        self.assertEqual(rows[1][0]['bbox'],list(cells[0]))

    def test_row_scope_overrides_document_semester_and_filters_other_track(self):
        doc=dict(year=2026,semester=1,section_semester=2,task='housing',institution='재단')
        scope=dict(year=2026,task='housing',institution='재단',track=None)
        self.assertFalse(scope_compatible(doc,semester=1,**scope))
        self.assertTrue(scope_compatible(doc,semester=2,**scope))
        doc.update(task='admissions',semester=None,section_semester=None,admission_track='실기/실적')
        self.assertFalse(scope_compatible(doc,year=2026,task='admissions',institution='재단',semester=None,track='논술'))

    def test_query_cleanup_preserves_negative_and_numeric_conditions(self):
        q=focused_query('2026년 주거안정장학금 2차 신청: 3월 17일 이후도 가능한가요? 제외 조건은?',2026,'한국장학재단')
        for word in ['2차','3월 17일','이후','제외']:self.assertIn(word,q)
        self.assertNotIn('2026년',q)

    def test_request_does_not_include_gold_and_has_configurable_budget(self):
        row=dict(question='질문',year=2026,institution='대학',task='admissions',gold_answer='SECRET')
        evidence=[dict(chunk_id='x',pdf_page=1,year=2026,text='근거')]
        body=make_request(row,evidence,max_tokens=384)
        self.assertEqual(body['max_tokens'],384)
        self.assertEqual(body['response_format']['type'],'json_schema')
        self.assertNotIn('SECRET',json.dumps(body))

    def test_calculation_requires_scope_and_first_month_pledge(self):
        row=dict(year=2026,semester=1,task='housing')
        facts=dict(year=2026,semester=1,month=3,selected=True,payment_approved=True,
                   duplicate_support=False,recognized_expense_krw=0)
        with self.assertRaisesRegex(ValueError,'pledge'):calculation_evidence(facts,row,None)
        with self.assertRaisesRegex(ValueError,'scope'):calculation_evidence(dict(facts,semester=2),row,None)
        docs=[dict(source_id='kosaf-2026-housing-plan-faq',pdf_page=10,kind='clause',text='지원 금액 20만원. 지원 범위 실제 지출'),
              dict(source_id='kosaf-2026-housing-plan-faq',pdf_page=22,kind='clause',text='서약서 제출 후 정액 지원')]
        calculation,evidence=calculation_evidence(dict(facts,pledge_submitted=True),row,SimpleNamespace(rows=docs))
        self.assertEqual(calculation['amount_krw'],200000)
        self.assertEqual([r['pdf_page'] for r in evidence],[10,22])

    def test_calculator_does_not_allow_llm_to_replace_payment_with_cap(self):
        calc=dict(amount_krw=120000,monthly_cap_krw=200000)
        validate_calculated_amount('상한은 20만원이고, 계산 결과는 120,000원입니다.',calc)
        with self.assertRaises(ValueError):validate_calculated_amount('20만원입니다.',calc)
        with self.assertRaises(ValueError):validate_calculated_amount('12만원과 추가 5만원입니다.',calc)

    def test_retrieval_budget_never_slices_exception_or_imports_wrong_year(self):
        r=PreprocessedRetriever.__new__(PreprocessedRetriever)
        r.config=dict(context_tokens=70,top_k=4,track_ranges={},reviewed_schedule=dict(source_id='mixed',pdf_page=25))
        r.tok=CharacterTokenizer();r.legacy=[];r.semantic=None
        base=dict(source_id='s',pdf_page=1,year=2026,semester=None,task='admissions',institution='대학',admission_track=None,section_semester=None)
        small='지원 조건입니다. 단, 추가 서류가 필요합니다.'
        large='지원 자격을 확인합니다. '*20+'단, 논술전형은 예외입니다.'
        r.rows=[dict(base,chunk_id='small',text=small,source_text=small),dict(base,chunk_id='large',text=large,source_text=large),
                dict(base,year=2025,chunk_id='wrongyear',text='지원',source_text='지원')]
        docs,log=r.retrieve('지원',year=2026,task='admissions',institution='대학')
        self.assertEqual([d['chunk_id'] for d in docs],['small'])
        self.assertEqual(docs[0]['text'],small)
        self.assertLessEqual(log['context_tokens'],70)
        self.assertTrue(any(d['chunk_id']=='large' for d in log['skipped']))

    def test_missing_evidence_abstains_without_generator(self):
        retriever=Mock();retriever.retrieve.return_value=([],{'seconds':0})
        with patch('mylittlellm.preprocessed_assistant.complete') as generator:
            value=ask_preprocessed('지원 조건?',year=2026,institution='대학',retriever=retriever)
        generator.assert_not_called()
        self.assertEqual(value['status'],'cannot_determine')
        self.assertFalse(value['llm_called'])


if __name__=='__main__':unittest.main()
