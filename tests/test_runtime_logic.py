import unittest
import io
import json
from unittest.mock import patch
from mylittlellm.assistant import validate_answer, ask
from mylittlellm.costs import housing_payment, net_cost
from mylittlellm.retrieval import retrieve, ambiguous_comparison


class RuntimeTests(unittest.TestCase):
    def housing(self, **overrides):
        return dict(year=2026, semester=1, month=4, selected=True, payment_approved=True,
                    duplicate_support=False, recognized_expense_krw=120000, **overrides)

    def test_payment_cap_and_first_month_exception(self):
        f=self.housing()
        self.assertEqual(housing_payment(f)["amount_krw"],120000)
        f["month"]=3
        self.assertEqual(housing_payment(f)["amount_krw"],200000)
        f.update(month=4,recognized_expense_krw=300000)
        self.assertEqual(housing_payment(f)["amount_krw"],200000)

    def test_unknown_eligibility_and_unimplemented_periods_rejected(self):
        for key,value in [("selected",None),("duplicate_support",True),("semester",2),
                          ("month",7),("recognized_expense_krw",-1),("recognized_expense_krw",True)]:
            f=self.housing();f[key]=value
            with self.assertRaises(ValueError): housing_payment(f)

    def test_loan_is_debt_not_discount(self):
        f=dict(period="test semester",same_period_confirmed=True,stacking_verified=True,
               tuition_krw=4000000,living_krw=2000000,tuition_grants_krw=2500000,
               living_grants_krw=500000,loan_krw=1000000)
        r=net_cost(f)
        self.assertEqual(r["net_cost_krw"],3000000)
        self.assertEqual(r["cash_needed_after_loan_krw"],2000000)
        self.assertEqual(r["debt_principal_krw"],1000000)
        f["tuition_grants_krw"]=5000000
        with self.assertRaises(ValueError): net_cost(f)

    def test_retrieval_never_crosses_year_institution_or_semester(self):
        rows=[dict(chunk_id=str(i),source_id=str(i),pdf_page=1,year=y,task="housing",institution=u,semester=s,text="주거비 월 상한 20만원")
              for i,(y,u,s) in enumerate([(2025,"A",1),(2026,"A",1),(2026,"B",1),(2026,"A",2)])]
        result=retrieve(rows,"주거비 상한",year=2026,task="housing",institution="A",semester=1)
        self.assertEqual([r["chunk_id"] for r in result],["1"])
        self.assertEqual(retrieve(rows,"주거비",year=2027,task="housing",semester=1),[])

    def test_hallucinated_citation_is_rejected(self):
        r=dict(status="answer",answer="답변",citations=["invented"],follow_up=[])
        with self.assertRaises(ValueError): validate_answer(r,[{"chunk_id":"real"}],"rag")
        r["citations"]=[]
        with self.assertRaises(ValueError): validate_answer(r,[{"chunk_id":"real"}],"rag")

    def test_flat_multi_year_change_table_is_quarantined(self):
        self.assertTrue(ambiguous_comparison("전형별 변경 사항 구분 2025학년도 2026학년도 추천 인원"))
        self.assertFalse(ambiguous_comparison("2026학년도 지원자격: 2024년 2월 이후 졸업자"))

    def test_calculator_adds_rule_conditions_not_just_cap_fragment(self):
        base=dict(source_id="kosaf-2026-housing-plan-faq",pdf_page=10,year=2026,
                  semester=1,task="housing",institution="한국장학재단",sha256="fixture",url="fixture")
        partial=dict(base,chunk_id="cap",text="지원 금액 월 최대 20만 원")
        full=dict(base,chunk_id="rule",text="지원 금액 월 최대 20만 원. 지원 범위: 실제 지출 비용")
        body={"choices":[{"finish_reason":"stop","message":{"content":json.dumps(
            dict(status="answer",answer="12만원",citations=["rule"],follow_up=[]))}}]}
        with patch("mylittlellm.assistant.retrieve",return_value=[partial]), \
             patch("mylittlellm.assistant.urllib.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value=io.StringIO(json.dumps(body))
            result=ask("주거비",year=2026,task="housing",semester=1,facts=self.housing(),index=[partial,full])
            sent=build.return_value.open.call_args.args[0]
            payload=json.loads(json.loads(sent.data)["messages"][1]["content"])
            self.assertTrue(any("실제 지출 비용" in x["text"] for x in payload["evidence"]))
            self.assertEqual(result["sources"][0]["chunk_id"],"rule")


if __name__ == "__main__": unittest.main()
