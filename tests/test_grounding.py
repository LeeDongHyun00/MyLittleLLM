import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from mylittlellm.claim_verifier import verify_claims
from mylittlellm.policy_rules import PolicyRules, compile_ast, solve_rule
from mylittlellm.grounded_assistant import GroundedAssistant, sufficiency_gate


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mylittlellm.retrieval import ROOT
        cls.config=json.loads((ROOT/'config/grounding_rules.json').read_text())
        cls.rules={r['id']:r for r in cls.config['rules']}

    def test_unknown_facts_remain_unknown(self):
        rule=self.rules['housing_expense_combination']
        self.assertEqual(solve_rule(rule,dict(deposit_conversion=True))['decision'],'unknown')
        self.assertEqual(solve_rule(rule,dict(deposit_conversion=True,housing_loan_interest=True))['decision'],'false')
        self.assertEqual(solve_rule(rule,dict(deposit_conversion=True,housing_loan_interest=False,rent_loan_interest=True))['decision'],'true')

    def test_academic_exceptions_and_boundaries(self):
        rule=self.rules['housing_academic']
        facts=dict(first_semester_exempt=False,disabled=False,care_leaver=False,graduation_or_excess=False,credits=12,required_credits=12,score=70)
        self.assertEqual(solve_rule(rule,facts)['decision'],'true')
        self.assertEqual(solve_rule(rule,dict(facts,score=69))['decision'],'false')
        self.assertEqual(solve_rule(rule,dict(facts,care_leaver=True,score=0,credits=11))['decision'],'false')
        self.assertEqual(solve_rule(rule,dict(disabled=True))['decision'],'true')
        self.assertEqual(solve_rule(rule,dict(graduation_or_excess=True))['decision'],'unknown')

    def test_type_and_domain_validation(self):
        rule=self.rules['housing_duplicate_support']
        for value in [True,-1,32,'1']:
            with self.subTest(value=value),self.assertRaises(ValueError):solve_rule(rule,dict(overlap_days=value))

    def test_school_and_personal_limits_both_apply(self):
        rule=self.rules['housing_benefit_count']
        self.assertEqual(solve_rule(rule,dict(academic_years=2,school_used=0,personal_used=8))['decision'],'false')
        self.assertEqual(solve_rule(rule,dict(academic_years=6,school_used=11,personal_used=11))['decision'],'true')
        self.assertEqual(solve_rule(rule,dict(academic_years=6,school_used=12,personal_used=12))['decision'],'false')

    def test_inconsistent_count_totals_do_not_prove_an_answer(self):
        rule=self.rules['housing_benefit_count']
        result=solve_rule(rule,dict(academic_years=4,school_used=4,personal_used=2))
        self.assertEqual(result['decision'],'unknown')
        self.assertEqual(result['reason'],'inconsistent_or_unresolved_facts')

    def test_negated_county_rule_and_explicit_exception(self):
        rule=self.rules['housing_county_distance']
        self.assertEqual(solve_rule(rule,dict(same_county=True,exception_approved=False))['decision'],'false')
        self.assertEqual(solve_rule(rule,dict(same_county=True,exception_approved=True))['decision'],'true')
        self.assertEqual(solve_rule(rule,dict(same_county=True))['decision'],'unknown')

    def test_certificate_exception_does_not_expand_to_pass_certificate(self):
        rule=self.rules['admissions_ged_certificate']
        self.assertEqual(solve_rule(rule,dict(essay_track=True,score_certificate=True))['decision'],'false')
        self.assertEqual(solve_rule(rule,dict(essay_track=True,score_certificate=False))['decision'],'true')

    def test_unknown_ast_operator_rejected(self):
        with self.assertRaises(ValueError):compile_ast({'python':['print(1)']},{})
        with self.assertRaises(ValueError):compile_ast({'var':'unregistered'}, {})

    def test_scope_and_changed_source_rejected(self):
        r=PolicyRules(index=[]);rule=r.rules['housing_academic']
        self.assertFalse(r.applicable(rule,dict(rule['scope'],year=2025)))
        with self.assertRaises(ValueError):r.documents(rule)

    def test_personal_event_requires_records_but_general_schedule_is_allowed(self):
        self.assertIsNotNone(sufficiency_gate(dict(question='제 계좌에 실제 입금되는 날짜를 알려주세요.'),[{}]))
        self.assertIsNone(sufficiency_gate(dict(question='사업계획에 안내된 일반 지급 기간은 언제입니까?'),[{}]))
        self.assertIsNone(sufficiency_gate(dict(question='계산상 금액만 묻고 실제 지급 완료는 묻지 않습니다.'),[{}]))
        self.assertIsNotNone(sufficiency_gate(dict(question='지원 조건?'),[]))

    def test_neural_veto_checks_every_sentence(self):
        verifier=Mock();verifier.score.side_effect=[dict(usable=True,support=.99,contradiction=0),dict(usable=True,support=.99,contradiction=.9)]
        result=verify_claims('첫 문장입니다. 두 번째 문장입니다.',[dict(chunk_id='x',text='근거')],verifier,threshold=.5)
        self.assertFalse(result['passed']);self.assertEqual(len(result['checks']),2)

    def test_oversize_evidence_is_not_assumed_verified(self):
        verifier=Mock();verifier.score.return_value=dict(usable=False,support=None,contradiction=None)
        self.assertFalse(verify_claims('답변',[dict(chunk_id='x',text='긴 근거')],verifier,threshold=.5)['passed'])

    def test_payment_past_tense_veto_precedes_neural_loading(self):
        assistant=GroundedAssistant.__new__(GroundedAssistant);assistant.verifier=None
        result,review=assistant.postcheck(dict(status='answer',answer='12만원이 지급되었습니다.',citations=['x']),[])
        self.assertEqual(result['status'],'cannot_determine');self.assertIsNone(assistant.verifier)

    def test_claim_checker_is_an_explicit_option(self):
        a=GroundedAssistant.__new__(GroundedAssistant)
        value=dict(status='cannot_determine',answer='근거 부족',citations=[],follow_up=[])
        a.plan=Mock(return_value=dict(preflight=None,proof=None,evidence=[]))
        a.generate=Mock(return_value=dict(llm_called=True));a.parse=Mock(return_value=value)
        a.postcheck=Mock(return_value=(value,dict(applied=True)))
        scope=dict(year=2026,task='housing',institution='한국장학재단',semester=1)
        a.ask('질문',**scope);a.postcheck.assert_not_called()
        a.ask('질문',claim_check=True,**scope);a.postcheck.assert_called_once()

if __name__=='__main__':unittest.main()
