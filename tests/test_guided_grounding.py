import itertools
import json
import unittest
from unittest.mock import Mock

from mylittlellm.guided_facts import collect_rule_facts, parse_answer
from mylittlellm.grounded_assistant import GroundedAssistant, requires_personal_record
from mylittlellm.policy_rules import PolicyRules, solve_rule
from mylittlellm.retrieval import ROOT


class GuidedGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config/grounding_rules.json').read_text())
        cls.rules={r['id']:r for r in cls.config['rules']}

    def test_susi_count_boundary_and_unknown(self):
        rule=self.rules['admissions_susi_application_limit']
        for n,expected in [(0,'true'),(5,'true'),(6,'false'),(7,'false')]:
            self.assertEqual(solve_rule(rule,{'counted_applications':n})['decision'],expected)
        self.assertEqual(solve_rule(rule,{})['missing'],['counted_applications'])

    def test_low_income_truth_table(self):
        rule=self.rules['admissions_low_income_status']
        for school,basic,near,single in itertools.product([False,True],repeat=4):
            facts=dict(education_qualified=school,basic_livelihood_eligible=basic,
                       near_poverty_eligible=near,single_parent_support_eligible=single)
            expected=school and (basic or near or single)
            self.assertEqual(solve_rule(rule,facts)['decision'],'true' if expected else 'false')

    def test_family_type_does_not_fill_legal_status(self):
        rule=self.rules['admissions_low_income_status']
        facts=dict(education_qualified=True,basic_livelihood_eligible=False,near_poverty_eligible=False)
        self.assertEqual(solve_rule(rule,facts)['missing'],['single_parent_support_eligible'])

    def test_vocational_threshold_and_exclusions(self):
        rule=self.rules['admissions_vocational_subject']
        facts=dict(eligible_school_graduate=True,meister_school=False,experiential_alternative_school=False,same_field_completed=False)
        for credits,expected in [(29,'false'),(30,'true'),(31,'true')]:
            self.assertEqual(solve_rule(rule,dict(facts,related_subject_credits=credits))['decision'],expected)
        self.assertEqual(solve_rule(rule,dict(facts,same_field_completed=True))['decision'],'true')
        for flag in ['meister_school','experiential_alternative_school']:
            self.assertEqual(solve_rule(rule,dict(facts,related_subject_credits=50,**{flag:True}))['decision'],'false')

    def test_question_planner_omits_irrelevant_credit_and_rent_fields(self):
        facts=dict(first_semester_exempt=False,disabled=False,care_leaver=False,graduation_or_excess=True)
        self.assertEqual(solve_rule(self.rules['housing_academic'],facts)['missing'],['score'])
        self.assertEqual(solve_rule(self.rules['housing_expense_combination'],dict(deposit_conversion=True))['missing'],['housing_loan_interest'])

    def test_no_questions_when_known_exception_decides(self):
        result=solve_rule(self.rules['housing_academic'],dict(disabled=True))
        self.assertEqual(result['decision'],'true');self.assertEqual(result['missing'],[])

    def test_jointly_constrained_missing_facts_remain_collectable(self):
        rule=dict(fields={'a':{'type':'bool'},'b':{'type':'bool'}},expression={'var':'a'},
                  constraints=[{'eq':[{'var':'a'},{'var':'b'}]}])
        result=solve_rule(rule,{})
        self.assertEqual(result['decision'],'unknown');self.assertEqual(result['missing'],['a','b'])

    def make_policy(self,rule_id):
        rule=self.rules[rule_id]
        index=[dict(chunk_id=s['chunk_id'],sha256=s['sha256'],text=' '.join(s['anchors']),pdf_page=1,year=2026) for s in rule['sources']]
        return PolicyRules(index=index),rule

    def test_missing_applicability_is_a_question_and_false_is_out_of_scope(self):
        policy,rule=self.make_policy('admissions_susi_application_limit')
        row=dict(rule['scope'],question='수시 지원 횟수는?')
        facts=dict(rule['scope'],rule_id=rule['id'],counted_applications=6)
        proof,_=policy.evaluate(row,facts)
        self.assertEqual(proof['reason'],'missing_applicability')
        self.assertEqual(proof['questions'][0]['field'],'counted_application')
        proof,_=policy.evaluate(row,dict(facts,counted_application=False))
        self.assertEqual(proof['reason'],'outside_rule_scope')

    def test_every_field_has_korean_question(self):
        for rule in self.rules.values():
            for field in rule['fields'].values():self.assertRegex(field['question'],r'[가-힣]')

    def test_intake_unknown_is_not_false_and_invalid_number_is_rejected(self):
        self.assertIsNone(parse_answer('모름',{'type':'bool'}))
        self.assertIs(parse_answer('아니요',{'type':'bool'}),False)
        field=dict(type='int',min=0,max=50)
        for answer in ['-1','1.5','100','1e1','많아요','12회']:
            with self.subTest(answer=answer),self.assertRaises(ValueError):parse_answer(answer,field)

    def test_guided_input_stops_after_exemption(self):
        policy,rule=self.make_policy('housing_academic')
        assistant=GroundedAssistant.__new__(GroundedAssistant)
        assistant.rules=policy;assistant.legacy=[]
        row=dict(rule['scope'],question='성적과 학점 요건이 충족되나요?')
        answers=Mock(side_effect=['네','네'])
        facts=collect_rule_facts(assistant,row,rule['id'],input_fn=answers,output_fn=Mock())
        self.assertEqual(answers.call_count,2)
        self.assertNotIn('score',facts);self.assertNotIn('credits',facts)
        self.assertEqual(assistant.plan(row,facts)['preflight']['decision'],'true')

    def test_guided_unknown_is_preserved_and_question_not_repeated(self):
        policy,rule=self.make_policy('admissions_susi_application_limit')
        assistant=GroundedAssistant.__new__(GroundedAssistant);assistant.rules=policy;assistant.legacy=[]
        row=dict(rule['scope'],question='수시 추가 지원 횟수?')
        answers=Mock(side_effect=['예','모름'])
        facts=collect_rule_facts(assistant,row,rule['id'],input_fn=answers,output_fn=Mock())
        self.assertNotIn('counted_applications',facts);self.assertEqual(answers.call_count,2)
        result=assistant.plan(row,facts)['preflight']
        self.assertEqual(result['status'],'cannot_determine')
        self.assertNotIn('counted_applications',result['follow_up'][0])

    def test_guided_scope_mismatch_does_not_prompt(self):
        policy,rule=self.make_policy('admissions_susi_application_limit')
        assistant=Mock(rules=policy);answer=Mock()
        with self.assertRaises(ValueError):collect_rule_facts(assistant,dict(rule['scope'],year=2025),rule['id'],input_fn=answer)
        answer.assert_not_called()

    def test_procedures_and_general_rules_do_not_require_personal_records(self):
        for question in ['제 지급 결과는 어디서 확인하나요?', '입금됐는지 확인하는 방법을 알려 주세요.',
                         '저는 기숙사생입니다. 지급 기준은 무엇인가요?', '최종 합격자 발표 절차를 설명해 주세요.',
                         '제 실제 지급 여부는 묻지 않습니다. 계산 조건만 알려 주세요.']:
            with self.subTest(question=question):self.assertFalse(requires_personal_record(question))

    def test_actual_events_and_mixed_requests_require_personal_records(self):
        for question in ['입금됐나요?', '제 계좌로 언제 들어오나요?', '제 지원 결과와 실제 입금 날짜를 알려 주세요.',
                         '확인 방법과 실제 입금됐는지 알려 주세요.', '저는 선발되었나요?',
                         '제 계좌에 실제 입금되는 날짜를 알려주세요.']:
            with self.subTest(question=question):self.assertTrue(requires_personal_record(question))


if __name__=='__main__':unittest.main()
