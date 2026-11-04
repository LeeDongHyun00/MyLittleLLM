"""Public development regressions; synthetic index records are not PDF evidence."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

from mylittlellm.grounded_assistant import GroundedAssistant
from mylittlellm.guided_facts import collect_rule_facts, parse_answer
from mylittlellm.policy_rules import PolicyRules, explain_false, false_explanation_nodes
from mylittlellm.retrieval import ROOT


# The source-bound paths and hashes come from the existing config. These inputs
# are developer-authored regressions, not new independent evaluation questions.
CASES = {
    'housing_academic': {
        'question': '성적·학점 요건은 충족하나요?',
        'true': {'first_semester_exempt': True},
        'false': {'first_semester_exempt': False, 'disabled': False,
                  'care_leaver': False, 'score': 69},
        'unknown': {'first_semester_exempt': False, 'disabled': False,
                    'graduation_or_excess': True, 'care_leaver': False},
        'missing': ['score'],
        'reason_text': '백분위 성적이 70점에 미달',
        'absent_reason': '이수학점이 학교 최소 이수학점에 미달',
    },
    'housing_expense_combination': {
        'question': '보증금 환산액과 이자를 청구하는 조건은?',
        'true': {'deposit_conversion': True, 'housing_loan_interest': False,
                 'rent_loan_interest': True},
        'false': {'deposit_conversion': True, 'housing_loan_interest': True},
        'unknown': {'deposit_conversion': True},
        'missing': ['housing_loan_interest'],
        'reason_text': '보증금 월환산액과 주택임차차입금 이자를 함께 청구',
    },
    'housing_duplicate_support': {
        'question': '같은 연월 중복수혜 제한은?',
        'true': {'same_year_month': True, 'other_program_is_conflicting': True, 'overlap_days': 0},
        'false': {'same_year_month': True, 'other_program_is_conflicting': True, 'overlap_days': 1},
        'unknown': {'same_year_month': True, 'other_program_is_conflicting': True},
        'missing': ['overlap_days'],
        'reason_text': '같은 연월에 자격이 하루 이상 겹치는 입력',
    },
    'housing_benefit_count': {
        'question': '다음 1회 수혜 횟수 요건은?',
        'true': {'academic_years': 6, 'school_used': 11, 'personal_used': 11},
        'false': {'academic_years': 4, 'school_used': 7, 'personal_used': 8},
        'unknown': {'academic_years': 4, 'school_used': 7},
        'missing': ['personal_used'],
        'reason_text': '개인 누적 수혜 횟수가 적용되는 개인별 한도',
        'absent_reason': '현재 학교의 수혜 횟수가 학제별 한도',
    },
    'housing_county_distance': {
        'question': '같은 군의 원거리 요건은?',
        'true': {'same_county': False},
        'false': {'same_county': True, 'exception_approved': False},
        'unknown': {'same_county': True},
        'missing': ['exception_approved'],
        'reason_text': '원거리 예외 인정은 받지 않은 것으로 입력',
    },
    'admissions_ged_certificate': {
        'question': '검정고시 증명서 제출 의무는?',
        'true': {'essay_track': True, 'score_certificate': False},
        'false': {'essay_track': True, 'score_certificate': True},
        'unknown': {'essay_track': True},
        'missing': ['score_certificate'],
        'reason_text': '제출 의무의 예외인 면제 조건',
        'absent_reason': '충족하지 않습니다',
    },
    'admissions_susi_application_limit': {
        'question': '수시 추가 지원 횟수 요건은?',
        'true': {'counted_applications': 5},
        'false': {'counted_applications': 6},
        'unknown': {},
        'missing': ['counted_applications'],
        'reason_text': '산입 대상 수시 원서접수가 이미 6회 이상',
    },
    'admissions_low_income_status': {
        'question': '저소득층학생전형의 학력·법정 지위 요건은?',
        'true': {'education_qualified': True, 'basic_livelihood_eligible': True},
        'false': {'education_qualified': False},
        'unknown': {'education_qualified': True, 'basic_livelihood_eligible': False,
                    'near_poverty_eligible': False},
        'missing': ['single_parent_support_eligible'],
        'reason_text': '학력 조건에 해당하지 않는 것으로 입력',
        'absent_reason': '지원대상자 조건이 모두 아니라고 입력',
    },
    'admissions_vocational_subject': {
        'question': '특성화고 동일계열 또는 전문교과 요건은?',
        'true': {'eligible_school_graduate': True, 'meister_school': False,
                 'experiential_alternative_school': False, 'same_field_completed': False,
                 'related_subject_credits': 30},
        'false': {'eligible_school_graduate': True, 'meister_school': False,
                  'experiential_alternative_school': False, 'same_field_completed': False,
                  'related_subject_credits': 29},
        'unknown': {'eligible_school_graduate': True, 'meister_school': False,
                    'experiential_alternative_school': False, 'same_field_completed': False},
        'missing': ['related_subject_credits'],
        'reason_text': '관련 전문교과도 30단위(학점)에 미달',
        'absent_reason': '출신으로 입력되어',
    },
}


def synthetic_policy():
    """Exercise validation with anchors, without asserting source reconstruction."""
    config=json.loads((ROOT/'config/grounding_rules.json').read_text())
    docs={}
    for rule in config['rules']:
        for source in rule['sources']:
            doc=docs.setdefault(source['chunk_id'],dict(
                chunk_id=source['chunk_id'],sha256=source['sha256'],text='',pdf_page=1,year=2026))
            doc['text']+=' '+' '.join(source['anchors'])
    return PolicyRules(index=list(docs.values()))


class L01BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.policy=synthetic_policy()
        self.assistant=GroundedAssistant.__new__(GroundedAssistant)
        self.assistant.rules=self.policy
        self.assistant.legacy=[]

    def inputs(self,rule_id,extra=None,*,applicability=True):
        rule=self.policy.rules[rule_id]
        row=dict(rule['scope'],question=CASES[rule_id]['question'])
        facts=dict(rule['scope'],rule_id=rule_id)
        if applicability:facts.update(rule.get('applicability',{}))
        facts.update(extra or {})
        return row,facts

    def test_nine_rule_true_false_unknown_matrix(self):
        self.assertEqual(set(CASES),set(self.policy.rules))
        for rule_id,case in CASES.items():
            for expected in ('true','false','unknown'):
                with self.subTest(rule=rule_id,decision=expected):
                    row,facts=self.inputs(rule_id,case[expected])
                    before=copy.deepcopy(facts)
                    proof,_=self.policy.evaluate(row,facts)
                    self.assertEqual(facts,before)
                    self.assertEqual(proof['decision'],expected)
                    self.assertIn(self.policy.rules[rule_id]['explanation'],proof['explanation'])
                    if expected=='unknown':
                        self.assertEqual(proof['reason'],'missing_facts')
                        self.assertEqual(proof['missing'],case['missing'])
                        self.assertEqual(proof['explanation'],self.policy.rules[rule_id]['explanation'])
                    elif expected=='false':
                        self.assertIn(case['reason_text'],proof['explanation'])
                        if 'absent_reason' in case:self.assertNotIn(case['absent_reason'],proof['explanation'])
                    else:
                        self.assertEqual(proof['explanation'],self.policy.rules[rule_id]['explanation'])

    def test_applicability_unknown_and_false_never_mean_ineligible(self):
        for rule_id,rule in self.policy.rules.items():
            for field in rule.get('applicability',{}):
                for value,reason in ((None,'missing_applicability'),(False,'outside_rule_scope')):
                    with self.subTest(rule=rule_id,field=field,value=value):
                        row,facts=self.inputs(rule_id,CASES[rule_id]['false'])
                        if value is None:del facts[field]
                        else:facts[field]=value
                        proof,_=self.policy.evaluate(row,facts)
                        self.assertEqual((proof['decision'],proof['reason']),('unknown',reason))
                        self.assertEqual(proof['explanation'],rule['explanation'])
                        self.assertEqual(proof['missing'],[field] if value is None else [])
                        self.assertEqual(self.assistant.plan(row,facts)['preflight']['status'],'cannot_determine')

    def test_inconsistent_counts_do_not_generate_false_explanations(self):
        row,facts=self.inputs('housing_benefit_count',dict(academic_years=4,school_used=8,personal_used=2))
        proof,_=self.policy.evaluate(row,facts)
        self.assertEqual((proof['decision'],proof['reason']),('unknown','inconsistent_facts'))
        self.assertEqual(explain_false(self.policy.rules['housing_benefit_count'],facts),'')
        result=self.assistant.plan(row,facts)['preflight']
        self.assertEqual(result['reason'],'inconsistent_rule_facts')
        self.assertEqual(result['status'],'cannot_determine')

    def test_known_academic_failures_are_separate_from_unknown_exemptions(self):
        rule_id='housing_academic'
        row,facts=self.inputs(rule_id,dict(first_semester_exempt=False,disabled=False,
            care_leaver=True,graduation_or_excess=False,credits=11,required_credits=12))
        proof,_=self.policy.evaluate(row,facts)
        self.assertEqual(proof['decision'],'false')
        self.assertIn('이수학점이 학교 최소 이수학점에 미달',proof['explanation'])
        self.assertNotIn('백분위 성적이 70점에 미달',proof['explanation'])
        # Unknown graduation exemption may still rescue the credit component.
        del facts['graduation_or_excess']
        proof,_=self.policy.evaluate(row,facts)
        self.assertEqual(proof['decision'],'unknown')
        self.assertEqual(proof['missing'],['graduation_or_excess'])

    def test_school_types_explain_only_the_known_exclusion(self):
        rule_id='admissions_vocational_subject'
        for field,reason in [('meister_school','마이스터고 출신'),
                             ('experiential_alternative_school','체험 위주 교육')]:
            with self.subTest(field=field):
                row,facts=self.inputs(rule_id,{field:True})
                proof,_=self.policy.evaluate(row,facts)
                self.assertEqual(proof['decision'],'false')
                self.assertIn(reason,proof['explanation'])
                self.assertNotIn('관련 전문교과도 30단위(학점)에 미달',proof['explanation'])
                self.assertNotIn('졸업(예정)자에 해당하지 않는 것으로 입력',proof['explanation'])
                # A confirmed subject equivalence never cancels a school exclusion.
                facts.update(same_field_completed=True,related_subject_credits=300)
                self.assertEqual(self.policy.evaluate(row,facts)[0]['decision'],'false')

    def test_legal_status_needs_all_alternatives_to_explain_failure(self):
        rule_id='admissions_low_income_status'
        row,facts=self.inputs(rule_id,dict(basic_livelihood_eligible=False,
            near_poverty_eligible=False,single_parent_support_eligible=False))
        proof,_=self.policy.evaluate(row,facts)
        self.assertEqual(proof['decision'],'false')
        self.assertIn('지원대상자 조건이 모두 아니라고 입력',proof['explanation'])
        self.assertNotIn('학력 조건에 해당하지 않는 것으로 입력',proof['explanation'])
        del facts['single_parent_support_eligible']
        proof,_=self.policy.evaluate(row,facts)
        self.assertEqual(proof['decision'],'unknown')
        self.assertIn('single_parent_support_eligible',proof['missing'])

    def test_scope_hash_and_anchor_guards_remain_required(self):
        for rule_id,case in CASES.items():
            row,facts=self.inputs(rule_id,case['false'])
            for scope_key,invalid in [('year',2025),('institution','다른 대학'),('task','other')]:
                with self.subTest(rule=rule_id,scope=scope_key):
                    with self.assertRaises(ValueError):self.policy.evaluate(dict(row,**{scope_key:invalid}),facts)
                    with self.assertRaises(ValueError):self.policy.evaluate(row,dict(facts,**{scope_key:invalid}))
            if 'semester' in row:
                with self.assertRaises(ValueError):self.policy.evaluate(dict(row,semester=2),facts)
            source=self.policy.rules[rule_id]['sources'][0]
            doc=self.policy.index[source['chunk_id']]
            for key,invalid in [('sha256','0'*64),('text','원문 앵커 없음')]:
                with self.subTest(rule=rule_id,invalid_source=key):
                    saved=doc[key];doc[key]=invalid
                    with self.assertRaises(ValueError):self.policy.evaluate(row,facts)
                    doc[key]=saved

    def test_every_typed_field_rejects_null_cross_type_and_fractional_inputs(self):
        for rule_id,rule in self.policy.rules.items():
            for field,spec in rule['fields'].items():
                invalid=[None,0,1,'false'] if spec['type']=='bool' else [None,True,False,1.5,'1',spec['min']-1,spec['max']+1]
                for value in invalid:
                    with self.subTest(rule=rule_id,field=field,value=value):
                        row,facts=self.inputs(rule_id,{field:value})
                        with self.assertRaises(ValueError):self.policy.evaluate(row,facts)

    def test_guided_legal_unknown_is_omitted_despite_question_prose(self):
        rule_id='admissions_low_income_status'
        row,_=self.inputs(rule_id)
        row['question']='저는 한부모 가구입니다. 저소득층학생전형의 요건은?'
        answers=Mock(side_effect=['예','예','아니요','아니요','모름'])
        facts=collect_rule_facts(self.assistant,row,rule_id,input_fn=answers,output_fn=Mock())
        self.assertEqual(answers.call_count,5)
        self.assertNotIn('single_parent_support_eligible',facts)
        self.assertIs(facts['near_poverty_eligible'],False)
        result=self.assistant.plan(row,facts)['preflight']
        self.assertEqual(result['status'],'cannot_determine')
        self.assertEqual(result['questions'][0]['field'],'single_parent_support_eligible')

    def test_guided_school_type_unknown_does_not_become_false(self):
        rule_id='admissions_vocational_subject'
        row,_=self.inputs(rule_id)
        answers=Mock(side_effect=['예','예','모름','아니요','예'])
        facts=collect_rule_facts(self.assistant,row,rule_id,input_fn=answers,output_fn=Mock())
        self.assertNotIn('meister_school',facts)
        self.assertNotIn('related_subject_credits',facts)
        self.assertEqual(answers.call_count,5)
        result=self.assistant.plan(row,facts)['preflight']
        self.assertEqual(result['status'],'cannot_determine')
        self.assertEqual([q['field'] for q in result['questions']],['meister_school'])

    def test_guided_retries_fraction_and_preserves_explicit_zero(self):
        rule_id='admissions_susi_application_limit'
        row,_=self.inputs(rule_id)
        answers=Mock(side_effect=['예','5.9','0'])
        output=Mock()
        facts=collect_rule_facts(self.assistant,row,rule_id,input_fn=answers,output_fn=output)
        self.assertIs(type(facts['counted_applications']),int)
        self.assertEqual(facts['counted_applications'],0)
        self.assertEqual(answers.call_count,3)
        self.assertIn('정수',output.call_args.args[0])
        self.assertEqual(self.assistant.plan(row,facts)['preflight']['decision'],'true')
        for answer in ('모름','/quit'):
            answers=Mock(side_effect=['예',answer])
            facts=collect_rule_facts(self.assistant,row,rule_id,input_fn=answers,output_fn=Mock())
            self.assertNotIn('counted_applications',facts)
        for answer in ('','모름','모르겠습니다','?'):
            self.assertIsNone(parse_answer(answer,dict(type='int',min=0,max=50)))

    def test_partial_proofs_cannot_override_personal_outcome_gate(self):
        for rule_id,case in CASES.items():
            row,facts=self.inputs(rule_id,case['true'])
            row['question']+=' 제 최종 합격 보장을 확정해 주세요.'
            with self.subTest(rule=rule_id):
                result=self.assistant.plan(row,facts)['preflight']
                self.assertEqual((result['status'],result['reason']),('cannot_determine','outcome_not_in_policy'))
        row,facts=self.inputs('housing_academic',CASES['housing_academic']['true'])
        row['question']+=' 제 계좌에 입금됐나요?'
        self.assertEqual(self.assistant.plan(row,facts)['preflight']['reason'],'personal_event_not_in_policy')

    def test_certificate_false_is_submission_exemption_not_rejection(self):
        row,facts=self.inputs('admissions_ged_certificate',CASES['admissions_ged_certificate']['false'])
        result=self.assistant.plan(row,facts)['preflight']
        self.assertEqual(result['decision'],'false')
        self.assertIn('해당 성적증명서 제출은 면제',result['answer'])
        self.assertNotIn('충족하지 않습니다',result['answer'])
        self.assertIn('합격증명서에 확대하지 않습니다',result['answer'])

    def test_optional_metadata_preserves_proof_shape_and_absence_behavior(self):
        rule_id='admissions_susi_application_limit'
        row,facts=self.inputs(rule_id,CASES[rule_id]['false'])
        new_proof,_=self.policy.evaluate(row,facts)
        del self.policy.rules[rule_id]['false_explanation']
        old_proof,_=self.policy.evaluate(row,facts)
        self.assertEqual(set(new_proof),set(old_proof))
        for key in old_proof:
            if key!='explanation':self.assertEqual(new_proof[key],old_proof[key])
        self.assertEqual(old_proof['explanation'],self.policy.rules[rule_id]['explanation'])

    def test_metadata_cannot_explain_an_or_alternative_as_a_failed_requirement(self):
        rule=copy.deepcopy(self.policy.rules['housing_county_distance'])
        rule['false_explanation']=[dict(conjunct_path=[0],text='잘못된 OR 이유')]
        with self.assertRaises(ValueError):false_explanation_nodes(rule)
        for path in ([True],[-1],[99],'0'):
            rule=copy.deepcopy(self.policy.rules['housing_academic'])
            rule['false_explanation']=[dict(conjunct_path=path,text='잘못된 경로')]
            with self.subTest(path=path),self.assertRaises(ValueError):false_explanation_nodes(rule)

    def test_solver_uncertainty_never_supplies_a_false_reason(self):
        import z3
        rule=self.policy.rules['admissions_susi_application_limit']
        solver=Mock()
        for outcomes in ([z3.unknown],[z3.sat,z3.unknown]):
            solver.check.side_effect=outcomes
            with patch('mylittlellm.policy_rules.rule_problem',return_value=(solver,{'counted_applications':z3.Int('count')},None)):
                self.assertEqual(explain_false(rule,{'counted_applications':6}),'')


if __name__=='__main__':unittest.main()
