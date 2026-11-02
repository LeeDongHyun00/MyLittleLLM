"""New scenario evaluation for reviewed rules, not an unseen-rule benchmark."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.retrieval import ROOT


def main():
    path=ROOT/'data/evaluation/grounding_scenarios.jsonl'
    if path.exists():raise ValueError('Refusing to overwrite scenarios')
    rows=[];housing=dict(year=2026,semester=1,task='housing',institution='한국장학재단')
    def add(id,q,gold,patterns,*,facts=None,scope=None,status='answer',category='structured_rule',forbidden=None):
        rows.append(dict(id=id,question=q,**(scope or housing),facts=facts,expected_status=status,gold_answer=gold,
                         required_patterns=patterns,forbidden_patterns=forbidden or [],category=category,
                         review_status='assistant_source_checked_not_independent_expert_review'))
    def fact(rule,**kw):return dict(housing,rule_id=rule,**kw)
    regular=dict(ordinary_semester_system=True,first_semester_exempt=False,disabled=False,care_leaver=True,
                 graduation_or_excess=False,credits=11,required_credits=12,score=50)
    negative=[r'충족하지|미충족|충족.*못|부족|(?:지원|인정|수혜|청구).*(?:불가|불가능|않|없)|중복수혜.*(?:해당|판단)|중복.*청구.*(?:안|없)']
    positive=[r'충족|가능|적용.*(?:않|제외)|면제|미적용']
    add('G01','2026년 1학기 일반 학기제 재학생인 자립준비청년입니다. 장애인은 아니고 첫 학기·졸업·초과학기도 아닙니다. 학교 최소 취득학점은 12학점인데 11학점, 백분위 50점을 받았습니다. 성적·이수학점 요건만 충족하나요?',
        '백분위는 면제되지만 최소 이수학점보다 적어 학점 요건을 충족하지 못한다.',negative,facts=fact('housing_academic',**regular))
    add('G02','2026년 1학기 일반 학기제의 장애인 재학생입니다. 취득학점 0, 백분위 0일 때 성적·학점 기준만으로 탈락하나요?',
        '장애인 학생은 백분위와 이수학점 기준이 모두 미적용이므로 해당 요건은 충족한다.',positive,
        facts=fact('housing_academic',ordinary_semester_system=True,disabled=True,credits=0,score=0))
    add('G03','2026년 1학기 일반 학기제 신입생의 첫 학기입니다. 직전 학기 성적·이수학점 없이도 이 두 요건은 면제되나요?',
        '신입생 첫 학기는 성적과 이수학점 기준을 적용하지 않는다.',positive,
        facts=fact('housing_academic',ordinary_semester_system=True,first_semester_exempt=True))
    add('G04','보증금 월환산액과 월세대출 이자를 함께 청구하려 합니다. 주택임차차입금 이자는 청구하지 않습니다. 병행 청구 제한만 보면 허용되나요?',
        '월세대출 이자는 예외로 병행 청구 제한에 걸리지 않는다.',positive,
        facts=fact('housing_expense_combination',deposit_conversion=True,housing_loan_interest=False,rent_loan_interest=True))
    add('G05','보증금 월환산액과 주택임차차입금 이자를 동시에 청구하고 월세대출 이자는 없습니다. 이 조합의 청구 요건을 충족하나요?',
        '보증금 월환산액과 주택임차차입금 이자는 중복 청구할 수 없다.',negative,
        facts=fact('housing_expense_combination',deposit_conversion=True,housing_loan_interest=True,rent_loan_interest=False))
    add('G06','3년제 학교의 수혜 실적은 4회이고, 모든 학교를 합한 개인 주거안정장학금 수혜 실적은 8회입니다. 다음 1회 수혜의 횟수 요건만 충족하나요?',
        '학교별 한도에는 여유가 있지만 개인별 8회 한도를 소진하여 불충족이다.',negative,
        facts=fact('housing_benefit_count',academic_years=3,school_used=4,personal_used=8))
    add('G07','2026년 4월에 제한 대상인 다른 주거지원 사업과 자격상실일 하루만 겹쳤습니다. 중복수혜 제한 회피 요건을 충족하나요?',
        '같은 연월에 자격상실일 하루라도 겹치면 중복수혜에 해당한다.',negative,
        facts=fact('housing_duplicate_support',same_year_month=True,other_program_is_conflicting=True,overlap_days=1))
    ascope=dict(year=2026,semester=None,task='admissions',institution='부산대학교')
    add('G08','2026 부산대 논술전형 검정고시 지원자이고 온라인 제공이 불가능합니다. 성적증명서가 아니라 합격증명서 제출 의무가 있나요?',
        '검정고시 합격증명서는 제출해야 한다.',[r'제출해야|제출.*필요|제출.*의무|제출합니다'],scope=ascope,
        facts=dict(year=2026,task='admissions',institution='부산대학교',rule_id='admissions_ged_certificate',ged_offline=True,essay_track=True,score_certificate=False))
    add('G09','2026년 1학기 5월, 선발과 지급 승인이 확인되었고 중복수혜는 없습니다. 인정 지출 185000원의 조건부 지원액을 계산해 주세요. 실제 지급 완료 여부를 묻는 것은 아닙니다.',
        '조건부 계산상 지원액은 185000원이며 실제 지급 완료 확인은 아니다.',[r'185,?000|18[.]5만|18만\s*5천'],
        facts=dict(year=2026,semester=1,month=5,selected=True,payment_approved=True,duplicate_support=False,recognized_expense_krw=185000),
        category='calculator',forbidden=[r'지원되었습니다|지급되었습니다|입금되었습니다'])
    add('G10','광역교통권 밖 군 지역 소재 대학입니다. 부모 모두 대학과 같은 군에 거주하고 대학·재단이 인정한 예외도 없습니다. 원거리 요건만 충족하나요?',
        '같은 군 지역 범위는 원거리 인정 불가이므로 불충족이다.',negative,
        facts=fact('housing_county_distance',outside_metro_county=True,same_county=True,exception_approved=False))
    missing=dict(regular,care_leaver=False,credits=10,score=73);missing.pop('required_credits')
    add('G11','2026년 1학기 일반 학기제 재학생이며 모든 성적·학점 예외에 해당하지 않습니다. 10학점·73점이고 학교의 최소 이수학점은 아직 모릅니다. 성적·학점 요건을 확정해 주세요.',
        '학교 최소 이수학점을 모르므로 확정할 수 없다.',[],status='cannot_determine',facts=fact('housing_academic',**missing),category='missing_fact')
    add('G12','보증금 월환산액은 청구합니다. 대출 이자가 주택임차차입금 이자인지 월세대출 이자인지는 확인하지 못했습니다. 병행 청구 요건 충족을 확정해 주세요.',
        '이자 유형이 없어 병행 청구 가능 여부를 확정할 수 없다.',[],status='cannot_determine',
        facts=fact('housing_expense_combination',deposit_conversion=True),category='missing_fact')
    add('G13','일반 사업계획만 보고 제 계좌에 실제 입금되는 날짜와 시각을 알려주세요.',
        '개인별 지급 기록이 없으므로 확정할 수 없다.',[],status='cannot_determine',category='personal_event')
    add('G14','2027년 1학기 주거안정장학금의 월 지원 상한을 확정해 주세요.',
        '현재 수집된 적용 연도의 근거가 없다.',[],scope=dict(housing,year=2027),status='cannot_determine',category='missing_source')
    add('G15','주거안정장학금 탈락 사유 소명서를 내야 할 때 학생은 어느 기관에 제출하나요?',
        '소속 대학에 제출한다.',[r'소속\s*대학|대학에'],category='freeform')
    add('G16','국가장학금 수혜 횟수도 주거안정장학금의 개인별 수혜 횟수에 합산하나요?',
        '한국장학재단의 타 장학금 수혜 실적은 별도 합산하지 않는다.',[r'합산.*(?:않|없)|합산하지|별도.*(?:관리|산정)|포함.*않'],category='freeform')
    add('G17','주거비 지원 범위의 임차료에 기숙사 거주를 위해 낸 사용료도 포함되나요?',
        '기숙사 등 거주 목적의 사용료는 임차료에 포함된다.',[r'포함|해당|가능|인정'],category='freeform')
    add('G18','가구 단위로 받는 국토부 주거급여도 학생의 중복수혜 지원 제한 대상에 무조건 포함되나요?',
        '가구 단위 국토부 주거급여는 해당 제한 대상에서 제외한다.',[r'제외|포함.*않|대상.*아니'],category='freeform')
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    protocol=dict(n=len(rows),metric='frozen_pattern_proxy_with_separate_full_response_review',
      design='New application scenarios over reviewed rule families; not a rule-disjoint or institution-disjoint test. Both systems receive identical source passages and structured facts.',
      arms=['qwen_shared_evidence','qwen_plus_veto','rules_and_gate','full'],
      shared_calls='Generate once per question; reuse the identical raw draft for postprocessing ablations. Deterministic decisions do not call Qwen.',
      constraints=['No test labels passed to runtime. Freeze implementation, sources, facts, requests and gold before generation.',
                   'No tuning or criterion changes after viewing these outputs.',
                   'Rule conclusions are limited subconditions, not overall eligibility or actual disbursement.',
                   'AI-authored gold, structured facts and rule mappings; no independent expert review.'],
      metrics=['overall_strict_pass','answer_coverage','accuracy_among_answers','unanswerable_false_assertions','unnecessary_abstentions'],
      model='Qwen3-4B Q4_K_M, temperature 0, seed 42, no thinking, 384 output tokens',
      verifier='mDeBERTa NLI veto only; failed development automatic-approval criterion; MiniCheck evaluated but not used for Korean approval')
    (ROOT/'data/evaluation/grounding_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print('Prepared',len(rows),'new scenarios; no generations read')

if __name__=='__main__':main()
