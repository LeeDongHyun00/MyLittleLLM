"""Author a new, explicitly non-independent engineering evaluation and review form."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.policy_rules import PolicyRules
from mylittlellm.grounded_assistant import GroundedAssistant
from mylittlellm.retrieval import ROOT


def main():
    target=ROOT/'data/evaluation/grounding_v2_scenarios.jsonl'
    if target.exists():raise ValueError('Refusing to overwrite scenarios')
    a=dict(year=2026,task='admissions',institution='부산대학교')
    h=dict(year=2026,task='housing',institution='한국장학재단',semester=1)
    positive=[r'충족|가능|지원할 수|대상.*해당']
    negative=[r'충족하지|불충족|불가|불가능|지원.*(?:없|안)|초과|해당하지|대상.*아니']
    forbidden=[r'합격(?:이|으로)?\s*확정|합격입니다']
    rows=[]
    def add(id,q,rule,values,gold,status='answer',decision='true',scope=None):
        scope=scope or a
        facts=dict(scope,rule_id=rule,**values) if rule else None
        rows.append(dict(id=id,question=q,**scope,facts=facts,expected_status=status,expected_decision=decision,
                         gold_answer=gold,required_patterns=(positive if decision=='true' else negative) if status=='answer' else [],
                         forbidden_patterns=forbidden+([r'충족하지|불충족|지원.*불가능'] if decision=='true' else []),
                         category='new_rule' if rule else 'gate_or_freeform',review_status='ai_authored_awaiting_independent_review'))
    add('V01','6회 제한에 포함되는 수시 전형으로 이미 다섯 번 접수했습니다. 이번 지원도 제한에 포함됩니다. 추가 한 번의 지원 횟수 요건만 충족하나요?',
        'admissions_susi_application_limit',dict(counted_application=True,counted_applications=5),'이번 지원은 여섯 번째이므로 횟수 요건만 충족한다.')
    add('V02','같은 대학 여러 전형을 포함해 6회 제한 대상 수시 원서를 여섯 번 접수했습니다. 이번도 제한 대상 전형입니다. 추가 한 번의 지원 횟수 요건을 충족하나요?',
        'admissions_susi_application_limit',dict(counted_application=True,counted_applications=6),'추가 지원은 일곱 번째여서 횟수 한도를 초과한다.',decision='false')
    add('V03','이번 지원은 6회 제한에 포함되는 수시 전형이지만, 이미 접수한 총횟수를 아직 모릅니다. 추가 지원 횟수 요건을 확정해 주세요.',
        'admissions_susi_application_limit',dict(counted_application=True),'기존 산입 횟수를 확인해야 한다.',status='cannot_determine',decision='unknown')
    add('V04','학생부종합 저소득층학생전형입니다. 법령상 동등 학력을 인정받았고 법정 차상위계층입니다. 수급자는 아닙니다. 학력·법정 대상자 요건만 충족하나요?',
        'admissions_low_income_status',dict(low_income_track=True,education_qualified=True,basic_livelihood_eligible=False,near_poverty_eligible=True),
        '인정된 동등 학력과 법정 차상위계층 자격으로 해당 두 요건을 충족한다.')
    add('V05','학생부종합 저소득층학생전형을 확인합니다. 정규 고교 졸업자지만 법정 수급권자·수급자, 차상위계층, 한부모가족지원법 지원대상자 모두 아닙니다. 해당 자격 요건을 충족하나요?',
        'admissions_low_income_status',dict(low_income_track=True,education_qualified=True,basic_livelihood_eligible=False,near_poverty_eligible=False,single_parent_support_eligible=False),
        '세 법정 대상자 범주 모두 아니므로 해당 자격 요건을 충족하지 못한다.',decision='false')
    add('V06','학생부종합 저소득층학생전형입니다. 고교 졸업자이고 법정 수급·차상위 대상은 아닙니다. 한부모 가구이지만 한부모가족지원법의 지원대상자인지는 모릅니다. 자격 요건을 확정해 주세요.',
        'admissions_low_income_status',dict(low_income_track=True,education_qualified=True,basic_livelihood_eligible=False,near_poverty_eligible=False),
        '가족 형태로 법정 자격을 추정하지 말고 한부모가족지원법 지원대상 여부를 확인해야 한다.',status='cannot_determine',decision='unknown')
    vf=dict(department_mapping_confirmed=True,eligible_school_graduate=True,meister_school=False,experiential_alternative_school=False,same_field_completed=False)
    add('V07','특성화고교졸업자전형의 모집단위별 전문교과 대응을 최신 공지로 확인했습니다. 인정되는 학교 졸업자이고 마이스터·체험형 대안 특성화고는 아닙니다. 동일계열 기준학과는 미이수지만 관련 전문교과 30단위를 이수했습니다. 학교유형·교과 요건만 충족하나요?',
        'admissions_vocational_subject',dict(vf,related_subject_credits=30),'확인된 비동일계열 관련 전문교과 30단위는 기준 이상이므로 해당 요건을 충족한다.')
    add('V08','특성화고교졸업자전형의 모집단위별 전문교과 대응을 최신 공지로 확인했습니다. 인정되는 학교 졸업자이며 제외 학교는 아닙니다. 동일계열 기준학과 미이수이고 관련 전문교과는 29단위입니다. 교과 요건을 충족하나요?',
        'admissions_vocational_subject',dict(vf,related_subject_credits=29),'관련 전문교과가 30단위 미만이라 해당 요건을 충족하지 못한다.',decision='false')
    no_mapping={k:v for k,v in vf.items() if k!='department_mapping_confirmed'}
    add('V09','특성화고교졸업자전형에서 인정되는 학교 졸업자이고 제외 학교는 아닙니다. 동일계열 기준학과 미이수이고 전문교과 35단위를 이수했지만 해당 모집단위의 관련 전문교과로 인정되는지 확인하지 못했습니다. 교과 요건을 확정해 주세요.',
        'admissions_vocational_subject',dict(no_mapping,related_subject_credits=35),'모집단위와 교과의 인정 관계를 최신 공지 또는 입학처에서 확인해야 한다.',status='cannot_determine',decision='unknown')
    add('V10','특성화고교졸업자전형의 교과 대응을 확인했고 관련 전문교과 50단위를 이수했습니다. 마이스터고 졸업자이며 체험형 대안학교는 아닙니다. 학교유형·교과 요건을 충족하나요?',
        'admissions_vocational_subject',dict(vf,meister_school=True,related_subject_credits=50),'관련 교과를 많이 이수해도 마이스터고는 제외 학교이다.',decision='false')
    add('V11','주거안정장학금 입금됐나요?',None,{},'사업계획만으로 개인 입금 완료를 확인할 수 없다.',status='cannot_determine',decision='unknown',scope=h)
    add('V12','저는 기숙사생입니다. 입금 기준으로 안내된 주거안정장학금의 일반적인 월 지원 상한은 얼마인가요?',None,{},'월 최대 20만원이다. 개인 입금 완료를 묻는 질문이 아니다.',scope=h)
    rows[-1]['required_patterns']=[r'20\s*만|200,?000'];rows[-1]['expected_decision']=None
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    protocol=dict(n=len(rows),design='Engineering scenarios for three newly implemented rule families and two gate questions; not unseen-rule generalization.',
                  arms=['qwen_same_evidence','expanded_rules_gate'],model='Qwen3-4B Q4_K_M; no training; temperature 0, seed 42, 384 output tokens, thinking off',
                  controls='Same source clauses and explicit scenario facts. One draft per scenario, reused for fallback. No answer labels sent to runtime.',
                  locked_before_generation=True,independent_expert_review=False,
                  limitations=['AI authored rules, facts and answer keys; new cases share reviewed rules and documents.',
                               'No natural-language slot extraction; guided intake asks explicit questions.',
                               'Pattern scores are proxies. Review every complete answer separately and preserve original scores.',
                               'Do not tune the implementation or patterns after reading these outputs. Report failures.'])
    (target.parent/'grounding_v2_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    directory=ROOT/'data/review';directory.mkdir(exist_ok=True)
    policy=PolicyRules();assistant=GroundedAssistant()
    # No generated response or proposed answer in the blind reviewer worksheet.
    with (directory/'grounding_v2_blind_review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','question','user_supplied_conditions','scope','source_references','reviewer_answer','reviewer_citations','reviewer_name','reviewed_at','notes'],lineterminator='\n')
        writer.writeheader()
        for row in rows:
            docs=policy.documents(policy.rules[row['facts']['rule_id']]) if row['facts'] else assistant.plan(row)['evidence']
            writer.writerow(dict(id=row['id'],question=row['question'],user_supplied_conditions=json.dumps(row['facts'],ensure_ascii=False),
                                 scope=json.dumps({k:row.get(k) for k in ['year','task','institution','semester']},ensure_ascii=False),
                                 source_references=' | '.join(f"{d['url']} PDF {d['pdf_page']}쪽" for d in docs)))
    print('Prepared',len(rows),'scenarios and an unfilled independent-review worksheet')


if __name__=='__main__':main()
