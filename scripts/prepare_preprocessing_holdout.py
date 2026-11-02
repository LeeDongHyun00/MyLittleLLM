"""Freeze new source-checked rule questions before any model answers are seen."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.hierarchical import sha
from mylittlellm.evaluation_v2 import compact
from mylittlellm.retrieval import ROOT,read_jsonl


def main():
    output=ROOT/'data/evaluation/preprocessing_holdout.jsonl'
    if output.exists():raise ValueError('Holdout exists; refusing overwrite')
    pnu='pusan-2026-susi';housing='kosaf-2026-housing-plan-faq'
    # id, group, source, page, question, answer, answer patterns, source patterns
    specs=[
      ('P01','pnu-name-id-document',pnu,23,'2026학년도 부산대 수시 제출서류의 성명이나 주민등록번호가 서로 다릅니다. 이 경우 추가로 어떤 서류를 내야 하나요?',
       '주민등록초본을 제출해야 합니다.',['주민등록초본'],['성명또는주민등록번호가다른경우주민등록초본']),
      ('P02','pnu-document-separate-sets',pnu,23,'2026학년도 부산대에서 하나의 지원자격으로 두 전형에 지원했습니다. 증빙서류는 한 묶음만 보내도 되나요, 전형별로 각각 내야 하나요?',
       '제출서류는 전형별로 각각 제출해야 합니다.',['(각각|전형별|별도).{0,15}(제출|내야)'],['2개이상의전형에지원하는경우제출서류는각각제출']),
      ('P03','pnu-essay-application-fee',pnu,33,'2026학년도 부산대 논술전형의 모집요강상 전형료는 얼마인가요? 면제 사유가 없는 경우입니다.',
       '60,000원입니다.',['(60,?000원|6만원)'],['논술전형.{0,30}60,?000']),
      ('P04','pnu-refund-account-holder',pnu,34,'2026학년도 부산대 원서접수에 입력하는 전형료 환불 계좌는 본인 명의만 되나요? 보호자 명의 계좌도 가능한가요?',
       '본인 또는 보호자 명의의 입출금이 자유로운 계좌를 입력할 수 있습니다.',['보호자','(가능|입력|사용|됩니다|됩니다)'],['본인또는보호자의계좌번호']),
      ('P05','pnu-residual-fee-refund-date',pnu,34,'2026학년도 부산대 입학전형료에 잔액이 생기면 언제까지 반환한다고 요강에 명시되어 있나요?',
       '2026년 4월 30일까지 반환합니다.',['2026','4월30일|4[./-]30'],['2026년4월30일까지반환']),
      ('P06','housing-regular-grade-credit',housing,9,'2026년 1학기 주거안정장학금에서 일반 재학생의 직전학기 이수학점과 100점 만점 성적 기준은 각각 얼마인가요? 학년제나 예외 대상은 아닙니다.',
       '직전학기 12학점 이상, 100점 만점 70점 이상입니다.',['12학점','70점'],['12학점','70점이상']),
      ('P07','housing-school-semester-count',housing,11,'2026년 1학기 주거안정장학금 시행계획에서 4년제 대학의 학제별 최대 지원 횟수는 몇 회인가요? 다른 학교 수혜 이력은 없습니다.',
       '4년제의 학제별 최대 지원 횟수는 8회입니다.',['8회'],['4년제','8회']),
      ('P08','housing-vacation-seasonal-class',housing,11,'2026년 1학기 주거안정장학금에서 방학 중 계절학기를 수강하는 경우도 지원할 수 있나요? 다른 요건은 모두 충족했습니다.',
       '계절학기를 수강하는 경우 방학 중에도 지원할 수 있습니다.',['(가능|지원할수|지원받을수|지원됩니다|지원대상)'],['계절학기를수강하는경우방학중에도지원가능']),
      ('P09','housing-deposit-conversion-rate',housing,10,'2026년 1학기 주거안정장학금에서 보증금을 월차임으로 환산할 때 준용하는 연 환산율은 몇 퍼센트인가요?',
       '연 4%입니다.',['4(%|퍼센트)'],['환산율.{0,6}연4%']),
      ('P10','housing-nonmetro-county-boundary',housing,8,'2026년 1학기 주거안정장학금에서 광역교통권에 속하지 않는 군 지역 소재 대학은 어느 범위까지 원거리로 인정하지 않나요?',
       '해당 군 지역 범위 이내까지 원거리로 인정하지 않습니다.',['해당군','(인정하지|인정불가|미인정|인정되지)'],['해당군지역범위이내까지만원거리인정불가']),
      ('P11','housing-unpublished-second-semester-schedule',housing,25,'수집된 2026년 주거안정장학금 시행계획의 추진 일정표만 보고 2학기 학생 신청 마감 날짜를 확정해 주세요.',
       '2학기 통합신청 일정을 고려하여 2학기 시행계획에 반영 예정이므로 이 자료만으로 확정할 수 없습니다.',[],[]),
      ('P12','housing-individual-transfer-date',housing,25,'2026년 1학기 주거안정장학금의 이 시행계획만으로 부산대가 제 계좌에 실제 입금할 날짜와 시각을 확정해 주세요.',
       '지급 일정은 대학별 자체 운영이므로 개인별 실제 입금 날짜와 시각을 확정할 수 없습니다.',[],[]),
    ]
    snapshots={r['source_id']:r for r in read_jsonl(ROOT/'data/retrieval_snapshot.jsonl')}
    old_groups=set()
    for file in ['data/evaluation/holdout_v1.jsonl','data/evaluation/rag_v2_gold.jsonl','data/examples/rag_triplets.jsonl']:
        old_groups.update(r['leakage_group'] for r in read_jsonl(ROOT/file))
    rows=[]
    for id,group,sid,page,question,answer,required,support in specs:
        if group in old_groups:raise ValueError('Old rule group reused')
        source=next(p for p in read_jsonl(ROOT/snapshots[sid]['extracted_path']) if p['pdf_page']==page)
        for pattern in support:
            if not re.search(pattern,compact(source['text'])):raise ValueError('Source pattern missing '+id+': '+pattern)
        rows.append(dict(id=id,leakage_group=group,year=2026,semester=(2 if id=='P11' else 1) if sid==housing else None,
                         institution='한국장학재단' if sid==housing else '부산대학교',task='housing' if sid==housing else 'admissions',
                         question=question,gold_answer=answer,required_patterns=required,support_patterns=support,
                         source_id=sid,gold_page=page,source_sha256=snapshots[sid]['sha256'],
                         expected_status='cannot_determine' if id in {'P11','P12'} else 'answer',
                         review_status='assistant_source_checked_not_independently_human_reviewed'))
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'gold_sha256':sha(output),'n':len(rows),
              'answerable':10,'unanswerable':2,'groups':len({r['leakage_group'] for r in rows}),
              'old_rule_groups_overlap':False,'source_documents_shared_with_development':True,
              'test_model_answers_seen':False,'rubric':'frozen answer/support patterns plus status; manual review reported separately',
              'independent_human_review':False,'scope':'new rule questions within known sources, not institution or document holdout'}
    (ROOT/'data/evaluation/preprocessing_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(protocol,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
