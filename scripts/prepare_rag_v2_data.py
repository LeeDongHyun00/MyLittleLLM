"""Freeze new rule-disjoint questions and build examples exclusively from old development data."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.retrieval import ROOT, read_jsonl
from mylittlellm.hierarchical import sha


def compact(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text))


def cases():
    out = []
    def add(id, group, year, sid, page, question, answer, facts, support):
        housing = sid.startswith('kosaf')
        out.append({'id':id,'leakage_group':group,'year':year,'semester':1 if housing else None,
                    'institution':'한국장학재단' if housing else '부산대학교','task':'housing' if housing else 'admissions',
                    'question':question,'gold_answer':answer,'required_patterns':facts,'support_patterns':support,
                    'source_id':sid,'gold_page':page,'expected_status':'answer',
                    'review_status':'assistant_source_checked_not_independently_human_reviewed'})
    p='pusan-2026-susi';h='kosaf-2026-housing-plan-faq'
    add('E01','pnu-registration-method',2026,p,31,
        '2026학년도 부산대 수시 합격자의 등록의사는 어떤 방식으로 확인하나요? 예치금을 내야 하나요?',
        '예치금 납부 없이 온라인 문서등록으로 등록의사를 확인합니다.',
        [r'온라인문서등록',r'예치금.{0,25}(없이|없|불필요|내지|납부하지)'],[r'예치금납부없이온라인문서등록'])
    add('E02','pnu-missing-assessment',2026,p,30,
        '2026학년도 부산대 수시에서 필수 면접고사 하나에 결시했습니다. 다른 전형요소를 치렀더라도 불합격 등 처리 대상인가요?',
        '네. 전형요소 중 하나라도 결시하면 합격취소·불합격 또는 지원자격 상실 관련 사유에 해당합니다.',
        [r'결시',r'불합격|합격취소|자격상실'],[r'어느하나라도결시'])
    add('E03','pnu-tuition-payment-deadline',2026,p,31,
        '2026학년도 부산대 수시 모집요강에 적힌 등록금 납부 최종 날짜와 시각은 언제인가요?',
        '2026년 2월 5일 16시입니다.',
        [r'2026',r'2(?:월|[./-])5',r'16(?::00|시)|오후4시'],[r'2026\.2\.3.{0,35}2\.5.{0,15}16:00'])
    add('E04','pnu-tuition-payment-deadline',2025,'pusan-2025-susi',28,
        '2025학년도 부산대 수시 모집요강에 적힌 등록금 납부 최종 날짜와 시각은 언제인가요?',
        '2025년 2월 12일 16시입니다.',
        [r'2025',r'2(?:월|[./-])12',r'16(?::00|시)|오후4시'],[r'2025\.2\.10.{0,30}12\..{0,15}16:00'])
    add('E05','pnu-withdrawal-finality',2026,p,32,
        '2026학년도 부산대 수시에서 등록 포기 신청이 이미 완료됐습니다. 완료된 등록 포기를 번복할 수 있나요?',
        '등록 포기가 완료되면 번복할 수 없습니다.',
        [r'번복.{0,20}(없|불가|불가능|못)|번복불가'],[r'등록포기가완료되면절대번복불가'])
    add('E06','pnu-fraud-readmission-bar',2026,p,30,
        '2026학년도 부산대 수시 요강에서 부정한 방법의 지원 때문에 입학이 취소되면, 취소일부터 몇 년간 부산대에 입학할 수 없나요?',
        '입학이 취소된 날부터 3년간 부산대학교에 입학할 수 없습니다.',
        [r'3년|삼년'],[r'입학이취소된날로부터3년간'])
    add('E07','pnu-ged-essay-transcript-exception',2026,p,24,
        '2026학년도 부산대 논술전형에 지원한 검정고시 출신자입니다. 온라인 제공이 안 되어 서류를 내야 할 때, 검정고시 성적증명서도 반드시 제출하나요? 합격증명서는 묻지 않습니다.',
        '논술전형은 검정고시 성적증명서를 제출할 필요가 없습니다.',
        [r'성적증명서',r'불필요|필요.{0,6}없|제출하지|면제|낼필요.{0,6}없'],[r'검정고시성적증명서.{0,8}논술전형의경우제출불필요'])
    add('E08','pnu-csat-online-consent',2026,p,24,
        '2026학년도 부산대 수시 원서를 접수했습니다. 수능성적 자료를 별도로 제출해야 하나요?',
        '별도 제출은 필요 없으며, 원서 접수 시 수능성적 자료 온라인 제공에 동의한 것으로 간주합니다.',
        [r'별도.{0,15}(없|불필요)|제출.{0,10}(필요없|불필요)|제출할필요가없',r'온라인.{0,15}동의|동의.{0,15}간주'],[r'수능성적:별도제출필요없음',r'수능성적자료온라인제공에동의'])
    add('H01','housing-receipt-retention',2026,h,22,
        '2026년 1학기 주거안정장학금 지급요청서에 적은 지출의 증빙서류는 언제부터 몇 년간 보관해야 하나요?',
        '장학금 지급일로부터 5년간 보관해야 합니다.',
        [r'지급일|지급받은날',r'5년|오년'],[r'장학금지급일로부터5년간보관'])
    add('H02','housing-family-consent',2026,h,18,
        '2026년 1학기 주거안정장학금을 신청합니다. 기초·차상위 자격 확인이 끝났어도 부모 주소 확인을 위한 가구원 동의가 필요한가요?',
        '네. 원거리 심사를 위한 부모 주소 확인 때문에 가구원 동의가 필요합니다.',
        [r'동의.{0,15}(필요|필수)|동의해야',r'주소|원거리'],[r'주소정보확인을위하여가구원동의필수'])
    add('H03','housing-school-info-correction',2026,h,16,
        '2026년 1학기 주거안정장학금 신청에서 학교 정보를 잘못 입력했습니다. 현재 재학 중인 대학이 대학 정보를 수정할 수 있나요?',
        '네. 현재 재학 중인 대학에서 대학 정보를 변경하여 수정할 수 있습니다.',
        [r'대학',r'수정.{0,15}(있|가능)|변경.{0,15}(있|가능)'],[r'현재재학중인대학에서학생의대학정보변경'])
    add('H04','housing-selection-appeal-destination',2026,h,21,
        '2026년 1학기 주거안정장학금 탈락 후 원거리 인정 여부에 소명할 사유가 있습니다. 탈락 사유 소명서는 어디에 제출해야 하나요?',
        '소속 대학에 주거안정장학금 탈락 사유 소명서를 제출해야 합니다.',
        [r'소속대학|재학중인대학|재학대학'],[r'소속대학에주거안정장학금탈락사유소명서'])
    add('H05','housing-document-issue-window',2026,h,19,
        '2026년 1학기 주거안정장학금 서류는 장학금 신청일을 기준으로 언제부터 발급된 서류를 인정하나요?',
        '학생의 장학금 신청일 전 1개월부터 발급된 서류를 인정합니다.',
        [r'신청일.{0,12}전.{0,5}(1개월|한달)|신청일.{0,10}(1개월|한달)전'],[r'장학금신청일전1개월부터발급된서류만인정'])
    add('H06','housing-offline-consent-option',2026,h,20,
        '2026년 1학기 주거안정장학금 가구원 동의에서 부모가 고령이라 인증서를 이용하기 어렵습니다. 오프라인 서류 동의가 가능한가요?',
        '가능합니다. 인증서 동의가 어려운 고령 가구원은 동의서류를 제출할 수 있으며 재단 상담센터에 문의합니다.',
        [r'가능|할수있|제출할수있',r'서류|오프라인'],[r'고령이거나.{0,130}동의서류로제출가능'])
    add('H07','housing-fraud-restriction',2026,h,19,
        '2026년 1학기 주거안정장학금에 위조 서류를 제출한 경우, 발견일부터 최대 몇 년 동안 장학금 제한자로 분류될 수 있나요?',
        '발견일부터 최대 2년 동안 제한자로 분류될 수 있습니다.',
        [r'2년|이년'],[r'발견일로부터최대2년'])
    add('H08','housing-second-application-period',2026,h,25,
        '2026년 1학기 주거안정장학금 시행계획의 추진 일정표 기준으로 학생 2차 신청의 마감 날짜는 언제인가요?',
        '계획표에 기재된 2차 신청 마감은 2026년 3월 17일입니다.',
        [r'2026',r'3(?:월|[./-])17'],[r'2차.{0,5}26\.2\.3.{0,5}3\.17'])
    return out


def main():
    target = ROOT/'data/evaluation/rag_v2_gold.jsonl'
    if target.exists():
        raise ValueError('Frozen gold already exists')
    snapshots = {r['source_id']:r for r in read_jsonl(ROOT/'data/retrieval_snapshot.jsonl')}
    old = read_jsonl(ROOT/'data/evaluation/holdout_v1.jsonl')
    training = read_jsonl(ROOT/'outputs/distillation-smoke-v1/examples.jsonl')
    used_groups = {r['leakage_group'] for r in old+training}
    used_questions = {compact(r['question']) for r in old+training}
    rows = cases()
    for row in rows:
        assert row['leakage_group'] not in used_groups and compact(row['question']) not in used_questions
        snap = snapshots[row['source_id']]
        if sha(ROOT/snap['raw_path']) != snap['sha256']:
            raise ValueError('Source changed')
        pages = read_jsonl(ROOT/snap['extracted_path'])
        text = next(p['text'] for p in pages if p['pdf_page']==row['gold_page'])
        for pattern in row['support_patterns']:
            if not re.search(pattern,compact(text)):
                raise ValueError(f"Source pattern absent in {row['id']}: {pattern}")
        for pattern in row['required_patterns']:
            if not re.search(pattern,compact(row['gold_answer'])):
                raise ValueError(f"Gold answer does not pass its own rubric: {row['id']} {pattern}")
        row['source_sha256'] = snap['sha256']
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    protocol = {'frozen_at':datetime.now(timezone.utc).isoformat(),'questions':len(rows),
                'rule_groups':len({r['leakage_group'] for r in rows}),'old_rule_group_overlap':0,
                'gold_sha256':sha(target),'independent_human_review':False,'test_used_for_prompt_selection':False,
                'scope':'All 16 questions are answerable procedural rules; not a new abstention or admissions prediction benchmark.',
                'primary_metric':'JSON completion + all required fact patterns + cited passages themselves match supporting clause patterns; not human semantic accuracy'}
    (target.parent/'rag_v2_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (target.parent/'rag_v2_gold.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','leakage_group','year','task','question','gold_answer','source_id','gold_page'],extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    # Reuse only already-exposed development questions as source-checked teacher examples.
    inputs = {r['id']:r for r in read_jsonl(ROOT/'outputs/evaluation-v1/inputs.jsonl')}
    gold = {r['id']:r for r in old}
    original_gold = read_jsonl(ROOT/'outputs/evaluation-v1/gold.jsonl')
    dev_path = ROOT/'data/evaluation/rag_v2_development.jsonl'
    if dev_path.exists():raise ValueError('Development subset already exists')
    dev_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in original_gold if r['id'] in ['A04','A12','S04','S06']))
    examples=[]
    for id in ['A02','A08','A16','S04','S06','U04']:
        r,g = inputs[id],gold[id]
        docs = [d for d in r['oracle_documents'] if d['year']==r['year']][:1] if id!='U04' else []
        doc_id='example:'+id
        examples.append({'id':'triplet-'+id,'leakage_group':g['leakage_group'],'question':r['question'],
          'origin_id':id,'provenance':[{k:d[k] for k in ['source_id','pdf_page','year','sha256']} for d in docs],
          'scope':{k:r[k] for k in ['year','institution','task']},
          'evidence':[{'chunk_id':doc_id,'pdf_page':d['pdf_page'],'year':d['year'],'text':d['text']} for d in docs],
          'response':{'status':'cannot_determine' if id=='U04' else 'answer','answer':g['gold_answer'],
                      'citations':[doc_id] if docs else [],'follow_up':['지원할 학년도를 알려주세요.'] if id=='U04' else []},
          'review_status':'assistant_source_checked_not_independently_human_reviewed'})
    r=next(r for r in training if r['id']=='demo-housing-2026-april')
    d=next(d for d in r['documents'] if d['source_id'].startswith('kosaf'))
    examples.append({'id':'triplet-housing','leakage_group':r['leakage_group'],'question':r['question'],
       'origin_id':r['id'],'provenance':[{k:d[k] for k in ['source_id','pdf_page','year','sha256'] if k in d}],
       'scope':dict(r['scope'],task='housing'), 'evidence':[{'chunk_id':'example:housing','pdf_page':d['pdf_page'],'year':2026,'text':d['text']}],
       'response':{'status':'answer','answer':r['answer'],'citations':['example:housing'],'follow_up':[]},
       'review_status':'assistant_source_checked_not_independently_human_reviewed'})
    path=ROOT/'data/examples/rag_triplets.jsonl'
    if path.exists():raise ValueError('Triplet bank already exists')
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
    print(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
