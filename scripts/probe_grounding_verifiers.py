"""Development-only Korean polarity probe; select before any fresh evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.claim_verifier import ClaimVerifier
from mylittlellm.retrieval import ROOT

PAIRS = [
 ('군 지역 범위 이내는 원거리로 인정하지 않습니다.', '군 지역 범위 이내는 원거리로 인정하지 않습니다.', '군 지역 범위 이내는 원거리로 인정합니다.'),
 ('논술전형 지원자는 검정고시 성적증명서를 제출하지 않습니다. 합격증명서는 제출합니다.', '논술전형 지원자는 검정고시 성적증명서를 제출할 필요가 없습니다.', '논술전형 지원자는 검정고시 합격증명서를 제출할 필요가 없습니다.'),
 ('주거비 지원의 월 상한은 20만원입니다.', '월 최대 지원 금액은 20만원입니다.', '월 최대 지원 금액은 30만원입니다.'),
 ('학생은 장학금 지급일로부터 5년간 지출 증빙서류를 보관해야 합니다.', '증빙 보관 기간은 장학금 지급일부터 5년입니다.', '증빙 보관 기간은 지급 요청서를 제출한 날부터 5년입니다.'),
 ('개인별 입금일은 대학별로 정하며 이 문서에는 특정 학생의 입금 기록이 없습니다.', '이 문서로는 특정 학생의 실제 입금일을 확인할 수 없습니다.', '학생의 계좌에 오늘 장학금이 입금되었습니다.'),
 ('2학기 신청 일정은 추후 시행계획에 반영할 예정입니다.', '2학기 신청 일정은 추후 계획에 반영될 예정입니다.', '2학기 신청 마감일은 9월 15일로 확정되었습니다.'),
 ('선발된 학생은 월 20만원 한도에서 인정된 지출을 지원받을 수 있습니다.', '인정된 지출은 월 20만원 한도에서 지원 가능합니다.', '학생에게 장학금 20만원이 이미 지급되었습니다.'),
 ('신청서 제출만으로 선발이 확정되지는 않습니다.', '신청서를 제출해도 선발이 확정되는 것은 아닙니다.', '신청서를 제출하면 선발이 확정됩니다.'),
]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['minicheck','multilingual_nli'],required=True)
    args=parser.parse_args();out=ROOT/'outputs/grounding-development';out.mkdir(exist_ok=True)
    path=out/(args.backend+'.json');
    if path.exists():raise ValueError('Refusing to overwrite development probe')
    verifier=ClaimVerifier(args.backend);rows=[]
    for i,(doc,positive,negative) in enumerate(PAIRS):
        for label,claim in [(1,positive),(0,negative)]:
            result=verifier.score(doc,claim)
            rows.append(dict(id=f'D{i+1}-{label}',language='ko',document=doc,claim=claim,label=label,**result))
            print(json.dumps(dict(id=rows[-1]['id'],label=label,**result)),flush=True)
    for label,claim in [(1,'The students are studying for exams.'),(0,'The students are on vacation.')]:
        doc='The students are in the library studying for their final exams.'
        rows.append(dict(id=f'EN-{label}',language='en',document=doc,claim=claim,label=label,**verifier.score(doc,claim)))
    candidates=[]
    for threshold in [.5,.7,.8,.9,.95]:
        ko=[r for r in rows if r['language']=='ko']
        accepted=lambda r:r['usable'] and r['support']>=threshold and (r['contradiction'] or 0)<.8
        candidates.append(dict(threshold=threshold,true_accept=sum(accepted(r) for r in ko if r['label']==1),
                               false_accept=sum(accepted(r) for r in ko if r['label']==0)))
    selected=next((r for r in candidates if r['false_accept']==0 and r['true_accept']>=6),None)
    value=dict(backend=args.backend,rows=rows,threshold_candidates=candidates,selected=selected,
               purpose='exposed_development_probe_not_a_reliability_guarantee',
               selection_rule='first threshold in [.5,.7,.8,.9,.95] with zero false accepts and at least 6/8 Korean true accepts')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'backend':args.backend,'selected':selected,'candidates':candidates}),flush=True)

if __name__=='__main__':main()
