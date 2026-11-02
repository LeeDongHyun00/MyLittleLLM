"""Publish reproducible small artifacts, never full retrieved policy text."""
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.hierarchical import sha
from mylittlellm.retrieval import ROOT,read_jsonl


def main():
    report=ROOT/'reports';run=ROOT/'outputs/preprocessing-holdout';dev=ROOT/'outputs/preprocessing-development'
    summary=json.loads((run/'summary.json').read_text());development=json.loads((dev/'summary.json').read_text())
    selection=json.loads((ROOT/'config/preprocessing_selected.json').read_text())
    for origin,prefix,names in [
        (run,'preprocessing_holdout_',['summary.json','responses.jsonl','scored.jsonl','lock.json']),
        (dev,'preprocessing_development_',['summary.json','responses.jsonl','scored.jsonl','lock.json'])]:
        for name in names:shutil.copyfile(origin/name,report/(prefix+name))
    shutil.copyfile(ROOT/'outputs/preprocessed/manifest.json',report/'preprocessing_index_manifest.json')
    inputs=read_jsonl(run/'inputs.jsonl')
    audit=[]
    for row in inputs:
        arms={}
        for arm,value in row['arms'].items():
            arms[arm]={k:v for k,v in value.items() if k!='evidence'}
            arms[arm]['documents']=[{k:d.get(k) for k in ['chunk_id','source_id','pdf_page','sha256','admission_track','section_semester','retrieval_method']} for d in value['evidence']]
        audit.append({'id':row['id'],'arms':arms})
    (report/'preprocessing_holdout_input_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    request_audit=[{'id':r['id'],'arm':r['arm'],'llm_called':r['llm_called'],
                    'payload_sha256':__import__('hashlib').sha256(json.dumps(r['payload'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
                   for r in read_jsonl(run/'requests.jsonl')]
    (report/'preprocessing_holdout_request_audit.json').write_text(json.dumps(request_audit,ensure_ascii=False,indent=2)+'\n')
    gold={r['id']:r for r in read_jsonl(ROOT/'data/evaluation/preprocessing_holdout.jsonl')}
    scored=read_jsonl(run/'scored.jsonl')
    with (report/'preprocessing_holdout_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['id','arm','question','gold_answer','answer','citations','status','valid','facts_pass','citation_clause_pass','strict_pass','seconds','finish_reason']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in scored:
            try:value=json.loads(r['content'])
            except ValueError:value={}
            writer.writerow(dict(id=r['id'],arm=r['arm'],question=gold[r['id']]['question'],gold_answer=gold[r['id']]['gold_answer'],
                                 answer=value.get('answer',r['content']),citations=json.dumps(value.get('citations',[]),ensure_ascii=False),
                                 status=value.get('status'),**{k:r['score'][k] for k in ['valid','facts_pass','citation_clause_pass','strict_pass']},
                                 seconds=r['elapsed_seconds'],finish_reason=r.get('finish_reason')))
    lines=['# 출력 설정·전처리·검색 병합 실제 평가','',
           'Qwen3-4B Q4_K_M 기본 모델을 CPU에서 실행했다. 추가 학습·QLoRA·새 증류는 하지 않았다. 기존 결과 파일을 유지하고 새 실행을 별도로 저장했다. 아래 수치는 고정 패턴 기반 자동 통과이며 전문가 판정 정확도가 아니다.','',
           '## 출력 설정 개발 비교','',
           '이미 공개된 8문항의 동일 BM25 상위 2개 근거로 비교했다. 160토큰 조건의 기존 응답 8개를 요청 동일성 확인 후 재사용하고, 384토큰 두 조건에서 16회 새로 생성했다.','',
           '| 조건 | 유효 출력 | 사실 패턴 | 인용 조항 | 전체 통과 | 잘림 |','|---|---:|---:|---:|---:|---:|']
    for arm,a in development['arms'].items():
        lines.append(f"| {arm} | {a['valid']}/8 | {a['facts_pass']}/8 | {a['citation_clause_pass']}/8 | {a['strict_pass']}/8 | {a['truncated']} |")
    lines.extend(['',f"자동 기준의 추천은 `{development['selected_prompt']}`였으나, 잘못된 추가 주장도 통과시키는 문제가 개발 검토에서 발견됐다. 새 검색 비교는 `{selection['prompt_name']}`·스키마와 출력 384토큰을 공통 조건으로 사용했다. 이는 사전 자동 선택 규칙에 대한 공개된 개발 단계 수정이며, 정확도 우위를 확인했다는 뜻이 아니다. [선택 근거](preprocessing_development_review.json)를 보존했다. 개발 중 별도 PDF 처리 작업이 일부 겹쳤으므로 개발 시간 차이는 순수한 설정 효과로 해석하지 않는다.",'',
                  '## 새 규칙 12문항 평가','',
                  '답변 가능 10문항과 확정 답변을 보류해야 하는 2문항이다. 질문·정답·채점 패턴을 먼저 고정하고, 설정·요청·입력 해시를 잠근 뒤 순서를 섞어 실행했다. 원문 문서는 개발 자료와 공유하지만 규칙 그룹 이름은 기존 질문·예시와 겹치지 않는다. 독립 전문가 검수는 받지 않았다.','',
                  '| 조건 | 근거 조항 포함 /10 | 유효 출력 /12 | 답변 가능 문항 통과 /10 | 보류 문항 통과 /2 | 전체 통과 /12 | 잘림 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for arm,a in summary['arms'].items():
        lines.append(f"| {arm} | {a['retrieval_clause_hits']} | {a['valid']} | {a['answerable_strict']} | {a['unanswerable_correct_abstentions']} | **{a['strict_pass']}/12** | {a['truncated']} |")
    lines.extend(['',f"새 시험의 실제 모델 호출은 {summary['new_model_calls']}회이고 시스템 판단은 {summary['total_decisions']}건이다. 근거가 없는 조건은 생성 호출 없이 보류한다. 모델 호출 실패와 잘림도 분모에서 제외하지 않았다.",'',
                  '| 조건 | 근거 토큰 중앙값 | 호출한 생성 시간 중앙값 | 잘못된 확정 답변 / 보류 문항 |',
                  '|---|---:|---:|---:|'])
    for arm,a in summary['arms'].items():
        lines.append(f"| {arm} | {a['median_evidence_tokens']} | {a['median_generation_seconds_called_only']:.2f}초 | {a['unsafe_assertions']}/2 |")
    lines.extend(['','생성 시간은 검색·모델 최초 로딩·문서 전처리 시간을 포함하지 않는다. 검색은 입력 준비 단계에서 따로 실행했고 문항별 시간은 입력 감사 파일에 보존했다.','',
                  '## 해석과 한계','',
                  '- concise_k2는 기존 BM25 상위 2개에 선택한 생성 설정을 적용한 조건이다. bm25_budget는 같은 BM25에서 최대 4개·1,800 근거 토큰, preprocessed는 같은 상한에 구조·범위·질의 정리·검색 병합을 함께 적용한 조건이다.',
                  '- 최대 토큰 상한이 같아도 실제 입력량은 다르다. 전처리, 범위 검사, 검색 선택의 개별 기여를 분리한 실험은 아니다.',
                  '- 12문항의 작은 연구용 표본이다. 새 대학·새 문서·연도 변화에 대한 일반화, 운영 환경의 전체 정확도를 추정하지 않는다.',
                  '- 일반 규정과 예외, 다른 표현의 동의어, 모든 추가 주장의 참거짓을 정규식으로 완전히 검사할 수 없다. 자동 점수와 별도 의미 검토를 구분한다.',
                  '- 기존 16문항을 사용한 개발 검색에서는 최종 색인의 고정 조항 패턴 적중이 15/16이었다. 이는 설정 선택에 사용한 개발 결과이며 새 시험 점수와 합산하지 않는다.',
                  '- 새 경로는 명시적 CLI 옵션이며 기본 파이프라인은 바꾸지 않는다. OCR·Docling·LLMLingua-2 재평가·샘플링 온도 비교는 이번 실행 범위에 포함하지 않았다.','',
                  '## 자료','',
                  '- [구현·명령어·범위](../docs/preprocessing.md)',
                  '- [새 시험셋](../data/evaluation/preprocessing_holdout.jsonl), [사전 고정 프로토콜](../data/evaluation/preprocessing_protocol.json)',
                  '- [문항별 답변 CSV](preprocessing_holdout_results.csv), [원시 응답](preprocessing_holdout_responses.jsonl), [전체 집계](preprocessing_holdout_summary.json)',
                  '- [새 실행 잠금](preprocessing_holdout_lock.json), [검색 입력 감사](preprocessing_holdout_input_audit.json), [요청 해시](preprocessing_holdout_request_audit.json)',
                  '- [개발 비교](preprocessing_development_summary.json), [개발 검색 점검](preprocessing_development_retrieval.json), [전처리 색인·페이지 감사](preprocessing_index_manifest.json)'])
    if (report/'preprocessing_qualitative_review.json').exists():
        lines.extend(['','[전체 응답의 별도 의미 검토](preprocessing_qualitative_review.json)를 함께 읽는다. 고정 자동 점수는 이 검토로 소급 변경하지 않았다.'])
    (report/'preprocessing_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'published':len(scored),'report':'reports/preprocessing_report.md'},ensure_ascii=False))


if __name__=='__main__':main()
