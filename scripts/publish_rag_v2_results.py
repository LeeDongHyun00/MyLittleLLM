"""Publish small audit artifacts without redistributing full source contexts."""
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.evaluation_v2 import ARMS, supported
from mylittlellm.hierarchical import sha
from mylittlellm.retrieval import ROOT, read_jsonl


def main():
    run = ROOT/'outputs/rag-v2-evaluation'
    if not (run/'summary.json').exists():
        raise ValueError('Finish all test requests before publishing or inspecting scores')
    report = ROOT/'reports'
    for name in ['summary.json','scored.jsonl','responses.jsonl','run_lock.json']:
        shutil.copyfile(run/name,report/('rag_v2_'+name))
    for source,dest in [
        ('outputs/rag-v2/manifest.json','rag_v2_index_manifest.json'),
        ('outputs/rag-v2-before-unicode-fix/manifest.json','rag_v2_initial_index_manifest.json'),
        ('outputs/retrieval_assets.json','rag_v2_retrieval_assets.json'),
        ('outputs/rag-v2-inputs/lock.json','rag_v2_input_lock.json'),
        ('outputs/rag-v2-prompt-selection/protocol.json','rag_v2_prompt_protocol.json'),
        ('outputs/rag-v2-prompt-selection/responses.jsonl','rag_v2_prompt_responses.jsonl')]:
        if (ROOT/source).exists():shutil.copyfile(ROOT/source,report/dest)
    inputs = {r['id']:r for r in read_jsonl(ROOT/'outputs/rag-v2-inputs/test.jsonl')}
    gold = {r['id']:r for r in read_jsonl(ROOT/'data/evaluation/rag_v2_gold.jsonl')}
    audit = []
    for id,row in inputs.items():
        item = {'id':id,'example_id':row['example']['id'] if row['example'] else None,
                'example_selection_seconds':row['example_seconds']}
        for mode in ['legacy','improved']:
            docs = row[mode]['evidence']
            item[mode] = {'seconds':row[mode]['seconds'],'clause_hit':supported(docs,gold[id]),
                          'context_tokens':row[mode].get('context_tokens'),
                          'documents':[{**{k:d[k] for k in ['chunk_id','source_id','pdf_page','year','sha256']},
                                        'retrieval_method':d.get('retrieval_method','legacy_bm25')} for d in docs]}
        audit.append(item)
    (report/'rag_v2_retrieval_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    requests = read_jsonl(run/'requests.jsonl')
    request_audit = [{'id':r['id'],'arm':r['arm'],
                      'payload_sha256':__import__('hashlib').sha256(json.dumps(r['payload'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                      'response_format':r['payload']['response_format']['type'],
                      'example_id':inputs[r['id']]['example']['id'] if r['arm']=='triplet' and inputs[r['id']]['example'] else None}
                     for r in requests]
    (report/'rag_v2_request_audit.json').write_text(json.dumps(request_audit,ensure_ascii=False,indent=2)+'\n')
    scored = read_jsonl(run/'scored.jsonl')
    with (report/'rag_v2_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','arm','question','gold_answer','answer','citations','valid','facts_pass',
                                                'citation_clause_pass','strict_pass','finish_reason','generation_seconds'],lineterminator='\n')
        writer.writeheader()
        for r in scored:
            try:value=json.loads(r['content'])
            except ValueError:value={}
            if not isinstance(value,dict):value={}
            writer.writerow({'id':r['id'],'arm':r['arm'],'question':gold[r['id']]['question'],'gold_answer':gold[r['id']]['gold_answer'],
                             'answer':value.get('answer',r['content']), 'citations':json.dumps(value.get('citations',[]),ensure_ascii=False),
                             **{k:r['score'][k] for k in ['valid','facts_pass','citation_clause_pass','strict_pass']},
                             'finish_reason':r.get('finish_reason'),'generation_seconds':r['elapsed_seconds']})
    summary=json.loads((run/'summary.json').read_text())
    names={'legacy':'기존 BM25','retrieval':'계층형·의미 검색','prompt_schema':'계층형·의미 검색+프롬프트·스키마','triplet':'계층형·의미 검색+프롬프트·스키마+예시'}
    lines=['# 계층형 검색·Qwen 재정렬·예시 적용 평가','',
           '같은 Qwen3-4B Q4_K_M 기본 모델로 새 서술형 시험 16문항·15개 규칙 그룹을 네 조건에서 비교했다. 개발 프롬프트 선택 8회와 새 시험 64회 생성 결과다. 아래 통과율은 고정된 자동 기준이며 독립 전문가가 판정한 의미 정확도가 아니다. 이전 객관식 28문항의 정확도와는 문항·지표가 달라 직접 비교하지 않는다.','',
           '| 조건 | 근거 조항 검색 | 유효 출력 | 필수 사실 | 인용 조항 | 전체 통과 | 잘림 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=summary['arms'][arm];n=a['n']
        lines.append(f"| {names[arm]} | {a['retrieval_clause_hit']}/{n} | {a['valid']}/{n} | {a['facts_pass']}/{n} | {a['citation_clause_pass']}/{n} | **{a['strict_pass']}/{n} ({a['strict_rate']:.1%})** | {a['truncated']} |")
    lines.extend(['','## 결과 해석','',
                  '전체 조합의 자동 통과율은 12.5%에서 43.8%로 올랐지만, 주된 변화는 160토큰 한도에서의 출력 완성이다. 기존 방식은 13/16건이 잘렸고 예시까지 추가한 조건은 0/16건이었다. 생성 모델의 지식이나 추론 정확도가 그만큼 향상됐다고 해석하면 안 된다.',
                  '', '새 검색의 고정 조항 패턴 적중은 12/16에서 8/16으로 떨어졌다. 검정고시 논술전형의 성적증명서 예외(E07), 증빙 보관 기간(H01), 발급일 기준(H05)에서 실제 오답이 나왔다. 문서에 붙인 1학기 메타데이터만으로 같은 문서 안의 2학기 표를 걸러내지 못하는 문제도 확인했다.',
                  '', '예시 조건의 6/16→7/16 차이는 E01 한 문항이며, 예시 없는 답의 올바른 “필요하지 않습니다”를 정규식이 놓친 결과다. 따라서 이 1점으로 예시가 의미 정확도를 높였다고 주장하지 않는다. E08에도 같은 표현 누락이 있고, E02는 다른 면접 안내 조항의 적용 범위를 검토해야 한다. 고정 점수는 수정하지 않고 [별도 질적 검토](rag_v2_qualitative_review.json)에 기록했다.',
                  '', '이번 결과로 새 검색 경로를 기본값으로 승격하지 않는다. 다음 비교의 우선순위는 기존 BM25에 짧은 프롬프트·스키마만 적용한 조건과, 절·전형·학기 단위로 적용 범위를 검수한 검색이다. 이 두 후속 비교는 이번 결과에 포함되지 않는다.',
                  '', '## 시간과 불확실성','',
                  '| 조건 | 검색 중앙값 | 생성 중앙값 | 검색·예시·생성 합산 추정 중앙값 | 전체 통과율의 그룹 bootstrap 95% 구간 |',
                  '|---|---:|---:|---:|---|'])
    for arm in ARMS:
        a=summary['arms'][arm];lo,hi=a['strict_cluster_95_ci']
        lines.append(f"| {names[arm]} | {a['median_retrieval_seconds']:.2f}초 | {a['median_generation_seconds']:.2f}초 | {a['median_estimated_warm_total_seconds']:.2f}초 | {lo:.1%}–{hi:.1%} |")
    c=summary['paired_comparisons']['legacy_vs_triplet'];lo,hi=c['paired_delta_cluster_95_ci']
    lines.extend(['',f"전체 개선 조건은 기존 대비 {c['improved']}문항 개선, {c['regressed']}문항 퇴보, {c['unchanged']}문항 동일했다. 통과율 차이의 규칙 그룹 bootstrap 95% 구간은 {lo*100:+.1f}∼{hi*100:+.1f}%p다.",
                  '', '검색 시간은 문항별 1회 측정하고 생성 시간과 합산했다. 최초 모델 로딩·전체 색인 생성은 이 합산 시간에 포함하지 않았다. CPU 4코어·메모리 16GiB이며, 실제로 호출한 임베딩·재정렬 모델은 동적 INT8으로 실행했다.',
                  '', '## 적용·해석 범위','',
                  '- [구현·재현 방법](../docs/rag_v2.md)에 논문 원본과 이 환경의 응용 구현 차이를 기록했다. 학생 모델 추가 학습이나 QLoRA는 수행하지 않았다.',
                  '- 부산대 입시 8문항·한국장학재단 주거 지원 8문항이다. 모두 답할 수 있는 절차 질문이며, 새 대학·새 문서·불확실성 처리 능력을 검증하지 않는다.',
                  '- 질문의 규칙 그룹은 기존 개발 자료와 분리했지만 원문 문서 자체는 공유한다. 예시는 기존 자료 7개에서만 선택했다.',
                  '- 검색 개선 조건은 계층·dense 검색·재정렬·문맥 예산 변화를 묶었고, 출력 개선 조건은 프롬프트와 JSON 스키마를 묶었다. 각 기술의 독립 효과로 읽지 않는다.',
                  '- 네 조건의 출력 한도는 160토큰이다. 기존 일반 CLI의 기본값 384토큰과는 다르므로, 이 결과를 기존 CLI 기본 설정의 성능으로 해석하지 않는다.',
                  '- 개발 프롬프트 두 후보 모두 기존 엄격 기준 0/4였다. 더 짧은 출력을 동률 기준으로 선택했으며, 개발 단계에서 정확도 우위를 확인한 것은 아니다. [개발 오류 검토](rag_v2_development_review.json)를 함께 확인한다.',
                  '- 실제 인용 텍스트의 조항 패턴까지 검사하지만 모든 추가 주장이나 부정 표현의 의미를 검증하지는 않는다. 결과를 본 뒤 고정 점수를 변경하지 않았다.',
                  '', '## 검토 자료','',
                  '- [64개 응답과 문항별 채점 CSV](rag_v2_results.csv), [전체 집계](rag_v2_summary.json), [원시 응답](rag_v2_responses.jsonl).',
                  '- [새 시험셋](../data/evaluation/rag_v2_gold.csv), [검색 근거 감사](rag_v2_retrieval_audit.json), [실행 해시](rag_v2_run_lock.json).',
                  '- [프롬프트 선택 결과](../config/rag_v2_selected.json), [개발 비교 응답](rag_v2_prompt_responses.jsonl), [원문 검토 기록](rag_v2_source_review.json).',''])
    runtime_path=report/'rag_v2_runtime_review.json'
    if runtime_path.exists():
        runtime=json.loads(runtime_path.read_text())
        before=runtime['initial_cli_peak_rss_kib']/1024**2
        after=runtime['final_cli_peak_rss_kib']/1024**2
        lines.extend(['## 실제 CLI 연결 검증','',
            f'64회 평가 후 별도 CLI 검사에서 생성 서버 연결이 끊기고 OOM kill 기록이 확인됐다. CLI 로더의 큰 FP32 복사본을 없애는 in-place INT8 변환과 사용하지 않는 메모리 반환을 적용했다. 이 두 검사에서 관찰한 CLI 프로세스의 최대 RSS는 {before:.2f}GiB에서 {after:.2f}GiB로 줄었다. 별도 생성 서버 메모리는 포함하지 않은 수치다.',
            '', '기존 개발 문항 A04의 검색 결과와 후보 점수는 기존 생성자와 차이가 0이었고, 재시도 후 서버 상태는 정상이며 OOM 횟수도 늘지 않았다. 기존 평가 코드·요청·정답 해시와 64회 결과는 변경하지 않았다.',
            '', 'CLI는 기존 월 지원 상한 예제에서 20만 원을 반환했으나, 선택한 인용 구간에는 상한 조항이 없었다. 따라서 연결·실행 검증 성공으로만 기록하며 근거 정확도 성공으로 세지 않는다. [초기 실패](rag_v2_cli_initial.json), [재시도](rag_v2_cli_smoke.json), [메모리·인용 검토](rag_v2_runtime_review.json)를 보존했다.',''])
    (report/'rag_v2_report.md').write_text('\n'.join(lines))
    print(json.dumps({'published_responses':len(scored),'report':'reports/rag_v2_report.md'},ensure_ascii=False))


if __name__=='__main__':main()
