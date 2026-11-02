"""Publish the frozen v2 responses and compact audits, not full policy passages."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.retrieval import ROOT,read_jsonl


def main():
    run=ROOT/'outputs/grounding-v2';reports=ROOT/'reports'
    summary=json.loads((run/'summary.json').read_text())
    lock=json.loads((run/'lock.json').read_text())
    plans=read_jsonl(run/'plans.jsonl');results=read_jsonl(run/'results.jsonl')
    gold={r['id']:r for r in read_jsonl(ROOT/'data/evaluation/grounding_v2_scenarios.jsonl')}
    for name in ['summary.json','lock.json','drafts.jsonl','results.jsonl']:
        shutil.copyfile(run/name,reports/('grounding_v2_'+name))
    audit=[dict(id=p['id'],route=(p['preflight'] or {}).get('reason','qwen_generation'),planning_seconds=p['planning_seconds'],
                request_sha256=hashlib.sha256(json.dumps(p['payload'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                sources=[{k:d.get(k) for k in ['chunk_id','source_id','pdf_page','sha256']} for d in p['evidence']]) for p in plans]
    (reports/'grounding_v2_input_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    with (reports/'grounding_v2_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['id','arm','question','proposed_gold','status','answer','follow_up','citations','strict_pass','llm_called','generation_seconds']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in results:
            value=row['value'];g=gold[row['id']]
            writer.writerow(dict(id=row['id'],arm=row['arm'],question=g['question'],proposed_gold=g['gold_answer'],status=value['status'],answer=value['answer'],
                                 follow_up=json.dumps(value['follow_up'],ensure_ascii=False),citations=json.dumps(value['citations'],ensure_ascii=False),
                                 strict_pass=row['score']['strict_pass'],llm_called=row['llm_called'],generation_seconds=row['generation_seconds']))
    baseline=summary['arms']['qwen_same_evidence'];improved=summary['arms']['expanded_rules_gate']
    lines=['# 한국어 조건 수집·입시 규칙 확대 평가','',
           f"새 12개 시나리오의 자동 통과는 동일 근거 Qwen {baseline['strict_pass']}/12, 확대 규칙·보류 {improved['strict_pass']}/12다. 이는 AI가 작성한 구조화 시나리오의 공학적 비교다. 독립 전문가 정확도나 새 대학 일반화 성능을 뜻하지 않는다.",'',
           '## 구현 변경','',
           '- 부산대 2026 수시의 추가 지원 횟수, 저소득층학생전형 법정 자격, 특성화고 학교·전문교과 요건을 추가했다. 총 9개 부분 판정 규칙이다.',
           '- `--guided-rule`로 JSON 파일 없이 한국어 문답을 수행한다. 미확인을 false나 0으로 채우지 않고, 결론에 영향을 줄 수 있는 조건을 묻는다. 면제 등으로 결론이 나면 질문을 멈춘다.',
           '- 적용 범위 미확인, 적용 범위 밖, 입력 모순, 연산 시간 초과를 구별한다. 내부 필드명 대신 한국어 확인 질문을 제공한다.',
           '- 실제 입금 여부와 일반 지급 기준·확인 절차를 구분하는 제한된 보류 규칙을 보완했다. 신경망 검증은 사용하지 않았고 모델 가중치도 학습하지 않았다.','',
           '## 실제 Qwen 비교','',
           'Qwen3-4B Q4_K_M, CPU, temperature 0, seed 42, thinking off, minimal 프롬프트·JSON 스키마·384 출력 토큰이다. 새 규칙 10문항과 개인 기록/일반 기준 질문 2문항을 사용했다. 답변 가능 8개, 보류 필요 4개다. 규칙 문항에서는 양쪽에 동일한 원문 조항과 명시적 시나리오 조건을 제공했고, 나머지는 같은 BM25 상위 2개를 사용했다.','',
           '| 조건 | 자동 통과 /12 | 답변 수 /12 | 보류 필요 문항에서 answer 상태 /4 | 불필요 보류 /8 | 경로상 Qwen 호출 수 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,arm in summary['arms'].items():
        lines.append(f"| {name} | {arm['strict_pass']} | {arm['answered']} | {arm['false_assertions']} | {arm['unnecessary_abstentions']} | {arm['llm_calls']} |")
    lines.extend(['',f"실제 Qwen 호출은 {summary['actual_model_calls']}회, 비교한 판단은 {summary['decisions']}건이다. 같은 생성 결과를 자유 질문 경로에서 재사용했다. API 호출 오류 {summary['generation_errors']}건, 출력 잘림 {summary['truncated']}건이다. 경로별 호출 수를 합산해 실제 생성 횟수로 계산하지 않는다.",'',
                  f"코드·요청·원문·정답을 고정한 커밋은 `{lock['code_commit']}`이다. 응답 확인 후 고정 채점 기준을 바꾸지 않는다. 기존 18개 평가 점수와 합산하지 않는다.",'',
                  '## 개발·회귀 점검','',
                  '- 코드 테스트 74개 통과. 지원 횟수와 30학점 경계, 법정 대상자 조합, 제외 학교, 미확인 적용 범위, 질문 생략과 미확인 유지 등을 점검했다.',
                  '- 기존 18개 시나리오의 요청과 판정 상태·결론·인용을 유지했다. 부족한 조건 안내는 한국어 질문으로 바뀐다. 기존 모델 답변을 재생성하거나 점수를 바꾸지 않았다.',
                  '- 보류 의도 개발 예시 12개는 이전 규칙 5/12, 보완 규칙 12/12였다. 이 예시는 수정에 사용한 개발 자료로 일반화 점수에 포함하지 않는다.',
                  '- 실제 CLI에 스크립트로 답을 입력해 5회 접수, 횟수 미확인, 첫 학기 면제의 흐름을 확인했다. 세 사례 모두 Qwen 호출 없이 처리됐다. 독립 사용성 시험은 아니다.','',
                  '## 해석과 남은 일','',
                  '규칙 검토·갱신과 사용자의 명시적 조건 확인에 비용이 든다. 자연어에서 조건을 자동 추출하거나 외부 개인 기록을 조회하지 않는다. 원문·정답 검수자는 현재 AI이며 독립 전문가 승인은 없다. 작은 새 시나리오에 대한 결과로 운영 정확도를 확정할 수 없다.',
                  '', '보류 필요 문항의 answer 상태는 명시적인 보류 실패를 세는 지표다. 일반 규정을 올바르게 설명하면서 개인 조건의 미확인을 밝히지 않은 경우도 포함되므로, 사실 날조나 실제 합격·지급 확정 주장 건수와 같지 않다.',
                  '', '의미 검토는 정규식 채점이 놓치는 추가 주장·범위·인용 오류를 별도로 기록한다. 공개된 사례를 튜닝에 사용한 이후에는 새로운 독립 자료로 다시 평가해야 한다.','',
                  '## 자료','',
                  '- [실행과 적용 범위](../docs/guided_grounding.md)',
                  '- [문항별 CSV](grounding_v2_results.csv), [원시 생성](grounding_v2_drafts.jsonl), [집계](grounding_v2_summary.json)',
                  '- [잠금 기록](grounding_v2_lock.json), [입력 감사](grounding_v2_input_audit.json)',
                  '- [개발·기존 18개 회귀 점검](grounding_v2_development.json), [CLI 기록](grounding_v2_cli.json)',
                  '- [독립 검수용 빈 양식](../data/review/grounding_v2_blind_review.csv): 제안 정답·모델 응답을 제외했다. 검수는 아직 완료되지 않았다.'])
    for name,label in [('grounding_v2_qualitative_review.json','전체 답변 의미 검토'),('grounding_v2_runtime.json','실행 환경·서버 상태')]:
        if (reports/name).exists():lines.append(f'- [{label}]({name})')
    review=reports/'grounding_v2_qualitative_review.json'
    if review.exists():
        lines.extend(['','## 전체 답변 검토에서 확인한 점',''])
        lines.extend(json.loads(review.read_text())['report_paragraphs'])
    (reports/'grounding_v2_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(published=len(results),report='reports/grounding_v2_report.md')))


if __name__=='__main__':main()
