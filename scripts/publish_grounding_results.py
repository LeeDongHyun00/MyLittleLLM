"""Publish small audits and own responses, excluding full policy contexts."""
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mylittlellm.retrieval import ROOT,read_jsonl


def main():
    run=ROOT/'outputs/grounding-evaluation';reports=ROOT/'reports'
    summary=json.loads((run/'summary.json').read_text())
    gold={r['id']:r for r in read_jsonl(ROOT/'data/evaluation/grounding_scenarios.jsonl')}
    results=read_jsonl(run/'results.jsonl');plans=read_jsonl(run/'plans.jsonl')
    for name in ['summary.json','lock.json','drafts.jsonl','results.jsonl']:
        shutil.copyfile(run/name,reports/('grounding_'+name))
    audit=[dict(id=p['id'],preflight_route=p['preflight'].get('reason') if p['preflight'] else None,
                request_sha256=__import__('hashlib').sha256(json.dumps(p['payload'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                sources=[{k:d.get(k) for k in ['chunk_id','source_id','pdf_page','sha256']} for d in p['evidence']]) for p in plans]
    (reports/'grounding_input_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    with (reports/'grounding_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','arm','category','question','gold_answer','status','answer','citations','strict_pass','llm_called','generation_seconds','verifier_seconds']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in results:
            value=r['value'];g=gold[r['id']]
            writer.writerow({**{k:r[k] for k in ['id','arm','category','llm_called','generation_seconds','verifier_seconds']},
                             **{k:g[k] for k in ['question','gold_answer']},**{k:value[k] for k in ['status','answer']},
                             'citations':json.dumps(value['citations'],ensure_ascii=False),'strict_pass':r['score']['strict_pass']})
    lines=['# 근거 충분성·규칙·주장 검증 실제 평가','',
           '**이 제한된 비교에서는 규칙·보류 검사가 개선됐고 신경망 검증 필터는 성능을 떨어뜨렸다.** 자동 기준은 기본 모델 13/18, 규칙·보류 18/18, 전체 결합 15/18이었다. 현재 `--pipeline grounded`는 규칙·보류 경로를 사용하고, 신경망 검증은 `--claim-check` 실험 옵션으로만 제공한다. 이는 공개된 비교를 보고 선택한 설정이며 별도의 독립 재검증 결과는 아니다.','',
           'Qwen3-4B Q4_K_M 기본 모델, CPU, 384 출력 토큰, temperature 0, seed 42, thinking 비활성화 조건이다. 새 학습은 하지 않았다. 아래 결과는 자동 패턴 기준이며 전문가 판정 정확도가 아니다.','',
           '## 비교 설계','',
           '검수한 규칙의 새 적용 시나리오 18개(답변 가능 14, 보류 필요 4)를 사전 고정했다. 기존 12문항의 점수와 합산하지 않는다. 새 규칙·대학·문서 일반화 검증이 아니다. 사람이 승인한 gold도 아니다. 규칙 선택과 조건은 구조화 입력으로 제공했다. 자연어에서 자동 추출한 성능이 아니다.','',
           '규칙 적용 문항에서는 원문 조항을 직접 조회하고 기본 모델에도 같은 근거와 조건을 제공했다. 자유 질문은 공통 BM25 상위 2개를 사용했다. 동일 생성 결과를 후처리 비교에 재사용하여 불필요한 중복 추론을 하지 않았다. 규칙·계산 분기는 생성 없이 조건부 문장을 출력한다.','',
           '| 조건 | 전체 자동 통과 /18 | 답변한 수 /18 | 답한 경우 자동 정답률 | 보류 필요 문항에 확정 답변 /4 | 불필요 보류 /14 | 경로별 Qwen 호출 수 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in summary['arms'].items():
        accuracy='—' if r['accuracy_among_answers'] is None else f"{100*r['accuracy_among_answers']:.1f}%"
        lines.append(f"| {arm} | {r['strict_pass']} | {r['answered']} | {accuracy} | {r['false_assertions']} | {r['unnecessary_abstentions']} | {r['llm_calls']} |")
    lines.extend(['',f"실제 Qwen 생성은 총 {summary['actual_model_calls']}회, 시스템 판단 비교는 {summary['decisions']}건이다. 생성 오류 {summary['generation_errors']}건, 출력 잘림 {summary['truncated']}건을 포함한다. 표의 경로별 호출 수를 합산한 값이 실제 생성 횟수는 아니다.",'',
                  '- qwen_shared_evidence: 동일 원문·조건을 받은 Qwen 답변.',
                  '- qwen_plus_veto: 같은 답변에 다국어 NLI 및 지급 완료 표현 검사를 적용.',
                  '- rules_and_gate: 검수된 AST·Z3·계산기와 사전 보류 검사. 범위 밖 자유 질문은 같은 Qwen 답변.',
                  '- full: rules_and_gate의 자유 질문에 검증 필터까지 적용. 규칙 증명 결과를 신경망 점수로 바꾸지 않음.','',
                  '## 범주별 자동 통과','',
                  '| 범주 | 문항 수 | Qwen | Qwen+검증 | 규칙·보류 | 전체 결합 |','|---|---:|---:|---:|---:|---:|'])
    arms=list(summary['arms'])
    for cat in summary['arms'][arms[0]]['categories']:
        values=[summary['arms'][a]['categories'][cat] for a in arms]
        lines.append(f"| {cat} | {values[0]['n']} | "+' | '.join(str(v['passed']) for v in values)+' |')
    lines.extend(['','## 검증 모델의 개발 결과','',
                  '공식 MiniCheck-Flan-T5-Large와 다국어 mDeBERTa NLI를 실제 다운로드해 한국어 8쌍과 영어 1쌍에 실행했다. 미리 정한 자동 승인 채택 조건(한국어 정답 6/8 이상 승인, 오답 승인 0/8)은 둘 다 실패했다.','',
                  '| 모델 | 한국어 정답 승인 /8, 임계값 0.5 | 한국어 오답 승인 /8 | 해석 |',
                  '|---|---:|---:|---|',
                  '| MiniCheck-Flan-T5-Large | 0 | 0 | 한국어 unknown 토큰 다수. 모두 거절하는 것은 정확도 개선이 아님 |',
                  '| mDeBERTa-v3-base-mnli-xnli | 8 | 2 | 보관 기간 시작점·지급 완료 표현 오판. 신뢰도 승인 기준으로 부적합 |','',
                  '다국어 NLI는 검증을 통과한 진실 판정기가 아니라 오답 차단을 시도하는 보조 필터로 평가했다. 이 선택과 임계값은 새 답변 생성 전에 고정했다. 지지 점수 0.5, 모순 점수 0.8을 사용하며, 512토큰을 넘는 원문은 절단하지 않고 검증 불가로 처리한다. 맞는 답도 보류할 수 있다.','',
                  '## 범위와 한계','',
                  '- Sufficient Context의 아이디어를 제한된 사전 보류 규칙으로 응용했다. 일반 근거 충분성을 자동 판별하는 논문 모델 전체를 재현하지 않았다.',
                  '- NL2Logic/Logic-LM의 구조를 검수된 AST와 실제 Z3 실행으로 응용했다. 자동 자연어→논리식 변환은 하지 않았다. 검수와 구조화 입력에 드는 비용을 모델 자체의 능력 향상으로 계산하지 않는다.',
                  '- 성적·학점, 청구 제한, 중복수혜, 횟수, 일부 원거리 조건, 증명서 의무의 부분 판정이다. 전체 자격·합격·송금을 보장하지 않는다.',
                  '- 생성 시간은 공유한 실제 호출 기록이다. 단 한 번의 순차 실행이므로 안정적인 처리량 비교로 해석하지 않는다. 모델 로딩과 조건 입력·규칙 검수 비용도 별도로 고려해야 한다.',
                  '- 정규식은 의미가 같은 표현을 놓치거나 추가 오답을 놓칠 수 있다. 원시 답변과 별도 의미 검토를 확인해야 한다.','',
                  '## 자료','',
                  '- [구현 범위·실행·필드 의미](../docs/grounding.md)',
                  '- [시험 시나리오](../data/evaluation/grounding_scenarios.jsonl), [고정 프로토콜](../data/evaluation/grounding_protocol.json)',
                  '- [문항별 CSV](grounding_results.csv), [원시 생성](grounding_drafts.jsonl), [집계](grounding_summary.json)',
                  '- [코드·모델·입력 잠금](grounding_lock.json), [입력 감사](grounding_input_audit.json)',
                  '- [MiniCheck 개발 결과](grounding_development_minicheck.json), [다국어 NLI 개발 결과](grounding_development_multilingual_nli.json)',
                  '- [실제 규칙·계산 CLI 확인](grounding_cli_smoke.json), [모순 입력 보류 CLI 확인](grounding_inconsistent_cli_smoke.json)',
                  '- [실행 환경·테스트·서버 상태](grounding_runtime.json)'])
    lines.extend(['','## 의미 검토와 실무 판단','',
                  '규칙·보류 경로는 월세대출 이자의 예외, 학교별 실적과 전체 누적 실적의 구분, 최소 학점·이자 유형 누락, 개인 입금 기록 부재를 처리했다. 기본 모델의 자동 실패 5건을 이 범위에서 해결했다. 이미 맞힌 원거리 문항까지 새로 개선한 것으로 세지 않는다.','',
                  '신경망 검증은 월세대출 예외 오답에 지지 점수 약 0.984를 주고, 수혜 횟수 오답도 통과시켰다. 동시에 정답을 여럿 거절했다. 전체 결합의 자유 질문 3건은 512토큰을 넘는 인용 구간 때문에 보류되어, 정확한 문맥 해석 실패와 길이 제한 효과를 구별해야 한다. 원문-주장 검사만으로 사용자의 구조화 조건을 올바르게 적용했는지 검증할 수도 없다.','',
                  '18/18은 부분 판정의 자동 점수다. 면제 대상자에게도 최소 학점 입력이 필수인 것처럼 읽히는 규칙 설명 2건이 의미 검토에서 발견됐다. 완료 후 해당 설명을 조건부 표현으로 고쳤고, 원래 답변과 점수는 보존했다. 기본 모델의 조건부 금액 답변은 숫자가 맞았지만 상한 조항만 인용해 실제 지출 지원 조항의 인용이 불완전했다.','',
                  '공개된 시나리오는 이제 개발 자료다. 다음 효과 검증은 새로운 독립 검수 자료와 자연어→조건 추출 단계까지 포함해 수행해야 한다.'])
    for name,label in [('grounding_qualitative_review.json','전체 응답의 의미 검토'),('grounding_post_eval_check.json','평가 이후 수정과 동등성 확인')]:
        if (reports/name).exists():lines.extend(['',f'[{label}]({name}) — 기존 자동 점수를 소급 변경하지 않는다.'])
    (reports/'grounding_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'published':len(results),'report':'reports/grounding_report.md'}))

if __name__=='__main__':main()
