"""Evidence gates, reviewed symbolic decisions and a fallible neural veto."""
import json
import os
import re
import time

from .assistant import validate_answer
from .claim_verifier import ClaimVerifier, verify_claims
from .policy_rules import PolicyRules
from .preprocessed_assistant import calculation_evidence
from .preprocessing import PreprocessedRetriever
from .rag_v2 import complete, request_body
from .retrieval import ROOT, read_jsonl, retrieve


def abstain(reason, detail, follow='적용 조건이나 공식 확인 자료를 보완해 주세요.'):
    return dict(status='cannot_determine',answer=detail,citations=[],follow_up=[follow],reason=reason)


def sufficiency_gate(row, evidence):
    if not evidence:return abstain('no_scoped_evidence','해당 대학·학년도·학기에 적용되는 근거가 없습니다.')
    q=row['question']
    if requires_personal_record(q):
        return abstain('personal_event_not_in_policy','일반 모집요강·사업계획만으로 개인의 실제 선발·지급 여부나 입금 날짜·시각을 확인할 수 없습니다.',
                       '해당 대학의 개인별 처리 내역을 확인해 주세요.')
    if re.search(r'(?:합격|선발).{0,15}(?:보장|확실|확정해)',q):
        return abstain('outcome_not_in_policy','규정의 조건 충족만으로 최종 합격·선발을 확정할 수 없습니다.')
    return None


def requires_personal_record(question):
    """Limited Korean intent gate; policy procedures are not personal records."""
    for clause in re.split(r'[.!?。！？\n]+',question):
        clause=re.sub(r'(?:입금|지급|수령|합격|선발)[^,;]{0,20}(?:묻지|질문하지|확인하지)\s*않[^,;]{0,5}', '', clause)
        procedural=re.search(r'방법|절차|어디(?:서|에서)|어떻게|메뉴|경로',clause)
        # A request for a procedure must not hide a separate request for an event.
        mixed=re.search(r'(?:와|과|그리고|또한|뿐만).*(?:실제|정확한|됐는지|되었는지)',clause)
        if procedural and not mixed:continue
        personal=re.search(r'(?<![가-힣])(?:내|제)\s*(?:계좌|통장|입금|지급|지원|합격|선발)|저에게|제게|저는|제가|나의',clause)
        event=re.search(r'(?:입금|지급|수령|합격|선발).{0,16}(?:됐|되었|했|완료|확정|여부|결과)|(?:계좌|통장).*(?:들어왔|들어오|들어올)',clause)
        timing=re.search(r'(?:입금|지급|수령).*(?:언제|날짜|시각)|(?:언제|날짜|시각).*(?:입금|지급|수령)',clause)
        direct_status=re.search(r'(?:입금|지급|수령|합격|선발)(?:이|은|는|을|이\s*)?\s*(?:됐|되었|했).{0,4}(?:나요|는지|어요|습니까)|입금됐어|선발됐어|합격했어',clause)
        if direct_status or (personal and (event or timing)):
            return True
    return False


def grounded_request(row, evidence, facts=None):
    body=request_body(row,evidence,prompt_name='minimal',max_tokens=384)
    if facts is not None:
        payload=json.loads(body['messages'][-1]['content'])
        payload['user_supplied_facts']=facts
        body['messages'][-1]['content']=json.dumps(payload,ensure_ascii=False)
        body['messages'][0]['content']+='\nuser_supplied_facts는 사용자가 제시한 시나리오 조건입니다. 실제 선발·지급 확인으로 해석하지 마세요.'
    return body


class GroundedAssistant:
    def __init__(self, verifier=None):
        self.rules=PolicyRules()
        self.legacy=read_jsonl(ROOT/'outputs/index.jsonl')
        self.verifier=verifier
        self.config=json.loads((ROOT/'config/grounding.json').read_text())

    def plan(self, row, facts=None):
        if not isinstance(row.get('question'),str) or not 0<len(row['question'])<=2000:raise ValueError('Invalid question')
        scope={k:row.get(k) for k in ['year','task','institution','semester']}
        evidence=retrieve(self.legacy,row['question'],k=2,**scope)
        decision=None;proof=None
        if facts is not None:
            try:
                if isinstance(facts,dict) and 'rule_id' in facts:
                    proof,evidence=self.rules.evaluate(row,facts)
                    if proof['decision']=='unknown':
                        if proof['reason'] in {'missing_facts','missing_applicability'}:
                            decision=abstain('missing_rule_facts','입력 조건만으로 '+proof['label']+'을 판정할 수 없습니다.',
                                             '판정에 필요한 조건을 확인해 주세요.')
                            decision['follow_up']=[q['prompt'] for q in proof['questions']]
                            decision['questions']=proof['questions']
                        elif proof['reason']=='outside_rule_scope':
                            decision=abstain('outside_rule_scope','입력한 상황은 이 규칙의 적용 범위에 포함되지 않습니다.',
                                             '해당 전형·학교 제도에 맞는 별도 규정이나 담당 기관을 확인해 주세요.')
                        elif proof['reason']=='solver_unresolved':
                            decision=abstain('solver_unresolved','정해진 처리 시간 안에 조건을 판정하지 못했습니다.')
                        else:
                            decision=abstain('inconsistent_rule_facts','서로 모순되거나 판정할 수 없는 입력 조건이 있어 답변을 보류합니다.',
                                             '학교별 실적과 전체 누적 실적 등 입력 조건을 다시 확인해 주세요.')
                    else:
                        if proof['rule_id']=='admissions_ged_certificate':
                            conclusion='해당 증명서를 제출해야 합니다.' if proof['decision']=='true' else '해당 성적증명서 제출은 면제입니다.'
                        else:
                            conclusion=proof['label']+('을 충족합니다.' if proof['decision']=='true' else '을 충족하지 않습니다.')
                        decision=dict(status='answer',answer=conclusion+' '+proof['explanation'],
                                      citations=proof['sources'],follow_up=[],reason='reviewed_rule',decision=proof['decision'])
                else:
                    retriever=PreprocessedRetriever(semantic=None)
                    calculation,evidence=calculation_evidence(facts,row,retriever)
                    proof=dict(calculation=calculation)
                    decision=dict(status='answer',answer=f"입력한 승인·지출 조건에 따른 계산상 지원액은 {calculation['amount_krw']:,}원입니다. 실제 지급 완료를 확인한 결과는 아닙니다.",
                                  citations=[d['chunk_id'] for d in evidence],follow_up=[],reason='verified_calculator')
            except (ValueError,TypeError,KeyError) as e:
                decision=abstain('invalid_or_unsupported_facts','구조화된 조건을 적용할 수 없습니다: '+str(e))
        # Even a valid partial-rule proof does not establish a personal outcome.
        gate=sufficiency_gate(row,evidence)
        if gate is not None:decision=gate
        return dict(row=row,facts=facts,evidence=evidence,preflight=decision,proof=proof,
                    payload=grounded_request(row,evidence,facts) if evidence else None)

    def generate(self, plan, base_url=None):
        if plan['payload'] is None:return dict(content=json.dumps(abstain('no_scoped_evidence','해당 범위의 근거가 없습니다.'),ensure_ascii=False),
                                              elapsed_seconds=0,finish_reason='no_evidence',llm_called=False)
        return dict(complete(plan['payload'],base_url or os.environ.get('QWEN_BASE_URL','http://127.0.0.1:8080/v1')),llm_called=True)

    def parse(self, response, evidence):
        try:
            if response.get('error') or response.get('finish_reason')=='length':raise ValueError('Generation failed or truncated')
            return validate_answer(json.loads(response['content']),evidence,'rag')
        except (ValueError,TypeError) as e:
            return dict(status='invalid_response',answer='모델 출력 검증에 실패했습니다.',citations=[],follow_up=[],error=str(e))

    def postcheck(self, value, evidence):
        if value['status']!='answer':return value,dict(applied=False,reason='not_an_answer')
        if re.search(r'(?:지원|지급|입금)(?:되었|됐| 완료)|수령했',value['answer']):
            return abstain('unsupported_personal_event','정책 문서만으로 실제 지급 완료를 확인할 수 없습니다.'),dict(applied=True,passed=False,reason='payment_event_veto')
        cited=[d for d in evidence if d['chunk_id'] in value['citations']]
        if self.verifier is None:self.verifier=ClaimVerifier(self.config['verifier_backend'])
        review=verify_claims(value['answer'],cited,self.verifier,threshold=self.config['support_threshold'],
                             contradiction_threshold=self.config['contradiction_threshold'])
        review.update(applied=True,mode='fallible_veto_not_truth_certification',calibration_passed=False)
        if not review['passed']:
            return abstain('claim_not_verified','생성한 답변을 인용 근거와 충분히 대조하지 못해 답변을 보류합니다.'),review
        return value,review

    def ask(self, question, *, year, task, institution, semester=None, facts=None, base_url=None, claim_check=False):
        start=time.monotonic();row=dict(question=question,year=year,task=task,institution=institution,semester=semester)
        plan=self.plan(row,facts)
        response=None;verification=dict(applied=False,reason='source_bound_deterministic_output')
        if plan['preflight'] is not None:value=plan['preflight']
        else:
            response=self.generate(plan,base_url)
            value=self.parse(response,plan['evidence'])
            if claim_check:value,verification=self.postcheck(value,plan['evidence'])
            else:verification=dict(applied=False,reason='optional_neural_veto_disabled_after_comparison')
        return dict(value,pipeline='grounded',llm_called=response is not None and response['llm_called'],
                    elapsed_seconds=round(time.monotonic()-start,4),proof=plan['proof'],verification=verification,
                    generation=response,sources=[{k:d[k] for k in ['chunk_id','source_id','pdf_page','sha256','url']}
                                                for d in plan['evidence'] if d['chunk_id'] in value['citations']])
