"""Experimental concise and preprocessed RAG entry points."""
import json
import os
import re
import time

from .assistant import validate_answer
from .costs import housing_payment
from .preprocessing import token_count
from .rag_v2 import complete, request_body
from .retrieval import ROOT, read_jsonl, retrieve


def calculation_evidence(facts, row, retriever):
    if row['task']!='housing':raise ValueError('Housing facts require a housing question')
    if facts.get('year')!=row['year'] or facts.get('semester')!=row.get('semester'):
        raise ValueError('Calculation facts and question scope differ')
    if facts.get('month')==3 and facts.get('pledge_submitted') is not True:
        raise ValueError('First-month payment requires confirmed pledge submission')
    calculation=housing_payment(facts)
    rules=[(10,['지원 금액']),(10,['지원 범위'])]
    if facts['month']==3:rules.append((22,['서약서','정액']))
    docs=[]
    for page,terms in rules:
        matches=[r for r in retriever.rows if r['source_id']==calculation['source_id'] and r['pdf_page']==page
                 and r['kind']=='clause' and all(term in r['text'] for term in terms)]
        if not matches:raise ValueError('Full calculator rule evidence is missing')
        selected=min(matches,key=lambda r:len(r['text']))
        if selected not in docs:docs.append(dict(selected,retrieval_method='calculator_rule_lookup'))
    docs=list({(d.get('chunk_id'),d['pdf_page'],d['text']):d for d in docs}.values())
    return calculation,docs


def validate_calculated_amount(answer, calculation):
    from decimal import Decimal
    amounts={Decimal(number.replace(',','').replace('−','-'))*(10000 if unit.startswith('만') else 1)
             for number,unit in re.findall(r'([+\-−]?\d[\d,]*(?:\.\d+)?)\s*(만\s*원|원)',answer)}
    allowed={calculation['amount_krw'],calculation['monthly_cap_krw']}
    if calculation['amount_krw'] not in amounts or not amounts<=allowed:
        raise ValueError('Generated money amounts differ from the verified calculation')


def make_request(row, evidence, *, prompt_name='minimal', max_tokens=384, calculation=None):
    body=request_body(row,evidence,prompt_name=prompt_name,max_tokens=max_tokens)
    if calculation is not None:
        payload=json.loads(body['messages'][-1]['content'])
        payload['verified_calculation']=calculation
        body['messages'][-1]['content']=json.dumps(payload,ensure_ascii=False)
        body['messages'][0]['content']+='\nverified_calculation이 있으면 그 금액과 적용 조건을 그대로 설명하세요. 실제 지급 확정으로 표현하지 마세요.'
        if body['response_format']['type']=='json_schema':
            body['response_format']['json_schema']['schema']['properties']['citations']['maxItems']=min(4,len(evidence))
    return body


def ask_preprocessed(question, *, year, task='admissions', institution=None, semester=None,
                     retriever=None, pipeline='preprocessed', max_tokens=384, base_url=None,
                     admission_track=None, facts=None):
    if pipeline not in {'concise','preprocessed'}:raise ValueError('Unknown pipeline')
    if not isinstance(question,str) or not 0<len(question)<=2000:raise ValueError('Question must contain 1–2000 characters')
    if not 64<=max_tokens<=1024:raise ValueError('max_tokens must be in 64–1024')
    if task not in {'admissions','housing','scholarship'}:raise ValueError('Unknown task')
    if pipeline=='concise' and (facts is not None or admission_track is not None):
        raise ValueError('Use preprocessed for structured facts and explicit track filtering')
    row=dict(question=question,year=year,task=task,institution=institution,semester=semester)
    start=time.monotonic();calculation=None
    if pipeline=='concise':
        evidence=retrieve(read_jsonl(ROOT/'outputs/index.jsonl'),question,year=year,task=task,
                          institution=institution,semester=semester,k=2)
        metadata={'method':'legacy_bm25_k2','seconds':time.monotonic()-start}
    else:
        if retriever is None:raise ValueError('Provide an initialized PreprocessedRetriever')
        evidence,metadata=retriever.retrieve(question,year=year,task=task,institution=institution,
                                             semester=semester,admission_track=admission_track)
        if facts is not None:
            calculation,mandatory=calculation_evidence(facts,row,retriever)
            remaining=retriever.config['context_tokens']-sum(token_count(d['text'],retriever.tok) for d in mandatory)
            if remaining<0:raise ValueError('Calculator conditions exceed evidence budget')
            combined=mandatory.copy();ids={d['chunk_id'] for d in mandatory}
            for d in evidence:
                size=token_count(d['text'],retriever.tok)
                if d['chunk_id'] not in ids and size<=remaining and len(combined)<retriever.config['top_k']:
                    combined.append(d);ids.add(d['chunk_id']);remaining-=size
            evidence=combined
            metadata['context_tokens']=retriever.config['context_tokens']-remaining
    if not evidence:
        return dict(status='cannot_determine',answer='해당 대학·연도·학기의 근거를 찾지 못했습니다.',
                    citations=[],follow_up=['해당 범위의 공식 안내 자료를 확인해 주세요.'],sources=[],
                    pipeline=pipeline,llm_called=False,retrieval=metadata,elapsed_seconds=time.monotonic()-start)
    response=complete(make_request(row,evidence,max_tokens=max_tokens,calculation=calculation),
                      base_url or os.environ.get('QWEN_BASE_URL','http://127.0.0.1:8080/v1'))
    try:
        if response.get('error') or response.get('finish_reason')=='length':raise ValueError(response.get('error') or 'Output truncated')
        result=validate_answer(json.loads(response['content']),evidence,'rag')
        if calculation is not None and result['status']=='answer':validate_calculated_amount(result['answer'],calculation)
    except (ValueError,TypeError) as error:
        result=dict(status='invalid_response',answer='모델 출력 검증에 실패했습니다.',citations=[],follow_up=[],
                    validation_error=str(error),raw_response=response['content'])
    result.update(pipeline=pipeline,llm_called=True,retrieval=metadata,verified_calculation=calculation,
                  max_tokens=max_tokens,usage=response.get('usage'),generation_seconds=response['elapsed_seconds'],
                  elapsed_seconds=round(time.monotonic()-start,3),
                  sources=[{k:r[k] for k in ['chunk_id','source_id','pdf_page','sha256','url']} for r in evidence if r['chunk_id'] in result['citations']])
    return result
