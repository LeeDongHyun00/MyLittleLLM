"""Ask explicit Korean questions; never infer personal facts from prose."""
import re


def parse_answer(text, field):
    text=text.strip()
    if text in {'','모름','모르겠음','모르겠어요','모르겠습니다','?'}:return None
    if field['type']=='bool':
        if text.lower() in {'예','네','맞음','맞아요','y','yes','true'}:return True
        if text.lower() in {'아니요','아니오','아님','아니에요','n','no','false'}:return False
        raise ValueError('예, 아니요, 모름 중 하나로 답해 주세요.')
    if not re.fullmatch(r'[0-9]+',text):raise ValueError('단위 없이 정수를 입력하거나 모름으로 답해 주세요.')
    value=int(text)
    if not field['min']<=value<=field['max']:
        raise ValueError(f"{field['min']}~{field['max']} 범위로 입력해 주세요.")
    return value


def collect_rule_facts(assistant, row, rule_id, initial=None, input_fn=input, output_fn=print):
    rule=assistant.rules.rules.get(rule_id)
    if rule is None:raise ValueError('지원하지 않는 규칙입니다. --list-rules로 목록을 확인하세요.')
    if not assistant.rules.applicable(rule,row):raise ValueError('선택한 규칙의 대학·학년도·학기가 질문 범위와 다릅니다.')
    facts=dict(initial) if initial is not None else dict(rule['scope'],rule_id=rule_id)
    if facts.get('rule_id')!=rule_id:raise ValueError('입력 파일과 선택한 규칙이 다릅니다.')
    skipped=set()
    output_fn('확인된 조건만 입력해 주세요. 모르면 모름, 중단하려면 /quit을 입력하세요.')
    while True:
        plan=assistant.plan(row,facts)
        decision=plan['preflight']
        if not decision or decision.get('reason')!='missing_rule_facts':return facts
        questions=[q for q in decision['questions'] if q['field'] not in skipped]
        if not questions:return facts
        question=questions[0]
        suffix=' [예/아니요/모름]' if question['type']=='bool' else f" [{question['min']}~{question['max']} 정수/모름]"
        try:
            answer=input_fn(question['prompt']+suffix+' > ')
        except (EOFError,KeyboardInterrupt):return facts
        if answer.strip()=='/quit':return facts
        try:
            value=parse_answer(answer,question)
        except ValueError as error:
            output_fn(str(error));continue
        if value is None:skipped.add(question['field'])
        else:facts[question['field']]=value
