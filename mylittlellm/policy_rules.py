"""Reviewed source-bound AST rules evaluated with Z3's three-valued entailment.

This is an application of Logic-LM/NL2Logic architecture, not an automatic
natural-language-to-logic reproduction. Facts are explicit user inputs; missing
facts stay symbolic. Unsatisfiable facts never imply an arbitrary answer.
"""
import json
import re
from pathlib import Path

from .retrieval import ROOT, read_jsonl


def compile_ast(node, variables, depth=0):
    import z3
    if depth > 20: raise ValueError('AST nesting exceeds limit')
    if type(node) is bool: return z3.BoolVal(node)
    if type(node) is int: return z3.IntVal(node)
    if not isinstance(node,dict) or len(node)!=1: raise ValueError('Invalid AST node')
    op, args = next(iter(node.items()))
    if op=='var':
        if args not in variables: raise ValueError('Unknown AST variable')
        return variables[args]
    if not isinstance(args,list): raise ValueError('AST arguments must be a list')
    values=[compile_ast(x,variables,depth+1) for x in args]
    unary={'not':z3.Not}
    binary={'eq':lambda a,b:a==b,'ge':lambda a,b:a>=b,'lt':lambda a,b:a<b,
            'mul':lambda a,b:a*b,'add':lambda a,b:a+b}
    if op in unary and len(values)==1:return unary[op](*values)
    if op in binary and len(values)==2:return binary[op](*values)
    if op=='ite' and len(values)==3:return z3.If(*values)
    if op=='and' and values:return z3.And(*values)
    if op=='or' and values:return z3.Or(*values)
    raise ValueError('Unsupported AST operation or arity')


def solve_rule(rule, facts):
    import z3
    solver=z3.Solver();solver.set(timeout=1000)
    variables={}
    for name,spec in rule['fields'].items():
        variables[name]=z3.Bool(name) if spec['type']=='bool' else z3.Int(name)
        if spec['type']=='int':solver.add(variables[name]>=spec['min'],variables[name]<=spec['max'])
        if name in facts:
            value=facts[name]
            expected=bool if spec['type']=='bool' else int
            if type(value) is not expected:raise ValueError('Wrong fact type: '+name)
            if expected is int and not spec['min']<=value<=spec['max']:raise ValueError('Fact out of range: '+name)
            solver.add(variables[name]==value)
    if solver.check()!=z3.sat:return dict(decision='unknown',reason='inconsistent_or_unresolved_facts')
    expression=compile_ast(rule['expression'],variables)
    solver.push();solver.add(z3.Not(expression));negative=solver.check();solver.pop()
    solver.push();solver.add(expression);positive=solver.check();solver.pop()
    decision='true' if negative==z3.unsat else 'false' if positive==z3.unsat else 'unknown'
    return dict(decision=decision,reason='symbolic_entailment' if decision!='unknown' else 'missing_facts',
                missing=[k for k in rule['fields'] if k not in facts],
                facts_used={k:facts[k] for k in rule['fields'] if k in facts})


class PolicyRules:
    def __init__(self, path=None, index=None):
        self.config=json.loads(Path(path or ROOT/'config/grounding_rules.json').read_text())
        self.rules={r['id']:r for r in self.config['rules']}
        self.index={r['chunk_id']:r for r in (index if index is not None else read_jsonl(ROOT/'outputs/preprocessed/index.jsonl'))}

    def documents(self, rule):
        result=[]
        for source in rule['sources']:
            doc=self.index.get(source['chunk_id'])
            if not doc or doc['sha256']!=source['sha256'] or any(s not in doc['text'] for s in source['anchors']):
                raise ValueError('Reviewed rule source changed or missing: '+source['chunk_id'])
            result.append(doc)
        return result

    def applicable(self, rule, row):
        return all(row.get(k)==v for k,v in rule['scope'].items())

    def evaluate(self, row, facts):
        if not isinstance(facts,dict):raise ValueError('Structured facts must be an object')
        rule=self.rules.get(facts.get('rule_id'))
        if rule is None:raise ValueError('Unknown rule_id')
        if not self.applicable(rule,row):raise ValueError('Rule and query scope differ')
        if any(facts.get(k)!=v for k,v in rule['scope'].items()):raise ValueError('Facts and rule scope differ')
        if not re.search(rule['question_pattern'],row['question']):raise ValueError('Question does not match selected rule topic')
        extras=set(facts)-set(rule['fields'])-set(rule['scope'])-{'rule_id'}
        if extras:raise ValueError('Unrecognized facts: '+', '.join(sorted(extras)))
        for k,v in rule.get('applicability',{}).items():
            if facts.get(k) is not v:raise ValueError('Rule applicability must be explicitly confirmed: '+k)
        docs=self.documents(rule)
        result=solve_rule(rule,facts)
        result.update(rule_id=rule['id'],label=rule['label'],sources=[d['chunk_id'] for d in docs],
                      explanation=rule['explanation'],review_status=self.config['review_status'])
        return result,docs
