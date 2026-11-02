import json
import unittest
from unittest.mock import Mock

import numpy as np

from mylittlellm.evaluation_v2 import score, supported
from mylittlellm.hierarchical import HierarchicalRetriever, scope_matches, token_windows
from mylittlellm.rag_v2 import request_body, schema
from mylittlellm.retrieval import ROOT, read_jsonl


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        gold = {'year':2026,'task':'housing','institution':'재단','semester':1,
                'expected_status':'answer','required_patterns':['5년'],
                'support_patterns':['지급일로부터5년간보관'],'gold_page':22}
        doc = {'chunk_id':'source:p3:h0','source_id':'source','pdf_page':3,'year':2026,
               'task':'housing','institution':'재단','semester':1,'text':'증빙을 지급일로부터 5년간 보관합니다.'}
        response = {'finish_reason':'stop','content':json.dumps({'status':'answer','answer':'지급일로부터 5년입니다.',
                    'citations':[doc['chunk_id']],'follow_up':[]})}
        return gold,doc,response

    def test_same_clause_on_another_page_is_accepted(self):
        gold,doc,response = self.fixture()
        self.assertTrue(score(response,gold,[doc])['strict_pass'])

    def test_correct_page_without_actual_clause_does_not_pass(self):
        gold,doc,response = self.fixture()
        doc.update(pdf_page=22,text='지원 대상자를 선발합니다.')
        result = score(response,gold,[doc])
        self.assertTrue(result['facts_pass'])
        self.assertFalse(result['strict_pass'])

    def test_wrong_scope_or_example_citation_and_truncation_fail(self):
        gold,doc,response = self.fixture()
        for key,value in [('year',2025),('semester',2),('institution','다른재단')]:
            self.assertFalse(score(response,gold,[dict(doc,**{key:value})])['strict_pass'])
        self.assertFalse(score(dict(response,finish_reason='length'),gold,[doc])['strict_pass'])
        value = json.loads(response['content']);value['citations']=['example:A02']
        self.assertFalse(score(dict(response,content=json.dumps(value)),gold,[doc])['strict_pass'])

    def test_reference_answer_never_enters_current_request(self):
        gold,doc,_ = self.fixture()
        row = dict(gold,question='증빙 보관기간은?',gold_answer='SECRET_REFERENCE')
        example = {'question':'예전 질문','scope':{},'evidence':[],
                   'response':{'status':'cannot_determine','answer':'확인 필요','citations':[],'follow_up':[]}}
        body = request_body(row,[doc],example=example)
        self.assertNotIn('SECRET_REFERENCE',json.dumps(body))
        self.assertEqual(schema([doc])['properties']['citations']['items']['enum'],[doc['chunk_id']])
        self.assertTrue(json.loads(body['messages'][1]['content'])['example_only'])
        self.assertNotIn('example_only',json.loads(body['messages'][-1]['content']))

    def test_new_rule_groups_do_not_overlap_old_questions_or_examples(self):
        gold = read_jsonl(ROOT/'data/evaluation/rag_v2_gold.jsonl')
        old = read_jsonl(ROOT/'data/evaluation/holdout_v1.jsonl')
        examples = read_jsonl(ROOT/'data/examples/rag_triplets.jsonl')
        groups = {r['leakage_group'] for r in gold}
        self.assertFalse(groups & {r['leakage_group'] for r in old})
        self.assertFalse(groups & {r['leakage_group'] for r in examples})
        self.assertFalse({r['question'] for r in gold} & {r['question'] for r in examples})
        self.assertEqual((len(gold),len(groups)),(16,15))


class HierarchyTests(unittest.TestCase):
    def test_unicode_windows_preserve_split_multitoken_characters(self):
        class MultiTokenTokenizer:
            def encode(self,text,**kwargs):return [0 for ch in text for _ in range(2 if ord(ch)>127 else 1)]
            def __call__(self,text,**kwargs):
                return {'offset_mapping':[(i,i+1) for i,ch in enumerate(text) for _ in range(2 if ord(ch)>127 else 1)]}
        text = '가나다ABCDE라마바'
        tokenizer = MultiTokenTokenizer()
        windows = list(token_windows(text,tokenizer,5,2))
        covered = set()
        for part,start,end in windows:
            self.assertEqual(part,text[start:end])
            self.assertNotIn('\ufffd',part)
            self.assertLessEqual(len(tokenizer.encode(part)),5)
            covered.update(range(start,end))
        self.assertEqual(covered,set(range(len(text))))

    def retriever(self):
        retriever = HierarchicalRetriever.__new__(HierarchicalRetriever)
        retriever.config = {'rrf_constant':60,'rerank_candidates':8,'context_tokens':20,'top_k':2}
        retriever.models = Mock()
        retriever.models.encode.return_value = np.array([[1.,0.]])
        retriever.models.tokenizer.encode.side_effect = lambda text,**kwargs:list(text)
        retriever.models.rerank.side_effect = lambda question,docs:[1.-i*.1 for i in range(len(docs))]
        meta = {'year':2026,'task':'admissions','institution':'대학','semester':None}
        retriever.parents = {'parent1':dict(meta,chunk_id='parent1',text='등록은 온라인으로'),
                             'parent2':dict(meta,chunk_id='parent2',text='등록 증빙 보관')}
        retriever.leaves = [dict(meta,chunk_id='a',parent_id='parent1',text='등록 온라인'),
                            dict(meta,chunk_id='b',parent_id='parent1',text='등록'),
                            dict(meta,chunk_id='c',parent_id='parent2',text='등록 증빙'),
                            dict(meta,chunk_id='wrong',parent_id='notpresent',year=2025,text='등록')]
        retriever.vectors = np.array([[1.,0.],[.9,.1],[.8,.2],[10.,0.]])
        return retriever

    def test_siblings_merge_and_other_year_is_excluded(self):
        retriever = self.retriever()
        docs,log = retriever.retrieve('등록',year=2026,task='admissions',institution='대학')
        self.assertEqual([d['chunk_id'] for d in docs],['parent1','parent2'])
        self.assertEqual(docs[0]['retrieval_method'],'auto_merge_parent')
        self.assertLessEqual(log['context_tokens'],20)
        self.assertNotIn('wrong',[r['chunk_id'] for r in log['candidates']])

    def test_parent_falls_back_to_leaf_within_budget(self):
        retriever = self.retriever();retriever.config['context_tokens']=6
        docs,log = retriever.retrieve('등록',year=2026,task='admissions',institution='대학')
        self.assertTrue(docs)
        self.assertEqual(docs[0]['retrieval_method'],'leaf_budget_fallback')
        self.assertLessEqual(log['context_tokens'],6)

    def test_missing_scope_errors_and_unavailable_scope_avoids_model(self):
        retriever = self.retriever()
        with self.assertRaises(ValueError):retriever.retrieve('등록',year=2026,task='admissions')
        with self.assertRaises(ValueError):retriever.retrieve('등록',year=2026,task='housing')
        docs,_ = retriever.retrieve('등록',year=2030,task='admissions',institution='대학')
        self.assertFalse(docs)
        retriever.models.encode.assert_not_called()

    def test_invalid_reranker_scores_do_not_silently_reorder_evidence(self):
        retriever = self.retriever()
        retriever.models.rerank.side_effect = lambda question,docs:[float('nan') for _ in docs]
        with self.assertRaises(ValueError):
            retriever.retrieve('등록',year=2026,task='admissions',institution='대학')


if __name__=='__main__':unittest.main()
