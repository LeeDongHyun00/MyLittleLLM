import json
import unittest
from mylittlellm.evaluation import prompt,score,paired_comparison,cluster_interval


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        row={'id':'fixture','question':'조건은?','year':2026,'institution':'대학','task':'admissions',
             'choices':{'A':'2명','B':'3명','C':'4명','D':'5명'},
             'oracle_documents':[{'id':'doc','year':2026,'institution':'대학','text':'2명'}],
             'retrieved_documents':[{'chunk_id':'doc:p1:c0','source_id':'doc','pdf_page':1,'year':2026,'text':'2명'}]}
        gold={'correct':'A','expected_action':'answer','year':2026,'required':['2명'],'forbidden':[],
              'evidence':[{'source_id':'doc','pdf_page':1,'year':2026}]}
        return row,gold

    def test_free_input_has_no_choices_or_gold(self):
        row,_=self.fixture(); content=json.loads(prompt(row,'free')[1]['content'])
        self.assertNotIn('choices',content)
        self.assertNotIn('correct',content)
        self.assertNotIn('reference_solution',content)
        self.assertEqual(content['evidence'][0]['text'],'2명')

    def test_mcq_does_not_accept_explanations_or_empty_output(self):
        row,gold=self.fixture()
        self.assertTrue(score({'mode':'mcq','content':'A'},gold,row)['correct'])
        for content in ['','A 또는 B','정답은 A','AB']:
            self.assertFalse(score({'mode':'mcq','content':content},gold,row)['correct'])

    def test_truncated_correct_prefix_and_invented_citation_fail(self):
        row,gold=self.fixture()
        value={'status':'answer','answer':'2명입니다.','citations':['doc:p1:c0'],'follow_up':[]}
        response={'mode':'free','content':json.dumps(value),'finish_reason':'stop'}
        self.assertTrue(score(response,gold,row)['strict_pass'])
        self.assertFalse(score(dict(response,finish_reason='length'),gold,row)['strict_pass'])
        value['citations']=['fake'];response['content']=json.dumps(value)
        self.assertFalse(score(response,gold,row)['strict_pass'])

    def test_wrong_year_citation_does_not_count_as_evidence(self):
        row,gold=self.fixture();gold['evidence'][0]['year']=2025
        response={'mode':'free','finish_reason':'stop','content':json.dumps({'status':'answer','answer':'2명','citations':['doc:p1:c0'],'follow_up':[]})}
        self.assertFalse(score(response,gold,row)['strict_pass'])

    def test_paired_comparison_counts_improvements_and_regressions(self):
        result=paired_comparison({'a':False,'b':True,'c':True},{'a':True,'b':False,'c':True})
        self.assertEqual((result['improved'],result['regressed'],result['unchanged']),(1,1,1))
        with self.assertRaises(ValueError):paired_comparison({'a':True},{'b':True})

    def test_grouped_interval_is_reproducible_and_bounded(self):
        rows=[{'group':'a','correct':True},{'group':'a','correct':False},{'group':'b','correct':True}]
        self.assertEqual(cluster_interval(rows,draws=100),cluster_interval(rows,draws=100))
        low,high=cluster_interval(rows,draws=100)
        self.assertTrue(0<=low<=high<=1)


if __name__=='__main__':unittest.main()
