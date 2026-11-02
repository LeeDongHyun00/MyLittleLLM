import importlib.util
import json
import math
import unittest

from mylittlellm.compression import guarded_compress, protected_spans, compress_documents
from mylittlellm.distillation_data import encode_supervised, messages, reviewed


class DropEverything:
    def compress_prompt_llmlingua2(self, context, **kwargs):
        return {'compressed_prompt': ''}


class TokenizerFixture:
    eos_token_id = 999
    def apply_chat_template(self, messages, **kwargs):
        return [10, 20, 30]
    def encode(self, text, **kwargs):
        return list(range(100, 100+len(text)))


class DataTests(unittest.TestCase):
    def example(self):
        return {'id': 'fixture', 'question': '2026년 조건?', 'scope': {'year': 2026},
                'documents': [{'id': 'doc:1', 'year': 2026, 'institution': 'fixture',
                               'pdf_page': 1, 'text': '연간 2명 이내'}],
                'answer': '2명', 'explanation': '원문의 상한을 적용', 'oracle_ids': ['doc:1']}

    def test_privileged_solution_never_in_student_context(self):
        example = self.example()
        student = json.loads(messages(example)[1]['content'])
        teacher = json.loads(messages(example, privileged=True)[1]['content'])
        self.assertNotIn('reference_solution', student)
        self.assertNotIn('oracle_ids', student)
        self.assertEqual(teacher['reference_solution']['answer'], example['answer'])

    def test_prompt_is_masked_and_completion_including_eos_is_supervised(self):
        batch = encode_supervised(TokenizerFixture(), self.example(), 'answer', 1000)
        self.assertEqual(batch['labels'][:3], [-100]*3)
        self.assertEqual(batch['input_ids'][3:], batch['labels'][3:])
        self.assertEqual(batch['labels'][-1], 999)

    def test_overlength_rejected_instead_of_cutting_conditions_or_answers(self):
        with self.assertRaises(ValueError):
            encode_supervised(TokenizerFixture(), self.example(), 'answer', 4)

    def test_approved_flag_without_reviewer_is_not_gold(self):
        self.assertFalse(reviewed({'review_status': 'approved'}))
        self.assertTrue(reviewed({'review_status': 'approved', 'reviewer_ids': ['human-reviewer']}))

    def test_compression_preserves_years_numbers_negations_exceptions(self):
        text = ('2026학년도 지원 안내. ※ 추천 인원 제한 없음. '
                '※ 단, 학교폭력 제4호 이상은 추천 불가. '
                '※ 2024년 4월 1일 기준 5% 이내이며 소수점 버림. '
                '※ 문화와 예술을 사랑하고 다양한 토론을 통해 새로운 생각을 나누는 풍부한 인재를 기다립니다.')
        result = guarded_compress(text, DropEverything())
        self.assertTrue(all(span in result['text'] for span in protected_spans(text)))
        self.assertIn('소수점 버림', result['text'])
        self.assertNotIn('풍부한 인재', result['text'])

    def test_empty_compression_falls_back(self):
        text = '문화와 예술을 사랑하고 다양한 토론을 통해 새로운 생각을 나누는 풍부한 인재를 기다립니다.'
        result = guarded_compress(text, DropEverything())
        self.assertTrue(result['fallback_to_original'])
        self.assertEqual(result['text'], text)

    def test_runtime_compression_keeps_source_identity_and_input_immutable(self):
        doc = {'chunk_id': '2026:p1:c0', 'sha256': 'frozen-source', 'year': 2026,
               'text': '2026학년도 제4호 이상은 추천 불가.'}
        original = dict(doc)
        result, report = compress_documents([doc], compressor=DropEverything(), tokenizer=TokenizerFixture())
        self.assertEqual(doc, original)
        self.assertEqual(result[0], original)
        self.assertEqual(report['documents'][0]['protected_spans_retained'], 1)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'training environment required')
class LossTests(unittest.TestCase):
    def test_jsd_is_zero_for_identical_distributions(self):
        import torch
        from mylittlellm.distillation_loss import jsd_loss
        values = torch.tensor([[0., 2., -1.], [4., -5., 1.]])
        self.assertAlmostEqual(jsd_loss(values, values).item(), 0., places=6)

    def test_teacher_detached_and_student_gradient_reduces_loss(self):
        import torch
        from mylittlellm.distillation_loss import jsd_loss
        student = torch.tensor([[1., -1., .5]], requires_grad=True)
        teacher = torch.tensor([[-1., 2., 0.]], requires_grad=True)
        loss = jsd_loss(student, teacher)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.norm().item(), 0.)
        self.assertGreater(loss.item(), 0.)
        self.assertLessEqual(loss.item(), math.log(2))
        self.assertLess(jsd_loss(student-student.grad*.2, teacher).item(), loss.item())

    def test_shape_mismatch_and_empty_rollout_rejected(self):
        import torch
        from mylittlellm.distillation_loss import jsd_loss
        with self.assertRaises(ValueError):
            jsd_loss(torch.zeros(2, 4), torch.zeros(3, 4))
        with self.assertRaises(ValueError):
            jsd_loss(torch.zeros(0, 4), torch.zeros(0, 4))


if __name__ == '__main__':
    unittest.main()
