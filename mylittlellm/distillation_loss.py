"""Full-vocabulary symmetric JSD with a detached privileged-context teacher."""
import math


def jsd_loss(student_logits, teacher_logits):
    import torch
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Expected matching [generated_tokens, vocabulary] logits")
    if not student_logits.shape[0]:
        raise ValueError("Empty rollout")
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    mixture_logp = torch.logaddexp(student_logp, teacher_logp) - math.log(2)
    # KL(P || M) + KL(Q || M). Do not substitute a KL on sampled tokens.
    loss = .5 * ((student_logp.exp() * (student_logp - mixture_logp)).sum(-1)
                 + (teacher_logp.exp() * (teacher_logp - mixture_logp)).sum(-1))
    return loss.mean()
