"""Muon: momentum orthogonalised by a Newton-Schulz iteration.

Follows the reference implementation (github.com/KellerJordan/Muon, muon.py, read 2026-11-18):
same quintic coefficients, five iterations, Nesterov momentum 0.95 and the sqrt(rows/cols)
scale. The reference runs the iteration in bfloat16 on GPU; here it stays in float32 on CPU.

Instead of following the raw gradient, each weight matrix moves along the nearest
semi-orthogonal matrix to its momentum, so every direction of the update has about the same
size. It is meant for 2-D hidden matrices only; embeddings, the output head and scalars keep
using Adam.
"""
import torch


def orthogonalize(matrix, steps=5):
    """Approximate U V^T of the SVD of `matrix` without computing an SVD."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=True, steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                                      steps=steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Muon is defined for weight matrices only")
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                buffer, grad = state["momentum"], parameter.grad
                buffer.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = orthogonalize(update, group["steps"])
                update = update * max(1.0, parameter.size(-2) / parameter.size(-1)) ** 0.5
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
