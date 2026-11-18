"""A small decoder-only transformer written from scratch (PyTorch, CPU).

Design choices that are deliberate, all of them cheap on a CPU:
- rotary position embedding: positions enter as rotations, i.e. orthogonal maps of the
  query/key vectors, so no position table has to be learned;
- pre-norm blocks with RMSNorm and normalised queries/keys (cosine attention), which keeps
  activations at unit scale without warm-up tricks;
- orthogonal initialisation of every matrix, zero initialisation of the two projections that
  write into the residual stream, so each block starts as the identity map;
- the operator-token embedding rows start orthonormal: one direction per rule.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class Config:
    def __init__(self, vocab_size, d_model=160, n_layer=4, n_head=5, context=256, init="orthogonal"):
        self.vocab_size, self.d_model, self.n_layer, self.n_head = vocab_size, d_model, n_layer, n_head
        self.context, self.init = context, init

    def to_dict(self):
        return dict(vars(self))


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotary(x, cos, sin):
    """Rotate consecutive coordinate pairs by a position-dependent angle (an orthogonal map)."""
    first, second = x[..., 0::2], x[..., 1::2]
    return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.head_dim = config.n_head, config.d_model // config.n_head
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        # Cosine attention needs a learned temperature; it starts sharp enough to attend.
        self.scale = nn.Parameter(torch.full((config.n_head, 1, 1), math.log(10.0)))

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = rotary(rms_norm(q), cos, sin), rotary(rms_norm(k), cos, sin)
        q = q * (self.scale.exp() / math.sqrt(self.head_dim))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.expand = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.project = nn.Linear(4 * config.d_model, config.d_model, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attention(rms_norm(x), cos, sin)
        return x + self.project(F.relu(self.expand(rms_norm(x))).square())


class SeedLM(nn.Module):
    def __init__(self, config, operator_slice=None):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        half = config.d_model // config.n_head // 2
        angle = torch.arange(config.context)[:, None] / (10000 ** (torch.arange(half) / half))[None, :]
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)
        self.reset(operator_slice)

    def reset(self, operator_slice=None):
        orthogonal = self.config.init == "orthogonal"
        for name, weight in self.named_parameters():
            if weight.ndim < 2:
                continue
            if orthogonal:
                nn.init.orthogonal_(weight)
            else:
                nn.init.normal_(weight, std=0.02)
            if name.endswith(("proj.weight", "project.weight")):
                nn.init.zeros_(weight)      # every block starts as the identity
        with torch.no_grad():
            self.embed.weight.mul_(0.5 if orthogonal else 1.0)
            if orthogonal and operator_slice is not None:
                # One orthonormal direction per rule (as many as the width allows).
                rows = self.embed.weight[operator_slice]
                nn.init.orthogonal_(rows)
                self.embed.weight[operator_slice] = rows

    def forward(self, tokens, labels=None):
        length = tokens.size(1)
        cos, sin = self.cos[:length], self.sin[:length]
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(rms_norm(x))
        if labels is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
