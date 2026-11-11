"""JIP's probe F_theta (paper Sec. "Probe Architecture").

Two branches score the conclusion of a response from cached backbone features:

* token branch: a two-layer Transformer over the response tokens with a learned embedding of the
  conclusion mask (``TokenBranch``);
* layer branch: per-layer normalized hidden states at the position that predicts the first conclusion
  token and at the last conclusion token, mixed by learned softmax weights over layers.

A sigmoid gate fuses the two states into the correctness score s. During training, branch dropout replaces
the gate by 1 (token only) or 0 (layer only) for a random half of the responses, so the fusion cannot rely on
the branch that fits the source domain best.

With imitation, the probe also predicts the judge's outputs from the fused state: a categorical distribution
q(L | h) over the four verdict levels, and one score s_k per reasoning step (the token branch with the step
mask, read as a reasoning-type span). Both are trained by their negative log-likelihood (the imitation loss).
The judge outputs are needed only during training; unlabeled responses contribute the imitation loss alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from jip.probes.token_branch import CONCLUSION, REASONING, TokenBranch

VERDICT_LEVELS = 4  # levels of the judge's verbalized chain verdict (edges in jip.registry.VERDICT_EDGES)


class JIPProbe(nn.Module):
    verdict_levels = VERDICT_LEVELS
    exact_imitation = True  # unweighted, unsmoothed step-label likelihood in the imitation loss

    def __init__(self, feature_dim: int, hidden_size: int, n_layers: int, head_dim: int, imitation: bool = True, branch_dropout: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.branch_dropout, self.imitation = branch_dropout, imitation
        self.token = TokenBranch(feature_dim, head_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_logits = nn.Parameter(torch.zeros(2, n_layers + 1))
        self.layer_proj = nn.Sequential(nn.Linear(2 * hidden_size, head_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Linear(2 * head_dim, head_dim)
        self.out = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, 1))
        if imitation:
            self.verdict = nn.Linear(head_dim, VERDICT_LEVELS)

    def layer_state(self, batch):
        x = torch.stack([batch["layers_anchor"], batch["layers_concl_last"]], dim=1).float()  # [B, 2, L+1, d]
        w = torch.softmax(self.layer_logits, dim=-1)  # [2, L+1]
        mixed = (self.layer_norm(x) * w[None, :, :, None]).sum(2)  # [B, 2, d]
        return self.layer_proj(mixed.flatten(1))

    def token_state(self, batch, span_key: str = "concl_mask", claim_type: int = CONCLUSION):
        return self.token(batch["tok"], batch["attn"], batch[span_key], claim_type)

    def state(self, batch, span_key: str = "concl_mask"):
        tok_h, lay_h = self.token_state(batch, span_key), self.layer_state(batch)
        g = torch.sigmoid(self.gate(torch.cat([tok_h, lay_h], -1)))
        if self.training and self.branch_dropout > 0:
            r = torch.rand(g.shape[0], 1, device=g.device)
            g = torch.where(r < self.branch_dropout / 2, torch.ones_like(g), torch.where(r < self.branch_dropout, torch.zeros_like(g), g))
        return g * tok_h + (1 - g) * lay_h

    def forward(self, batch, span_key: str = "concl_mask"):
        """Logit of P(correct) for the response (span ``batch[span_key]`` is the conclusion)."""
        return self.out(self.state(batch, span_key)).reshape(-1)

    def forward_all(self, batch):
        """(answer logit, [B, levels] verdict-level logits); probes with imitation only."""
        h = self.state(batch)
        return self.out(h).reshape(-1), self.verdict(h)

    def step_logits(self, batch):
        """Score of the span in ``batch["concl_mask"]`` as a reasoning step.

        With imitation, the probe reads the token-branch state of the span marked as a reasoning-type claim; without
        it, the step span is scored exactly like a conclusion.
        """
        if self.imitation:
            return self.out(self.token_state(batch, claim_type=REASONING)).reshape(-1)
        return self.forward(batch)
