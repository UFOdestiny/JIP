"""SAPLMA (Azaria & Mitchell, 2023): the second probe of JIP's evidence.

A feed-forward probe on one layer's hidden state at the last conclusion token, trained on the same labeled source
as JIP's probe; the layer is selected on validation AUROC (``pipeline.train.select_saplma_layer``).
"""

from __future__ import annotations

import torch.nn as nn


class SAPLMA(nn.Module):
    def __init__(self, hidden_size: int, layer: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer = layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, head_dim // 4),
            nn.ReLU(),
            nn.Linear(head_dim // 4, 1),
        )

    def forward(self, batch, span_key: str = "concl_mask"):
        return self.mlp(batch["layers_concl_last"][:, self.layer].float()).reshape(-1)
