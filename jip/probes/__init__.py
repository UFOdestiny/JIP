"""Trainable probes: JIP's probe, its imitation ablations, and SAPLMA (the second probe of JIP's evidence).

Every probe is sized to the largest width (a multiple of 8) whose trainable parameters stay within
``PARAM_BUDGET``. A probe key is also the name of its training run (``ckpt/<key>``, ``preds/<key>``);
``pipeline.train --tag`` appends a suffix.

Further probes (the supervised baselines of the paper) register themselves in ``PROBES`` and ``ARCHS`` from the
optional ``baselines`` package; ``spec`` and ``build`` import it on first use of an unknown key.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch.nn as nn

from jip.probes.jip_probe import JIPProbe
from jip.probes.saplma import SAPLMA

PARAM_BUDGET = 2_000_000
JIP_FEATURES = ("tok", "layers_anchor", "layers_concl_last")


@dataclass(frozen=True)
class ProbeSpec:
    arch: str
    kwargs: dict = field(default_factory=dict)
    features: Tuple[str, ...] = ("tok",)
    imitation: Optional[Tuple[float, float]] = None  # (step-label weight, verdict weight) of the imitation loss
    step_scores: bool = True  # whether the probe scores each reasoning step
    pool_size: int = 0  # unlabeled questions per OOD domain that join training (0: labeled source only)
    pool_mode: str = "imitation"  # pool targets: "imitation" (step labels and verdict), "steps", "verdict", or "pseudo"


IMITATION = (0.5, 0.5)  # lambda_r, lambda_v
PROBES: Dict[str, ProbeSpec] = {
    # JIP's probe: imitation on the labeled source and on unlabeled 6,000-question pools of the five OOD domains.
    "jip": ProbeSpec("jip", features=JIP_FEATURES, imitation=IMITATION, pool_size=6000),
    # Ablations: imitation on the labeled source only, smaller pools, one imitation target on the pools, no imitation.
    "jip-source": ProbeSpec("jip", features=JIP_FEATURES, imitation=IMITATION),
    "jip-pool2k": ProbeSpec("jip", features=JIP_FEATURES, imitation=IMITATION, pool_size=2000),
    "jip-pool-steps": ProbeSpec("jip", features=JIP_FEATURES, imitation=IMITATION, pool_size=6000, pool_mode="steps"),
    "jip-pool-verdict": ProbeSpec("jip", features=JIP_FEATURES, imitation=IMITATION, pool_size=6000, pool_mode="verdict"),
    "jip-no-imitation": ProbeSpec("jip", dict(imitation=False), JIP_FEATURES),
    # The last-token probe whose score joins JIP's evidence.
    "saplma": ProbeSpec("saplma", features=("layers_concl_last",), step_scores=False),
}

# arch -> constructor(dims, width, **kwargs)
ARCHS: Dict[str, Callable[..., nn.Module]] = {
    "jip": lambda dims, w, **kw: JIPProbe(dims["feature_dim"], dims["hidden_size"], dims["n_layers"], w, **kw),
    "saplma": lambda dims, w, **kw: SAPLMA(dims["hidden_size"], dims["saplma_layer"], w),
}


def _register_optional() -> None:
    try:
        importlib.import_module("baselines.probes")
    except ModuleNotFoundError as exc:
        if exc.name != "baselines":
            raise


def spec(method: str) -> ProbeSpec:
    if method not in PROBES:
        _register_optional()
    if method not in PROBES:
        raise KeyError(f"unknown probe {method!r}; available: {', '.join(PROBES)}")
    return PROBES[method]


def build(method: str, dims: dict, head_dim: int) -> nn.Module:
    s = spec(method)
    return ARCHS[s.arch](dims, head_dim, **s.kwargs)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def size_to_budget(method: str, dims: dict, budget: int = PARAM_BUDGET, lo: int = 32, hi: int = 1024, step: int = 8) -> int:
    """Largest width (a multiple of 8, so 8 attention heads divide it) whose probe stays within ``budget``."""
    best = lo
    for width in range(lo, hi + 1, step):
        if width % 8:
            continue
        if count_params(build(method, dims, width)) > budget:
            break
        best = width
    return best
