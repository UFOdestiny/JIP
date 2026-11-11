"""Benchmark loaders, prompts, and response parsing."""

from jip.data.babi import BAbI
from jip.data.hotpotqa import HotpotQA
from jip.data.math import GSM8K, MATH
from jip.data.musique import MuSiQue
from jip.data.strategyqa import StrategyQA

LOADERS = {cls.name: cls for cls in (HotpotQA, MuSiQue, StrategyQA, BAbI, GSM8K, MATH)}


def load_dataset(name: str, split: str, max_samples: int = 0, root=None):
    """One split of a prepared benchmark; ``root`` overrides ``JIP_DATASETS`` (e.g. the unlabeled pools)."""
    return LOADERS[name](split=split, max_samples=max_samples, root=root)
