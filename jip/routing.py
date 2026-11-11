"""Routing judge calls by expected information gain (paper Sec. "Routing Judge Calls").

For a response with judge-free evidence a, the probe's imitation q(l | h) predicts the judge's verdict level l, and
p(y | a, l) is the prediction of JIP + Judge with the judge output J replaced by the average judge output of the
validation responses at level l. With H the binary entropy and p_bar(y | a) = sum_l q(l | h) p(y | a, l),

    EIG(a) = H(p_bar(y | a)) - sum_l q(l | h) H(p(y | a, l)).

Calling the judge on the responses with the largest gain minimizes the expected log-loss for a given number of
calls. Responses with a call are scored by JIP + Judge, the others by JIP.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from jip.registry import VERDICT_EDGES

EPS = 1e-6


def binary_entropy(p) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def verdict_levels(v) -> np.ndarray:
    """Level index of the judge's verdict (missing verdicts count as 0.5)."""
    return np.digitize(np.nan_to_num(v, nan=0.5), VERDICT_EDGES)


def imitation_distribution(verdict_logq: np.ndarray) -> np.ndarray:
    """q(l | h) from the probe's predicted log-probabilities of the verdict levels (``verdict_logq``)."""
    q = np.exp(verdict_logq)
    return q / q.sum(1, keepdims=True)


def level_judge_outputs(judge_val: np.ndarray, levels_val: np.ndarray, n_levels: int, defaults: Sequence[float] = (0.5, 0.0)) -> list:
    """Average judge output (columns of ``judge_val``) of the validation responses at each verdict level."""
    out = []
    for j in range(n_levels):
        sel = levels_val == j
        out.append([np.nanmean(judge_val[sel, c]) if sel.any() else d for c, d in enumerate(defaults)])
    return out


def level_predictions(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, judge_cols: Sequence[int], level_outputs: list) -> np.ndarray:
    """[n, levels] predictions p(y | a, l) of JIP + Judge with its judge columns set to each level's average output."""
    pj = []
    for values in level_outputs:
        Xl = X.copy()
        for c, v in zip(judge_cols, values):
            Xl[:, c] = v
        pj.append(predict(Xl))
    return np.column_stack(pj)


def expected_information_gain(q: np.ndarray, p_levels: np.ndarray):
    """(EIG, p_bar) per response from q(l | h) [n, levels] and p(y | a, l) [n, levels]; EIG >= 0 by concavity of H."""
    p_bar = (q * p_levels).sum(1)
    return binary_entropy(p_bar) - (q * binary_entropy(p_levels)).sum(1), p_bar


def call_mask(order: np.ndarray, budget: float) -> np.ndarray:
    """Boolean mask of the responses that receive a judge call: the first ``budget`` share of ``order``
    (``np.argsort(-eig)`` for routing by information gain)."""
    k = int(round(budget * len(order)))
    m = np.zeros(len(order), bool)
    m[order[:k]] = True
    return m
