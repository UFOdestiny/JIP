"""Evaluation of JIP: evidence assembly per dataset cell, the calibrator, scoring, and aggregation over seeds.

Protocol: every calibrator and decision threshold is fit on the validation split
of the evaluated dataset and applied once to the evaluated split (``test`` or the
fresh held-out ``test2``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from jip import calibrator as C
from jip import chain_features as EV
from jip import metrics as M
from jip import registry
from jip.artifacts import index_by_uid, pred_path, read_jsonl, root
from jip.samples import load_judge_task, load_samples, verdict_value

FEATURE_KEYS = ["tok", "ll", "top1", "margin", "ent"]


def verdict_score(rec: dict) -> float:
    """Judge-Final score: the chain verdict, with P(true) as a tie-breaker (the verbalized probability is coarse)."""
    v = verdict_value(rec)
    return np.nan if v is None else v + 1e-3 * (rec.get("p_true") or 0.0)


def available_methods(backbone: str) -> Dict[str, List[int]]:
    """method -> seeds with predictions for ``backbone``."""
    out = {}
    preds = root() / "preds"
    for mdir in sorted(preds.iterdir()) if preds.exists() else []:
        seeds = sorted(int(p.name[4:]) for p in (mdir / backbone).glob("seed*") if p.is_dir() and p.name[4:].isdigit() and any(p.rglob("*.jsonl")))
        if seeds:
            out[mdir.name] = seeds
    return out


def load_preds(method: str, backbone: str, seed: int, dataset: str, split: str) -> Optional[dict]:
    path = pred_path(method, backbone, seed, dataset, split)
    return index_by_uid(read_jsonl(path)) if path.exists() else None


class Cell:
    """Validation and evaluated split of one (backbone, dataset), with cached evidence building blocks."""

    def __init__(self, backbone: str, dataset: str, split: str, hidden_dim: int, judge: str = registry.MAIN_JUDGE):
        self.backbone, self.dataset, self.split, self.hidden_dim = backbone, dataset, split, hidden_dim
        self.judge = judge  # judge whose chain verdict enters the evidence at inference
        self.val = load_samples(backbone, dataset, "validation", feature_keys=FEATURE_KEYS)
        self.test = load_samples(backbone, dataset, split, feature_keys=FEATURE_KEYS)
        self._static, self._cache = {}, {}

    def samples(self, split: str):
        if split == "validation":
            return self.val
        assert split == self.split, f"cell holds {self.split!r}, not {split!r}"
        return self.test

    def labels(self, split: str) -> np.ndarray:
        return np.asarray([s.y for s in self.samples(split)])

    def scores(self, split: str, preds: dict) -> np.ndarray:
        return np.asarray([preds[s.uid]["logit"] if s.uid in preds else np.nan for s in self.samples(split)], dtype=np.float64)

    def span_features(self, split: str) -> np.ndarray:
        if ("span", split) not in self._cache:
            self._cache["span", split] = EV.matrix(
                [EV.span_features(steps=s.steps, conclusion=s.conclusion, feat=s.feat) for s in self.samples(split)], EV.SPAN_FEATURE_NAMES
            )
        return self._cache["span", split]

    def judge_outputs(self, split: str) -> np.ndarray:
        """[chain verdict, log-odds of P(true)] of the answer-blind judge (one call), NaN where it failed."""
        if ("judge", split) not in self._cache:
            rec = load_judge_task(self.judge, "final_blind", self.backbone, self.dataset, split)
            v = np.asarray([verdict_score(rec.get(s.uid, {})) for s in self.samples(split)], dtype=np.float64)
            pt = np.asarray([rec.get(s.uid, {}).get("p_true") or np.nan for s in self.samples(split)], dtype=np.float64)
            pt = np.clip(np.nan_to_num(pt, nan=0.5), 1e-12, 1 - 1e-12)
            self._cache["judge", split] = np.column_stack([v, np.log(pt / (1 - pt))])
        return self._cache["judge", split]

    def _static_features(self, s):
        if s.uid not in self._static:
            self._static[s.uid] = EV.static_features(steps=s.steps, conclusion=s.conclusion, n_hops=s.gen["n_hops"], feat=s.feat, hidden_dim=self.hidden_dim)
        return self._static[s.uid]

    def u_matrix(self, split: str, preds: dict) -> np.ndarray:
        """u(r) of every response (probe step scores, chain structure, token statistics, judge step labels)."""
        rows = []
        for s in self.samples(split):
            pr = preds[s.uid]
            rows.append(
                EV.sample_features(
                    logit=pr["logit"],
                    step_logits=pr.get("step_logits"),
                    steps=s.steps,
                    conclusion=s.conclusion,
                    n_hops=s.gen["n_hops"],
                    feat=s.feat,
                    hidden_dim=self.hidden_dim,
                    step_labels=s.step_labels,
                    static=self._static_features(s),
                )
            )
        return EV.matrix(rows)

    def with_predicted_verdict(self, X: np.ndarray, split: str, preds: dict) -> np.ndarray:
        """Append the probe's imitation outputs (predicted verdict v_hat and q(L | h)) to judge-free evidence."""
        if not preds or "verdict_hat" not in next(iter(preds.values())):
            return X
        X = np.column_stack([X, [preds[s.uid].get("verdict_hat", np.nan) for s in self.samples(split)]])
        if "verdict_logq" in next(iter(preds.values())):  # categorical probes: the full imitation distribution q(L | h)
            X = np.column_stack([X, np.exp([preds[s.uid]["verdict_logq"] for s in self.samples(split)])])
        return X


def evidence_b(U, judge, extra):
    """Evidence [A; J] of JIP + Judge: probe score, chain structure and feature gap, the judge's verdict and its token
    log-odds of P(true) (one judge call), span token features, the imitation outputs and the second-probe score."""
    return np.column_stack([U[:, EV.EVIDENCE_COLUMNS], judge, extra])


def evidence_a(U, judge, extra):
    """Judge-free evidence A of JIP: the same without the judge."""
    return np.column_stack([U[:, EV.EVIDENCE_COLUMNS], extra])


EVIDENCE = {"B": evidence_b, "A": evidence_a}


class ProbeEvidence:
    """Probe scores and evidence matrices of one (probe, seed) on the validation and evaluated split of a cell."""

    def __init__(
        self,
        cell: Cell,
        preds_val: dict,
        preds_test: dict,
        use_predicted_verdict: bool = True,
        probes_val: tuple = (),
        probes_test: tuple = (),
    ):
        """``probes_*``: predictions of further internal-state probes (same seed) whose scores join the evidence."""
        sp = cell.split
        self.sv, self.st = cell.scores("validation", preds_val), cell.scores(sp, preds_test)
        self.Uv = cell.u_matrix("validation", preds_val)
        self.Ut = cell.u_matrix(sp, preds_test)
        self.Ev, self.Et = cell.span_features("validation"), cell.span_features(sp)
        n_span = self.Ev.shape[1]
        if use_predicted_verdict:
            self.Ev = cell.with_predicted_verdict(self.Ev, "validation", preds_val)
            self.Et = cell.with_predicted_verdict(self.Et, sp, preds_test)
        for pv, pt in zip(probes_val, probes_test):
            self.Ev = np.column_stack([self.Ev, cell.scores("validation", pv)])
            self.Et = np.column_stack([self.Et, cell.scores(sp, pt)])
        self.n_compact = self.Ev.shape[1] - n_span  # imitation outputs and second-probe scores (last columns of Ev)
        self.jv, self.jt = cell.judge_outputs("validation"), cell.judge_outputs(sp)

    def matrices(self, tier: str):
        build = EVIDENCE[tier]
        return build(self.Uv, self.jv, self.Ev), build(self.Ut, self.jt, self.Et)

    def compact_columns(self, tier: str):
        """Columns of the evidence used by the compact candidate of the calibrator: judge outputs (with the judge),
        the predicted verdict, and the second-probe scores."""
        n_u = len(EV.EVIDENCE_COLUMNS)
        width = self.matrices(tier)[0].shape[1]
        cols = list(range(width - self.n_compact, width))
        if tier == "B":
            cols = [n_u, n_u + 1] + cols
        return cols


def final_evidence(cell: "Cell", backbone: str, seed: int, probe: Optional[str] = None) -> Optional[ProbeEvidence]:
    """Evidence of JIP for one probe seed: the predictions of the probe run ``probe`` (default FINAL_PROBE) plus the
    scores of EXTRA_PROBES."""
    sp = cell.split
    head = [load_preds(probe or registry.FINAL_PROBE, backbone, seed, cell.dataset, s) for s in ("validation", sp)]
    probes = [[load_preds(p, backbone, seed, cell.dataset, s) for s in ("validation", sp)] for p in registry.EXTRA_PROBES]
    if any(x is None for x in head) or any(x is None for pr in probes for x in pr):
        return None
    return ProbeEvidence(cell, head[0], head[1], probes_val=tuple(p[0] for p in probes), probes_test=tuple(p[1] for p in probes))


def final_calibrator(ev: ProbeEvidence, tier: str):
    """Factory of JIP's calibrator for ``ev`` and ``tier`` ("A": JIP, "B": JIP + Judge)."""
    cols = ev.compact_columns(tier)
    return lambda: C.JIPCalibrator(cols)


def eval_scores(y_val, s_val, y_te, s_te, calibrated_te=None) -> dict:
    """Discrimination on the evaluated split (accuracy threshold from validation), plus calibration of
    ``calibrated_te`` if given."""
    ok_v, ok_t = np.isfinite(s_val), np.isfinite(s_te)
    out = M.discrimination(y_te[ok_t], s_te[ok_t], M.best_threshold(y_val[ok_v], s_val[ok_v]))
    out["n_test"] = int(ok_t.sum())
    if calibrated_te is not None:
        out.update(M.calibration(y_te[ok_t], calibrated_te[ok_t]))
    return out


def fit_eval(cls, y_val, s_val, X_val, y_te, s_te, X_te):
    """Fit a calibrator on validation evidence and score its held-out probabilities."""
    m = cls().fit(s_val, X_val, y_val)
    p_val, p_te = m.predict(s_val, X_val), m.predict(s_te, X_te)
    return eval_scores(y_val, p_val, y_te, p_te, p_te), m, p_te


def scalar_calibrated(name, y_val, s_val, s_te) -> np.ndarray:
    ok = np.isfinite(s_val)
    m = C.SCALAR[name]().fit(s_val[ok], y_val[ok])
    out = np.full_like(s_te, np.nan)
    good = np.isfinite(s_te)
    out[good] = m.predict(s_te[good])
    return out


def agg(rows: List[dict]) -> dict:
    """Mean/std over seeds for a list of metric dicts."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows])), "n_seeds": len(rows)} for k in keys}


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
