"""Compile-time statistical calibration: fitted state for compiled artifacts.

Two pieces of standard, defensible machinery, fitted once at compile time
from teacher-labeled sample records and baked into the artifact as constants:

* **Temperature scaling** (Guo et al., 2017) — one scalar ``T`` fitted by
  minimizing negative log-likelihood, turning the decision model's raw
  probabilities into calibrated ones. Applied to probabilities as
  ``p_i^(1/T)`` renormalized (equivalent to scaling logits).
* **Split conformal prediction** — a quantile ``q`` of nonconformity scores
  (``1 - p_T(true label)``) at miscoverage level ``alpha``. At runtime the
  artifact forms the prediction set ``{label : p_T(label) >= 1 - q}`` and
  dispatches only when that set is a singleton; otherwise it falls back.
  This carries the standard split-conformal marginal guarantee: the true
  label lands in the prediction set with probability >= 1 - alpha (over
  exchangeable data), so a singleton dispatch is wrong with probability
  <= alpha, marginally.

Everything here is pure Python on top of probabilities the runtime already
produces — no torch, no new dependencies, honest small-sample caveats: with
``n`` calibration points the quantile granularity is ``1/(n+1)``, so tiny
samples give coarse guarantees. The compiler surfaces ``n`` in the artifact
metadata for exactly that reason.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

#: Search grid for the temperature scalar (log-spaced, inclusive bounds).
_T_MIN, _T_MAX, _T_STEPS = 0.25, 4.0, 60


def apply_temperature(probs: Sequence[float], temperature: float) -> List[float]:
    """Rescale a probability vector by ``p^(1/T)`` and renormalize."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    floor = 1e-12
    powered = [max(p, floor) ** (1.0 / temperature) for p in probs]
    total = sum(powered)
    return [p / total for p in powered]


def fit_temperature(
    distributions: Sequence[Sequence[float]], true_indices: Sequence[int]
) -> float:
    """Fit the NLL-minimizing temperature over a labeled calibration set."""
    if len(distributions) != len(true_indices) or not distributions:
        raise ValueError("need equal, non-empty distributions and labels")
    best_t, best_nll = 1.0, float("inf")
    for step in range(_T_STEPS + 1):
        t = _T_MIN * (_T_MAX / _T_MIN) ** (step / _T_STEPS)
        nll = 0.0
        for probs, true_index in zip(distributions, true_indices):
            calibrated = apply_temperature(probs, t)
            nll -= math.log(max(calibrated[true_index], 1e-12))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return round(best_t, 4)


def conformal_quantile(
    distributions: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    temperature: float,
    alpha: float,
) -> float:
    """Split-conformal quantile of nonconformity scores at level ``alpha``.

    Score per calibration point: ``1 - p_T(true label)``. Returns the
    ``ceil((n+1)(1-alpha))/n`` empirical quantile (capped at 1.0), the
    standard finite-sample-valid choice.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    scores = sorted(
        1.0 - apply_temperature(probs, temperature)[true_index]
        for probs, true_index in zip(distributions, true_indices)
    )
    n = len(scores)
    if n == 0:
        raise ValueError("need at least one calibration point")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0  # too few points for this alpha: everything conformal
    # Round UP at 1e-6: conservative (never shrinks the prediction set below
    # the fitted guarantee) and immune to float-boundary dispatch flakiness.
    return min(1.0, math.ceil(scores[rank - 1] * 1e6) / 1e6)


def prediction_set(
    probs: Sequence[float], temperature: float, quantile: float
) -> Tuple[List[int], List[float]]:
    """Indices whose calibrated probability clears the conformal bar."""
    calibrated = apply_temperature(probs, temperature)
    included = [i for i, p in enumerate(calibrated) if p >= 1.0 - quantile]
    return included, calibrated


def calibration_report(
    distributions: Sequence[Sequence[float]],
    true_indices: Sequence[int],
    temperature: float,
    quantile: float,
    alpha: float,
) -> Dict[str, object]:
    """Summary baked into artifact metadata: honest, inspectable numbers."""
    n = len(distributions)
    singleton = correct_singleton = 0
    for probs, true_index in zip(distributions, true_indices):
        included, _ = prediction_set(probs, temperature, quantile)
        if len(included) == 1:
            singleton += 1
            if included[0] == true_index:
                correct_singleton += 1
    return {
        "n_calibration": n,
        "alpha": alpha,
        "temperature": temperature,
        "conformal_quantile": quantile,
        "quantile_granularity": round(1.0 / (n + 1), 4),
        "dispatch_rate_on_calibration": round(singleton / n, 4) if n else 0.0,
        "dispatch_accuracy_on_calibration": (
            round(correct_singleton / singleton, 4) if singleton else None
        ),
    }
