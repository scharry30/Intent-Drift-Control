"""KL projection of a finite candidate distribution."""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy
from .validation import number


def project(pi, vectors, weights, chi=0.5):
    pi = np.asarray(pi, dtype=float)
    v = np.asarray(vectors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        pi.ndim != 1
        or not len(pi)
        or v.shape != (len(pi), 3)
        or (weights.shape != (3,))
    ):
        raise ValueError("Invalid projection dimensions")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError("Projection needs finite positive support")
    if (
        abs(pi.sum() - 1) > 1e-12
        or np.any(weights <= 0)
        or abs(weights.sum() - 1) > 1e-12
    ):
        raise ValueError("Invalid mass or weights")
    number(chi)
    if chi >= 1:
        raise ValueError("chi must be strictly below one")
    anchor = int(np.argmax(v @ weights))
    target = (1 - chi) * (pi @ v) + chi * v[anchor]

    def dual(z):
        logweights = np.log(pi) + v @ z
        normalizer = logsumexp(logweights)
        q = np.exp(logweights - normalizer)
        return (float(normalizer - z @ target), v.T @ q - target)

    fit = minimize(
        dual,
        np.zeros(3),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] * 3,
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 5000, "maxls": 80},
    )
    z = fit.x
    normalizer = logsumexp(np.log(pi) + v @ z)
    q = np.exp(np.log(pi) + v @ z - normalizer)
    residual = q @ v - target
    kl = float(np.sum(xlogy(q, q / pi)))
    complementarity = float(np.max(np.abs(z * residual)))
    gap = kl - float(z @ target - normalizer)
    ok = (
        np.all(np.isfinite(q))
        and np.all(q > 0)
        and (abs(q.sum() - 1) <= 1e-12)
        and (residual.min() >= -1e-07)
        and (complementarity <= 1e-06)
        and (abs(gap) <= 1e-06)
    )
    result = {
        "status": "PASS_NUMERICAL_CONTRACT" if ok else "SOLVER_INVALID",
        "q": q.tolist(),
        "pi": pi.tolist(),
        "vectors": v.tolist(),
        "anchor": anchor,
        "target": target.tolist(),
        "moment": (q @ v).tolist(),
        "residual": residual.tolist(),
        "multipliers": z.tolist(),
        "kl": kl,
        "complementarity": complementarity,
        "primal_dual_gap": gap,
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "semantic_recovery_guarantee": False,
        "tolerances": {
            "primal": 1e-07,
            "normalization": 1e-12,
            "complementarity": 1e-06,
            "gap": 1e-06,
        },
    }
    if not ok:
        raise ValueError(
            "KL solver did not meet frozen residual tolerances: " + json.dumps(result)
        )
    return result
