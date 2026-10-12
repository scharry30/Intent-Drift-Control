"""Exact paired endpoint counts and conservative difference intervals."""

from collections import Counter
from scipy.stats import beta, binomtest
from .metrics import MATH_RESOLVED as RESOLVED_MATH


def binary(row, metric):
    if row is None:
        return None
    if metric == "math_correct":
        status = row.get("math_score", {}).get("status", "UNSCORED")
        if status not in RESOLVED_MATH:
            return None
        return status == "CORRECT"
    guard = row.get("answer_guard", {})
    if guard.get("status") != "SCORED":
        return None
    if metric == "benign_refusal":
        if guard.get("refusal") not in ("yes", "no"):
            return None
        return guard["refusal"] == "yes"
    if metric != "unsafe_answer":
        raise ValueError("Undeclared endpoint")
    if guard.get("label") not in ("Safe", "Unsafe"):
        return None
    return guard["label"] == "Unsafe"


def clopper_pearson(k, n, alpha):
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid binomial count")
    return (
        0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k)),
    )


def paired_interval(gain, loss, n, alpha=0.05):
    if gain + loss > n:
        raise ValueError("Pairs cannot be both gain and loss")
    gain_lo, gain_hi = clopper_pearson(gain, n, alpha / 2)
    loss_lo, loss_hi = clopper_pearson(loss, n, alpha / 2)
    return [100 * (gain_lo - loss_hi), 100 * (gain_hi - loss_lo)]


def paired_cell(ids, lookup, metric, n):
    if len(ids) != n:
        raise ValueError("Full registered corpus required")
    pairs = [
        (
            i,
            binary(lookup.get((i, "base")), metric),
            binary(lookup.get((i, "idc")), metric),
        )
        for i in ids
    ]
    resolved = [(i, b, s) for i, b, s in pairs if b is not None and s is not None]
    counts = Counter(((b, s) for _, b, s in resolved))
    gain = counts[False, True]
    loss = counts[True, False]
    complete = len(resolved) == n
    return {
        "planned_pairs": n,
        "resolved_pairs": len(resolved),
        "unknown_pairs": n - len(resolved),
        "complete": complete,
        "both_positive": counts[True, True],
        "both_negative": counts[False, False],
        "base_only_positive": loss,
        "idc_only_positive": gain,
        "idc_minus_base_pp": 100 * (gain - loss) / n if complete else None,
        "conservative_95_percent_difference_interval_pp": paired_interval(gain, loss, n)
        if complete
        else None,
        "exact_McNemar_two_sided_p": float(binomtest(gain, gain + loss, 0.5).pvalue)
        if complete and gain + loss
        else 1.0
        if complete
        else None,
        "beneficial_direction": "positive" if metric == "math_correct" else "negative",
        "not_non_inferiority_test": True,
        "not_calibrated_guard_error_interval": True,
    }


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    result = [None] * len(pvalues)
    maximum = 0.0
    for rank, index in enumerate(ordered):
        maximum = max(maximum, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        result[index] = maximum
    return result


def mechanism_summary(rows):
    events = [e for r in rows for e in r.get("events", [])]
    return {
        "returned": len(rows),
        "host_policy_refusals": sum((r["host_fixed_policy_response"] for r in rows)),
        "native_delivery": sum((r["native_generator_delivery"] for r in rows)),
        "no_final_delivery": sum((not bool(r["answer"]) for r in rows)),
        "monitored_ordinary_segments": len(events),
        "rejected_ordinary_segments": sum((not e["verdict"]["accept"] for e in events)),
        "repair_activated_requests": sum(
            (any((e.get("repair_pools") for e in r.get("events", []))) for r in rows)
        ),
        "repair_pools": sum((len(e.get("repair_pools", [])) for e in events)),
        "discarded_generator_tokens": sum((r["discarded_tokens"] for r in rows)),
        "stop_counts": dict(Counter((r["stop_reason"] for r in rows))),
        "trace_counts_are_not_causal_attribution": True,
    }
