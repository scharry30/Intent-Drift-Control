"""Multiplicity-aware selection of admissible repair candidates."""

from collections import Counter
import numpy as np
from .projection import project


def select_pool(
    draws, assessments, rejected, operating, uniform, chi=0.5, law="projection"
):
    if not draws:
        raise ValueError("An actual nonempty repair pool is required")
    if not 0 <= uniform < 1:
        raise ValueError("Uniform selector draw must be in [0,1)")
    counts = Counter((tuple(ids) for ids in draws))
    admissible = [
        ids
        for ids in sorted(counts)
        if operating.verdict(assessments.get(ids))["accept"]
    ]
    strict = []
    if rejected is not None:
        strict = [
            ids
            for ids in admissible
            if assessments[ids].closure == rejected.closure
            and assessments[ids].positive_lo - rejected.positive_hi > 0
            and (rejected.negative_lo - assessments[ids].negative_hi > 0)
        ]
    if strict:
        support, route = (strict, "STRICT_DIRECTIONAL")
    else:
        closures = [ids for ids in admissible if assessments[ids].closure]
        support, route = (
            (closures, "ADMISSIBLE_CLOSURE_FALLBACK")
            if closures
            else (
                [ids for ids in admissible if not assessments[ids].closure],
                "ADMISSIBLE_CONTINUATION_FALLBACK",
            )
        )
    if not support:
        return {
            "selected_ids": None,
            "route": "EMPTY_SUPPORT",
            "draw_count": len(draws),
            "distinct_count": len(counts),
            "strict_certificate": False,
        }
    mass = sum((counts[ids] for ids in support))
    pi = [counts[ids] / mass for ids in support]
    vectors = [assessments[ids].vector for ids in support]
    projection = project(pi, vectors, operating.weights, chi)
    if law == "projection":
        probabilities = projection["q"]
    elif law == "empirical":
        probabilities = pi
    elif law == "best":
        probabilities = [float(i == projection["anchor"]) for i in range(len(support))]
    elif law == "mixture":
        probabilities = [
            (1 - chi) * p + chi * (i == projection["anchor"]) for i, p in enumerate(pi)
        ]
    else:
        raise ValueError("Unknown selector law")
    selected = min(
        int(np.searchsorted(np.cumsum(probabilities), uniform, side="right")),
        len(support) - 1,
    )
    return {
        "selected_ids": list(support[selected]),
        "route": route,
        "draw_count": len(draws),
        "distinct_count": len(counts),
        "support": [list(ids) for ids in support],
        "support_counts": [counts[ids] for ids in support],
        "mask_mass": mass / len(draws),
        "probabilities": probabilities,
        "uniform": uniform,
        "law": law,
        "strict_certificate": bool(strict),
        "projection": projection,
    }
