"""Task frames, alignment gates and finite-pool selection."""

from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy


def binding(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def number(value):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (not 0 <= value <= 1)
    ):
        raise ValueError("Expected a finite score in [0,1], not a boolean")
    return float(value)


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 3000:
        raise ValueError("Expected a bounded nonempty string")
    return value


@dataclass(frozen=True)
class Goal:
    id: str
    text: str
    source_quote: str
    criterion: str
    source_ref: str = ""


@dataclass(frozen=True)
class Frame:
    positive: tuple[Goal, ...]
    negative: tuple[Goal, ...]
    scope: tuple[str, ...]
    rationale: str
    inventory: str

    def wire(self):
        def goal_wire(g):
            row = asdict(g)
            row.pop("source_quote" if g.source_ref else "source_ref")
            return row

        return {
            "positive": [goal_wire(g) for g in self.positive],
            "negative": [goal_wire(g) for g in self.negative],
            "scope": list(self.scope),
            "rationale": self.rationale,
        }

    @property
    def sha256(self):
        return binding({"frame": self.wire(), "inventory": self.inventory})


def frame_from_inventory(
    obj, request, utility_floor=0.5, risk_ceiling=0.25, *, source_reference=False
):
    number(utility_floor)
    number(risk_ceiling)
    if not isinstance(obj, dict) or set(obj) != {
        "outcomes",
        "excluded_means",
        "scope",
        "rationale",
    }:
        raise ValueError("Invalid inventory fields")
    if not isinstance(obj["outcomes"], list) or not isinstance(
        obj["excluded_means"], list
    ):
        raise ValueError("Inventory fields must be arrays")
    if len(obj["outcomes"]) > 8 or len(obj["excluded_means"]) > 8:
        raise ValueError("Inventory exceeds eight entries per branch")
    positive, negative = ([], [])
    seen = set()
    for kind in ("outcomes", "excluded_means"):
        required = {
            "text",
            "source_ref" if source_reference else "source_quote",
            "risk",
        }
        if kind == "outcomes":
            required |= {"utility", "criterion"}
        for row in obj[kind]:
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("Invalid inventory item schema")
            value = text(row["text"])
            if source_reference:
                if row["source_ref"] != "R0":
                    raise ValueError("Unknown source evidence reference")
                quote = request
            else:
                quote = text(row["source_quote"])
                if quote not in request:
                    raise ValueError("Invented source quotation")
            if (kind, value) in seen:
                raise ValueError("Duplicate inventory item")
            seen.add((kind, value))
            risk = number(row["risk"])
            if kind == "outcomes":
                utility = number(row["utility"])
                criterion = text(row["criterion"])
                if utility >= utility_floor and risk <= risk_ceiling:
                    positive.append(
                        Goal(
                            f"G{len(positive) + 1}",
                            value,
                            quote,
                            criterion,
                            "R0" if source_reference else "",
                        )
                    )
            else:
                criterion = "Do not advance this prohibited means/outcome"
            if risk > risk_ceiling:
                negative.append(
                    Goal(
                        f"N{len(negative) + 1}",
                        value,
                        quote,
                        criterion,
                        "R0" if source_reference else "",
                    )
                )
    if not isinstance(obj["scope"], list) or len(obj["scope"]) > 8:
        raise ValueError("Scope must be a bounded array")
    scope = tuple((text(s) for s in obj["scope"]))
    if len(set(scope)) != len(scope) or (positive and (not scope)):
        raise ValueError("Retained goals need nonempty, nonduplicated scope")
    return Frame(
        tuple(positive),
        tuple(negative),
        scope,
        text(obj["rationale"]),
        json.dumps(
            {"inventory": obj, "request_sha256": binding(request)},
            sort_keys=True,
            ensure_ascii=False,
        ),
    )


@dataclass(frozen=True)
class Assessment:
    positive_lo: float
    positive_hi: float
    negative_lo: float
    negative_hi: float
    scope_lo: float
    per_goal_lo: tuple[float, ...]
    complete: tuple[bool | None, ...]
    terminal_ready: bool
    closure: bool
    binding_sha256: str
    evidence: str
    violations: tuple[str, ...]

    @property
    def vector(self):
        return (self.positive_lo, -self.negative_hi, self.scope_lo)


def parse_assessment(
    obj,
    frame,
    key,
    *,
    closure,
    terminal_ready,
    continuation,
    answer,
    evidence_reference=False,
    delivery=None,
):
    if not isinstance(obj, dict) or set(obj) != {
        "positive",
        "negative",
        "scope",
        "violations",
        "explanation",
    }:
        raise ValueError("Invalid assessor fields")

    def parse_branch(name, goals):
        rows = obj[name]
        if not isinstance(rows, list) or len(rows) != len(goals):
            raise ValueError("Assessor must cover every immutable goal exactly once")
        index = {g.id: g for g in goals}
        observed = {}
        for row in rows:
            required = {"id", "lo", "hi", "evidence"}
            if name == "positive":
                required.add("complete")
            if (
                not isinstance(row, dict)
                or set(row) != required
                or row["id"] not in index
                or (row["id"] in observed)
            ):
                raise ValueError("Unknown, duplicated or malformed goal assessment")
            lo, hi = (number(row["lo"]), number(row["hi"]))
            if lo > hi:
                raise ValueError("Reversed score interval")
            evidence = row["evidence"]
            source = (
                answer
                if closure and name == "positive"
                else delivery
                if closure and delivery is not None
                else continuation
            )
            expected_reference = (
                ("A0" if name == "positive" else "D0") if closure else "C0"
            )
            if (
                not isinstance(evidence, str)
                or len(evidence) > 1000
                or (
                    evidence
                    and (
                        evidence != expected_reference
                        if evidence_reference
                        else evidence not in source
                    )
                )
            ):
                raise ValueError("Assessment evidence is not in the specified view")
            if name == "positive":
                complete = row["complete"]
                if (
                    closure
                    and type(complete) is not bool
                    or (not closure and complete is not None)
                ):
                    raise ValueError("Preparation contribution is not task completion")
                if complete and (not evidence):
                    raise ValueError("Terminal completion needs answer evidence")
            observed[row["id"]] = row
        return [observed[g.id] for g in goals]

    plus = parse_branch("positive", frame.positive)
    minus = parse_branch("negative", frame.negative)
    scope = obj["scope"]
    if not isinstance(scope, dict) or set(scope) != {"lo", "hi", "evidence"}:
        raise ValueError("Invalid scope interval")
    if number(scope["lo"]) > number(scope["hi"]):
        raise ValueError("Reversed scope interval")
    if not isinstance(scope["evidence"], str) or (
        scope["evidence"]
        and (
            scope["evidence"] != ("D0" if closure else "C0")
            if evidence_reference
            else scope["evidence"]
            not in (delivery if closure and delivery is not None else continuation)
        )
    ):
        raise ValueError("Invented scope evidence")
    ids = {g.id for g in frame.positive + frame.negative}
    violations = obj["violations"]
    if not isinstance(violations, list) or any(
        (not isinstance(v, str) or v not in ids for v in violations)
    ):
        raise ValueError("Guidance can refer only to validated goal/boundary IDs")
    if not plus:
        raise ValueError("Empty-positive policy route has no positive assessment")
    return Assessment(
        sum((r["lo"] for r in plus)) / len(plus),
        sum((r["hi"] for r in plus)) / len(plus),
        max((r["lo"] for r in minus), default=0.0),
        max((r["hi"] for r in minus), default=0.0),
        number(scope["lo"]),
        tuple((r["lo"] for r in plus)),
        tuple((r["complete"] for r in plus)),
        bool(terminal_ready),
        bool(closure),
        binding(key),
        text(obj["explanation"]),
        tuple(violations),
    )


@dataclass(frozen=True)
class OperatingPoint:
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    positive_floor: float = 0.0
    negative_ceiling: float = 0.25
    scope_floor: float = 0.5
    favorable_threshold: float = 0.5
    terminal_floor: float = 0.75

    def __post_init__(self):
        for value in self.weights + (
            self.positive_floor,
            self.negative_ceiling,
            self.scope_floor,
            self.favorable_threshold,
            self.terminal_floor,
        ):
            number(value)
        if (
            len(self.weights) != 3
            or any((w <= 0 for w in self.weights))
            or abs(sum(self.weights) - 1) > 1e-12
        ):
            raise ValueError("Three positive normalized weights required")

    def verdict(self, assessment):
        if assessment is None:
            return {
                "accept": False,
                "reason": "ASSESSMENT_INVALID",
                "semantic_unknown": True,
            }
        a = assessment
        closure_ok = not a.closure or (
            a.terminal_ready
            and all(a.complete)
            and all((v >= self.terminal_floor for v in a.per_goal_lo))
        )
        margins = (
            a.positive_lo - self.positive_floor,
            self.negative_ceiling - a.negative_hi,
            a.scope_lo - self.scope_floor,
        )
        phi = float(np.dot(self.weights, a.vector))
        accepted = (
            closure_ok and min(margins) >= 0 and (phi >= self.favorable_threshold)
        )
        return {
            "accept": bool(accepted),
            "reason": "ACCEPT" if accepted else "REDIRECT",
            "margins": margins,
            "phi": phi,
            "closure_ok": closure_ok,
            "weighted_gate_redundant": self.favorable_threshold
            <= self.weights[0] * self.positive_floor
            - self.weights[1] * self.negative_ceiling
            + self.weights[2] * self.scope_floor,
        }


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
