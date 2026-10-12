"""Strict frame inventory and assessment validation."""

import json
from .types import Frame, Goal, Assessment
from .validation import number, text
from ..utils.hashing import binding


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
