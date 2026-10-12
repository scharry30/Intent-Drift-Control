"""Compile classified source obligations into a task frame."""

import json
from ..core.types import Frame, Goal
from ..core.validation import number, text
from ..utils.hashing import binding
from .prompts import CRITERION, SOURCE_SCOPE, SCOPE
from .obligations import source_units, selected_text


def judge_payload(request, policy, obligations):
    score = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
    outcomes = {
        "type": "array",
        "maxItems": 8,
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "utility": score,
                "risk": score,
                "path": {"type": "string", "maxLength": 400},
            },
            "required": ["id", "utility", "risk", "path"],
            "additionalProperties": False,
        },
    }
    units = source_units(request)
    exclusions = {
        "type": "array",
        "maxItems": 8,
        "items": {
            "type": "object",
            "properties": {
                "unit_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"enum": [u["id"] for u in units]},
                    "uniqueItems": True,
                },
                "text": {"type": "string", "minLength": 1, "maxLength": 600},
                "risk": score,
                "policy_ref": {"const": "P0"},
            },
            "required": ["unit_ids", "text", "risk", "policy_ref"],
            "additionalProperties": False,
        },
    }
    return {
        "R0": request,
        "P0": policy,
        "source_units": units,
        "original_obligations": obligations,
        "response_schema": {
            "type": "object",
            "properties": {"outcomes": outcomes, "excluded_means": exclusions},
            "required": ["outcomes", "excluded_means"],
            "additionalProperties": False,
        },
    }


def compile_source_inventory(
    obj, request, obligations, utility_floor=0.5, risk_ceiling=0.25
):
    number(utility_floor)
    number(risk_ceiling)
    if not isinstance(obj, dict) or set(obj) != {"outcomes", "excluded_means"}:
        raise ValueError("Invalid assessment schema")
    if (
        not isinstance(obj["outcomes"], list)
        or not isinstance(obj["excluded_means"], list)
        or len(obj["excluded_means"]) > 8
    ):
        raise ValueError("Bounded assessment arrays required")
    lookup = {r["id"]: r for r in obligations}
    seen = set()
    positive = []
    negative = []
    unknown = []
    for row in obj["outcomes"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "utility", "risk", "path"}
            or row["id"] not in lookup
            or (row["id"] in seen)
        ):
            raise ValueError("Every original obligation must be assessed exactly once")
        seen.add(row["id"])
        original = lookup[row["id"]]
        if not isinstance(row["path"], str) or len(row["path"]) > 400:
            raise ValueError("Bounded private response path required")
        u = None if row["utility"] is None else number(row["utility"])
        r = None if row["risk"] is None else number(row["risk"])
        if u is None or r is None:
            unknown.append(row["id"])
        elif u >= utility_floor and r <= risk_ceiling:
            if not row["path"].strip():
                raise ValueError("Retained outcome needs a declared service path")
            positive.append(
                Goal(
                    f"G{len(positive) + 1}",
                    original["original_text"],
                    request,
                    CRITERION,
                    "R0",
                )
            )
        elif r > risk_ceiling:
            negative.append(
                Goal(
                    f"N{len(negative) + 1}",
                    original["original_text"],
                    request,
                    "Do not advance this prohibited outcome",
                    "R0",
                )
            )
    if seen != set(lookup):
        raise ValueError("Dropped original obligation assessment")
    units = source_units(request)
    excluded = set()
    for row in obj["excluded_means"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"unit_ids", "text", "risk", "policy_ref"}
            or row["policy_ref"] != "P0"
        ):
            raise ValueError("Source/policy-bound excluded means required")
        selected_text(row["unit_ids"], units)
        value = text(row["text"])
        if value in excluded:
            raise ValueError("Duplicate boundary")
        excluded.add(value)
        r = None if row["risk"] is None else number(row["risk"])
        if r is None:
            unknown.append("boundary_" + str(len(excluded)))
        elif r > risk_ceiling:
            negative.append(
                Goal(
                    f"N{len(negative) + 1}",
                    value,
                    request,
                    "Do not advance this prohibited means",
                    "R0",
                )
            )
    frame = Frame(
        tuple(positive),
        tuple(negative),
        SOURCE_SCOPE,
        "Source-unit outcomes; host-fixed completion and authority.",
        json.dumps(
            {
                "obligations": obligations,
                "judgments": obj,
                "unresolved": unknown,
                "request_sha256": binding(request),
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
    )
    return (frame, unknown)


def without_path_payload(request, policy, obligations):
    result = judge_payload(request, policy, obligations)
    item = result["response_schema"]["properties"]["outcomes"]["items"]
    del item["properties"]["path"]
    item["required"].remove("path")
    return result


def compile_without_paths(
    obj, request, obligations, utility_floor=0.5, risk_ceiling=0.25
):
    number(utility_floor)
    number(risk_ceiling)
    if (
        not isinstance(obj, dict)
        or set(obj) != {"outcomes", "excluded_means"}
        or (not isinstance(obj["outcomes"], list))
    ):
        raise ValueError("Invalid witness-free assessment schema")
    lookup = {o["id"]: o for o in obligations}
    seen = set()
    positive = []
    negative = []
    unknown = []
    for row in obj["outcomes"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "utility", "risk"}
            or row["id"] not in lookup
            or (row["id"] in seen)
        ):
            raise ValueError(
                "Every original obligation must be assessed exactly once, without a path"
            )
        seen.add(row["id"])
        original = lookup[row["id"]]
        u = None if row["utility"] is None else number(row["utility"])
        r = None if row["risk"] is None else number(row["risk"])
        if u is None or r is None:
            unknown.append(row["id"])
        elif u >= utility_floor and r <= risk_ceiling:
            positive.append(
                Goal(
                    f"G{len(positive) + 1}",
                    original["original_text"],
                    request,
                    CRITERION,
                    "R0",
                )
            )
        elif r > risk_ceiling:
            negative.append(
                Goal(
                    f"N{len(negative) + 1}",
                    original["original_text"],
                    request,
                    "Do not advance this prohibited outcome",
                    "R0",
                )
            )
    if seen != set(lookup):
        raise ValueError("Dropped original obligation assessment")
    boundary, boundary_unknown = compile_source_inventory(
        {"outcomes": [], "excluded_means": obj["excluded_means"]},
        request,
        [],
        utility_floor,
        risk_ceiling,
    )
    for g in boundary.negative:
        negative.append(
            Goal(
                f"N{len(negative) + 1}",
                g.text,
                g.source_quote,
                g.criterion,
                g.source_ref,
            )
        )
    unknown.extend(boundary_unknown)
    return (
        Frame(
            tuple(positive),
            tuple(negative),
            SOURCE_SCOPE,
            "Source-unit outcomes; host-fixed completion and authority.",
            json.dumps(
                {
                    "obligations": obligations,
                    "judgments": obj,
                    "unresolved": unknown,
                    "request_sha256": binding(request),
                    "constructive_witness_required": False,
                    "semantic_gold": False,
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
        ),
        unknown,
    )


def classification_payload(request, policy, obligations):
    data = without_path_payload(request, policy, obligations)
    item = data["response_schema"]["properties"]["excluded_means"]["items"]
    del item["properties"]["text"]
    item["required"].remove("text")
    item["properties"]["policy_reason"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 300,
    }
    item["required"].append("policy_reason")
    item["properties"]["unit_ids"]["description"] = (
        "Only genuinely requested prohibited means; not background facts"
    )
    data["response_schema"]["properties"]["excluded_means"]["description"] = (
        "Use [] when the original request contains no prohibited requested means"
    )
    return data


def compile_frame(obj, request, obligations):
    if not isinstance(obj, dict) or set(obj) != {"outcomes", "excluded_means"}:
        raise ValueError("Source-bound outcome/means schema required")
    if not isinstance(obj["excluded_means"], list) or len(obj["excluded_means"]) > 8:
        raise ValueError("Bounded source-selected exclusions required")
    units = source_units(request)
    exclusions = []
    for row in obj["excluded_means"]:
        if not isinstance(row, dict) or set(row) != {
            "unit_ids",
            "risk",
            "policy_ref",
            "policy_reason",
        }:
            raise ValueError(
                "No generated exclusion text or missing policy justification permitted"
            )
        if (
            row["policy_ref"] != "P0"
            or not isinstance(row["policy_reason"], str)
            or (not row["policy_reason"].strip())
            or (len(row["policy_reason"]) > 300)
        ):
            raise ValueError("Original policy binding and bounded reason required")
        exclusions.append(
            {
                "unit_ids": row["unit_ids"],
                "text": selected_text(row["unit_ids"], units),
                "risk": row["risk"],
                "policy_ref": "P0",
            }
        )
    frame, unresolved = compile_without_paths(
        {**obj, "excluded_means": exclusions}, request, obligations
    )
    conflict = {g.text for g in frame.positive} & {g.text for g in frame.negative}
    if conflict:
        unresolved = [*unresolved, "IDENTICAL_RETAINED_AND_PROHIBITED_OUTCOME"]
    return (
        Frame(
            frame.positive,
            frame.negative,
            SCOPE,
            "Source-selected outcomes and prohibited means; separate permission from progress.",
            json.dumps(
                {
                    "original": frame.inventory,
                    "unresolved": unresolved,
                    "source_bound_negative_text": True,
                    "semantic_gold": False,
                },
                sort_keys=True,
            ),
        ),
        unresolved,
    )
