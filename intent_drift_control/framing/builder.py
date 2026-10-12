"""Request-level triage followed by source-bound frame construction."""

import json
from ..core.types import Frame, Goal
from ..utils.hashing import binding
from ..utils.random import seed_ticket
from ..config.defaults import FRAME_SEED
from .policy import POLICY
from .prompts import CRITERION, SCOPE, EXTRACTION_ARMS, EXTRACT, JUDGE, TRIAGE
from .obligations import extract_payload, parse_obligations
from .compiler import classification_payload, compile_frame


def extraction_payload(request):
    data = extract_payload(request, POLICY, EXTRACTION_ARMS[1])
    data["response_schema"]["properties"]["outcomes"]["uniqueItems"] = True
    return data


def build_source_frame(engine, request):
    obj, extraction = engine.json_call(
        EXTRACT,
        extraction_payload(request),
        phase="unique_source_obligation_extract",
        budget=2048,
        seed=seed_ticket(FRAME_SEED, request, "source_obligation_extract"),
    )
    obligations = parse_obligations(obj, request)
    obj, judgment = engine.json_call(
        JUDGE,
        classification_payload(request, POLICY, obligations),
        phase="source_bound_obligation_judge",
        budget=2048,
        seed=seed_ticket(FRAME_SEED, request, "source_obligation_judge_without_path"),
    )
    frame, unresolved = compile_frame(obj, request, obligations)
    if unresolved:
        raise ValueError("Unresolved source-bound frame: " + ",".join(unresolved))
    return frame


def build_frame(engine, request):
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Original request required")
    payload = {
        "R0": request,
        "P0": POLICY,
        "response_schema": {
            "type": "object",
            "properties": {"fully_serviceable": {"type": ["boolean", "null"]}},
            "required": ["fully_serviceable"],
            "additionalProperties": False,
        },
    }
    obj, call = engine.json_call(
        TRIAGE,
        payload,
        phase="whole_request_serviceability",
        budget=128,
        seed=seed_ticket(FRAME_SEED, request, "whole_request_serviceability_v1"),
    )
    if (
        not isinstance(obj, dict)
        or set(obj) != {"fully_serviceable"}
        or type(obj["fully_serviceable"]) not in (bool, type(None))
    ):
        raise ValueError(
            "Strict fallible whole-request feasibility declaration required"
        )
    decision = obj["fully_serviceable"]
    if decision is not True:
        return build_source_frame(engine, request)
    inventory = json.dumps(
        {
            "original_request_sha256": binding(request),
            "fully_serviceable": True,
            "routing_is_fallible_not_semantic_gold": True,
            "triage_call_id": call["call_id"],
        },
        sort_keys=True,
    )
    frame = Frame(
        (Goal("G1", request, request, CRITERION, "R0"),),
        (),
        SCOPE,
        "Retain the entire original serviceable task without reconstructing its outcomes.",
        inventory,
    )
    return frame
