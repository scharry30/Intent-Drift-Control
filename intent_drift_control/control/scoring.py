"""Per-goal assessments bound to the exact candidate and prefix."""

from dataclasses import asdict
from ..core import binding, parse_assessment
from ..utils.records import append
from ..utils.random import seed_ticket
from ..framing.policy import POLICY
from .prompts import ASSESSOR, assessment_schema


def judgment_key(engine, request, frame, accepted, ids):
    protocol = engine.protocol(accepted, ids)
    return (
        {
            "request": request,
            "policy_sha256": binding(POLICY),
            "frame_sha256": frame.sha256,
            "accepted_ids": list(accepted),
            "candidate_ids": list(ids),
            "mode": "CLOSURE" if protocol["closure"] else "CONTINUATION",
            "assessor_prompt_sha256": binding(ASSESSOR),
            "response_schema_sha256": binding(
                assessment_schema(frame, protocol["closure"])
            ),
            "schema_version": 4,
        },
        protocol,
    )


def assess(engine, request, frame, accepted, ids, master, memo):
    key, protocol = judgment_key(engine, request, frame, accepted, ids)
    hashed = binding(key)
    if hashed in memo:
        return memo[hashed]
    if not protocol["valid"]:
        memo[hashed] = None
        append(
            engine.out / "assessments.jsonl",
            {"binding": key, "status": "PROTOCOL_INVALID"},
        )
        return None
    candidate_view = engine.tokenizer.decode(ids, skip_special_tokens=False)
    payload = {
        "request": request,
        "policy": POLICY,
        "frame": frame.wire(),
        "accepted_prefix": engine.tokenizer.decode(accepted, skip_special_tokens=False),
        "evidence_views": {
            "C0": candidate_view,
            **(
                {"A0": protocol["answer"], "D0": protocol["new_delivery"]}
                if protocol["closure"]
                else {}
            ),
        },
        "mode": key["mode"],
        "response_schema": assessment_schema(frame, protocol["closure"]),
    }
    try:
        obj, record = engine.json_call(
            ASSESSOR,
            payload,
            phase="assessor",
            budget=2048,
            seed=seed_ticket(master, hashed, "assessor"),
        )
        result = parse_assessment(
            obj,
            frame,
            key,
            closure=protocol["closure"],
            terminal_ready=protocol["terminal_ready"],
            continuation=candidate_view,
            answer=protocol["answer"],
            evidence_reference=True,
            delivery=protocol["new_delivery"],
        )
        append(
            engine.out / "assessments.jsonl",
            {
                "case_id": engine.case_id,
                "role": engine.role,
                "binding": key,
                "call_id": record["call_id"],
                "status": "VALID_DECLARED_SCORES",
                "assessment": asdict(result),
                "raw_assessment": obj,
                "semantic_gold": False,
            },
        )
    except ValueError as exc:
        result = None
        append(
            engine.out / "assessments.jsonl",
            {
                "case_id": engine.case_id,
                "role": engine.role,
                "binding": key,
                "status": "INVALID",
                "error": str(exc),
                "semantic_unknown": True,
            },
        )
    memo[hashed] = result
    return result
