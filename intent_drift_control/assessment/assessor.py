"""Evaluate typed declarations against the native candidate state."""

from dataclasses import asdict
import json
from ..core import Assessment, binding
from ..control.scoring import judgment_key
from ..framing.policy import POLICY
from ..utils.records import append
from ..utils.random import seed_ticket
from ..utils.json import normalize_json_text, unique_object
from .wire import ARMS, score_interface, compile_wire


ASSESSOR_OUTPUT_CONTRACT = "\nReturn exactly one JSON object satisfying response_schema for the ACTUAL mode.\nThe response_schema is a contract for your output, not text to reproduce.\nDo not echo or summarize request, policy, frame, accepted_prefix, evidence_views,\nresponse_schema, or these instructions. Emit each required field exactly once,\nuse only the field names and immutable IDs present in response_schema, and emit\nno extra fields or IDs. Start with { and end with }. No Markdown fence or prose.\n".strip()


def assessor_output_contract(arm, frame, closure):
    positive_ids = [g.id for g in frame.positive]
    negative_ids = [g.id for g in frame.negative]
    mode = "CLOSURE" if closure else "CONTINUATION"
    ids = f"This is {mode}. Required positive IDs, exactly once: {positive_ids!r}. Required negative IDs, exactly once: {negative_ids!r}."
    if arm == ARMS[0]:
        shape = '\nFor this ARRAY wire, the top-level keys are exactly positive, negative, scope,\nand violations; there is no top-level completion key. Each positive array item\nhas exactly id, lo, hi, evidence, complete. Each negative array item has exactly\nid, lo, hi, evidence: NEVER include complete on a negative item, even when the\nnegative item is harmless. In CONTINUATION, positive complete is null; in\nCLOSURE it is boolean. Evidence is mode-bound: use only C0 or "" in\nCONTINUATION, and A0 for a completed positive or D0/"" for negative and scope\nin CLOSURE; never use P0 or R0. Use [] only when the schema\'s minItems is zero,\nand otherwise include every required ID once.\n'
    else:
        shape = "\nFor this HOST-SLOT wire, positive and negative are objects keyed by the exact\nimmutable IDs in response_schema. Each score slot has exactly lo, hi, evidence;\ncompletion appears only in the separate top-level completion object for CLOSURE.\n"
    return ASSESSOR_OUTPUT_CONTRACT + "\n" + ids + shape


class Assessor:
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError("Unregistered native score interface")
        self.arm = arm

    def state(self, engine, request, frame, accepted, ids):
        original, p = judgment_key(engine, request, frame, accepted, ids)
        payload = {
            "request": request,
            "policy": POLICY,
            "frame": frame.wire(),
            "accepted_prefix": engine.tokenizer.decode(
                accepted, skip_special_tokens=False
            ),
            "evidence_views": {
                "C0": engine.tokenizer.decode(ids, skip_special_tokens=False),
                **(
                    {"A0": p["answer"], "D0": p["new_delivery"]} if p["closure"] else {}
                ),
            },
            "mode": original["mode"],
        }
        state = {"source_key": original, "source_payload": payload}
        system, schema = score_interface(frame, p["closure"], self.arm)
        if self.arm == ARMS[0]:
            system = system.replace(
                "complete MUST be null.",
                "In CONTINUATION, positive complete is null; negative items have no complete field.",
            )
            system = system.replace(
                "complete MUST be boolean per goal.",
                "In CLOSURE, positive complete is boolean; negative items have no complete field.",
            )
        system = system + "\n" + assessor_output_contract(self.arm, frame, p["closure"])
        key = {
            **original,
            "assessor_prompt_sha256": binding(system),
            "response_schema_sha256": binding(schema),
            "schema_version": 5 if self.arm == ARMS[0] else 6,
            "assessment_wire_arm": self.arm,
            "raw_JSON_parser": "STRICT_DUPLICATE_KEYS_WITH_BOUNDED_SYNTAX_NORMALIZATION",
            "native_state_adapter_version": 2,
        }
        return (state, p, key, system, schema)

    def judgment_key(self, engine, request, frame, accepted, ids):
        _, protocol, key, _, _ = self.state(engine, request, frame, accepted, ids)
        return (key, protocol)

    def __call__(self, engine, request, frame, accepted, ids, master, memo):
        state, p, key, system, schema = self.state(
            engine, request, frame, accepted, ids
        )
        h = binding(key)
        if h in memo:
            return memo[h]
        common = {
            "case_id": engine.case_id,
            "role": engine.role,
            "binding": key,
            "semantic_gold": False,
        }
        if not p["valid"]:
            append(
                engine.out / "assessments.jsonl",
                {**common, "status": "PROTOCOL_INVALID"},
            )
            memo[h] = None
            return None
        record = None
        obj = None
        result = None
        try:
            payload = {**state["source_payload"], "response_schema": schema}
            record = engine.call(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                phase="assessor",
                thinking=False,
                budget=2048,
                seed=seed_ticket(master, h, "assessor"),
                greedy=False,
            )
            if not record["eos"]:
                raise ValueError("Auxiliary response unfinished at cap")
            text, normalized_outer_braces = normalize_json_text(record["text"])
            obj = json.loads(text, object_pairs_hook=unique_object)
            if not isinstance(obj, dict):
                raise ValueError("Expected one JSON object")
            compiled = compile_wire(obj, state, frame, key, p, self.arm)
            result = Assessment(**compiled["assessment"])
            append(
                engine.out / "assessments.jsonl",
                {
                    **common,
                    "status": "VALID_DECLARED_SCORES",
                    "call_id": record["call_id"],
                    "raw_assessment": obj,
                    "assessment": asdict(result),
                    "normalized_outer_braces": normalized_outer_braces,
                    "compiled_wire_metadata": {
                        k: v for k, v in compiled.items() if k != "assessment"
                    },
                },
            )
        except (ValueError, TypeError) as exc:
            append(
                engine.out / "assessments.jsonl",
                {
                    **common,
                    "status": "INVALID",
                    "error": str(exc),
                    "call_id": record["call_id"] if record else None,
                    "raw_assessment": obj,
                    "semantic_unknown": True,
                },
            )
        memo[h] = result
        return result
