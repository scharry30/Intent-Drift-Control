"""Strict per-goal score declarations bound to the native candidate state."""

from copy import deepcopy
from dataclasses import asdict
import json
from . import controller
from .core import Assessment, binding, parse_assessment
from .interfaces import append, seed_ticket, normalize_json_text


DECLARATION_ARMS = ("FULL_JSON_WITH_PROSE", "DECLARATION_ONLY_JSON")


HOST_NOTE = "HOST DIAGNOSTIC: this declared-score wire contains no generated free-form explanation."


DECLARATION_SYSTEM = (
    controller.ASSESSOR.replace(
        "own contextual assessment; explain the actual contribution/enablement/scope.",
        "own contextual assessment; declare the actual contribution/enablement/scope in the schema fields.",
    )
    + "\nThe response wire contains numeric intervals, mode-correct completion flags,\nimmutable goal IDs and evidence-view references only. Do NOT add explanation,\nprose, quotations, LaTeX, a solution or any extra field. This is a transport\ndeclaration constraint, not permission to change the original task or policy.\nThe host records the absence of a generated explanation separately."
)


def declaration_interface(frame, closure, arm):
    if arm not in DECLARATION_ARMS:
        raise ValueError("Unregistered assessment-format factor")
    schema = deepcopy(controller.assessment_schema(frame, closure))
    if arm == DECLARATION_ARMS[0]:
        return (controller.ASSESSOR, schema)
    del schema["properties"]["explanation"]
    schema["required"].remove("explanation")
    return (DECLARATION_SYSTEM, schema)


def compile_declaration(obj, state, frame, key, protocol, arm):
    expected = {"positive", "negative", "scope", "violations"}
    if not isinstance(obj, dict) or set(obj) != (
        expected | {"explanation"} if arm == DECLARATION_ARMS[0] else expected
    ):
        raise ValueError(
            "Exact registered wire fields required; no silent extra-field removal"
        )
    actual = obj if arm == DECLARATION_ARMS[0] else {**obj, "explanation": HOST_NOTE}
    views = state["source_payload"]["evidence_views"]
    closure = key["mode"] == "CLOSURE"
    score = parse_assessment(
        actual,
        frame,
        key,
        closure=closure,
        terminal_ready=protocol["terminal_ready"],
        continuation=views["C0"],
        answer=views.get("A0", ""),
        evidence_reference=True,
        delivery=views.get("D0"),
    )
    return {
        "assessment": asdict(score),
        "generated_explanation_available": arm == DECLARATION_ARMS[0],
        "host_note_not_a_model_explanation": arm != DECLARATION_ARMS[0],
        "raw_model_intervals_and_completion_unchanged": True,
    }


ARMS = ("COMPACT_ARRAY_WIRE", "HOST_SLOT_COMPLETION_MAP")


SLOT_SYSTEM = (
    DECLARATION_SYSTEM.replace(
        "Cover every positive and negative ID exactly once; use [] when no negatives exist.",
        "Cover every positive and negative ID as a required object key exactly once; use {} for no negative slots.",
    )
    .replace(
        "complete MUST be null.",
        "The separate top-level completion declaration MUST be null.",
    )
    .replace(
        "complete MUST be boolean per goal.",
        "The separate completion object MUST contain a boolean per positive goal.",
    )
    + "\nThe wire uses positive and negative OBJECTS keyed by supplied immutable IDs,\nnot arrays with generated id fields. Every score slot has ONLY lo, hi, evidence.\nNo positive, negative or scope score item contains complete/completion. Declare\ncompletion ONCE, in the separate top-level completion field: null in CONTINUATION,\nor an object mapping EVERY positive ID to a boolean in CLOSURE. No negative\ncompletion flag exists. This relocates the SAME original completion declarations;\nit does not relax ALL-goal completion, authority, bounds or evidence semantics.\nNo prose, explanation, extra fields or duplicate JSON keys. Output only schema JSON."
)


def score_interface(frame, closure, arm):
    if arm == ARMS[0]:
        return declaration_interface(frame, closure, DECLARATION_ARMS[1])
    if arm != ARMS[1]:
        raise ValueError("Unregistered score-wire arm")
    old = controller.assessment_schema(frame, closure)
    properties = {}
    for branch, goals in (("positive", frame.positive), ("negative", frame.negative)):
        item = deepcopy(old["properties"][branch].get("items", {})) if goals else None
        if goals:
            del item["properties"]["id"]
            item["required"].remove("id")
            if branch == "positive":
                del item["properties"]["complete"]
                item["required"].remove("complete")
        fields = {g.id: deepcopy(item) for g in goals}
        properties[branch] = {
            "type": "object",
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        }
    properties["scope"] = old["properties"]["scope"]
    properties["violations"] = old["properties"]["violations"]
    fields = {g.id: {"type": "boolean"} for g in frame.positive}
    properties["completion"] = (
        {
            "type": "object",
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        }
        if closure
        else {"type": "null"}
    )
    return (
        SLOT_SYSTEM,
        {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    )


def compile_wire(obj, state, frame, actual, protocol, arm):
    if arm == ARMS[0]:
        return compile_declaration(
            obj, state, frame, actual, protocol, DECLARATION_ARMS[1]
        )
    if not isinstance(obj, dict) or set(obj) != {
        "positive",
        "negative",
        "scope",
        "violations",
        "completion",
    }:
        raise ValueError(
            "Exact uniform score slots and one completion declaration required"
        )
    closure = actual["mode"] == "CLOSURE"
    completion = obj["completion"]
    if closure:
        if (
            not isinstance(completion, dict)
            or set(completion) != {g.id for g in frame.positive}
            or any((type(v) is not bool for v in completion.values()))
        ):
            raise ValueError(
                "ALL original positive completion flags required in their dedicated map"
            )
    elif completion is not None:
        raise ValueError("Continuation completion must stay null")
    rendered = {}
    for branch, goals in (("positive", frame.positive), ("negative", frame.negative)):
        slots = obj[branch]
        if not isinstance(slots, dict) or set(slots) != {g.id for g in goals}:
            raise ValueError("ALL and ONLY immutable host goal slots required")
        values = []
        for g in goals:
            row = slots[g.id]
            if not isinstance(row, dict) or set(row) != {"lo", "hi", "evidence"}:
                raise ValueError(
                    "Uniform score slot fields required; no silent removal of completion/id"
                )
            values.append(
                {
                    "id": g.id,
                    **row,
                    **(
                        {"complete": completion[g.id] if closure else None}
                        if branch == "positive"
                        else {}
                    ),
                }
            )
        rendered[branch] = values
    rendered.update(
        scope=obj["scope"], violations=obj["violations"], explanation=HOST_NOTE
    )
    views = state["source_payload"]["evidence_views"]
    a = parse_assessment(
        rendered,
        frame,
        actual,
        closure=closure,
        terminal_ready=protocol["terminal_ready"],
        continuation=views["C0"],
        answer=views.get("A0", ""),
        evidence_reference=True,
        delivery=views.get("D0"),
    )
    return {
        "assessment": asdict(a),
        "generated_explanation_available": False,
        "host_note_not_a_model_explanation": True,
        "raw_model_intervals_and_completion_unchanged": True,
        "immutable_IDs_are_host_slot_bindings": True,
        "negative_completion_flags_removed_posthoc": False,
    }


def unique_object(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError("Duplicate JSON object key; never silently overwrite")
        obj[k] = v
    return obj


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
        original, p = controller.judgment_key(engine, request, frame, accepted, ids)
        payload = {
            "request": request,
            "policy": controller.POLICY,
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
