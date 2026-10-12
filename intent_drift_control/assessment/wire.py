"""Compile host-bound ID slots and mode-correct completion fields."""

from copy import deepcopy
from dataclasses import asdict
from ..core import parse_assessment
from ..control.prompts import assessment_schema
from .declarations import (
    DECLARATION_ARMS,
    DECLARATION_SYSTEM,
    HOST_NOTE,
    declaration_interface,
    compile_declaration,
)


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
    old = assessment_schema(frame, closure)
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
