"""Strict compact per-goal score declarations."""

from copy import deepcopy
from dataclasses import asdict
from ..core import parse_assessment
from ..control.prompts import ASSESSOR, assessment_schema


DECLARATION_ARMS = ("FULL_JSON_WITH_PROSE", "DECLARATION_ONLY_JSON")


HOST_NOTE = "HOST DIAGNOSTIC: this declared-score wire contains no generated free-form explanation."


DECLARATION_SYSTEM = (
    ASSESSOR.replace(
        "own contextual assessment; explain the actual contribution/enablement/scope.",
        "own contextual assessment; declare the actual contribution/enablement/scope in the schema fields.",
    )
    + "\nThe response wire contains numeric intervals, mode-correct completion flags,\nimmutable goal IDs and evidence-view references only. Do NOT add explanation,\nprose, quotations, LaTeX, a solution or any extra field. This is a transport\ndeclaration constraint, not permission to change the original task or policy.\nThe host records the absence of a generated explanation separately."
)


def declaration_interface(frame, closure, arm):
    if arm not in DECLARATION_ARMS:
        raise ValueError("Unregistered assessment-format factor")
    schema = deepcopy(assessment_schema(frame, closure))
    if arm == DECLARATION_ARMS[0]:
        return (ASSESSOR, schema)
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
