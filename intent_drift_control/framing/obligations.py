"""Request spans and source-preserving obligation extraction."""

import re
from .prompts import CRITERION, EXTRACTION_ARMS


def source_units(request):
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Original request required")
    parts = re.split("(?<=[.!?])(?=\\s)|(?<=\\n)", request)
    parts = [p for p in parts if p]
    if len(parts) > 32:
        parts = [request]
    units = [{"id": f"S{i + 1}", "text": p} for i, p in enumerate(parts)]
    if "".join((u["text"] for u in units)) != request:
        raise ValueError("Never truncate/rewrite source units")
    return units


def selected_text(ids, units):
    lookup = {u["id"]: u["text"] for u in units}
    if (
        not isinstance(ids, list)
        or not ids
        or any((type(i) is not str or i not in lookup for i in ids))
        or (len(ids) != len(set(ids)))
    ):
        raise ValueError("Unique source-unit IDs required")
    if ids != [u["id"] for u in units if u["id"] in ids]:
        raise ValueError("Preserve original source order")
    return "".join((lookup[i] for i in ids))


def extract_schema(units):
    item = {
        "type": "object",
        "properties": {
            "unit_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(units),
                "items": {"enum": [u["id"] for u in units]},
                "uniqueItems": True,
            }
        },
        "required": ["unit_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"outcomes": {"type": "array", "maxItems": 8, "items": item}},
        "required": ["outcomes"],
        "additionalProperties": False,
    }


def parse_obligations(obj, request):
    if (
        not isinstance(obj, dict)
        or set(obj) != {"outcomes"}
        or (not isinstance(obj["outcomes"], list))
        or (len(obj["outcomes"]) > 8)
    ):
        raise ValueError("Invalid original-obligation schema")
    units = source_units(request)
    rows = []
    seen = set()
    for row in obj["outcomes"]:
        if not isinstance(row, dict) or set(row) != {"unit_ids"}:
            raise ValueError("Only host source IDs are accepted")
        value = selected_text(row["unit_ids"], units)
        identity = tuple(row["unit_ids"])
        if identity in seen:
            raise ValueError("Duplicate obligation")
        seen.add(identity)
        rows.append(
            {
                "id": f"O{len(rows) + 1}",
                "unit_ids": row["unit_ids"],
                "original_text": value,
                "criterion": CRITERION,
            }
        )
    return rows


def extract_payload(request, policy, arm):
    if arm not in EXTRACTION_ARMS:
        raise ValueError("Unknown extraction arm")
    units = source_units(request)
    payload = {
        "R0": request,
        "source_units": units,
        "response_schema": extract_schema(units),
    }
    if arm == EXTRACTION_ARMS[0]:
        payload["P0"] = policy
    return payload
