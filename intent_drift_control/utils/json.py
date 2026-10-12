"""Transport-level JSON normalization and duplicate-key rejection."""


def normalize_json_text(value):
    if not isinstance(value, str):
        raise TypeError("JSON response must be text")
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip().lower() in {"```", "```json"}
            and (lines[-1].strip() == "```")
        ):
            value = "\n".join(lines[1:-1]).strip()
    normalized_outer_braces = False
    if value.startswith("{{") and value.endswith("}}"):
        value = value[1:-1].strip()
        normalized_outer_braces = True
    return (value, normalized_outer_braces)


def unique_object(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError("Duplicate JSON object key; never silently overwrite")
        obj[k] = v
    return obj
