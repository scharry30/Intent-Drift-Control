"""Repair guidance constructed from violated source obligations."""


def guidance(frame, rejected):
    selected = set(rejected.violations) if rejected else set()
    references = [g for g in frame.positive + frame.negative if g.id in selected]
    clauses = [g.id + ": " + g.text for g in references]
    return (
        "Resume from the exact accepted prefix; the rejected text is absent. Preserve the original legitimate outcomes, avoid excluded means, and use the permitted scope. Do not add a new <think> tag. "
        + "; ".join(clauses)
    )
