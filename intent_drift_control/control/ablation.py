"""Ordinary-phase component masks with unchanged repair and closure gates."""

from ..utils.interfaces import ResourceEnded


MASKS = tuple((f"{i:03b}" for i in range(8)))


def ordinary_verdict(assessment, protocol, point, mask):
    if mask not in MASKS:
        raise ValueError("Exactly one frozen three-branch mask required")
    if mask == "111" or protocol["closure"]:
        return point.verdict(assessment)
    if not protocol["valid"]:
        return {"accept": False, "reason": "PROTOCOL_INVALID", "mask": mask}
    if mask == "000":
        return {"accept": True, "reason": "ORDINARY_PROTOCOL_ONLY", "mask": mask}
    if assessment is None:
        return {"accept": False, "reason": "ACTIVE_BRANCH_UNKNOWN", "mask": mask}
    a = assessment
    values = (a.positive_lo, 1 - a.negative_hi, a.scope_lo)
    bounds = (
        a.positive_lo >= point.positive_floor,
        a.negative_hi <= point.negative_ceiling,
        a.scope_lo >= point.scope_floor,
    )
    active = [i for i, bit in enumerate(mask) if bit == "1"]
    mass = sum((point.weights[i] for i in active))
    score = sum((point.weights[i] * values[i] / mass for i in active))
    cutoff = point.favorable_threshold + point.weights[1]
    return {
        "accept": all((bounds[i] for i in active)) and score >= cutoff,
        "reason": "MASK_ACCEPT"
        if all((bounds[i] for i in active)) and score >= cutoff
        else "MASK_REDIRECT",
        "mask": mask,
        "favorable_score": score,
        "original_full_arm_theta": cutoff,
    }


class PhaseEngine:
    def __init__(self, engine):
        self.engine = engine
        self.generator_phase = None
        self.last_protocol = None

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def call(self, messages, **kwargs):
        if kwargs["thinking"]:
            self.generator_phase = kwargs["phase"]
        return self.engine.call(messages, **kwargs)

    def json_call(self, *args, **kwargs):
        return self.engine.json_call(*args, **kwargs)

    def protocol(self, accepted, ids):
        self.last_protocol = self.engine.protocol(accepted, ids)
        return self.last_protocol


class OrdinaryOnlyPoint:
    def __init__(self, point, engine, mask):
        self.point = point
        self.engine = engine
        self.mask = mask

    def __getattr__(self, name):
        return getattr(self.point, name)

    def verdict(self, assessment):
        if self.engine.generator_phase == "ordinary_generator":
            return ordinary_verdict(
                assessment, self.engine.last_protocol, self.point, self.mask
            )
        return self.point.verdict(assessment)


class StopOnly:
    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, n):
        return getattr(self.engine, n)

    def call(self, messages, **kw):
        if kw.get("phase") == "repair_generator":
            raise ResourceEnded("STOP_ONLY_ABLATION_NO_REPAIR")
        return self.engine.call(messages, **kw)
