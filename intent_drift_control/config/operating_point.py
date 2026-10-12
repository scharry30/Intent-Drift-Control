"""Continuation and closure acceptance thresholds."""

from dataclasses import dataclass
import numpy as np
from ..core.validation import number


@dataclass(frozen=True)
class OperatingPoint:
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    positive_floor: float = 0.0
    negative_ceiling: float = 0.25
    scope_floor: float = 0.5
    favorable_threshold: float = 0.5
    terminal_floor: float = 0.75

    def __post_init__(self):
        for value in self.weights + (
            self.positive_floor,
            self.negative_ceiling,
            self.scope_floor,
            self.favorable_threshold,
            self.terminal_floor,
        ):
            number(value)
        if (
            len(self.weights) != 3
            or any((w <= 0 for w in self.weights))
            or abs(sum(self.weights) - 1) > 1e-12
        ):
            raise ValueError("Three positive normalized weights required")

    def verdict(self, assessment):
        if assessment is None:
            return {
                "accept": False,
                "reason": "ASSESSMENT_INVALID",
                "semantic_unknown": True,
            }
        a = assessment
        closure_ok = not a.closure or (
            a.terminal_ready
            and all(a.complete)
            and all((v >= self.terminal_floor for v in a.per_goal_lo))
        )
        margins = (
            a.positive_lo - self.positive_floor,
            self.negative_ceiling - a.negative_hi,
            a.scope_lo - self.scope_floor,
        )
        phi = float(np.dot(self.weights, a.vector))
        accepted = (
            closure_ok and min(margins) >= 0 and (phi >= self.favorable_threshold)
        )
        return {
            "accept": bool(accepted),
            "reason": "ACCEPT" if accepted else "REDIRECT",
            "margins": margins,
            "phi": phi,
            "closure_ok": closure_ok,
            "weighted_gate_redundant": self.favorable_threshold
            <= self.weights[0] * self.positive_floor
            - self.weights[1] * self.negative_ceiling
            + self.weights[2] * self.scope_floor,
        }
