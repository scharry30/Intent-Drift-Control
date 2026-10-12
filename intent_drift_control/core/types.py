"""Source-linked goals, task frames and interval assessments."""

from dataclasses import asdict, dataclass
from ..utils.hashing import binding


@dataclass(frozen=True)
class Goal:
    id: str
    text: str
    source_quote: str
    criterion: str
    source_ref: str = ""


@dataclass(frozen=True)
class Frame:
    positive: tuple[Goal, ...]
    negative: tuple[Goal, ...]
    scope: tuple[str, ...]
    rationale: str
    inventory: str

    def wire(self):
        def goal_wire(g):
            row = asdict(g)
            row.pop("source_quote" if g.source_ref else "source_ref")
            return row

        return {
            "positive": [goal_wire(g) for g in self.positive],
            "negative": [goal_wire(g) for g in self.negative],
            "scope": list(self.scope),
            "rationale": self.rationale,
        }

    @property
    def sha256(self):
        return binding({"frame": self.wire(), "inventory": self.inventory})


@dataclass(frozen=True)
class Assessment:
    positive_lo: float
    positive_hi: float
    negative_lo: float
    negative_hi: float
    scope_lo: float
    per_goal_lo: tuple[float, ...]
    complete: tuple[bool | None, ...]
    terminal_ready: bool
    closure: bool
    binding_sha256: str
    evidence: str
    violations: tuple[str, ...]

    @property
    def vector(self):
        return (self.positive_lo, -self.negative_hi, self.scope_lo)
