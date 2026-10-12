"""Finite-number and nonempty-text contracts."""

import math


def number(value):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (not 0 <= value <= 1)
    ):
        raise ValueError("Expected a finite score in [0,1], not a boolean")
    return float(value)


def text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 3000:
        raise ValueError("Expected a bounded nonempty string")
    return value
