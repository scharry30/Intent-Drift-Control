"""Stable content-based cohort selection independent of model outcomes."""

from ..utils.hashing import binding


def choose(rows, salt, count):
    return sorted(rows, key=lambda r: binding([salt, r["id"]]))[:count]
