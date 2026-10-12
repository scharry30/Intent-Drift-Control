"""Deterministic seed tickets for independent model calls."""

from .hashing import binding


def seed_ticket(master, *parts):
    return int(binding([master, *parts])[:8], 16) % (2**31 - 1)
