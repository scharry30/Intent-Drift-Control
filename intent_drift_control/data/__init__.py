"""Corpus metadata and offline record handling."""

from .sources import SOURCES
from .records import read_rows
from .cohort import choose


__all__ = ["SOURCES", "read_rows", "choose"]
