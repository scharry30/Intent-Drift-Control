"""Abstract inference contract and deterministic call bindings."""

import json
import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from .core import binding


class WindowEnded(RuntimeError):
    pass


class ResourceEnded(RuntimeError):
    pass


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def seed_ticket(master, *parts):
    return int(binding([master, *parts])[:8], 16) % (2**31 - 1)


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


class Tokenizer(Protocol):
    def decode(
        self, ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str: ...


class InferenceEngine(Protocol):
    out: Path
    case_id: str
    role: str
    tokenizer: Tokenizer
    episode_cost: Mapping[str, int | float]

    def call(self, messages: list[dict], **kwargs) -> dict: ...

    def json_call(self, system: str, payload: dict, **kwargs) -> tuple[dict, dict]: ...

    def protocol(self, accepted: Sequence[int], ids: Sequence[int]) -> dict: ...

    def decode(self, accepted: Sequence[int]) -> dict: ...
