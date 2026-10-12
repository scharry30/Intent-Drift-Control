"""Inference and tokenizer contracts supplied by the application."""

from pathlib import Path
from typing import Mapping, Protocol, Sequence


class WindowEnded(RuntimeError):
    pass


class ResourceEnded(RuntimeError):
    pass


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
