"""Exact padding, row stops and finite-language batch callbacks."""

from concurrent.futures import Future
from dataclasses import dataclass, field
import time


def left_pad(sequences, pad_id, torch, device):
    if not sequences or any((not s for s in sequences)):
        raise ValueError("Nonempty exact per-row token inputs required")
    width = max(map(len, sequences))
    inputs = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long, device=device
    )
    masks = torch.zeros_like(inputs)
    for index, ids in enumerate(sequences):
        inputs[index, width - len(ids) :] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        masks[index, width - len(ids) :] = 1
    return (inputs, masks)


def unpad_generated(ids, eos_ids, pad_id):
    ids = list(ids)
    eos_ids = set(eos_ids)
    endings = [i for i, token in enumerate(ids) if token in eos_ids]
    if not endings:
        return ids
    end = endings[0]
    if any((token != pad_id for token in ids[end + 1 :])):
        raise ValueError("Finished batch row contains nonpadding continuation")
    return ids[: end + 1]


@dataclass
class Request:
    slot: object
    kind: str
    start: dict
    inputs: list
    deadline: float
    monotonic_deadline: float
    grammar: object = None
    future: Future = field(default_factory=Future)

    @property
    def bucket(self):
        s = self.start
        return (
            self.kind,
            s["max_new_tokens"],
            s.get("greedy", True),
            s.get("thinking", False),
        )


class RowStop:
    def __init__(
        self,
        torch,
        requests,
        width,
        stopped_rows,
        *,
        eos_ids=(),
        wall=time.time,
        mono=time.monotonic,
    ):
        self.torch = torch
        self.requests = requests
        self.width = width
        self.stopped_rows = stopped_rows
        self.lengths = {}
        self.wall = wall
        self.mono = mono
        self.eos_ids = set(eos_ids)
        self.native_completed = {}
        self.completed_times = {}

    def __call__(self, input_ids, scores, **kwargs):
        now, tick = (self.wall(), self.mono())
        for row, r in enumerate(self.requests):
            if row in self.native_completed:
                continue
            if row not in self.lengths and (
                now >= r.deadline or tick >= r.monotonic_deadline
            ):
                self.lengths[row] = input_ids.shape[1] - self.width
                self.stopped_rows.add(row)
            if row not in self.lengths and int(input_ids[row, -1]) in self.eos_ids:
                self.native_completed[row] = input_ids.shape[1] - self.width
                self.completed_times[row] = now
        return self.torch.tensor(
            [i in self.stopped_rows for i in range(len(self.requests))],
            dtype=self.torch.bool,
            device=input_ids.device,
        )


def generated_row(ids, row, stop, eos_ids, pad_id):
    values = list(ids)
    if row in stop.lengths:
        length = stop.lengths[row]
        if any((t != pad_id for t in values[length:])):
            raise ValueError("Nonpadding content after per-row stop")
        values = values[:length]
    return unpad_generated(values, eos_ids, pad_id)


def batch_grammar(requests, width, pad_id, stopped_rows):
    def allowed(row, input_ids):
        if row in stopped_rows:
            return [pad_id]
        suffix = input_ids.tolist()[width:]
        grammar = requests[row].grammar
        cropped = unpad_generated(suffix, grammar.eos_ids, pad_id)
        return grammar.allowed(cropped)

    return allowed
