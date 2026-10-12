"""Coalesce ready requests by decoding mode and auxiliary phase."""

import threading
import time
from ..config.defaults import HOST_TASKS
from ..utils.interfaces import ResourceEnded
from ..utils.records import append
from .accounting import FreshPrefillTransport


class ThroughputTransport(FreshPrefillTransport):
    def drain(self, *, timeout=0.05):
        from collections import deque
        from queue import Empty

        if threading.get_ident() != self.owner:
            raise RuntimeError("CUDA owner mismatch")
        if not self.pending:
            try:
                self.pending.append(self.queue.get(timeout=timeout))
            except Empty:
                return False
        while len(self.pending) < HOST_TASKS:
            try:
                self.pending.append(self.queue.get_nowait())
            except Empty:
                break
        if len(self.pending) < self.maximum_batch:
            until = time.monotonic() + 0.02
            while len(self.pending) < HOST_TASKS and time.monotonic() < until:
                try:
                    self.pending.append(
                        self.queue.get(timeout=max(0.0001, until - time.monotonic()))
                    )
                except Empty:
                    break
        first = self.pending.popleft()
        batch = [first]
        rest = deque()
        limit = (
            self.native_generator_batch_limit
            if first.kind == "generator"
            else self.maximum_batch
        )

        def phase(r):
            return r.start["phase"] if r.kind == "auxiliary" else r.kind

        for r in self.pending:
            width = max((len(q.inputs) for q in batch + [r]))
            work = (len(batch) + 1) * (width + r.start["max_new_tokens"])
            if (
                r.bucket == first.bucket
                and phase(r) == phase(first)
                and (len(batch) < limit)
                and (work <= self.pad_work_limit)
                and ((len(batch) + 1) * width <= self.prefill_work_limit)
            ):
                batch.append(r)
            else:
                rest.append(r)
        self.pending = rest
        ready = []
        for r in batch:
            try:
                r.slot.check()
                if (
                    time.time() >= r.deadline
                    or time.monotonic() >= r.monotonic_deadline
                ):
                    raise ResourceEnded("Call budget ended in model queue")
                ready.append(r)
            except BaseException as exc:
                append(
                    r.slot.out / "no_call_events.jsonl",
                    {
                        **r.start,
                        "reason": str(exc),
                        "physical_model_call": False,
                        "queue_only": True,
                    },
                )
                r.future.set_exception(exc)
        if not ready:
            return True
        try:
            output = self.generate(ready)
        except BaseException as exc:
            for r in ready:
                r.future.set_exception(exc)
            self.abort(exc)
            raise
        for r, row in zip(ready, output, strict=True):
            r.future.set_result(row)
        return True
