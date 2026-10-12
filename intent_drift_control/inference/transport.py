"""Single-owner model transport with independent episode requests."""

from collections import deque
from queue import Queue, Empty
import threading
import time
from ..utils.hashing import binding
from ..utils.records import append
from ..utils.interfaces import ResourceEnded
from ..evaluation.guard import parse_native
from .batching import left_pad, RowStop, generated_row, batch_grammar
from .multinomial import IndependentRowSampler


class SharedTransport:
    def __init__(
        self,
        engine,
        guard,
        *,
        maximum_batch=4,
        pad_work_limit=65536,
        prefill_work_limit=32768,
        native_generator_batch_limit=1,
    ):
        self.engine = engine
        self.guard = guard
        self.torch = engine.torch
        self.out = engine.out
        self.maximum_batch = maximum_batch
        self.pad_work_limit = pad_work_limit
        self.prefill_work_limit = prefill_work_limit
        self.native_generator_batch_limit = native_generator_batch_limit
        self.owner = threading.get_ident()
        self.queue = Queue()
        self.pending = deque()
        self.sequence = 0
        self.abort_error = None
        self.status_lock = threading.Lock()

    def submit(self, request):
        if threading.get_ident() == self.owner:
            raise RuntimeError("Owner cannot block waiting for its own model request")
        with self.status_lock:
            if self.abort_error is not None:
                raise RuntimeError("Shared model owner aborted") from self.abort_error
            self.queue.put(request)
        return request.future.result()

    def abort(self, error):
        with self.status_lock:
            self.abort_error = error
            pending = list(self.pending)
            self.pending.clear()
            while True:
                try:
                    pending.append(self.queue.get_nowait())
                except Empty:
                    break
            for r in pending:
                if not r.future.done():
                    r.future.set_exception(error)

    def drain(self, *, timeout=0.05):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the owner may execute CUDA calls")
        if not self.pending:
            try:
                self.pending.append(self.queue.get(timeout=timeout))
            except Empty:
                return False
        until = time.monotonic() + 0.02
        while len(self.pending) < self.maximum_batch and time.monotonic() < until:
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
        for r in self.pending:
            width = max((len(q.inputs) for q in batch + [r]))
            work = (len(batch) + 1) * (width + r.start["max_new_tokens"])
            if (
                r.bucket == first.bucket
                and len(batch) < limit
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
            outputs = self.generate(ready)
        except BaseException as exc:
            for r in ready:
                r.future.set_exception(exc)
            self.abort(exc)
            raise
        for r, result in zip(ready, outputs, strict=True):
            r.future.set_result(result)
        return True

    def generate(self, requests):
        from transformers import LogitsProcessorList, StoppingCriteriaList

        if threading.get_ident() != self.owner:
            raise RuntimeError("Model owner violated")
        kind = requests[0].kind
        if kind == "generator" and len(requests) > self.native_generator_batch_limit:
            raise ValueError("Registered native generator batch limit violated")
        if any((r.bucket != requests[0].bucket for r in requests)):
            raise ValueError("Incompatible decoding modes cannot share a batch")
        model = self.guard.model if kind == "guard" else self.engine.model
        tokenizer = self.guard.tokenizer if kind == "guard" else self.engine.tokenizer
        eos = self.guard.eos_ids if kind == "guard" else self.engine.eos_ids
        torch = self.torch
        device = str(next(model.parameters()).device)
        physical_started_unix = time.time()
        physical_started_mono = time.monotonic()
        for r in requests:
            timeout = r.start.get(
                "physical_call_budget_seconds",
                180 if kind in ("guard", "typed") else 3600,
            )
            r.deadline = min(r.deadline, physical_started_unix + timeout)
            r.monotonic_deadline = min(
                r.monotonic_deadline, physical_started_mono + timeout
            )
        pad = tokenizer.eos_token_id
        tensor, masks = left_pad([r.inputs for r in requests], pad, torch, device)
        width = tensor.shape[1]
        stopped = set()
        stop = RowStop(torch, requests, width, stopped, eos_ids=eos)
        greedy = requests[0].start.get("greedy", True)
        sampler = None
        kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "num_beams": 1,
            "eos_token_id": sorted(eos),
            "pad_token_id": pad,
            "max_new_tokens": requests[0].start["max_new_tokens"],
            "stopping_criteria": StoppingCriteriaList([stop]),
        }
        if not greedy:
            thinking = requests[0].start["thinking"]
            sampler = IndependentRowSampler(
                torch,
                [r.start["seed"] for r in requests],
                width,
                eos,
                pad,
                device=device,
                temperature=0.6 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8,
                top_k=20,
                stopped_rows=stopped,
                rng_states=[
                    r.slot.ordinary_rng_state
                    if r.start.get("rng_policy") == "CONTINUOUS_ORDINARY_V1"
                    else None
                    for r in requests
                ],
            )
            kwargs["logits_processor"] = LogitsProcessorList([sampler])
        if kind == "typed":
            kwargs["prefix_allowed_tokens_fn"] = batch_grammar(
                requests, width, pad, stopped
            )
        self.sequence += 1
        batch_start = {
            "batch_id": self.sequence,
            "kind": kind,
            "members": [
                {
                    "episode_id": r.slot.episode_id,
                    "call_id": r.start["call_id"],
                    "input_ids": r.inputs,
                    "input_sha256": binding(r.inputs),
                    "seed": r.start["seed"],
                }
                for r in requests
            ],
            "batch_size": len(requests),
            "padded_input_width": width,
            "physical_padded_prefill_positions": tensor.numel(),
            "logical_unpadded_prefill_tokens": sum(
                map(len, (r.inputs for r in requests))
            ),
            "attention_mask_sha256": binding(masks.tolist()),
            "started_unix": time.time(),
            "sampler": "GREEDY"
            if greedy
            else "INDEPENDENT_PER_ROW_TORCH_MULTINOMIAL_V1",
            "hf_do_sample": False,
            "per_row_sampler_is_not_stock_global_rng": not greedy,
            "cross_row_kernel_numerical_equivalence_not_assumed": True,
            "max_new_tokens": kwargs["max_new_tokens"],
            "cache_strategy": "FRESH_EXACT_PREFILL_PER_BATCH",
            "cuda_allocated_before_bytes": torch.cuda.memory_allocated()
            if device.startswith("cuda")
            else None,
        }
        append(self.out / "physical_batch_starts.jsonl", batch_start)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        try:
            with torch.inference_mode():
                output = model.generate(
                    input_ids=tensor, attention_mask=masks, **kwargs
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            suffixes = output[:, width:].tolist()
            ids = [generated_row(s, i, stop, eos, pad) for i, s in enumerate(suffixes)]
        except BaseException as exc:
            row = {
                **batch_start,
                "status": "TRANSPORT_ERROR",
                "cost_complete": False,
                "finished_unix": time.time(),
                "shared_batch_latency_s": time.perf_counter() - tick,
                "error": f"{type(exc).__name__}: {exc}",
            }
            append(self.out / "physical_batches.jsonl", row)
            for r in requests:
                append(
                    r.slot.out
                    / ("guard_calls.jsonl" if kind == "guard" else "model_calls.jsonl"),
                    {
                        **r.start,
                        "batch_id": self.sequence,
                        "status": "TRANSPORT_ERROR",
                        "cost_complete": False,
                        "error": row["error"],
                        "latency_s": None,
                        "finished_unix": row["finished_unix"],
                    },
                )
            raise
        elapsed = time.perf_counter() - tick
        finished = time.time()
        batch = {
            **batch_start,
            "status": "GENERATED",
            "cost_complete": True,
            "finished_unix": finished,
            "shared_batch_latency_s": elapsed,
            "generated_ids_by_row": ids,
            "deadline_stopped_lengths": stop.lengths,
            "native_completed_lengths": stop.native_completed,
            "native_completed_unix": stop.completed_times,
            "rng_draw_counts": dict(sampler.draw_counts) if sampler else {},
            "logical_actual_generated_tokens": sum(map(len, ids)),
            "physical_padded_decode_positions": len(requests) * len(suffixes[0]),
            "structural_padding_decode_positions": len(requests) * len(suffixes[0])
            - sum(map(len, ids)),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device.startswith("cuda")
            else None,
            "shared_batch_time_not_additive_across_episode_rows": True,
            "no_exact_allocated_seconds_claim": True,
        }
        append(self.out / "physical_batches.jsonl", batch)
        results = []
        for index, (r, generated) in enumerate(zip(requests, ids, strict=True)):
            rng_after = sampler.generators[index].get_state() if sampler else None
            if r.start.get("rng_policy") == "CONTINUOUS_ORDINARY_V1":
                r.slot.ordinary_rng_state = rng_after
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            eos_present = bool(generated and generated[-1] in eos)
            interrupted = index in stop.lengths
            result = {
                **r.start,
                "input_ids": r.inputs,
                "status": "GENERATED",
                "generated_ids": generated,
                "generated_tokens": len(generated),
                "output_sha256": binding(generated),
                "raw": tokenizer.decode(generated, skip_special_tokens=False),
                "text": text,
                "eos": eos_present,
                "batch_id": self.sequence,
                "batch_row": index,
                "latency_s": None,
                "shared_batch_latency_s": elapsed,
                "finished_unix": finished,
                "cost_complete": True,
                "deadline_interrupted": interrupted,
                "physical_execution_started_unix": physical_started_unix,
                "physical_execution_deadline_unix": r.deadline,
                "queue_wait_s": physical_started_mono - r.start["queued_monotonic"]
                if "queued_monotonic" in r.start
                else None,
                "native_completed_unix": stop.completed_times.get(index),
                "rng_state_after_sha256": binding(rng_after.tolist())
                if rng_after is not None
                else None,
                "physical_batch_time_shared_nonadditive": True,
            }
            if kind == "typed":
                error = "PER_ROW_TOKEN_BOUNDARY_DEADLINE" if interrupted else None
                try:
                    r.grammar.validate(generated)
                except ValueError as exc:
                    error = str(exc)
                result.update(
                    typed_syntax_valid=error is None, typed_syntax_error=error
                )
            if kind == "guard":
                result["parsed"] = (
                    {"label": "Unknown", "status": "TOKEN_BOUNDARY_DEADLINE_REACHED"}
                    if interrupted
                    else parse_native(text, eos_present)
                )
            append(
                r.slot.out
                / ("guard_calls.jsonl" if kind == "guard" else "model_calls.jsonl"),
                result,
            )
            results.append(result)
        return results
