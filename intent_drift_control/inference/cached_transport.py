"""Batch native continuations using private exact-prefix host-cache leases."""

import threading
import time
from ..utils.hashing import binding
from ..utils.records import append
from ..utils.interfaces import ResourceEnded
from .cache import CacheLease, reusable, pack_cache, row_cache, copy_cache
from .batching import left_pad, RowStop, generated_row
from .sampling import IndependentRowSampler
from .vectorized import VectorizedTransport


VERSION = "EIGHT_ROW_EXACT_PREFIX_HOST_CACHE_CDF_v4"


class CachedTransport(VectorizedTransport):
    def __init__(self, *args, host_task_limit=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.host_task_limit = host_task_limit

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
        while len(self.pending) < self.host_task_limit:
            try:
                self.pending.append(self.queue.get_nowait())
            except Empty:
                break
        if len(self.pending) < self.maximum_batch:
            until = time.monotonic() + 0.02
            while len(self.pending) < self.host_task_limit and time.monotonic() < until:
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

        first_resume = first.kind == "generator" and reusable(
            first.slot, first.inputs, first.start["phase"]
        )
        for r in self.pending:
            candidate_resume = r.kind == "generator" and reusable(
                r.slot, r.inputs, r.start["phase"]
            )
            same_cache_mode = (
                first.kind != "generator" or first_resume == candidate_resume
            )
            width = max((len(q.inputs) for q in batch + [r]))
            work = (len(batch) + 1) * (width + r.start["max_new_tokens"])
            if (
                r.bucket == first.bucket
                and phase(r) == phase(first)
                and same_cache_mode
                and (len(batch) < limit)
                and (work <= self.pad_work_limit)
                and (
                    first_resume or (len(batch) + 1) * width <= self.prefill_work_limit
                )
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

    def generate(self, requests):
        if requests[0].kind != "generator":
            return super().generate(requests)
        if (
            threading.get_ident() != self.owner
            or not 1 <= len(requests) <= self.native_generator_batch_limit
        ):
            raise ValueError(
                "Only the registered owner may decode its bounded row batch"
            )
        if any((r.bucket != requests[0].bucket for r in requests)):
            raise ValueError("Incompatible modes cannot share generator work")
        from transformers import LogitsProcessorList, StoppingCriteriaList

        torch = self.torch
        model = self.engine.model
        tokenizer = self.engine.tokenizer
        eos = self.engine.eos_ids
        device = str(next(model.parameters()).device)
        leases = [getattr(r.slot, "native_pending_kv", None) for r in requests]
        resume = all((reusable(r.slot, r.inputs, r.start["phase"]) for r in requests))
        tensor, masks = left_pad(
            [r.inputs for r in requests], tokenizer.eos_token_id, torch, device
        )
        width = tensor.shape[1]
        kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "eos_token_id": sorted(eos),
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": requests[0].start["max_new_tokens"],
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        cache_lengths = [0] * len(requests)
        if resume:
            cache_lengths = [lease.past.get_seq_length() for lease in leases]
            packed, cache_width = pack_cache(leases, torch)
            if cache_width != width - 1:
                raise ValueError(
                    "Resumed batch must feed exactly its last emitted tokens"
                )
            kwargs["past_key_values"] = copy_cache(packed, device)
            del packed
        parent_calls = [lease.parent_call_id if resume else None for lease in leases]
        parent_batches = [lease.parent_batch_id if resume else None for lease in leases]
        for r in requests:
            r.slot.native_pending_kv = None
        del leases
        now = time.time()
        mono = time.monotonic()
        stopped = set()
        for r in requests:
            seconds = r.start.get("physical_call_budget_seconds", 3600)
            r.deadline = min(r.deadline, now + seconds)
            r.monotonic_deadline = min(r.monotonic_deadline, mono + seconds)
        stop = RowStop(torch, requests, width, stopped, eos_ids=eos)
        kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
        sampler = None
        if not requests[0].start.get("greedy", True):
            sampler = IndependentRowSampler(
                torch,
                [r.start["seed"] for r in requests],
                width,
                eos,
                tokenizer.eos_token_id,
                device=device,
                temperature=0.6,
                top_p=0.95,
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
        self.sequence += 1
        start = {
            "batch_id": self.sequence,
            "kind": "generator",
            "batch_size": len(requests),
            "members": [
                {
                    "episode_id": r.slot.episode_id,
                    "call_id": r.start["call_id"],
                    "input_sha256": binding(r.inputs),
                    "input_ids": r.inputs,
                    "seed": r.start["seed"],
                }
                for r in requests
            ],
            "started_unix": now,
            "padded_input_width": width,
            "cached_prefix_tokens_by_row": cache_lengths,
            "cache_strategy": "MASKED_EXACT_NATIVE_BATCH_RESUME"
            if resume
            else "FRESH_EXACT_BATCH_PREFILL",
            "logical_unpadded_prefill_tokens": sum(
                (len(r.inputs) - n for r, n in zip(requests, cache_lengths))
            ),
            "physical_padded_prefill_positions": len(requests)
            * (1 if resume else width),
            "attention_mask_sha256": binding(masks.tolist()),
            "max_new_tokens": kwargs["max_new_tokens"],
            "sampler": "GREEDY"
            if sampler is None
            else "INDEPENDENT_ROW_INVERSE_CDF_V1",
            "lease_storage_device": "cpu",
            "host_copy_preserves_dtype_and_values": True,
            "legacy_multinomial_bitwise_equivalence_claimed": False,
            "cross_row_BF16_token_equivalence_not_assumed": True,
            "backend_version": VERSION,
        }
        append(self.out / "physical_batch_starts.jsonl", start)
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
            suffixes = output.sequences[:, width:].tolist()
            ids = [
                generated_row(v, i, stop, eos, tokenizer.eos_token_id)
                for i, v in enumerate(suffixes)
            ]
            past = output.past_key_values
            if past.get_seq_length() != width + len(suffixes[0]) - 1:
                raise ValueError("Returned cache and actual output positions disagree")
            for index, (r, value) in enumerate(zip(requests, ids)):
                if (
                    value
                    and (not r.start.get("disable_native_resume", False))
                    and (r.start["phase"] == "ordinary_generator")
                    and (index not in stop.lengths)
                    and (value[-1] not in eos)
                ):
                    own = row_cache(
                        past,
                        index,
                        width - len(r.inputs),
                        len(r.inputs) + len(value) - 1,
                    )
                    r.slot.native_pending_kv = CacheLease(
                        r.slot.episode_id,
                        tuple(r.inputs + value),
                        copy_cache(own, "cpu", copy=True),
                        r.start["call_id"],
                        self.sequence,
                    )
        except BaseException as exc:
            for r in requests:
                r.slot.native_pending_kv = None
            append(
                self.out / "physical_batches.jsonl",
                {
                    **start,
                    "status": "TRANSPORT_ERROR",
                    "cost_complete": False,
                    "finished_unix": time.time(),
                    "shared_batch_latency_s": time.perf_counter() - tick,
                    "error": str(exc),
                },
            )
            raise
        elapsed = time.perf_counter() - tick
        finished = time.time()
        append(
            self.out / "physical_batches.jsonl",
            {
                **start,
                "status": "GENERATED",
                "cost_complete": True,
                "finished_unix": finished,
                "shared_batch_latency_s": elapsed,
                "generated_ids_by_row": ids,
                "logical_actual_generated_tokens": sum(map(len, ids)),
                "physical_padded_decode_positions": len(requests) * len(suffixes[0]),
                "structural_padding_decode_positions": len(requests) * len(suffixes[0])
                - sum(map(len, ids)),
                "deadline_stopped_lengths": stop.lengths,
                "native_completed_unix": stop.completed_times,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if device.startswith("cuda")
                else None,
                "rng_draw_counts": dict(sampler.draw_counts) if sampler else {},
                "shared_batch_time_not_additive_across_episode_rows": True,
            },
        )
        results = []
        for index, (r, value) in enumerate(zip(requests, ids)):
            rng = sampler.generators[index].get_state() if sampler else None
            if r.start.get("rng_policy") == "CONTINUOUS_ORDINARY_V1":
                r.slot.ordinary_rng_state = rng
            row = {
                **r.start,
                "input_ids": r.inputs,
                "status": "GENERATED",
                "batch_id": self.sequence,
                "batch_row": index,
                "generated_ids": value,
                "generated_tokens": len(value),
                "output_sha256": binding(value),
                "raw": tokenizer.decode(value, skip_special_tokens=False),
                "text": tokenizer.decode(value, skip_special_tokens=True).strip(),
                "eos": bool(value and value[-1] in eos),
                "deadline_interrupted": index in stop.lengths,
                "finished_unix": finished,
                "latency_s": None,
                "shared_batch_latency_s": elapsed,
                "cost_complete": True,
                "cached_prefix_tokens": cache_lengths[index],
                "actual_processed_input_tokens": len(r.inputs) - cache_lengths[index],
                "logical_conditioning_tokens": len(r.inputs),
                "cache_strategy": start["cache_strategy"],
                "cache_parent_call_id": parent_calls[index],
                "cache_parent_batch_id": parent_batches[index],
                "native_completed_unix": stop.completed_times.get(index),
                "queue_wait_s": mono - r.start["queued_monotonic"],
                "rng_state_after_sha256": binding(rng.tolist())
                if rng is not None
                else None,
                "sampling_backend": "GREEDY"
                if sampler is None
                else "INDEPENDENT_ROW_INVERSE_CDF_V1",
                "lease_storage_device": "cpu",
                "backend_version": VERSION,
            }
            append(r.slot.out / "model_calls.jsonl", row)
            results.append(row)
        return results
