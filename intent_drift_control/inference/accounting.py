"""Charge exact fresh-prefill work without inventing per-row timings."""

from ..utils.hashing import binding
from ..utils.records import append
from .transport import SharedTransport


VERSION = "PAPER_FULL_PUBLIC_FRESH_BATCHED_BACKEND_v2"


def accounted(row, request):
    if row["input_ids"] != request.inputs or row["input_sha256"] != binding(
        request.inputs
    ):
        raise ValueError("Actual row conditioning changed")
    if row["generated_tokens"] != len(row["generated_ids"]):
        raise ValueError("Actual output token accounting changed")
    return {
        **row,
        "actual_processed_input_tokens": len(request.inputs),
        "logical_conditioning_tokens": len(request.inputs),
        "cached_prefix_tokens": 0,
        "no_native_KV_lease_or_cross_episode_cache": True,
        "backend_version": VERSION,
    }


class FreshPrefillTransport(SharedTransport):
    def generate(self, requests):
        rows = super().generate(requests)
        if requests[0].kind != "generator":
            return rows
        result = []
        for row, request in zip(rows, requests, strict=True):
            value = accounted(row, request)
            append(
                request.slot.out / "fresh_prefill_account_receipts.jsonl",
                {
                    "call_id": row["call_id"],
                    "batch_id": row["batch_id"],
                    "batch_row": row["batch_row"],
                    "actual_input_sha256": row["input_sha256"],
                    "actual_processed_input_tokens": len(request.inputs),
                    "cached_prefix_tokens": 0,
                    "generated_tokens": row["generated_tokens"],
                    "physical_batch_latency_not_divided": True,
                    "backend_version": VERSION,
                },
            )
            result.append(value)
        return result
