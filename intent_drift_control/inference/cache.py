"""Private exact-prefix cache leases and lossless per-row cache packing."""

from dataclasses import dataclass


@dataclass
class CacheLease:
    owner: str
    conditional_ids: tuple
    past: object
    parent_call_id: int | None = None
    parent_batch_id: int | None = None

    def accepts(self, owner, exact_inputs):
        return self.owner == owner and self.conditional_ids == tuple(exact_inputs)


def cache_action(slot, request):
    lease = getattr(slot, "native_pending_kv", None)
    ordinary = request.start["phase"] == "ordinary_generator"
    if (
        ordinary
        and lease is not None
        and lease.accepts(slot.episode_id, request.inputs)
    ):
        length = lease.past.get_seq_length()
        if length != len(request.inputs) - 1:
            raise ValueError(
                "Pending native KV must omit exactly the last emitted token"
            )
        return (lease.past, length, "RESUME_EXACT_NOW_ACCEPTED_ORDINARY_NATIVE_KV")
    slot.native_pending_kv = None
    return (
        None,
        0,
        "FRESH_AFTER_PREFIX_OR_PROMPT_CHANGE"
        if lease is not None
        else "FRESH_NATIVE_PREFILL",
    )


def reusable(slot, inputs, phase):
    lease = getattr(slot, "native_pending_kv", None)
    return (
        not getattr(slot, "disable_native_resume", False)
        and phase == "ordinary_generator"
        and (lease is not None)
        and lease.accepts(slot.episode_id, inputs)
        and (lease.past.get_seq_length() == len(inputs) - 1)
    )


def pack_cache(leases, torch):
    from transformers import DynamicCache

    legacy = [lease.past.to_legacy_cache() for lease in leases]
    if not legacy or len({len(x) for x in legacy}) != 1:
        raise ValueError("Cache layer counts must agree")
    width = max((lease.past.get_seq_length() for lease in leases))
    layers = []
    for index in range(len(legacy[0])):
        pairs = []
        for tensor_index in range(2):
            rows = [
                torch.nn.functional.pad(
                    value[index][tensor_index],
                    (0, 0, width - value[index][tensor_index].shape[-2], 0),
                )
                for value in legacy
            ]
            if any((row.shape[0] != 1 for row in rows)):
                raise ValueError("A row cache cannot contain another episode row")
            pairs.append(torch.cat(rows, dim=0))
        layers.append(tuple(pairs))
    return (DynamicCache.from_legacy_cache(tuple(layers)), width)


def row_cache(past, index, left_padding, logical_length):
    from transformers import DynamicCache

    if logical_length <= 0 or left_padding < 0:
        raise ValueError("A positive logical cache length is required")
    if left_padding + logical_length > past.get_seq_length():
        raise ValueError("Cannot manufacture uncomputed KV positions")
    return DynamicCache.from_legacy_cache(
        tuple(
            (
                tuple(
                    (
                        value[
                            index : index + 1,
                            :,
                            left_padding : left_padding + logical_length,
                            :,
                        ]
                        for value in layer
                    )
                )
                for layer in past.to_legacy_cache()
            )
        )
    )


def copy_cache(past, device, *, copy=False):
    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(
        tuple(
            (
                tuple((value.detach().to(device=device, copy=copy) for value in layer))
                for layer in past.to_legacy_cache()
            )
        )
    )
