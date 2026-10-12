"""Generator token partitions and matched latency summaries."""

import statistics


def token_count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Work counts must be nonnegative integers")
    return value


def charged_generator_tokens(row):
    tracked = token_count(row["generator_tokens"])
    cost = row["cost"]
    if tracked and "generator_tokens" not in cost:
        raise ValueError("Positive generated work needs the saved backend ledger")
    charged = token_count(cost.get("generator_tokens", 0))
    if charged < tracked:
        raise ValueError("Backend generated work is below completed-call work")
    return charged


def generator_partition(row):
    generated = charged_generator_tokens(row)
    tracked = row["generator_tokens"]
    discarded = token_count(row["discarded_tokens"])
    accepted = row.get("accepted_ids")
    if accepted is None and generated == 0:
        accepted = []
    if not isinstance(accepted, list):
        raise ValueError("Positive work needs an explicit accepted prefix")
    committed = len(accepted)
    if committed + discarded > tracked:
        raise ValueError("Invalid raw generator partition")
    return {
        "generator": generated,
        "committed": committed,
        "discarded": discarded,
        "uncommitted": generated - committed - discarded,
        "controller_tracked": tracked,
        "interrupted_call_delta": generated - tracked,
    }


def quantile(values, q):
    if not values:
        return None
    v = sorted(values)
    index = (len(v) - 1) * q
    low = int(index)
    return v[low] + (v[min(low + 1, len(v) - 1)] - v[low]) * (index - low)


def costs(rows):
    by = {
        (r["id"], r["method"]): r
        for r in rows
        if r["study"] == "MAIN" and r["method"] in ("base", "idc")
    }
    common = {i for i, m in by if (i, "base") in by and (i, "idc") in by}
    valid = {
        i
        for i in common
        if all(
            (
                not by[i, m].get("engineering_failure")
                and by[i, m].get("endpoint_wall_s_before_evaluation") is not None
                for m in ("base", "idc")
            )
        )
    }
    result = []
    for method in ("base", "idc"):
        selected = [by[i, method] for i in sorted(valid)]
        record = {
            "method": method,
            "matched_returned_pairs": len(common),
            "cost_complete_pairs": len(valid),
            "excluded_pairs_unknown_cost": len(common - valid),
            "queue_inclusive_not_isolated_latency": True,
            "shared_frame_standalone_token_charge": True,
            "endpoint_evaluator_excluded": True,
        }
        for field in (
            "generator_tokens",
            "auxiliary_tokens",
            "prefill_tokens",
            "calls",
        ):
            values = [
                r.get("cost", {}).get(field, 0) + r.get("frame_cost", {}).get(field, 0)
                for r in selected
            ]
            record[field + "_mean"] = statistics.mean(values) if values else None
        walls = [r["endpoint_wall_s_before_evaluation"] for r in selected]
        record.update(
            latency_median_s=quantile(walls, 0.5), latency_p95_s=quantile(walls, 0.95)
        )
        result.append(record)
    return result
