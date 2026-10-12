"""Answer-only numeric and symbolic math scoring."""

import collections
import re


MATH_DATASETS = ("aime2024", "gsm8k", "math500")


def gold_answer(row):
    if row["dataset"] not in MATH_DATASETS:
        raise ValueError("Not a math dataset")
    answer = str(row["source_row"]["answer"])
    if row["dataset"] == "gsm8k":
        match = re.search("####\\s+(-?[0-9.,]+)", answer)
        if not match:
            raise ValueError("Invalid GSM gold: no official #### target")
        answer = match.group(1).replace(",", "")
    return answer.strip()


def score_math(row):
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    from math_verify.utils import timeout
    from sympy import simplify

    result = {
        "id": row["id"],
        "method": row["method"],
        "dataset": row["dataset"],
        "split": row["split"],
        "generator_valid": bool(row["engineering_valid"]),
        "correct": False,
        "status": "UNSCORED",
        "gold": gold_answer(row),
    }
    if not row["engineering_valid"]:
        return {**result, "status": "GENERATOR_INVALID"}
    answer = row["answer"]
    if not isinstance(answer, str) or not answer.strip():
        return {**result, "status": "EMPTY_FINAL"}
    if len(answer) > 100000:
        return {**result, "status": "ANSWER_LENGTH_LIMIT"}
    try:
        gold = parse(
            "$" + result["gold"] + "$",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=3,
            raise_on_error=True,
        )
        predicted = parse(
            answer,
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=3,
            raise_on_error=True,
        )
        result.update(
            gold_parsed=[str(x) for x in gold],
            predicted_parsed=[str(x) for x in predicted],
        )
        if not gold:
            return {**result, "status": "GOLD_PARSE_FAILED"}
        if not predicted:
            return {**result, "status": "PREDICTION_EXTRACTION_FAILED"}
        if row["dataset"] in ("aime2024", "gsm8k"):

            @timeout(3)
            def exact_numeric():
                return (
                    len(gold) == len(predicted) == 1
                    and (not predicted[0].free_symbols)
                    and bool(simplify(gold[0].doit() - predicted[0].doit()) == 0)
                )

            correct = exact_numeric()
        else:
            correct = verify(
                gold,
                predicted,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                timeout_seconds=3,
                raise_on_error=True,
            )
        return {
            **result,
            "correct": bool(correct),
            "status": "CORRECT" if correct else "INCORRECT",
        }
    except Exception as exc:
        return {
            **result,
            "status": "GRADER_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def aggregate_math(planned_keys, cases, scores):
    case_index = {c["id"]: c for c in cases}
    planned = {
        tuple(k) for k in planned_keys if case_index[k[0]]["dataset"] in MATH_DATASETS
    }
    score_index = {(r["id"], r["method"]): r for r in scores}
    if len(score_index) != len(scores):
        raise ValueError("Duplicate math scores")
    if set(score_index) - planned:
        raise ValueError("Unplanned math scores")
    groups = collections.defaultdict(list)
    for key in planned:
        c = case_index[key[0]]
        groups[c["dataset"], c["split"], key[1]].append(key)
    result = []
    for group, keys in sorted(groups.items()):
        observed = [score_index[k] for k in keys if k in score_index]
        counts = collections.Counter((r["status"] for r in observed))
        result.append(
            {
                "dataset": group[0],
                "split": group[1],
                "method": group[2],
                "planned_n": len(keys),
                "observed_n": len(observed),
                "pending_n": len(keys) - len(observed),
                "status_counts": dict(counts),
                "correct_n": counts["CORRECT"],
                "accuracy_all_planned_lower_bound": counts["CORRECT"] / len(keys),
                "accuracy_observed_all": counts["CORRECT"] / len(observed)
                if observed
                else None,
                "accuracy_valid_generator": counts["CORRECT"]
                / sum((r["generator_valid"] for r in observed))
                if any((r["generator_valid"] for r in observed))
                else None,
                "gold_parse_or_grader_error_n": counts["GOLD_PARSE_FAILED"]
                + counts["GRADER_ERROR"],
                "not_official_pdf_grader": True,
            }
        )
    return {
        "planned_math_n": len(planned),
        "scored_n": len(scores),
        "groups": result,
        "all_planned_scored": set(score_index) == planned,
    }
