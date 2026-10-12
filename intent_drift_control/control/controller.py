"""Accepted-prefix execution with monitored proposals and bounded repair."""

import random
import time
from ..core import Assessment, binding, select_pool
from ..config import OperatingPoint
from ..config.defaults import GENERATOR_BUDGET
from ..utils.records import append
from ..utils.random import seed_ticket
from ..utils.interfaces import ResourceEnded, WindowEnded
from ..framing.builder import build_frame
from .prompts import generation_messages
from .scoring import assess, judgment_key
from .repair import guidance


def execute(
    engine,
    request,
    role,
    master,
    *,
    frame=None,
    point=None,
    law="projection",
    assessor=None,
):
    point = point or OperatingPoint()
    if role != "idc":
        raise ValueError("Only the intent-drift controller is included")
    score, score_key = (assess, judgment_key)
    if assessor is not None:
        if (
            role != "idc"
            or not callable(assessor)
            or (not callable(getattr(assessor, "judgment_key", None)))
        ):
            raise ValueError(
                "Explicit Intent Drift Control assessor must provide evaluation and judgment_key"
            )
        score_key = assessor.judgment_key

        def score(engine, request, frame, before, ids, master, memo):
            key, protocol = score_key(engine, request, frame, before, ids)
            original, host_protocol = judgment_key(engine, request, frame, before, ids)
            if protocol != host_protocol or any(
                (
                    key.get(k) != original[k]
                    for k in (
                        "request",
                        "policy_sha256",
                        "frame_sha256",
                        "accepted_ids",
                        "candidate_ids",
                        "mode",
                    )
                )
            ):
                raise ValueError(
                    "Custom score key changed the exact request/frame/prefix/protocol"
                )
            value = assessor(engine, request, frame, before, ids, master, memo)
            if value is not None and (
                not isinstance(value, Assessment)
                or value.binding_sha256 != binding(key)
                or value.closure != host_protocol["closure"]
                or (value.terminal_ready != host_protocol["terminal_ready"])
            ):
                raise ValueError(
                    "Custom assessment does not bind its actual key and native mode"
                )
            return value

    accepted = []
    work = 0
    discarded = 0
    events = []
    memo = {}
    terminal = "ENGINEERING_INVALID"
    started = time.time()
    try:
        if frame is None:
            frame = build_frame(engine, request, master)
        if frame is not None and (not frame.positive):
            return {
                "thinking": "",
                "answer": "I cannot assist with the prohibited request.",
                "engineering_valid": False,
                "policy_reply": True,
                "stop_reason": "EMPTY_POSITIVE_POLICY",
                "accepted_ids": [],
                "generator_tokens": 0,
                "discarded_tokens": 0,
                "frame_sha256": frame.sha256,
                "events": [],
                "cost": dict(engine.episode_cost),
            }
        for state in range(64):
            remaining = GENERATOR_BUDGET - work
            if remaining <= 0:
                terminal = "GENERATOR_BUDGET"
                break
            before = tuple(accepted)
            ordinary = engine.call(
                generation_messages(request, frame),
                phase="ordinary_generator",
                thinking=True,
                budget=min(256, remaining),
                accepted=before,
                seed=seed_ticket(master, request, "ordinary", state),
                greedy=False,
            )
            ids = tuple(ordinary["generated_ids"])
            work += len(ids)
            rejected = score(engine, request, frame, before, ids, master, memo)
            verdict = point.verdict(rejected)
            event = {
                "state": state,
                "prefix_before": list(before),
                "ordinary_call_id": ordinary["call_id"],
                "candidate_ids": list(ids),
                "verdict": verdict,
                "repair_pools": [],
            }
            selected = ids if verdict["accept"] else None
            if selected is None:
                discarded += len(ids)
                for pool_id in range(2):
                    draws = []
                    call_ids = []
                    for slot in range(4):
                        remaining = GENERATOR_BUDGET - work
                        if remaining <= 0:
                            break
                        record = engine.call(
                            generation_messages(
                                request, frame, guidance(frame, rejected)
                            ),
                            phase="repair_generator",
                            thinking=True,
                            budget=min(256, remaining),
                            accepted=before,
                            seed=seed_ticket(
                                master, request, "repair", state, pool_id, slot
                            ),
                            greedy=False,
                        )
                        candidate = tuple(record["generated_ids"])
                        draws.append(candidate)
                        call_ids.append(record["call_id"])
                        work += len(candidate)
                    if len(draws) != 4:
                        discarded += sum(map(len, draws))
                        terminal = "INCOMPLETE_POOL_BUDGET"
                        event["repair_pools"].append(
                            {
                                "pool": pool_id,
                                "call_ids": call_ids,
                                "draws": [list(d) for d in draws],
                                "status": "INCOMPLETE_NOT_SELECTED",
                            }
                        )
                        break
                    scores = {
                        d: score(engine, request, frame, before, d, master, memo)
                        for d in sorted(set(draws))
                    }
                    uniform = random.Random(
                        seed_ticket(master, request, "selector", state, pool_id)
                    ).random()
                    choice = select_pool(
                        draws, scores, rejected, point, uniform, 0.5, law
                    )
                    event["repair_pools"].append(
                        {
                            "pool": pool_id,
                            "call_ids": call_ids,
                            "draws": [list(d) for d in draws],
                            "selection": choice,
                        }
                    )
                    selected = (
                        tuple(choice["selected_ids"])
                        if choice["selected_ids"] is not None
                        else None
                    )
                    discarded += sum(map(len, draws)) - (
                        len(selected) if selected else 0
                    )
                    if selected is not None:
                        selected_score = scores[selected]
                        key, _ = score_key(engine, request, frame, before, selected)
                        if selected_score.binding_sha256 != binding(key):
                            raise ValueError(
                                "Commit binding changed; no fresh semantic reassessment permitted"
                            )
                        break
            if selected is None:
                terminal = (
                    terminal
                    if terminal == "INCOMPLETE_POOL_BUDGET"
                    else "NO_ADMISSIBLE_REPAIR"
                )
                event["prefix_after"] = list(before)
                event["committed"] = False
                events.append(event)
                append(
                    engine.out / "control_events.jsonl",
                    {"case_id": engine.case_id, **event},
                )
                break
            accepted.extend(selected)
            event["prefix_after"] = list(accepted)
            event["selected_ids"] = list(selected)
            event["committed"] = True
            events.append(event)
            append(
                engine.out / "control_events.jsonl",
                {"case_id": engine.case_id, **event},
            )
            if engine.protocol(before, selected)["closure"]:
                terminal = "EOS"
                break
        else:
            terminal = "COMMIT_CAP"
    except WindowEnded:
        raise
    except (ValueError, ResourceEnded) as exc:
        terminal = f"{type(exc).__name__}: {exc}"
    decoded = engine.decode(accepted)
    valid = (
        terminal == "EOS" and decoded["reasoning_closed"] and bool(decoded["answer"])
    )
    return {
        **decoded,
        "answer": decoded["answer"] if valid else "",
        "unserved_answer": "" if valid else decoded["answer"],
        "engineering_valid": bool(valid),
        "policy_reply": False,
        "stop_reason": terminal,
        "accepted_ids": accepted,
        "generator_tokens": work,
        "discarded_tokens": discarded,
        "frame_sha256": frame.sha256 if frame else None,
        "events": events,
        "cost": dict(engine.episode_cost),
        "pipeline_latency_s": time.time() - started,
        "estimated_completion_not_independent_H": role == "idc",
    }
