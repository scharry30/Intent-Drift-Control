"""Charge physical native prefill and resume only private accepted state."""

import time
from ..utils.hashing import binding
from ..utils.records import append
from ..utils.interfaces import ResourceEnded
from .episode import Episode
from .batching import Request


class NativeEpisode(Episode):
    def _request(
        self,
        kind,
        messages,
        *,
        inputs,
        thinking,
        phase,
        budget,
        seed,
        accepted=(),
        greedy=True,
        grammar=None,
        purpose=None,
    ):
        if kind != "generator":
            return super()._request(
                kind,
                messages,
                inputs=inputs,
                thinking=thinking,
                phase=phase,
                budget=budget,
                seed=seed,
                accepted=accepted,
                greedy=greedy,
                grammar=grammar,
                purpose=purpose,
            )
        if type(budget) is not int or budget <= 0:
            raise ResourceEnded("Positive integer call cap required")
        if len(inputs) + budget > min(32768, self.model.config.max_position_embeddings):
            raise ResourceEnded("No silent native session truncation")
        self.reserve(len(inputs), budget, auxiliary=not thinking)
        self.sequence += 1
        requested_seed = seed
        continuous = not greedy and phase in ("ordinary_generator", "native_generator")
        if continuous:
            if self.ordinary_initial_seed is None:
                self.ordinary_initial_seed = seed
            seed = self.ordinary_initial_seed
        now = time.time()
        mono = time.monotonic()
        start = {
            "call_id": self.sequence,
            "episode_id": self.episode_id,
            "case_id": self.case_id,
            "role": self.role,
            "kind": kind,
            "phase": phase,
            "thinking": thinking,
            "greedy": greedy,
            "seed": seed,
            "messages": messages,
            "accepted_ids": list(accepted),
            "input_tokens": len(inputs),
            "input_sha256": binding(inputs),
            "max_new_tokens": budget,
            "started_unix": now,
            "deadline_unix": self.episode_deadline,
            "queued_monotonic": mono,
            "physical_call_budget_seconds": 3600,
            "cache_strategy": "EXACT_ACCEPTED_NATIVE_SESSION_OR_FRESH_IF_NOT_IDENTICAL",
            "prefill_budget_reservation_full_context_upper_bound": len(inputs),
            "temperature": None if greedy else 0.6 if thinking else 0.7,
            "top_p": None if greedy else 0.95 if thinking else 0.8,
            "top_k": None if greedy else 20,
            "rng_policy": "CONTINUOUS_ORDINARY_V1"
            if continuous
            else "INDEPENDENT_PHASE_TICKETS_V1",
            "requested_legacy_phase_seed": requested_seed,
            "seed_is_initial_stream_seed": continuous,
            "rng_state_before_sha256": binding(self.ordinary_rng_state.tolist())
            if continuous and self.ordinary_rng_state is not None
            else None,
            "purpose": purpose,
            "controller_feedback": None,
        }
        append(self.out / "call_starts.jsonl", start)
        result = self.shared.submit(
            Request(
                self,
                kind,
                start,
                inputs,
                self.episode_deadline,
                self.monotonic_deadline,
            )
        )
        self.episode_cost.update(
            calls=1,
            prefill_tokens=result["actual_processed_input_tokens"],
            logical_conditioning_tokens=len(inputs),
            native_cached_prefix_tokens=result["cached_prefix_tokens"],
        )
        self.episode_cost.update(
            {
                "generator_tokens" if thinking else "auxiliary_tokens": result[
                    "generated_tokens"
                ]
            }
        )
        self.check()
        return result
