"""Private episode state sharing externally supplied model instances."""

from collections import Counter
from pathlib import Path
import time
from ..utils.hashing import binding
from ..utils.records import append
from ..utils.interfaces import ResourceEnded, WindowEnded
from .native_state import NativeState
from .batching import Request
from .grammar import FiniteJsonGrammar


class Episode(NativeState):
    def __init__(
        self,
        shared,
        output,
        case_id,
        role,
        episode_id,
        *,
        wall_seconds=3600,
        ordinary_rng_policy="CONTINUOUS_ORDINARY_V1",
    ):
        base = shared.engine
        self.shared = shared
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.torch = base.torch
        self.tokenizer = base.tokenizer
        self.model = base.model
        self.start_think = base.start_think
        self.end_think = base.end_think
        self.eos_ids = base.eos_ids
        self.case_id = case_id
        self.role = role
        self.episode_id = episode_id
        self.sequence = 0
        self.deadline = base.deadline
        self.episode_deadline = min(self.deadline, time.time() + wall_seconds)
        self.monotonic_deadline = time.monotonic() + min(
            wall_seconds, max(0.0, self.deadline - time.time())
        )
        self.episode_cost = Counter()
        self.external_cost = Counter()
        if ordinary_rng_policy not in (
            "CONTINUOUS_ORDINARY_V1",
            "INDEPENDENT_PHASE_TICKETS_V1",
        ):
            raise ValueError("Explicit registered RNG policy required")
        self.ordinary_rng_policy = ordinary_rng_policy
        self.ordinary_rng_state = None
        self.ordinary_initial_seed = None

    def check(self):
        if time.time() >= self.deadline:
            raise WindowEnded("Frozen fifteen-hour authorization ended")
        if (
            time.time() >= self.episode_deadline
            or time.monotonic() >= self.monotonic_deadline
        ):
            raise ResourceEnded("Per-episode wall budget ended")

    def reserve(self, tokens, budget, *, auxiliary):
        self.check()
        if (
            self.episode_cost["prefill_tokens"]
            + self.external_cost["prefill_tokens"]
            + tokens
            > 2000000
        ):
            raise ResourceEnded("Combined logical input budget exhausted")
        if (
            auxiliary
            and self.episode_cost["auxiliary_tokens"]
            + self.external_cost["auxiliary_tokens"]
            + budget
            > 131072
        ):
            raise ResourceEnded("Combined auxiliary output budget exhausted")

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
        if type(budget) is not int or budget <= 0:
            raise ResourceEnded("Positive integer call cap required")
        model = self.shared.guard.model if kind == "guard" else self.model
        if len(inputs) + budget > min(32768, model.config.max_position_embeddings):
            raise ResourceEnded("No silent context truncation")
        deployment = kind != "guard" or purpose == "deployment"
        if deployment:
            self.reserve(len(inputs), budget, auxiliary=not thinking)
        else:
            self.check()
        self.sequence += 1
        requested_seed = seed
        continuous = (
            not greedy
            and phase in ("ordinary_generator", "native_generator")
            and (self.ordinary_rng_policy == "CONTINUOUS_ORDINARY_V1")
        )
        if continuous:
            if self.ordinary_initial_seed is None:
                self.ordinary_initial_seed = seed
            seed = self.ordinary_initial_seed
        now = time.time()
        mono = time.monotonic()
        deadline = self.episode_deadline
        monotonic_deadline = min(
            self.monotonic_deadline, mono + max(0.0, deadline - now)
        )
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
            "deadline_unix": deadline,
            "cache_strategy": "FULL_EXACT_PREFILL_NO_FOREIGN_OR_REJECTED_KV",
            "queued_monotonic": mono,
            "physical_call_budget_seconds": 180 if kind in ("guard", "typed") else 3600,
            "temperature": None if greedy else 0.6 if thinking else 0.7,
            "top_p": None if greedy else 0.95 if thinking else 0.8,
            "top_k": None if greedy else 20,
            "purpose": purpose,
            "controller_feedback": purpose == "deployment" if kind == "guard" else None,
        }
        start.update(
            rng_policy="CONTINUOUS_ORDINARY_V1"
            if continuous
            else "INDEPENDENT_PHASE_TICKETS_V1",
            requested_legacy_phase_seed=requested_seed,
            seed_is_initial_stream_seed=continuous,
            rng_state_before_sha256=binding(self.ordinary_rng_state.tolist())
            if continuous and self.ordinary_rng_state is not None
            else None,
        )
        if grammar:
            start.update(
                output_options=list(grammar.options),
                output_options_sha256=binding(grammar.options),
                constraint="FINITE_CANONICAL_JSON_PREFIX_TRIE_NOT_SEMANTIC_GOLD",
            )
        append(
            self.out
            / ("guard_call_starts.jsonl" if kind == "guard" else "call_starts.jsonl"),
            start,
        )
        result = self.shared.submit(
            Request(self, kind, start, inputs, deadline, monotonic_deadline, grammar)
        )
        if deployment:
            account = self.external_cost if kind == "guard" else self.episode_cost
            account.update(calls=1, prefill_tokens=len(inputs))
            account.update(
                {
                    "generator_tokens" if thinking else "auxiliary_tokens": result[
                        "generated_tokens"
                    ]
                }
            )
        self.check()
        return result

    def call(
        self, messages, *, phase, thinking, budget, seed, accepted=(), greedy=False
    ):
        return self._request(
            "generator",
            messages,
            inputs=self.prompt_ids(messages, thinking=thinking, accepted=accepted),
            phase=phase,
            thinking=thinking,
            budget=budget,
            seed=seed,
            accepted=accepted,
            greedy=greedy,
        )

    def typed_call(self, messages, options, *, phase, seed):
        grammar = FiniteJsonGrammar(self.tokenizer, options, self.eos_ids)
        return self._request(
            "typed",
            messages,
            inputs=self.prompt_ids(messages, thinking=False),
            thinking=False,
            phase=phase,
            budget=grammar.maximum_tokens,
            seed=seed,
            grammar=grammar,
        )
