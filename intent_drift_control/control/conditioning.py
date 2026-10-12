"""Keep original ordinary prompts while retaining internal frame monitoring."""

from ..utils.hashing import binding
from ..utils.records import append
from .prompts import generation_messages


VERSION = "MONITOR_ONLY_FRAME_ORDINARY_GENERATOR_TRANSPARENCY_CANDIDATE_v1"


class OrdinaryTransparent:
    def __init__(self, engine, request, frame):
        self.engine, self.request, self.frame = (engine, request, frame)

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def call(self, messages, **kwargs):
        if kwargs.get("phase") == "ordinary_generator":
            if kwargs.get("thinking") is not True or messages != generation_messages(
                self.request, self.frame
            ):
                raise ValueError("Exact original full-loop ordinary messages required")
            actual = generation_messages(self.request)
            append(
                self.out / "ordinary_conditioning_delta.jsonl",
                {
                    "version": VERSION,
                    "case_id": self.case_id,
                    "frame_sha256_remains_monitor_dependency": self.frame.sha256,
                    "requested_Frame_prompt_sha256": binding(messages),
                    "actual_Base_prompt_sha256": binding(actual),
                    "accepted_ids_sha256": binding(list(kwargs.get("accepted", ()))),
                    "native_sampler_KV_RNG_backend_unchanged": True,
                    "no_dataset_kind_gold_or_expected_answer_routing": True,
                },
            )
            messages = actual
        return self.engine.call(messages, **kwargs)
