"""Independent per-row multinomial sampling for the original transport."""

from collections import Counter


class IndependentRowSampler:
    def __init__(
        self,
        torch,
        seeds,
        input_width,
        eos_ids,
        pad_id,
        *,
        device,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        stopped_rows=None,
        rng_states=None,
    ):
        from transformers import (
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        if not seeds or any((type(s) is not int or s < 0 for s in seeds)):
            raise ValueError(
                "Explicit nonnegative integer per-row seed tickets required"
            )
        self.torch = torch
        self.input_width = input_width
        self.eos_ids = set(eos_ids)
        self.pad_id = pad_id
        self.stopped_rows = stopped_rows if stopped_rows is not None else set()
        self.generators = [
            torch.Generator(device=device).manual_seed(seed) for seed in seeds
        ]
        if rng_states is not None:
            if len(rng_states) != len(seeds):
                raise ValueError("One optional private RNG state per row is required")
            for generator, state in zip(self.generators, rng_states, strict=True):
                if state is not None:
                    generator.set_state(state)
        self.warpers = (
            TemperatureLogitsWarper(temperature),
            TopKLogitsWarper(top_k),
            TopPLogitsWarper(top_p),
        )
        self.draw_counts = Counter()
        self.finished_rows = set()

    def __call__(self, input_ids, scores):
        if scores.shape[0] != len(self.generators):
            raise ValueError("Batch membership changed inside a generation call")
        warped = scores.float().clone()
        for transform in self.warpers:
            warped = transform(input_ids, warped)
        result = self.torch.full_like(warped, float("-inf"))
        last_tokens = (
            input_ids[:, -1].tolist()
            if input_ids.shape[1] > self.input_width
            else [None] * len(self.generators)
        )
        for row, generator in enumerate(self.generators):
            if last_tokens[row] in self.eos_ids:
                self.finished_rows.add(row)
            if row in self.stopped_rows or row in self.finished_rows:
                token = self.pad_id
            else:
                probabilities = self.torch.softmax(warped[row : row + 1], dim=-1)
                token = int(
                    self.torch.multinomial(probabilities, 1, generator=generator).item()
                )
                self.draw_counts[row] += 1
            result[row, token] = 0.0
        return result
