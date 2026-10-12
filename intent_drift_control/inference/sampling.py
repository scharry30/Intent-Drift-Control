"""Independent inverse-CDF sampling under unchanged decoding warpers."""

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

        assert seeds and all((type(s) is int and s >= 0 for s in seeds))
        self.torch = torch
        self.input_width = input_width
        self.eos_ids = set(eos_ids)
        self.pad_id = pad_id
        self.stopped_rows = stopped_rows if stopped_rows is not None else set()
        self.finished_rows = set()
        self.generators = [
            torch.Generator(device=device).manual_seed(seed) for seed in seeds
        ]
        if rng_states is not None:
            assert len(rng_states) == len(seeds)
            for generator, state in zip(self.generators, rng_states, strict=True):
                if state is not None:
                    generator.set_state(state)
        self.warpers = (
            TemperatureLogitsWarper(temperature),
            TopKLogitsWarper(top_k),
            TopPLogitsWarper(top_p),
        )
        self.draw_counts = Counter()

    def __call__(self, input_ids, scores):
        torch = self.torch
        if scores.shape[0] != len(self.generators):
            raise ValueError("Batch membership changed inside decoding")
        warped = scores.float().clone()
        for warper in self.warpers:
            warped = warper(input_ids, warped)
        last = (
            input_ids[:, -1].tolist()
            if input_ids.shape[1] > self.input_width
            else [None] * len(self.generators)
        )
        active = []
        uniforms = []
        for row, generator in enumerate(self.generators):
            if last[row] in self.eos_ids:
                self.finished_rows.add(row)
            if row in self.stopped_rows or row in self.finished_rows:
                continue
            active.append(row)
            uniforms.append(torch.rand((), generator=generator, device=scores.device))
            self.draw_counts[row] += 1
        result = torch.full_like(warped, float("-inf"))
        tokens = torch.full(
            (len(self.generators),), self.pad_id, dtype=torch.long, device=scores.device
        )
        if active:
            index = torch.tensor(active, dtype=torch.long, device=scores.device)
            probabilities = torch.softmax(warped.index_select(0, index), dim=-1)
            cumulative = probabilities.cumsum(-1)
            cumulative = cumulative / cumulative[:, -1:]
            draws = torch.stack(uniforms).reshape(-1, 1)
            chosen = torch.searchsorted(
                cumulative.contiguous(), draws.contiguous(), right=True
            ).squeeze(-1)
            tokens.index_copy_(0, index, chosen.clamp_max(scores.shape[-1] - 1))
        result.scatter_(1, tokens[:, None], 0.0)
        return result
