"""Native thinking-token views and exact accepted-prefix terminal protocol."""

import json
from ..utils.records import append
from ..utils.json import normalize_json_text


class NativeState:
    def prompt_ids(self, messages, *, thinking, accepted=()):
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        if thinking:
            marker = "<|im_start|>assistant\n"
            if marker in rendered:
                suffix = rendered.rsplit(marker, 1)[-1]
                if not suffix:
                    rendered += "<think>\n"
                elif suffix.strip() != "<think>":
                    raise ValueError("Unexpected native thinking prompt")
            elif rendered.endswith("<think>\n") or rendered.endswith("<think>"):
                pass
            else:
                raise ValueError("Unexpected native thinking prompt")
        elif rendered.endswith("<think>\n"):
            rendered += "</think>\n"
        elif rendered.endswith("<think>"):
            rendered += "</think>\n"
        return self.tokenizer.encode(rendered, add_special_tokens=False) + list(
            accepted
        )

    def json_call(self, system, payload, *, phase, budget, seed):
        output_contract = "\nReturn only the value for the schema. Do not echo R0, P0, source_units, original_obligations, response_schema, or these instructions. Start with { and end with }. Output JSON only, with no explanation."
        row = self.call(
            [
                {"role": "system", "content": system + output_contract},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            phase=phase,
            thinking=False,
            budget=budget,
            seed=seed,
            greedy=False,
        )
        try:
            if not row["eos"]:
                raise ValueError("Auxiliary response unfinished at cap")
            value, normalized_outer_braces = normalize_json_text(row["text"])
            obj = json.loads(value)
            if not isinstance(obj, dict):
                raise ValueError("Expected one JSON object")
        except (ValueError, TypeError) as exc:
            append(
                self.out / "auxiliary_results.jsonl",
                {"call_id": row["call_id"], "status": "INVALID", "error": str(exc)},
            )
            raise ValueError(str(exc)) from exc
        append(
            self.out / "auxiliary_results.jsonl",
            {
                "call_id": row["call_id"],
                "status": "PARSED",
                "object": obj,
                "normalized_outer_braces": normalized_outer_braces,
            },
        )
        return (obj, row)

    def decode(self, accepted):
        ids = list(accepted)
        if self.end_think in ids:
            split = ids.index(self.end_think)
            return {
                "thinking": self.tokenizer.decode(
                    ids[:split], skip_special_tokens=True
                ).strip(),
                "answer": self.tokenizer.decode(
                    ids[split + 1 :], skip_special_tokens=True
                ).strip(),
                "reasoning_closed": True,
            }
        return {
            "thinking": self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "answer": "",
            "reasoning_closed": False,
        }

    def protocol(self, accepted, candidate):
        prefix = tuple(accepted)
        ids = tuple(candidate)
        combined = prefix + ids
        closure = bool(ids and ids[-1] in self.eos_ids)
        valid = (
            bool(ids)
            and self.start_think not in combined
            and (combined.count(self.end_think) <= 1)
            and (not any((t in self.eos_ids for t in combined[:-1])))
        )
        decoded = self.decode(combined)
        ready = (
            valid
            and closure
            and decoded["reasoning_closed"]
            and bool(decoded["answer"])
        )
        if self.end_think in prefix:
            new_reasoning = ""
        elif self.end_think in ids:
            new_reasoning = self.tokenizer.decode(
                ids[: ids.index(self.end_think)], skip_special_tokens=True
            )
        else:
            new_reasoning = self.tokenizer.decode(ids, skip_special_tokens=True)
        delivery = (
            new_reasoning + "\n\n" + decoded["answer"]
            if closure
            else self.tokenizer.decode(ids, skip_special_tokens=False)
        )
        return {
            "valid": valid,
            "closure": closure,
            "terminal_ready": ready,
            "new_delivery": delivery,
            **decoded,
        }
