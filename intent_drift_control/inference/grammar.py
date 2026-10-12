"""Finite JSON token language with explicit terminal declarations."""


class FiniteJsonGrammar:
    def __init__(self, tokenizer, options, eos_ids):
        self.options = tuple(options)
        if not self.options or len(set(self.options)) != len(self.options):
            raise ValueError("A nonempty distinct finite output language is required")
        self.eos_ids = tuple(sorted(set(eos_ids)))
        if not self.eos_ids:
            raise ValueError("Native EOS IDs are required")
        self.sequences = tuple(
            (tuple(tokenizer.encode(s, add_special_tokens=False)) for s in self.options)
        )
        if any((not s or any((t in self.eos_ids for t in s)) for s in self.sequences)):
            raise ValueError("Content cannot contain native EOS")
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("Distinct options must have distinct token encodings")
        self.children = {}
        self.terminals = set(self.sequences)
        for sequence in self.sequences:
            for i, token in enumerate(sequence):
                self.children.setdefault(sequence[:i], set()).add(token)
        self.maximum_tokens = max(map(len, self.sequences)) + 1

    def allowed(self, suffix):
        prefix = tuple(suffix)
        if prefix and prefix[-1] in self.eos_ids:
            if prefix[:-1] not in self.terminals:
                raise ValueError("EOS before a complete declaration")
            return list(self.eos_ids)
        choices = set(self.children.get(prefix, ()))
        if prefix in self.terminals:
            choices.update(self.eos_ids)
        if not choices:
            raise ValueError("Generated tokens left the finite language")
        return sorted(choices)

    def callback(self, input_length):
        def allowed_tokens(batch_id, input_ids):
            return self.allowed(input_ids.tolist()[input_length:])

        return allowed_tokens

    def validate(self, tokens):
        ids = tuple(tokens)
        if not ids or ids[-1] not in self.eos_ids or ids[:-1] not in self.terminals:
            raise ValueError("Missing EOS or incomplete typed declaration")
        return self.options[self.sequences.index(ids[:-1])]
