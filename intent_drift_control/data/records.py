"""Read dataset records without loading or downloading a corpus."""

import json
from pathlib import Path


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
