"""Memory-mapped packed token blocks from uint16 shards."""

import json
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

import numpy as np


class TokenBlocks:
    """Fixed-length blocks over the concatenated shards listed in manifest.json."""

    def __init__(self, directory, sequence_length=4096):
        directory = Path(directory)
        shards = json.loads((directory / 'manifest.json').read_text())['shards']
        self.ends = list(accumulate(shard['tokens'] for shard in shards))
        self.token_count = self.ends[-1]
        self.sequence_length = sequence_length
        self.maps = [np.memmap(directory / shard['file'], mode='r', dtype='<u2') for shard in shards]

    def __len__(self):
        # Every full block also needs one next-token target.
        return (self.token_count - 1) // self.sequence_length

    def __getitem__(self, block):
        start = block * self.sequence_length
        remaining = self.sequence_length + 1
        shard = bisect_right(self.ends, start)
        offset = start - (self.ends[shard - 1] if shard else 0)
        chunks = []
        while remaining:
            part = self.maps[shard][offset:offset + remaining]
            chunks.append(part)
            remaining -= len(part)
            shard += 1
            offset = 0
        values = np.concatenate(chunks).astype(np.int64)
        return values[:-1], values[1:]
