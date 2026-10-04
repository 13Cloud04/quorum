"""Passage store: 2.7 M passages kept as one UTF-8 blob plus an offset array, both
memory-mapped. Loading 2.7 M Python strings would cost about 3 GB of RAM; this costs
nothing until a passage is read.

Row r of the store is row r of the embedding matrix and label r in the vector index.
"""
from __future__ import annotations

import numpy as np

from . import DATA


class PassageStore:
    def __init__(self, prefix=DATA / "passages"):
        self.blob = np.memmap(f"{prefix}.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(f"{prefix}.offsets.npy", mmap_mode="r")   # n + 1 entries
        self.title_end = np.load(f"{prefix}.title_end.npy", mmap_mode="r")
        self.ids = np.load(f"{prefix}.ids.npy", mmap_mode="r")           # BEIR "docN" -> N

    def __len__(self):
        return len(self.offsets) - 1

    def _raw(self, r):
        return bytes(self.blob[self.offsets[r]:self.offsets[r + 1]])

    def title(self, r):
        return self._raw(r)[:self.title_end[r] - self.offsets[r]].decode()

    def text(self, r):
        return self._raw(r)[self.title_end[r] - self.offsets[r]:].decode()

    def get(self, r):
        raw = self._raw(r)
        cut = self.title_end[r] - self.offsets[r]
        return raw[:cut].decode(), raw[cut:].decode()


def build(parquet=DATA / "nq_corpus.parquet", prefix=DATA / "passages"):
    """One pass over the parquet file, one row group (about a million rows) at a time."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(parquet)
    n = f.metadata.num_rows
    offsets = np.zeros(n + 1, dtype=np.int64)
    title_end = np.zeros(n, dtype=np.int64)
    ids = np.zeros(n, dtype=np.int64)
    pos = row = 0
    with open(f"{prefix}.bin", "wb") as out:
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g)
            for did, title, text in zip(t["_id"].to_pylist(), t["title"].to_pylist(),
                                        t["text"].to_pylist()):
                tb, xb = title.encode(), text.encode()
                out.write(tb)
                out.write(xb)
                offsets[row] = pos
                title_end[row] = pos + len(tb)
                ids[row] = int(did[3:])            # "doc123" -> 123
                pos += len(tb) + len(xb)
                row += 1
    offsets[n] = pos
    assert row == n
    np.save(f"{prefix}.offsets.npy", offsets)
    np.save(f"{prefix}.title_end.npy", title_end)
    np.save(f"{prefix}.ids.npy", ids)
    return n
