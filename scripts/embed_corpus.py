"""Embed all 2.68 M passages into data/emb.f32 (float32 [n, 384], 4.1 GB).

About an hour on an M5. Resumable: chunks already written are recorded in
data/emb.done and skipped, so the job can be stopped and restarted.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import DATA                      # noqa: E402
from quorum.embed import DIM, Embedder       # noqa: E402
from quorum.store import PassageStore        # noqa: E402

CHUNK = 1 << 16


def main():
    store = PassageStore()
    n = len(store)
    path, done_path = DATA / "emb.f32", DATA / "emb.done"
    mode = "r+" if path.exists() else "w+"
    out = np.memmap(path, dtype=np.float32, mode=mode, shape=(n, DIM))
    done = set(map(int, done_path.read_text().split())) if done_path.exists() else set()
    emb = Embedder()
    t0, rows = time.time(), 0
    for c, start in enumerate(range(0, n, CHUNK)):
        if c in done:
            continue
        stop = min(start + CHUNK, n)
        out[start:stop] = emb.passages(store.get(r) for r in range(start, stop))
        out.flush()
        with open(done_path, "a") as f:
            f.write(f"{c}\n")
        rows += stop - start
        rate = rows / (time.time() - t0)
        left = (n - stop) / rate / 60
        print(f"chunk {c:3d}  rows {stop:>9,}  {rate:6.0f}/s  ~{left:4.0f} min left", flush=True)


if __name__ == "__main__":
    main()
