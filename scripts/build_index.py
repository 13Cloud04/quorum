"""Build the HNSW index over all 2.68 M passage vectors with vecdb, then check its
recall@20 against exact search for 200 dev questions."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, retrieve                 # noqa: E402
from quorum.embed import Embedder                 # noqa: E402
from quorum.store import PassageStore             # noqa: E402


def main():
    n = len(PassageStore())
    if "--check-only" in sys.argv:
        import vecdb
        ix = vecdb.Index.load(str(retrieve.INDEX_PATH))
    else:
        t = time.time()
        ix = retrieve.build(n)
        print(f"built {n:,} vectors in {time.time() - t:.0f}s; "
              f"{ix.memory_bytes / 2**30:.2f} GiB in memory", flush=True)
    qv = Embedder().queries([q.text for q in data.clean_split("dev")])
    exact = ix.brute_force(qv, k=20)        # labels only
    for ef in (50, 100, 200):
        got, _ = ix.search(qv, k=20, ef=ef, threads=0)
        recall = np.mean([len(set(a) & set(b)) / 20 for a, b in zip(got, exact)])
        print(f"ef={ef}: recall@20 vs exact {recall:.4f}")


if __name__ == "__main__":
    main()
