"""Dense retrieval over the 2.68 M-passage index, built with vecdb (my own HNSW library,
../vecdb). Poison passages are inserted into the same index the way an attacker's
documents would be ingested, with labels starting at the corpus size so they can be
told apart when scoring. The defences never look at labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import DATA
from .embed import DIM

INDEX_PATH = DATA / "nq.vecdb"
M, EF_CONSTRUCTION = 16, 100
EF_SEARCH = 200


@dataclass
class Hit:
    label: int
    score: float          # cosine similarity to the question


def corpus_vectors(n):
    return np.memmap(DATA / "emb.f32", dtype=np.float32, mode="r", shape=(n, DIM))


def build(n, threads=0, poison_room=10_000):
    import vecdb

    ix = vecdb.Index(dim=DIM, max_elements=n + poison_room, metric="ip", M=M,
                     ef_construction=EF_CONSTRUCTION)
    vecs = corpus_vectors(n)
    step = 1 << 18
    for s in range(0, n, step):
        ix.add(np.ascontiguousarray(vecs[s:s + step]),
               labels=np.arange(s, min(s + step, n), dtype=np.uint64), threads=threads)
    ix.save(str(INDEX_PATH))
    return ix


class Retriever:
    def __init__(self, n_corpus, index=None):
        import vecdb

        self.n = n_corpus
        self.ix = index or vecdb.Index.load(str(INDEX_PATH))
        self.vecs = corpus_vectors(n_corpus)
        self.poison_vecs = np.zeros((0, DIM), np.float32)

    def add_poisons(self, vectors):
        """Insert poison passages; they get labels n, n+1, ... in insertion order."""
        start = self.n + len(self.poison_vecs)
        labels = np.arange(start, start + len(vectors), dtype=np.uint64)
        self.ix.add(np.ascontiguousarray(vectors, dtype=np.float32), labels=labels)
        self.poison_vecs = np.vstack([self.poison_vecs, vectors])
        return labels

    def vector(self, label):
        return self.vecs[label] if label < self.n else self.poison_vecs[label - self.n]

    def search(self, qvecs, k=20, ef=EF_SEARCH):
        labels, dist = self.ix.search(np.ascontiguousarray(qvecs, dtype=np.float32), k=k,
                                      ef=max(ef, k), threads=0)
        return [[Hit(int(l), 1.0 - float(d)) for l, d in zip(lr, dr) if l >= 0]
                for lr, dr in zip(labels, dist)]
