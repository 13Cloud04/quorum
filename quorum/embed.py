"""Dense embeddings with BAAI/bge-small-en-v1.5 (33 M parameters, 384 dimensions) on the
Apple GPU through PyTorch's MPS backend. Vectors are L2-normalised, so cosine similarity
is a dot product.

bge expects an instruction in front of queries but not in front of passages. Poison
passages are embedded exactly like corpus passages, as the attacker's text would be.
"""
from __future__ import annotations

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 256     # passages average about 110 tokens; under 1% are cut


def passage_input(title, text):
    return f"{title} {text}".strip()


class Embedder:
    def __init__(self, device=None):
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(MODEL, device=device)
        self.model.max_seq_length = MAX_TOKENS
        if device == "mps":
            self.model.half()

    def _encode(self, texts, batch_size):
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)

    def queries(self, questions, batch_size=128):
        return self._encode([QUERY_PREFIX + q for q in questions], batch_size)

    def passages(self, pairs, batch_size=128):
        """pairs: iterable of (title, text)."""
        return self._encode([passage_input(t, x) for t, x in pairs], batch_size)
