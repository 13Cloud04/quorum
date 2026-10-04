"""Questions, gold answers and gold passages.

BEIR's NQ release has questions and the passages that answer them, but no short answers.
NQ-open (the same Google questions, used across the RAG literature) has the answers, so
the two are joined on normalised question text.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from functools import lru_cache

from . import DATA
from .answers import normalize


@dataclass
class Question:
    qid: str
    text: str
    answers: list                      # gold aliases
    gold_rows: list = field(default_factory=list)
    target: str | None = None          # the attacker's answer (attacked questions only)
    adv_texts: list = field(default_factory=list)   # PoisonedRAG's poison bodies


@lru_cache(maxsize=1)
def _beir():
    import pyarrow.parquet as pq

    q = pq.read_table(DATA / "nq_queries.parquet").to_pydict()
    queries = dict(zip(q["_id"], q["text"]))
    qrels = {}
    with open(DATA / "nq_qrels.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], []).append(int(row["corpus-id"][3:]))
    o = pq.read_table(DATA / "nq_open_validation.parquet").to_pydict()
    nq_open = {normalize(t): a for t, a in zip(o["question"], o["answer"])}
    return queries, qrels, nq_open


def targets():
    """PoisonedRAG's 100 attacked NQ questions with their attacker-chosen answers."""
    queries, qrels, nq_open = _beir()
    raw = json.loads((DATA / "poisonedrag_nq.json").read_text())
    out = []
    for qid, t in raw.items():
        answers = list(nq_open.get(normalize(t["question"]), []))
        if t["correct answer"] not in answers:
            answers.append(t["correct answer"])
        out.append(Question(qid, t["question"], answers, qrels.get(qid, []),
                            t["incorrect answer"], t["adv_texts"]))
    return out


def clean_split(name):
    """Unattacked questions with NQ-open answers, disjoint from the targets.
    'dev' (200) is the only data used to choose defence settings; 'test' (500) is
    reported."""
    queries, qrels, nq_open = _beir()
    skip = {t.qid for t in targets()}
    pool = sorted(qid for qid, text in queries.items()
                  if qid not in skip and normalize(text) in nq_open)
    random.Random(0).shuffle(pool)
    rows = {"dev": pool[:200], "test": pool[200:700]}[name]
    return [Question(q, queries[q], list(nq_open[normalize(queries[q])]), qrels.get(q, []))
            for q in rows]
