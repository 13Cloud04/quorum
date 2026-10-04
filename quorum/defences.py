"""Defences. Each takes a question and its ranked passages and returns an Outcome.

    vanilla        PoisonedRAG's setting: top-k passages in one prompt.
    guard          Plain RAG's answer, but refuse when that answer rests only on passages
                   that are near-copies of each other. (Designed after the first results;
                   validated on a held-out attack set, see README.)
    isolate_vote   RobustRAG's idea (Xiang et al., 2024): answer from each passage on its
                   own, then take the most common answer.
    quorum         This project: filter passages that carry the question or an instruction,
                   answer from each passage alone, merge near-duplicate passages into one
                   source, and answer only when independent sources agree; when they
                   disagree, abstain instead of guessing.

The defences see passage text and vectors, never which passages are poisoned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .answers import contains, group, is_idk, tokens
from .llm import rag_prompt

ISOLATED_PROMPT = (
    "Answer the question using only the context below. Reply with the short answer only "
    "(a name, number, date or short phrase), not a sentence. If the context does not "
    "answer the question, reply \"I don't know\".\n\nContext: {context}\n\nQuestion: "
    "{question}\n\nShort answer:")


@dataclass
class Passage:
    label: int
    title: str
    text: str
    vec: np.ndarray
    poison: bool = False           # for scoring only; defences must not read it

    @property
    def body(self):
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Outcome:
    answer: str                    # "" when the system abstains
    abstained: bool = False
    used: list = field(default_factory=list)       # labels the answer rests on
    dropped: list = field(default_factory=list)    # labels removed by filters
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------- filters

def echoes(question, text, frac=0.7, min_run=4):
    """Does the passage contain the opening of the question word for word? PoisonedRAG's
    black-box attack pastes the whole question into each poison, because that is what
    gets it retrieved. Encyclopedic text rarely contains "how many episodes are in ...".

    The match must start at the question's first word. Matching any long run instead also
    hit titles quoted in questions ("who sang never gonna let you go"): 82 of 4,201 gold
    passages, against 18 for the prefix rule (all 18 for keyword queries such as
    "third president of the united states of america")."""
    q, t = tokens(question), tokens(text)
    need = max(min_run, int(np.ceil(frac * len(q))))
    if len(q) < need:
        return False
    head = q[:need]
    return any(t[j:j + need] == head for j in range(len(t) - need + 1))


class InjectionFilter:
    """agentgate's prompt-injection detector (../agentgate): signature rules plus a
    TF-IDF classifier.

    Two changes for documents. Each sentence is also scored on its own, because one
    injected sentence inside a paragraph of ordinary text gets diluted: passage-level
    scoring caught 91 of 500 injected poisons, sentence-level 194. And the threshold is
    agentgate's warning level (0.6), not its blocking level (0.9) (158 caught), because
    dropping one of 20 retrieved passages costs little. The price, on 3,000 random corpus
    passages: 12 flagged (0.4%), against 5 for whole-passage scoring
    (scripts/filter_stats.py)."""

    SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

    def __init__(self, threshold=0.6):
        import sys
        from . import ROOT

        sys.path.insert(0, str(ROOT.parent / "agentgate"))
        from agentgate.detector import RULE_THRESHOLD, Detector

        self.detector = Detector()
        self.rule_threshold = RULE_THRESHOLD
        self.threshold = threshold

    def score(self, text):
        d = self.detector.detect(text)
        ml = max([d.ml] + [self.detector.ml_score(s) for s in self.SENTENCE.split(text) if s.strip()])
        return d.rule_score, ml

    def __call__(self, text):
        rule, ml = self.score(text)
        return rule >= self.rule_threshold or ml >= self.threshold


# ---------------------------------------------------------------- answering

class Answerer:
    """Asks the LLM about each passage on its own. Answers are cached per
    (question, passage), so every defence and setting reuses them."""

    def __init__(self, llm):
        self.llm = llm

    def isolated(self, question, passages):
        prompts = [ISOLATED_PROMPT.format(context=p.body, question=question) for p in passages]
        return self.llm.generate(prompts, max_tokens=24)

    def joint(self, question, passages):
        return self.llm.generate([rag_prompt(question, [p.body for p in passages])],
                                 max_tokens=48)[0]


def vanilla(question, passages, answerer, k=5):
    top = passages[:k]
    a = answerer.joint(question, top)
    return Outcome(a, is_idk(a), [p.label for p in top])


def isolate_vote(question, passages, answerer, k=10):
    """Plain majority over per-passage answers; every passage is one vote."""
    top = passages[:k]
    answers = answerer.isolated(question, top)
    voters = [i for i, a in enumerate(answers) if not is_idk(a)]
    if not voters:
        return Outcome("", True, detail={"answers": answers})
    groups = group([answers[i] for i in voters])
    best = max(groups, key=len)
    return Outcome(answers[voters[best[0]]], False, [top[voters[i]].label for i in best],
                   detail={"answers": answers})


# ---------------------------------------------------------------- quorum

@dataclass
class QuorumSettings:
    k: int = 10                 # passages answered (after filtering)
    echo: bool = True           # drop passages that contain the question
    inject: bool = True         # drop passages agentgate flags as injection
    tau: float = 0.85           # cosine above which two passages count as one source
    min_sources: int = 1        # independent sources the answer needs
    conflict: float = 0.5       # abstain if runner-up has >= conflict * winner's sources
    cluster: bool = True        # False: every passage is its own source


def sources(vecs, tau):
    """Single-linkage clusters of passages whose cosine similarity is >= tau
    (union-find over the pairs). Returns a cluster id per passage."""
    n = len(vecs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        sim = np.asarray(vecs) @ np.asarray(vecs).T
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= tau:
                    parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def quorum(question, passages, answerer, s=QuorumSettings(), inj=None):
    dropped, kept = [], []
    for p in passages:
        if (s.echo and echoes(question, p.text)) or (s.inject and inj and inj(p.body)):
            dropped.append(p.label)
        else:
            kept.append(p)
    top = kept[:s.k]
    answers = answerer.isolated(question, top)
    voters = [i for i, a in enumerate(answers) if not is_idk(a)]
    detail = {"answers": answers, "dropped": dropped}
    if not voters:
        return Outcome("", True, dropped=dropped, detail=detail | {"why": "no passage answers"})

    cid = sources([top[i].vec for i in voters], s.tau) if s.cluster else list(range(len(voters)))
    # One vote per source. A source whose passages disagree votes with its best-ranked one.
    first = {}
    for pos, c in enumerate(cid):
        first.setdefault(c, pos)
    reps = sorted(first.values())                          # voter position of each source
    rep_answers = [answers[voters[r]] for r in reps]
    groups = group(rep_answers)
    groups.sort(key=lambda g: (-len(g), g[0]))
    win = groups[0]
    run = len(groups[1]) if len(groups) > 1 else 0
    win_sources = {cid[reps[i]] for i in win}
    support = [top[voters[pos]].label for pos in range(len(voters)) if cid[pos] in win_sources]
    detail |= {"sources": len(reps), "votes": [len(g) for g in groups],
               "clusters": cid, "voters": voters}
    if len(win) < s.min_sources:
        return Outcome("", True, support, dropped, detail | {"why": "too few sources"})
    if run and run >= s.conflict * len(win):
        return Outcome("", True, support, dropped, detail | {"why": "sources disagree"})
    return Outcome(rep_answers[win[0]], False, support, dropped, detail)


# ---------------------------------------------------------------- guard

def guard(question, passages, answerer, s=QuorumSettings(), inj=None, min_dup=2):
    """Answer like plain RAG (filters, then top 5 in one prompt), then check where the
    answer came from: ask each of the top-k passages on its own, keep those whose answer
    appears in the joint answer, and merge near-copies. If every supporting passage falls
    in one group of `min_dup` or more near-copies, the answer rests on what is effectively
    a single source that was repeated, which is what planted passages look like: refuse."""
    kept = [p for p in passages
            if not (s.echo and echoes(question, p.text)) and not (s.inject and inj and inj(p.body))]
    dropped = [p.label for p in passages if p not in kept]
    out = vanilla(question, kept, answerer, 5)
    out.dropped = dropped
    if out.abstained:
        return out
    top = kept[:s.k]
    answers = answerer.isolated(question, top)
    support = [i for i, a in enumerate(answers) if not is_idk(a) and contains(out.answer, a)]
    if len(support) >= min_dup and len(set(sources([top[i].vec for i in support], s.tau))) == 1:
        return Outcome("", True, [top[i].label for i in support], dropped,
                       {"why": "answer rests on near-copies only", "answers": answers})
    return out
