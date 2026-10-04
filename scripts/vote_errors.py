"""Why per-passage voting loses clean accuracy: compare plain RAG (top 5 in one prompt)
with quorum's vote on the 500 clean test questions. Reads the LLM cache only, so run
scripts/evaluate.py first.

    python scripts/vote_errors.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, experiment, systems                 # noqa: E402
from quorum.answers import contains, contains_any            # noqa: E402
from quorum.defences import Answerer, InjectionFilter, quorum, vanilla   # noqa: E402
from quorum.llm import LLM                                   # noqa: E402
from quorum.store import PassageStore                        # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import cases_for                               # noqa: E402


def main():
    store, llm = PassageStore(), LLM()
    ans, inj, s = Answerer(llm), InjectionFilter(), systems.tuned_settings()
    c, shown = Counter(), 0
    for q, ps in cases_for("clean", data.clean_split("test"), store):
        v, o = vanilla(q.text, ps, ans, 5), quorum(q.text, ps, ans, s, inj)
        rv, ro = experiment.score(q, v), experiment.score(q, o)
        c["vote right, plain RAG wrong"] += ro == "correct" and rv != "correct"
        if rv != "correct" or ro == "correct":
            continue
        votes = o.detail.get("votes") or []
        c["plain RAG right, vote wrong"] += 1
        c["  a passage on its own gave the right answer, and lost the vote"] += any(
            contains_any(a, q.answers) for a in o.detail.get("answers", []))
        c["  the vote was a tie, settled by rank"] += len(votes) > 1 and votes[0] == votes[1]
        c["  the vote's answer is part of a gold answer (\"Close\" for \"Glenn Close\")"] += \
            bool(o.answer) and any(contains(g, o.answer) for g in q.answers)
        if shown < 5:
            print(f"{q.text}\n    gold {q.answers[0]!r}; plain RAG {v.answer[:70]!r}; "
                  f"vote {o.answer!r}, votes per answer {votes}")
            shown += 1
    print(f"\n({llm.calls} model calls)")
    for k, n in c.items():
        print(f"{n:4d}  {k}")


if __name__ == "__main__":
    main()
