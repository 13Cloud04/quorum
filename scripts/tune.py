"""Stage 2: choose quorum's two settings using clean dev questions only. No attack data
is read here, so the attack results are not fitted to the attacks.

  tau       Cosine above which two passages count as one source. Set to the 99th
            percentile of similarity between passages from *different* Wikipedia articles
            retrieved for the same question, so distinct sources are merged at most 1% of
            the time.
  conflict  Abstain when the runner-up answer has at least this fraction of the winner's
            sources. The most cautious value whose dev accuracy stays within 3 points of
            the same pipeline with no conflict rule.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, experiment, systems             # noqa: E402
from quorum.defences import Answerer, InjectionFilter, QuorumSettings, quorum   # noqa: E402
from quorum.llm import LLM                               # noqa: E402
from quorum.store import PassageStore                    # noqa: E402

CONFLICTS = [0.34, 0.5, 0.67, 0.75, 1.0]
BUDGET = 0.03


def main():
    store = PassageStore()
    world = experiment.World(store)
    hits = experiment.load_retrieval("clean")
    dev = data.clean_split("dev")
    cases = [(q, [world.passage(l) for l, _ in hits[q.qid]]) for q in dev]

    sims = []
    for _, ps in cases:
        top = ps[:10]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                if top[i].title != top[j].title:
                    sims.append(float(top[i].vec @ top[j].vec))
    tau = round(float(np.percentile(sims, 99)), 3)
    print(f"cross-article pairs: {len(sims)}; median {np.median(sims):.3f}; "
          f"p99 {tau}; max {max(sims):.3f}")

    llm, inj = LLM(), InjectionFilter()
    answerer = Answerer(llm)
    experiment.prefetch(answerer, cases)
    base = QuorumSettings(tau=tau)

    def acc(s):
        outs = [experiment.score(q, quorum(q.text, ps, answerer, s, inj)) for q, ps in cases]
        return outs.count("correct") / len(outs), outs.count("abstain") / len(outs)

    ref, ref_abs = acc(replace(base, conflict=99.0))
    print(f"no conflict rule: accuracy {ref:.3f}, abstain {ref_abs:.3f}")
    chosen = None
    for c in CONFLICTS:
        a, ab = acc(replace(base, conflict=c))
        ok = a >= ref - BUDGET
        print(f"conflict {c:4}: accuracy {a:.3f}, abstain {ab:.3f} {'ok' if ok else ''}")
        if ok and chosen is None:
            chosen = c
    s = replace(base, conflict=chosen if chosen is not None else 99.0)
    systems.save_settings(s)
    print("chosen:", s)


if __name__ == "__main__":
    main()
