"""Exploratory, designed AFTER seeing results/evaluate.log: keep plain RAG's answer (so no
clean-accuracy cost) but refuse when that answer rests on one group of near-duplicate
passages. Printed for every condition; not part of the main results."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, experiment, systems                        # noqa: E402
from quorum.answers import contains, is_idk                          # noqa: E402
from quorum.defences import Answerer, InjectionFilter, Outcome, sources, vanilla   # noqa: E402
from quorum.llm import LLM                                           # noqa: E402
from quorum.store import PassageStore                                # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import cases_for                                       # noqa: E402


def alarm(q, ps, ans, inj, s, min_dup):
    kept = systems.filtered(q, ps, inj)
    out = vanilla(q, kept, ans, 5)
    if out.abstained:
        return out
    top = kept[:s.k]
    answers = ans.isolated(q, top)
    voters = [i for i, a in enumerate(answers) if not is_idk(a) and contains(out.answer, a)]
    if not voters:
        return out
    cid = sources([top[i].vec for i in voters], s.tau)
    if len(set(cid)) == 1 and len(voters) >= min_dup:
        return Outcome("", True, detail={"why": "answer rests on near-duplicates only"})
    return out


def main():
    store, s = PassageStore(), systems.tuned_settings()
    ans, inj = Answerer(LLM()), InjectionFilter()
    plan = [(c, data.targets()) for c in ("clean", "poisonedrag", "paraphrase", "inject", "diverse")]
    plan += [("clean-test", data.clean_split("test")), ("dev", data.clean_split("dev"))]
    for name, qs in plan:
        cases = cases_for("clean" if name in ("clean-test", "dev") else name, qs, store)
        for m in (2, 3):
            c = Counter(experiment.score(q, alarm(q.text, ps, ans, inj, s, m)) for q, ps in cases)
            print(f"{name:12s} min_dup={m}  " + "  ".join(f"{k} {c.get(k, 0):3d}" for k in
                                                         ("correct", "attacker", "abstain", "other")))


if __name__ == "__main__":
    main()
