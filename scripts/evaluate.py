"""Stage 3: run every system on every condition and write results/results.json.

    python scripts/evaluate.py            # 100 attacked questions x 5 conditions, 500 clean test,
                                          # then the held-out questions x 5 conditions

Each question ends in one of: correct, attacker (the attacker's chosen answer), abstain,
other (a wrong answer the attacker did not choose).
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import RESULTS, attacks, data, experiment, systems   # noqa: E402
from quorum.defences import Answerer, InjectionFilter           # noqa: E402
from quorum.llm import LLM                                      # noqa: E402
from quorum.store import PassageStore                           # noqa: E402

CONDITIONS = ("clean",) + attacks.ATTACKS
HELDOUT = ("ho-clean",) + tuple("ho-" + a for a in attacks.ATTACKS)


def cases_for(condition, questions, store):
    world = experiment.World(store, condition)
    hits = experiment.load_retrieval(condition)
    return [(q, [world.passage(l) for l, _ in hits[q.qid]]) for q in questions if q.qid in hits]


def retrieval_stats(cases, n):
    top5 = [sum(p.poison for p in ps[:5]) for _, ps in cases]
    gold = [any(p.label in q.gold_rows for p in ps[:10]) for q, ps in cases if q.gold_rows]
    return {"poisons_in_top5_mean": sum(top5) / len(top5),
            "questions_with_poison_in_top5": sum(t > 0 for t in top5),
            "gold_in_top10": sum(gold) / max(1, len(gold))}


def run(name, cases, table, n):
    out = {}
    for sys_name, fn in table.items():
        t = time.time()
        rows = []
        for q, ps in cases:
            o = fn(q.text, ps)
            rows.append({"qid": q.qid, "result": experiment.score(q, o), "answer": o.answer,
                         "abstain_reason": o.detail.get("why"),
                         "poisons_used": sum(l >= n for l in o.used),
                         "poisons_dropped": sum(l >= n for l in o.dropped)})
        c = Counter(r["result"] for r in rows)
        out[sys_name] = {"counts": dict(c), "n": len(rows), "rows": rows}
        print(f"  {name:12s} {sys_name:28s} " + "  ".join(
            f"{k} {c.get(k, 0):3d}" for k in ("correct", "attacker", "abstain", "other"))
            + f"   ({time.time() - t:.1f}s)", flush=True)
    return out


def main():
    store = PassageStore()
    llm = LLM()
    answerer = Answerer(llm)
    inj = InjectionFilter()
    table = systems.systems(answerer, inj=inj)
    targets = data.targets()
    results = {"settings": systems.tuned_settings().__dict__, "conditions": {}, "retrieval": {}}
    heldout = data.heldout_targets()
    plan = [(c, c, targets) for c in CONDITIONS] + [("clean-test", "clean", data.clean_split("test"))]
    plan += [(c, "clean" if c == "ho-clean" else c, heldout) for c in HELDOUT]
    for name, condition, qs in plan:
        cases = cases_for(condition, qs, store)
        t = time.time()
        experiment.prefetch(answerer, cases, joint_ks=(5, 10))
        experiment.prefetch(answerer, [(q, systems.filtered(q.text, ps, inj)) for q, ps in cases],
                            joint_ks=(5,), isolated=False)
        print(f"{name}: {len(cases)} questions, LLM prefetch {time.time() - t:.0f}s "
              f"({llm.calls} model calls so far)", flush=True)
        results["retrieval"][name] = retrieval_stats(cases, len(store))
        results["conditions"][name] = run(name, cases, table, len(store))
    (RESULTS / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
