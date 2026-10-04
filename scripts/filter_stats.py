"""How often each filter fires, on poisons and on real passages. No LLM calls.

    python scripts/filter_stats.py

  copied-question filter   every attack's poisons, and every gold passage in BEIR-NQ
                           (passages that answer one of its 3,452 questions)
  injection filter         the `inject` poisons and 3,000 random corpus passages, scored
                           on the whole passage and sentence by sentence, at agentgate's
                           warning (0.6) and blocking (0.9) thresholds
  merging                  how many sources each question's fakes count as at quorum's tau
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import DATA, attacks, data, systems        # noqa: E402
from quorum.defences import InjectionFilter, echoes, sources   # noqa: E402
from quorum.store import PassageStore                   # noqa: E402


def main():
    store = PassageStore()
    queries, qrels, _ = data._beir()
    qtext = {t.qid: t.text for t in data.targets()} | {t.qid: t.text for t in data.heldout_targets()}

    print("copied-question filter")
    for name in attacks.ATTACKS + tuple("ho-" + a for a in attacks.ATTACKS):
        ps = json.loads((DATA / f"attack_{name}.json").read_text())
        hit = sum(echoes(qtext[p["qid"]], p["text"]) for p in ps)
        print(f"  {name:16s} poisons removed {hit:4d} of {len(ps)}")
    pairs = [(queries[q], r) for q, rows in qrels.items() if q in queries for r in rows]
    hit = sum(echoes(q, store.text(r)) for q, r in pairs)
    print(f"  {'gold passages':16s} wrongly removed {hit} of {len(pairs)} ({100 * hit / len(pairs):.2f}%)")

    tau = systems.tuned_settings().tau
    print(f"merging: sources per question's fakes at tau={tau}")
    for name in attacks.ATTACKS + tuple("ho-" + a for a in attacks.ATTACKS):
        ps, vecs = json.loads((DATA / f"attack_{name}.json").read_text()), np.load(DATA / f"attack_{name}.npy")
        by = defaultdict(list)
        for i, p in enumerate(ps):
            by[p["qid"]].append(i)
        n = [len(set(sources(vecs[ix], tau))) for ix in by.values()]
        print(f"  {name:16s} {len(ps) / len(by):.2f} fakes per question count as {np.mean(n):.2f} "
              f"sources; all in one source for {sum(c == 1 for c in n)} of {len(n)} questions")

    inj = InjectionFilter()
    det = inj.detector
    poisons = [p["text"] for p in json.loads((DATA / "attack_inject.json").read_text())]
    rng = random.Random(0)
    real = []
    for r in rng.sample(range(len(store)), 3000):
        title, text = store.get(r)
        real.append(f"{title}. {text}" if title else text)

    def scores(texts):
        out = []
        for t in texts:
            d = det.detect(t)
            sent = max([d.ml] + [det.ml_score(s) for s in inj.SENTENCE.split(t) if s.strip()])
            out.append((d.rule_score >= inj.rule_threshold, d.ml, sent))
        return out

    sp, sr = scores(poisons), scores(real)
    print("injection filter (rules OR classifier >= threshold)")
    for mode, i in (("whole passage", 1), ("each sentence", 2)):
        for th in (0.6, 0.9):
            caught = sum(s[0] or s[i] >= th for s in sp)
            fp = sum(s[0] or s[i] >= th for s in sr)
            print(f"  {mode:14s} threshold {th}: poisons caught {caught:3d} of {len(sp)}, "
                  f"real passages flagged {fp} of {len(sr)}")


if __name__ == "__main__":
    main()
