"""Seconds per question, end to end (embed the question, search, LLM), with the LLM
cache turned off, for plain RAG and for both quorum modes. One question at a time, as a user
would ask them.

    python scripts/bench_latency.py [n_questions]
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, experiment, systems      # noqa: E402
from quorum.defences import Answerer, InjectionFilter, guard, quorum, vanilla   # noqa: E402
from quorum.embed import Embedder                 # noqa: E402
from quorum.llm import LLM                        # noqa: E402
from quorum.retrieve import Retriever             # noqa: E402
from quorum.store import PassageStore             # noqa: E402


def main(n):
    store, emb = PassageStore(), Embedder()
    r, world = Retriever(len(store)), experiment.World(store)
    llm = LLM(cache=False)
    answerer, inj, s = Answerer(llm), InjectionFilter(), systems.tuned_settings()
    qs = data.clean_split("test")[:n]
    llm.generate(["warm up"], max_tokens=4)
    times = {"vanilla k=5": [], "quorum vote": [], "quorum guard": []}
    for q in qs:
        for name in times:
            t = time.perf_counter()
            hits = r.search(emb.queries([q.text]), k=experiment.DEPTH)[0]
            ps = [world.passage(h.label) for h in hits]
            if name == "quorum vote":
                quorum(q.text, ps, answerer, s, inj)
            elif name == "quorum guard":
                guard(q.text, ps, answerer, s, inj)
            else:
                vanilla(q.text, ps, answerer, 5)
            times[name].append(time.perf_counter() - t)
    for name, ts in times.items():
        print(f"{name:12s} median {statistics.median(ts):.2f}s  p90 "
              f"{sorted(ts)[int(0.9 * len(ts)) - 1]:.2f}s  over {len(ts)} questions")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
