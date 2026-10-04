"""Stage 1: retrieve the top 20 passages for every question, clean and under each attack.

    python scripts/retrieve.py clean            # targets + dev + test, no poisons
    python scripts/retrieve.py poisonedrag      # targets, with that attack's poisons inserted
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, experiment                    # noqa: E402
from quorum.embed import Embedder                      # noqa: E402
from quorum.llm import LLM                             # noqa: E402
from quorum.store import PassageStore                  # noqa: E402

TAU = float(dict(a.split("=") for a in sys.argv[2:]).get("tau", 0.85))


def main(condition):
    store, emb = PassageStore(), Embedder()
    qs = data.targets()
    if condition == "clean":
        qs = qs + data.clean_split("dev") + data.clean_split("test")
    t = time.time()
    out = experiment.run_retrieval(condition, qs, store, emb, llm=LLM(), tau=TAU)
    n = len(store)
    poisoned = sum(any(l >= n for l, _ in hs[:5]) for hs in out.values())
    print(f"{condition}: {len(out)} questions in {time.time() - t:.0f}s; "
          f"questions with a poison in the top 5: {poisoned}")


if __name__ == "__main__":
    main(sys.argv[1])
