"""Stage 1: retrieve the top 20 passages for every question, clean and under each attack.

    python scripts/retrieve.py clean            # targets + dev + test, no poisons
    python scripts/retrieve.py poisonedrag      # targets, with that attack's poisons inserted
    python scripts/retrieve.py clean heldout    # the held-out questions, no poisons
    python scripts/retrieve.py ho-paraphrase    # held-out questions, held-out poisons
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import data, experiment, systems           # noqa: E402
from quorum.embed import Embedder                      # noqa: E402
from quorum.llm import LLM                             # noqa: E402
from quorum.store import PassageStore                  # noqa: E402

# The adaptive attacker knows quorum's merging threshold (chosen by scripts/tune.py).
TAU = float(dict(a.split("=") for a in sys.argv[2:] if "=" in a).get("tau", systems.tuned_settings().tau))


def main(condition):
    store, emb = PassageStore(), Embedder()
    qs = data.questions_for(condition)
    if condition == "clean":
        qs = (data.heldout_targets() if "heldout" in sys.argv else
              qs + data.clean_split("dev") + data.clean_split("test"))
    t = time.time()
    out = experiment.run_retrieval(condition, qs, store, emb, llm=LLM(), tau=TAU)
    n = len(store)
    poisoned = sum(any(l >= n for l, _ in hs[:5]) for hs in out.values())
    print(f"{condition}: {len(out)} questions in {time.time() - t:.0f}s; "
          f"questions with a poison in the top 5: {poisoned}")


if __name__ == "__main__":
    main(sys.argv[1])
