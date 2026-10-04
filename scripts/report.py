"""Turn results/results.json into the markdown tables used in the README
(results/RESULTS.md)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quorum import RESULTS        # noqa: E402

ATTACKS = ["poisonedrag", "paraphrase", "inject", "diverse"]
KEYS = ("correct", "attacker", "abstain", "other")


def pct(r, key):
    return f"{100 * r['counts'].get(key, 0) / r['n']:.0f}%"


def grid(res, conds, key, title, systems=None):
    systems = systems or list(res["conditions"][conds[0]].keys())
    out = [f"**{title}**", "", "| system | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for s in systems:
        out.append(f"| {s} | " + " | ".join(pct(res["conditions"][c][s], key) for c in conds) + " |")
    return "\n".join(out)


def clean_table(res, cond, title):
    rows = [f"**{title}**", "", "| system | correct | abstained | other wrong |", "|---|---|---|---|"]
    for s, r in res["conditions"][cond].items():
        rows.append(f"| {s} | " + " | ".join(pct(r, k) for k in ("correct", "abstain", "other")) + " |")
    return "\n".join(rows)


def main():
    res = json.loads((RESULTS / "results.json").read_text())
    n = {c: next(iter(v.values()))["n"] for c, v in res["conditions"].items()}
    ho = ["ho-" + a for a in ATTACKS]
    parts = [
        f"Settings (chosen on clean dev questions): `{json.dumps(res['settings'])}`",
        "## PoisonedRAG's 100 questions (the paper's fakes, and three attacks built on them)",
        grid(res, ATTACKS, "attacker", f"Attack success rate: the attacker's answer was given ({n['poisonedrag']} questions; lower is better)"),
        grid(res, ["clean"] + ATTACKS, "correct", "Correct"),
        grid(res, ["clean"] + ATTACKS, "abstain", "Abstained (\"I'm not sure\")"),
        "## The cost when nobody attacks",
        clean_table(res, "clean-test", f"{n['clean-test']} clean test questions"),
        "## Held-out set (made after the first results; fakes written by the local model)",
        grid(res, ho, "attacker", f"Attack success rate ({n['ho-poisonedrag']} questions; lower is better)"),
        grid(res, ["ho-clean"] + ho, "correct", "Correct"),
        grid(res, ["ho-clean"] + ho, "abstain", "Abstained"),
    ]
    rows = ["## Retrieval", "", "| condition | questions with a poison in the top 5 | poisons in the top 5 (mean) | gold passage in the top 10 |",
            "|---|---|---|---|"]
    for c, r in res["retrieval"].items():
        rows.append(f"| {c} | {r['questions_with_poison_in_top5']} | {r['poisons_in_top5_mean']:.2f} | {100 * r['gold_in_top10']:.0f}% |")
    parts.append("\n".join(rows))
    text = "\n\n".join(parts) + "\n"
    (RESULTS / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
