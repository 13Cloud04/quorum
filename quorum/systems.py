"""The systems compared in the results: baselines, quorum, and quorum with one part
removed at a time."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

from . import RESULTS
from .defences import InjectionFilter, QuorumSettings, echoes, isolate_vote, quorum, vanilla

SETTINGS_PATH = RESULTS / "settings.json"


def tuned_settings():
    """Quorum's settings, chosen on clean dev questions by scripts/tune.py."""
    if SETTINGS_PATH.exists():
        return QuorumSettings(**json.loads(SETTINGS_PATH.read_text()))
    return QuorumSettings()


def save_settings(s):
    SETTINGS_PATH.write_text(json.dumps(asdict(s), indent=1))


def filtered(question, passages, inj):
    return [p for p in passages if not echoes(question, p.text) and not inj(p.body)]


def systems(answerer, s=None, inj=None):
    s = s or tuned_settings()
    inj = inj or InjectionFilter()
    return {
        "vanilla k=5": lambda q, ps: vanilla(q, ps, answerer, 5),
        "vanilla k=10": lambda q, ps: vanilla(q, ps, answerer, 10),
        "filters + vanilla k=5": lambda q, ps: vanilla(q, filtered(q, ps, inj), answerer, 5),
        "isolate-vote k=10": lambda q, ps: isolate_vote(q, ps, answerer, 10),
        "quorum": lambda q, ps: quorum(q, ps, answerer, s, inj),
        "quorum - echo filter": lambda q, ps: quorum(q, ps, answerer, replace(s, echo=False), inj),
        "quorum - injection filter": lambda q, ps: quorum(q, ps, answerer, replace(s, inject=False), inj),
        "quorum - clustering": lambda q, ps: quorum(q, ps, answerer, replace(s, cluster=False), inj),
        "quorum - conflict rule": lambda q, ps: quorum(q, ps, answerer, replace(s, conflict=99.0), inj),
    }
