"""Answer normalisation, matching and grouping.

Scoring follows the RAG literature (and PoisonedRAG): a response "contains" an answer if
the answer's normalised tokens appear as a contiguous run in the normalised response.
Matching on whole tokens, not characters, keeps "24" from matching "2024".
"""
from __future__ import annotations

import re
import string
import unicodedata

_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—"}
_ARTICLES = {"a", "an", "the"}
_IDK = re.compile(
    r"\b(i don'?t know|i do not know|unknown|not (?:mentioned|specified|provided|stated|given)"
    r"|(?:does|do) not (?:mention|specify|say|state|provide|contain)|no information|cannot (?:be )?"
    r"(?:determined|answer|find)|can'?t (?:answer|find))\b", re.I)


def tokens(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    return [t for t in s.split() if t not in _ARTICLES]


def normalize(s):
    return " ".join(tokens(s))


def contains(response, answer):
    r, a = tokens(response), tokens(answer)
    if not a:
        return False
    n = len(a)
    return any(r[i:i + n] == a for i in range(len(r) - n + 1))


def contains_any(response, answers):
    return any(contains(response, a) for a in answers)


def is_idk(response):
    return not tokens(response) or bool(_IDK.search(response))


def _numbers(toks):
    return {t for t in toks if any(ch.isdigit() for ch in t)}


def same_answer(a, b):
    """Do two short answers say the same thing? Equal after normalisation, or one is
    contained in the other ("1972" / "december 1972"), and they never disagree on a
    number ("23 episodes" vs "24 episodes" are different answers)."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    if _numbers(ta) != _numbers(tb) and _numbers(ta) and _numbers(tb):
        return False
    if ta == tb:
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    n = len(short)
    return any(long_[i:i + n] == short for i in range(len(long_) - n + 1))


def group(answers):
    """Greedy grouping: each answer joins the first group whose first member it matches.
    Returns a list of lists of indices into `answers`."""
    groups = []
    for i, a in enumerate(answers):
        for g in groups:
            if same_answer(answers[g[0]], a):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups
