"""quorum: retrieval-augmented generation that is attacked with poisoned documents and
defended by making sources agree before it answers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
