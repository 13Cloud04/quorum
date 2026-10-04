"""Download everything the experiments need into data/ (about 1 GB, plus 2.5 GB of models
in the Hugging Face cache).

  - BEIR Natural Questions: 2,681,468 Wikipedia passages, 3,452 test questions, and
    qrels saying which passages answer each question
  - NQ-open validation: short gold answers (BEIR has none), joined to BEIR by question text
  - PoisonedRAG's published attack passages for 100 NQ test questions (MIT licence,
    USENIX Security 2025), pinned to a commit and checked by SHA-256
  - the embedding model and the LLM
"""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DATA = Path(__file__).resolve().parent.parent / "data"
POISONEDRAG_COMMIT = "f660d72174f06b13fae5163ce656e7b235db858f"
POISONEDRAG_SHA256 = "44df711454a9bada08e72e9e4a003a2cc845c43707ac93a3493e5168ec415cf2"
POISONEDRAG_URL = ("https://raw.githubusercontent.com/sleeepeer/PoisonedRAG/"
                   f"{POISONEDRAG_COMMIT}/results/adv_targeted_results/nq.json")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def get(repo, filename, out, repo_type="dataset"):
    out = DATA / out
    if not out.exists():
        src = hf_hub_download(repo, filename, repo_type=repo_type)
        shutil.copy(src, out)
    print(f"{out.name:28s} {out.stat().st_size / 2**20:8.1f} MB")


def main():
    DATA.mkdir(exist_ok=True)
    get("BeIR/nq", "corpus/corpus-00000-of-00001.parquet", "nq_corpus.parquet")
    get("BeIR/nq", "queries/queries-00000-of-00001.parquet", "nq_queries.parquet")
    get("BeIR/nq-qrels", "test.tsv", "nq_qrels.tsv")
    get("google-research-datasets/nq_open", "nq_open/validation-00000-of-00001.parquet",
        "nq_open_validation.parquet")

    out = DATA / "poisonedrag_nq.json"
    if not out.exists():
        with urllib.request.urlopen(POISONEDRAG_URL, timeout=60) as r:
            out.write_bytes(r.read())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    if digest != POISONEDRAG_SHA256:
        raise SystemExit(f"{out.name}: sha256 {digest} does not match the pinned file")
    targets = json.loads(out.read_text())
    print(f"{out.name:28s} {len(targets)} targets, sha256 {digest[:16]}")

    for repo in (EMBED_MODEL, LLM):
        path = snapshot_download(repo)
        print(f"{repo:45s} -> {path}")


if __name__ == "__main__":
    main()
