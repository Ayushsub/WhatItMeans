"""Config loading. Read once at startup, passed down through the stages."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def sources() -> list[dict[str, Any]]:
    """Feed list with per-source defaults already merged in.

    Merging here means no downstream stage has to remember that
    snippet_max_words might be absent on an individual source.
    """
    cfg = _load("sources.yml")
    defaults = cfg.get("defaults", {})
    merged = [{**defaults, **src} for src in cfg["sources"]]
    # A source that omits `publisher` gets its own id, which is the correct
    # answer for any publisher running a single feed. Defaulting here rather
    # than downstream means an omitted field degrades to the old per-feed
    # behaviour instead of raising or silently counting as an empty publisher.
    for src in merged:
        src.setdefault("publisher", src["id"])
    return merged


@functools.lru_cache(maxsize=1)
def watchlist() -> dict[str, Any]:
    return _load("watchlist.yml")


@functools.lru_cache(maxsize=1)
def compliance() -> dict[str, Any]:
    return _load("compliance.yml")


# --- Tunables -------------------------------------------------------------
# Grouped here rather than scattered as literals, because these are the knobs
# you actually turn when tuning output quality and LLM spend.

RETENTION_HOURS = 24  # hard ceiling; nothing older is ever rendered

# Per-publisher article ceiling, applied after normalization.
#
# sources.yml caps entries PER FEED, which does not constrain a newsroom that
# runs six of them. Measured without this: Economic Times, BusinessLine and
# Mint together supplied 81% of the corpus purely because they publish the most
# feeds. That is a structural bias toward whatever those three desks cover, not
# an editorial judgement anyone made.
#
# Trimmed newest-first, so a busy newsroom keeps its freshest reporting.
MAX_ARTICLES_PER_PUBLISHER = 75
NEAR_DUPLICATE_THRESHOLD = 0.82  # title 3-gram Jaccard -> same article
CLUSTER_TIME_WINDOW_HOURS = 18  # articles further apart than this never merge
MAX_ANALYZED_PER_RUN = 10  # the single knob controlling LLM spend
TRIAGE_CANDIDATE_COUNT = 15  # how many clusters PROMPT 2 ranks

# Clustering similarity. Two backends, two thresholds — they are NOT
# interchangeable numbers and must not be unified.
#
# Measured on 271 live articles (see plan Part 10):
#   embeddings  should-merge 0.82-0.92 | controls 0.47-0.62 | p99 0.74
#   tfidf       could not separate them at ANY threshold — paraphrases like
#               "Gold Climbs Above $4,400" vs "Gold prices rise Rs 6,600/10g"
#               share almost no tokens, so lowering the threshold produced a
#               16-article blob before it merged the true pair.
# Embeddings are therefore the default and TF-IDF is only a degraded fallback.
EMBED_SIMILARITY_THRESHOLD = 0.78  # sits in the 0.74-0.82 gap
TFIDF_SIMILARITY_THRESHOLD = 0.42  # fallback only; known to under-merge

# BAAI/bge-small-en-v1.5: 384-dim, ~130MB, 34ms/article on a laptop CPU.
# This is an EMBEDDING model, not an LLM — it runs fine on modest hardware and
# is cached in CI (see .github/workflows/pipeline.yml).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
USE_EMBEDDINGS = True  # set False to force the TF-IDF fallback
