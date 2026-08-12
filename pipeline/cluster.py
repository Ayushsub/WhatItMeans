"""[Stages 3-4] Near-duplicate removal, then clustering into stories.

This is the rate-limit solution (plan Part 1). 300-600 articles collapse to
40-80 clusters, and only the top-ranked few get an LLM call. Clustering quality
therefore controls both output quality AND inference spend.

Approach: sentence embeddings + agglomerative single-link, constrained by a
time window. Still no vector DB and no retrieval (plan Part 2, RAG no-fit) —
these are vectors used for grouping, which is not RAG.

WHY EMBEDDINGS AND NOT TF-IDF. The plan specified TF-IDF first and said to
upgrade only on observed failure. That failure was observed on the first real
run: 271 articles produced 254 clusters, and the same gold story sat in two
separate clusters consuming two of six LLM slots. Sweeping the TF-IDF
threshold from 0.44 down to 0.20 never merged it — paraphrased headlines
("Gold Climbs Above $4,400" vs "Gold prices rise Rs 6,600/10g") share almost
no tokens, so the threshold hit an unrelated 16-article blob before it hit the
true pair. Lexical similarity cannot fix that; it is a vocabulary problem.
TF-IDF is retained as a fallback for when fastembed is unavailable.

Single-link is deliberate: news stories chain (A covers the RBI decision, B
covers the decision + bank reaction, C covers bank reaction). Complete-link
would split that chain into three clusters.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from datetime import timedelta

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from models import Article, Cluster

from . import config

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Financial-news boilerplate. Left in the default English stop list these
# dominate similarity and merge unrelated market-wrap stories.
_EXTRA_STOPWORDS = {
    "said", "says", "say", "will", "may", "new", "news", "update", "updates",
    "report", "reports", "reported", "today", "week", "month", "year", "day",
    "market", "markets", "stock", "stocks", "share", "shares", "crore", "lakh",
    "rs", "inr", "per", "cent", "percent", "pc", "india", "indian", "live",
    "latest", "top", "check", "know", "here", "amid", "ahead", "after", "before",
}


def _title_ngrams(title: str, n: int = 3) -> set[str]:
    """Word 3-grams for near-duplicate detection."""
    words = _WORD_RE.findall(title.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def drop_near_duplicates(articles: list[Article]) -> list[Article]:
    """[Stage 3] Remove syndicated re-posts of the identical story.

    Wire copy gets republished verbatim across outlets. Keeping all copies
    inflates the source_count signal that ranking and confidence depend on, so
    duplicates from the SAME PUBLISHER are dropped; the same story from a
    DIFFERENT publisher is genuine corroboration and is kept for clustering.

    Scoped by masthead rather than by feed: one publisher runs several desks,
    and the identical ET story syndicated into Markets, Stocks and Banking
    would otherwise survive as three articles under three different source_ids.
    """
    kept: list[Article] = []
    grams: list[tuple[set[str], str]] = []

    for art in sorted(articles, key=lambda a: a.published_at):
        g = _title_ngrams(art.title)
        dupe = False
        for other_g, other_pub in grams:
            if other_pub != art.masthead:
                continue
            if _jaccard(g, other_g) >= config.NEAR_DUPLICATE_THRESHOLD:
                dupe = True
                break
        if not dupe:
            kept.append(art)
            grams.append((g, art.masthead))

    if len(articles) != len(kept):
        log.info("near-duplicate removal: %d -> %d articles", len(articles), len(kept))
    return kept


def _tfidf_similarity(articles: list[Article]) -> np.ndarray:
    """Fallback backend. TF-IDF over title (weighted) + snippet.

    The title is repeated twice so it outweighs the snippet — headlines carry
    the story identity, snippets carry incidental detail.
    """
    corpus = [f"{a.title} {a.title} {a.snippet}" for a in articles]
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.6,  # a term in >60% of a news batch is boilerplate
        sublinear_tf=True,
        strip_accents="unicode",
    )
    matrix = vec.fit_transform(corpus)
    # Strip our extra stopwords post-hoc by zeroing their columns; simpler than
    # merging vocabularies and keeps sklearn's own English list intact.
    vocab = vec.vocabulary_
    kill = [vocab[w] for w in _EXTRA_STOPWORDS if w in vocab]
    if kill:
        matrix = matrix.tolil()
        matrix[:, kill] = 0
        matrix = matrix.tocsr()
    # L2-normalized by TfidfVectorizer, so the dot product IS cosine similarity.
    return (matrix @ matrix.T).toarray()


_embedder = None


def _embed_similarity(articles: list[Article]) -> np.ndarray | None:
    """Primary backend. Returns None if fastembed is unavailable, so a missing
    optional dependency degrades clustering instead of breaking the run."""
    global _embedder
    try:
        from fastembed import TextEmbedding
    except ImportError:
        log.warning("fastembed not installed; falling back to TF-IDF clustering")
        return None

    try:
        if _embedder is None:
            _embedder = TextEmbedding(model_name=config.EMBED_MODEL)
        texts = [f"{a.title}. {a.snippet}" for a in articles]
        vecs = np.array(list(_embedder.embed(texts)), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - model download/runtime failure
        log.warning("embedding failed (%s); falling back to TF-IDF", e)
        return None

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    return vecs @ vecs.T


def cluster_articles(articles: list[Article]) -> list[Cluster]:
    """[Stage 4] Group articles into stories via time-windowed single-link."""
    if not articles:
        return []
    if len(articles) == 1:
        return [_make_cluster(articles)]

    sim = _embed_similarity(articles) if config.USE_EMBEDDINGS else None
    if sim is not None:
        threshold = config.EMBED_SIMILARITY_THRESHOLD
        backend = "embeddings"
    else:
        sim = _tfidf_similarity(articles)
        threshold = config.TFIDF_SIMILARITY_THRESHOLD
        backend = "tfidf"

    np.fill_diagonal(sim, 0.0)
    window = timedelta(hours=config.CLUSTER_TIME_WINDOW_HOURS)

    # Union-find over pairs above threshold and inside the time window.
    parent = list(range(len(articles)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = np.argwhere(sim >= threshold)
    merges = 0
    for i, j in pairs:
        if i >= j:
            continue
        if abs(articles[i].published_at - articles[j].published_at) > window:
            continue
        union(int(i), int(j))
        merges += 1

    groups: dict[int, list[Article]] = defaultdict(list)
    for idx, art in enumerate(articles):
        groups[find(idx)].append(art)

    clusters = [_make_cluster(members) for members in groups.values()]
    clusters.sort(key=lambda c: c.newest_at, reverse=True)

    multi = sum(1 for c in clusters if len(c.articles) > 1)
    log.info(
        "clustered %d articles -> %d clusters (%d multi-article, %d merges, "
        "backend=%s threshold=%.2f)",
        len(articles), len(clusters), multi, merges, backend, threshold,
    )
    return clusters


def _make_cluster(members: list[Article]) -> Cluster:
    """Cluster id is derived from the earliest article, so a cluster keeps its
    identity across runs as later articles join it. That stability is what lets
    rank.py award a novelty bonus only to genuinely new stories."""
    lead = min(members, key=lambda a: a.published_at)
    cid = hashlib.sha1(f"cluster:{lead.id}".encode()).hexdigest()[:12]
    return Cluster(id=cid, articles=sorted(members, key=lambda a: a.published_at))
