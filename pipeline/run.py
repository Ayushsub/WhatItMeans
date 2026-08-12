"""Pipeline entry point.

Stages 1-6 run with ZERO LLM calls, which is what makes clustering tunable
without spending inference (plan Part 10):

    python -m pipeline.run --no-llm --dump out/

Full run:

    python -m pipeline.run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import archive, config, render
from .cluster import cluster_articles, drop_near_duplicates
from .fetch import fetch_all
from .normalize import normalize_all
from .rank import rank, select_for_analysis
from .tag import tag_all


def _load_dotenv() -> None:
    """Load .env for local runs. CI sets real environment variables, which
    always win — this never overwrites an existing value."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _setup_logging(verbose: bool) -> None:
    # Windows consoles default to cp1252 and blow up on the rupee sign, which
    # appears constantly in Indian headlines. Force UTF-8 on the stream.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)-20s %(message)s",
        stream=sys.stderr,
    )


def run(
    no_llm: bool = False,
    dump: Path | None = None,
    limit: int | None = None,
    push: bool = False,
) -> dict:
    log = logging.getLogger("run")
    started = datetime.now(timezone.utc)

    # --- Stages 1-6: no inference ---------------------------------------
    results = fetch_all(config.sources())
    articles = normalize_all(results)
    if not articles:
        log.error("no articles survived normalization; aborting")
        return {"ok": False, "reason": "no articles"}

    articles = drop_near_duplicates(articles)
    clusters = cluster_articles(articles)
    clusters = tag_all(clusters)

    # What the live site already covers. Without this the novelty penalty in
    # rank.py never fires, and a running story is re-analyzed every 20 minutes
    # — burning an LLM slot and putting the same story on the page twice.
    previous = archive.load_previous(render.SITE)
    seen_ids, seen_urls = archive.published_signature(previous)

    ranked = rank(clusters, seen_ids=seen_ids, seen_urls=seen_urls)
    selected, candidates = select_for_analysis(ranked, limit=limit)

    log.info(
        "selected %d clusters for analysis out of %d candidates "
        "(%d stories already live)",
        len(selected), len(candidates), len(previous),
    )

    if dump:
        _dump(dump, results, articles, ranked, selected, started)
        log.info("dumped inspection artifacts to %s", dump)

    if no_llm:
        log.info("--no-llm set: stopping before inference")
        return {
            "ok": True,
            "articles": len(articles),
            "clusters": len(ranked),
            "selected": len(selected),
        }

    # --- Stages 7-12: inference and publish -----------------------------
    from llm.providers import describe_configuration

    from .analyze import analyze_selected
    from .comply import enforce
    from .ledger import Ledger
    from .publish import publish, verify_retention
    from .render import render_site

    log.info("llm keys: %s", describe_configuration())

    ledger = Ledger()
    analyzed = analyze_selected(selected, ledger, candidates=candidates)
    approved = enforce(analyzed)
    ledger.record(approved)

    site = render_site(approved, ranked)
    published = publish(site, push=push)
    retention_ok = verify_retention() if push else True

    return {
        "ok": published and retention_ok,
        "articles": len(articles),
        "clusters": len(ranked),
        "analyzed": len(analyzed),
        "published": len(approved),
        "blocked_by_compliance": len(analyzed) - len(approved),
        "pushed": push and published,
    }


def _dump(out: Path, results, articles, ranked, selected, started) -> None:
    """Write inspection artifacts. This is the tuning surface for clustering."""
    out.mkdir(parents=True, exist_ok=True)

    (out / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": r.source["id"],
                    "status": r.status,
                    "entries": len(r.entries),
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_ids = {c.id for c in selected}
    (out / "clusters.json").write_text(
        json.dumps(
            [
                {
                    "id": c.id,
                    "selected": c.id in selected_ids,
                    "score": c.score,
                    "breakdown": c.score_breakdown,
                    "n_articles": len(c.articles),
                    # n_publishers is the corroboration signal; n_feeds is only
                    # here so a gap between the two is visible when tuning.
                    "n_publishers": c.source_count,
                    "n_feeds": len({a.source_id for a in c.articles}),
                    "tags": c.tags.model_dump(),
                    "titles": [
                        {"pub": a.masthead, "src": a.source_id, "title": a.title}
                        for a in c.articles
                    ],
                }
                for c in ranked
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (out / "summary.json").write_text(
        json.dumps(
            {
                "started": started.isoformat(),
                "articles": len(articles),
                "clusters": len(ranked),
                "multi_article_clusters": sum(1 for c in ranked if len(c.articles) > 1),
                "selected": len(selected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Newsx pipeline")
    p.add_argument("--no-llm", action="store_true", help="stop after ranking")
    p.add_argument("--dump", type=Path, help="write inspection artifacts here")
    p.add_argument("--limit", type=int, help="override max clusters analyzed")
    p.add_argument(
        "--push",
        action="store_true",
        default=os.environ.get("PUBLISH") == "1",
        help="force-push ./site to gh-pages (CI uses PUBLISH=1)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _load_dotenv()
    _setup_logging(args.verbose)
    result = run(
        no_llm=args.no_llm, dump=args.dump, limit=args.limit, push=args.push
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
