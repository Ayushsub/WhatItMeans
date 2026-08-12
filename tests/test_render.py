"""Offline render smoke test.

Renders the full site from synthetic analyses joined to REAL clusters, so the
templates and JSON API can be verified without spending any inference.
Run: python tests/test_render.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import Article, Cluster, PublishedItem, SourceTier, Tags  # noqa: E402
from pipeline.render import render_site  # noqa: E402
from tests.test_compliance import _analysis  # noqa: E402

# Render to a scratch dir, NEVER to ./site — otherwise running this test
# silently replaces real published output with fixture data.
OUT = ROOT / ".test-site"


def _fake_cluster(n: int) -> Cluster:
    now = datetime.now(timezone.utc)
    arts = [
        Article(
            id=f"a{n}{i}",
            title=f"Rupee slips to 95.28 as crude climbs, report {i}",
            snippet="The rupee closed 11 paise lower against the dollar as Brent held near $87.",
            url=f"https://example.com/story-{n}-{i}",
            source_id=f"src{i}",
            source_name=["Mint", "Economic Times", "BusinessLine"][i % 3],
            publisher=["mint", "economic_times", "hindu_businessline"][i % 3],
            source_tier=SourceTier.MAJOR,
            published_at=now - timedelta(hours=n, minutes=i * 7),
            fetched_at=now,
        )
        for i in range(3)
    ]
    c = Cluster(id=f"cluster{n:02d}", articles=arts)
    c.tags = Tags(
        entities=["RBI"],
        tickers=["RELIANCE"],
        sectors=["Oil & Gas", "IT & Tech"],
        macro_themes=["Currency", "Crude & Energy"],
    )
    return c


def _check_publisher_scoping(failures: list[str]) -> None:
    """Corroboration must count MASTHEADS, not feeds.

    One newsroom runs several feeds (ET Markets, ET Stocks, ET Banking). If
    source_count counted feeds, a single publisher carrying a story across its
    own desks would read as independent corroboration — which is both a false
    claim on the page ("reported by 3 sources") and a distortion of the largest
    term in the ranking score.
    """
    now = datetime.now(timezone.utc)

    def art(i: int, feed: str, pub: str, title: str) -> Article:
        return Article(
            id=f"p{i}",
            title=title,
            snippet="Snippet.",
            url=f"https://example.com/p{i}",
            source_id=feed,
            source_name="Economic Times",
            publisher=pub,
            source_tier=SourceTier.MAJOR,
            published_at=now - timedelta(minutes=i),
            fetched_at=now,
        )

    # Three different FEEDS, one publisher -> one source.
    same = Cluster(
        id="same",
        articles=[
            art(1, "et_markets", "economic_times", "Rupee slips to 95.28"),
            art(2, "et_stocks", "economic_times", "Rupee weakens past 95 on crude"),
            art(3, "et_banking", "economic_times", "Banks flag rupee pressure"),
        ],
    )
    if same.source_count != 1:
        failures.append(
            f"source_count={same.source_count} for 3 feeds of ONE publisher, expected 1"
        )

    # Three different publishers -> three sources.
    diff = Cluster(
        id="diff",
        articles=[
            art(4, "et_markets", "economic_times", "Rupee slips to 95.28"),
            art(5, "mint_markets", "mint", "Rupee weakens past 95 on crude"),
            art(6, "hbl_markets", "hindu_businessline", "Banks flag rupee pressure"),
        ],
    )
    if diff.source_count != 3:
        failures.append(
            f"source_count={diff.source_count} for 3 distinct publishers, expected 3"
        )

    # An article with no publisher set falls back to its feed id, which is the
    # correct answer for a single-feed publisher and never counts as blank.
    solo = Article(
        id="solo",
        title="Solo",
        snippet="",
        url="https://example.com/solo",
        source_id="rbi_press",
        source_name="RBI",
        source_tier=SourceTier.WIRE,
        published_at=now,
        fetched_at=now,
    )
    if solo.masthead != "rbi_press":
        failures.append(f"masthead fallback broken: got {solo.masthead!r}")

    # The near-duplicate filter must also be publisher-scoped: the same story
    # syndicated across one newsroom's desks is one article, not three.
    from pipeline.cluster import drop_near_duplicates

    syndicated = [
        art(7, "et_markets", "economic_times", "Rupee slips to 95.28 against dollar"),
        art(8, "et_stocks", "economic_times", "Rupee slips to 95.28 against dollar"),
        art(9, "mint_markets", "mint", "Rupee slips to 95.28 against dollar"),
    ]
    kept = drop_near_duplicates(syndicated)
    if len(kept) != 2:
        failures.append(
            f"drop_near_duplicates kept {len(kept)} of 3 syndicated copies, expected 2 "
            "(one per publisher)"
        )


def _check_archive_merge(failures: list[str]) -> None:
    """The same story must not appear twice under two cluster ids.

    A cluster's id derives from its earliest article, so anything that changes
    which article is earliest mints a new id for the same story. Observed live:
    one gold story published twice, 40 minutes apart, both on the homepage.
    Shared source URLs are the stable identity.
    """
    from pipeline.archive import merge, published_signature

    now = datetime.now(timezone.utc)

    def item(cid: str, urls: list[str], updated_min: int) -> PublishedItem:
        return PublishedItem(
            cluster_id=cid,
            analysis=_analysis(),
            tags=Tags(),
            sources=[{"name": "Mint", "url": u, "published_at": now.isoformat()} for u in urls],
            published_at=now - timedelta(hours=1),
            updated_at=now - timedelta(minutes=updated_min),
            confidence="medium",
        )

    old = item("old01", ["https://example.com/gold-a", "https://example.com/gold-b"], 40)
    new = item("new01", ["https://example.com/gold-b", "https://example.com/gold-c"], 0)
    merged = merge([new], [old])
    if len(merged) != 1:
        failures.append(
            f"merge kept {len(merged)} items for one story under two ids, expected 1"
        )
    elif merged[0].cluster_id != "new01":
        failures.append(f"merge kept the STALE analysis ({merged[0].cluster_id})")

    # Genuinely different stories must both survive.
    other = item("other1", ["https://example.com/rupee-a"], 10)
    if len(merge([new], [other])) != 2:
        failures.append("merge collapsed two unrelated stories")

    ids, urls = published_signature([old, other])
    if ids != {"old01", "other1"}:
        failures.append(f"published_signature ids wrong: {ids}")
    if "https://example.com/gold-a" not in urls:
        failures.append("published_signature dropped a source url")


def main() -> int:
    approved = [(_fake_cluster(i), _analysis()) for i in range(1, 4)]
    site = render_site(approved, out_dir=OUT)

    failures: list[str] = []
    _check_publisher_scoping(failures)
    _check_archive_merge(failures)

    def must_exist(rel: str) -> Path | None:
        p = site / rel
        if not p.exists():
            failures.append(f"missing: {rel}")
            return None
        return p

    index = must_exist("index.html")
    must_exist("about.html")
    must_exist("contact.html")
    must_exist("static/style.css")
    must_exist("robots.txt")
    must_exist(".nojekyll")
    feed = must_exist("api/v1/feed.json")

    for i in range(1, 4):
        must_exist(f"story/cluster{i:02d}.html")
        must_exist(f"api/v1/clusters/cluster{i:02d}.json")

    # --- content checks ---------------------------------------------------
    if index:
        html = index.read_text(encoding="utf-8")
        for needle, why in [
            ("<title>", "title tag"),
            ("viewport", "mobile viewport"),
            ("not investment advice", "disclaimer text"),
            ("gist", "tier-2 gist block"),
            ("hook", "tier-1 hook"),
        ]:
            if needle not in html:
                failures.append(f"index.html missing {why} ({needle!r})")

    story = site / "story" / "cluster01.html"
    if story.exists():
        html = story.read_text(encoding="utf-8")
        for needle, why in [
            ("What happened", "section 1"),
            ("Who is affected", "section 2"),
            ("Which sectors", "section 3"),
            ("transmission mechanism", "section 4"),
            ("Market reaction", "section 5"),
            ("bull case", "section 6a"),
            ("bear case", "section 6b"),
            ("Key variables to watch", "section 7"),
            ("What optimists are arguing", "bull attribution label"),
            ("What pessimists are arguing", "bear attribution label"),
            ("Sources", "source attribution"),
            ("not investment advice", "disclaimer"),
        ]:
            if needle.lower() not in html.lower():
                failures.append(f"story page missing {why} ({needle!r})")
        # Every source must link out to the publisher.
        if "https://example.com/story-1-0" not in html:
            failures.append("story page does not link back to the source article")

    if feed:
        data = json.loads(feed.read_text(encoding="utf-8"))
        if data.get("count") != 3:
            failures.append(f"feed.json count={data.get('count')}, expected 3")
        if "disclaimer" not in data:
            failures.append("feed.json missing disclaimer")
        if data["items"] and data["items"][0].get("tier") != "free":
            failures.append("feed.json items missing the monetization tier field")

    print(f"rendered test site to {OUT}")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all render checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
