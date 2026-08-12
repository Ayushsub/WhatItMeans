"""[Stage 11] Render the static site, the JSON API, and social prompts.

Three consumers, one source of truth:
  web pages        humans, published to gh-pages
  /api/v1/*.json   a future paid API (plan Part 7) — versioned path from day one
  social_prompts/  LOCAL-ONLY draft LinkedIn/Instagram post prompts, written to
                   ROOT/social_prompts (gitignored) — never inside SITE, never
                   committed, never published. Read and used only on whatever
                   machine actually runs the pipeline locally.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm import load_prompt
from models import Analysis, Cluster, PublishedItem

from . import archive, config

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
_DEFAULT_SITE = ROOT / "site"
SITE = _DEFAULT_SITE  # published output location

SITE_NAME = "WhatItMeans"

_PLATFORM_FORMATS = {
    "linkedin": "1200-1500 chars, line breaks between sections, max 3 hashtags at the end",
    "instagram": "<=2200 chars, tighter lines, 5-8 hashtags",
}


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(WEB / "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _human_time(dt: datetime) -> str:
    """Relative time. Everything on the site is <24h old, so absolute
    timestamps would carry less information than 'about 3 hours ago'."""
    delta = datetime.now(timezone.utc) - dt
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    return dt.strftime("%d %b, %H:%M UTC")


def build_items(approved: list[tuple[Cluster, Analysis]]) -> list[PublishedItem]:
    now = datetime.now(timezone.utc)
    items: list[PublishedItem] = []
    for cluster, analysis in approved:
        items.append(
            PublishedItem(
                cluster_id=cluster.id,
                analysis=analysis,
                tags=cluster.tags,
                sources=[
                    {
                        "name": a.source_name,
                        "headline": a.title,
                        "url": str(a.url),
                        "published_at": a.published_at.isoformat(),
                    }
                    for a in cluster.articles
                ],
                published_at=cluster.lead.published_at,
                updated_at=now,
                confidence=analysis.confidence,
            )
        )
    items.sort(key=lambda i: i.published_at, reverse=True)
    return items


def _social_prompt(item: PublishedItem, platform: str) -> str:
    """Fill the stored template. NOT executed here — this is an artifact for a
    separate system to consume later."""
    return (
        load_prompt("social_template.txt")
        .replace("{PLATFORM}", platform)
        .replace("{FORMAT_SPEC}", _PLATFORM_FORMATS[platform])
        .replace(
            "{ANALYSIS_JSON}",
            json.dumps(item.analysis.model_dump(), ensure_ascii=False, indent=2),
        )
    )


def render_site(
    approved: list[tuple[Cluster, Analysis]],
    all_clusters: list[Cluster] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Render everything into out_dir (default ./site). Returns that directory.

    out_dir exists so tests can render to a scratch path — otherwise running
    the render test would silently overwrite real published output with
    fixture data.
    """
    SITE = out_dir or _DEFAULT_SITE

    # Carry forward still-fresh stories BEFORE the directory is wiped. Without
    # this the site would only ever contain the current run's handful of
    # stories instead of a rolling 24h feed. Skipped for test renders, which
    # pass an explicit out_dir and want deterministic contents.
    items = build_items(approved)
    if out_dir is None:
        items = archive.merge(items, archive.load_previous(SITE))

    env = _env()
    comp = config.compliance()
    now = datetime.now(timezone.utc)

    # Full rebuild every run. Anything not regenerated ceases to exist, which
    # is how the retention rule is enforced by construction (plan Part 3).
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "story").mkdir(parents=True)
    (SITE / "api" / "v1" / "clusters").mkdir(parents=True)
    shutil.copytree(WEB / "static", SITE / "static")

    # Social prompts are a LOCAL-ONLY artifact — never published, never
    # committed. publish.py copies the whole SITE tree to gh-pages, so this
    # must live outside SITE in production or it would become public. Test
    # renders (out_dir given) keep it under the scratch SITE for assertion
    # convenience; that path is never published.
    SOCIAL_DIR = SITE / "social_prompts" if out_dir is not None else ROOT / "social_prompts"
    if SOCIAL_DIR.exists():
        shutil.rmtree(SOCIAL_DIR)
    SOCIAL_DIR.mkdir(parents=True)

    base_ctx = {
        "site_name": SITE_NAME,
        "disclaimer": comp["disclaimer"],
        "grievance": comp["grievance"],
        "retention_hours": config.RETENTION_HOURS,
        "updated_human": now.strftime("%d %b %Y, %H:%M UTC"),
    }

    # index
    (SITE / "index.html").write_text(
        env.get_template("index.html").render(
            **base_ctx,
            rel="",
            items=[_item_ctx(i) for i in items],
        ),
        encoding="utf-8",
    )

    # story pages
    tmpl = env.get_template("story.html")
    for item in items:
        ctx = _item_ctx(item)
        (SITE / "story" / f"{item.cluster_id}.html").write_text(
            tmpl.render(**base_ctx, rel="../", item=ctx, a=item.analysis),
            encoding="utf-8",
        )

    # static pages
    for page in ("about.html", "contact.html"):
        (SITE / page).write_text(
            env.get_template(page).render(**base_ctx, rel=""), encoding="utf-8"
        )

    # --- JSON API (versioned path from day one) -------------------------
    feed = {
        "version": "1.0",
        "generated_at": now.isoformat(),
        "retention_hours": config.RETENTION_HOURS,
        "disclaimer": comp["disclaimer"]["short"],
        "count": len(items),
        "items": [
            {
                "cluster_id": i.cluster_id,
                "headline": i.analysis.headline,
                "hook": i.analysis.hook,
                "gist": i.analysis.gist,
                "confidence": i.analysis.confidence,
                "tier": i.tier,
                "published_at": i.published_at.isoformat(),
                "url": f"story/{i.cluster_id}.html",
                "api": f"api/v1/clusters/{i.cluster_id}.json",
            }
            for i in items
        ],
    }
    (SITE / "api" / "v1" / "feed.json").write_text(
        json.dumps(feed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for item in items:
        (SITE / "api" / "v1" / "clusters" / f"{item.cluster_id}.json").write_text(
            json.dumps(item.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Social prompts (local-only artifact, never published) -----------
    for item in items:
        (SOCIAL_DIR / f"{item.cluster_id}.json").write_text(
            json.dumps(
                {
                    "cluster_id": item.cluster_id,
                    "headline": item.analysis.headline,
                    "generated_at": now.isoformat(),
                    "prompts": {
                        p: _social_prompt(item, p) for p in _PLATFORM_FORMATS
                    },
                    "analysis": item.analysis.model_dump(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    (SITE / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    # .nojekyll stops GitHub Pages running Jekyll over the output, which would
    # otherwise strip files and directories beginning with an underscore.
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    log.info("rendered %d stories to %s", len(items), SITE)
    return SITE


def _item_ctx(item: PublishedItem) -> dict:
    """Template context for one item — adds the humanized timestamp."""
    return {
        "cluster_id": item.cluster_id,
        "analysis": item.analysis,
        "tags": item.tags,
        "sources": item.sources,
        "published_at": item.published_at,
        "published_human": _human_time(item.published_at),
        "confidence": item.confidence,
        "tier": item.tier,
    }
