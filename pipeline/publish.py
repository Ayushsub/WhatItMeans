"""[Stage 12] Publish to gh-pages as a single orphan commit.

THIS IS THE RETENTION MECHANISM (plan Part 3). Most designs "delete" data by
removing rows while git history keeps every version forever. Here the published
branch is rewritten as ONE commit with no parent on every run, so no article
text older than the most recent run exists anywhere in it.

  main       source code only — article data NEVER lands here
  gh-pages   published site, force-pushed as a fresh orphan commit each run
  data       the derived-facts ledger (no article text, so normal history is
             fine and useful)

Running locally without --push just builds ./site and does nothing to git.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PAGES_BRANCH = "gh-pages"

# The orphan commit is built on a throwaway local branch, never on PAGES_BRANCH
# itself. `git checkout --orphan gh-pages` fails with "a branch named 'gh-pages'
# already exists" on every run after the first, which would break publishing
# permanently once the branch exists locally. The local name is irrelevant
# because we push an explicit HEAD:gh-pages refspec.
_BUILD_BRANCH = "pages-build"


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("git %s", " ".join(args[1:]))
    return subprocess.run(
        args, cwd=cwd, check=check, capture_output=True, text=True, encoding="utf-8"
    )


def verify_retention(cwd: Path | None = None) -> bool:
    """Assert the published branch holds exactly one commit.

    This is the check from plan Part 10 — run it in CI so a regression in the
    publish step surfaces as a failed build rather than a silent retention
    breach.
    """
    cwd = cwd or ROOT
    # Check the REMOTE branch: that is what is actually published. The local
    # ref may not exist at all after a clean CI checkout.
    ref = f"origin/{PAGES_BRANCH}"
    try:
        _run(["git", "fetch", "-q", "origin", PAGES_BRANCH], cwd, check=False)
        out = _run(["git", "rev-list", "--count", ref], cwd, check=False)
        if out.returncode != 0:
            out = _run(["git", "rev-list", "--count", PAGES_BRANCH], cwd, check=False)
            ref = PAGES_BRANCH
    except FileNotFoundError:
        log.warning("git not available; skipping retention check")
        return True
    if out.returncode != 0:
        log.info("branch %s does not exist yet", PAGES_BRANCH)
        return True
    count = int((out.stdout or "0").strip() or 0)
    if count > 1:
        log.error(
            "RETENTION BREACH: %s has %d commits, expected 1. Old article data "
            "is reachable in git history.", ref, count,
        )
        return False
    log.info("retention check OK: %s has %d commit", ref, count)
    return True


def publish(site_dir: Path, push: bool = False, cwd: Path | None = None) -> bool:
    """Force-push site_dir to gh-pages as a single parentless commit."""
    cwd = cwd or ROOT

    if not push:
        log.info("built %s (not pushed; pass --push or set PUBLISH=1)", site_dir)
        return True

    if not (site_dir / "index.html").exists():
        log.error("refusing to publish: %s has no index.html", site_dir)
        return False

    # An empty build means every provider was rate-limited or every story was
    # blocked. Publishing it would replace a working site with a blank page.
    # Keeping the previous build is strictly better: its stories are still
    # inside the retention window, and the next run republishes normally.
    story_count = len(list((site_dir / "story").glob("*.html")))
    if story_count == 0:
        log.warning(
            "refusing to publish an empty site (0 stories) — keeping the "
            "previously published build"
        )
        return True

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    worktree = ROOT / ".pages-worktree"

    try:
        # A detached worktree keeps the main checkout untouched, so a failed
        # publish can never leave the source tree on the wrong branch.
        _run(["git", "worktree", "remove", "--force", str(worktree)], cwd, check=False)
        _run(["git", "worktree", "add", "--detach", str(worktree)], cwd)

        # Orphan branch: no parent, so no history to carry old data.
        _run(["git", "branch", "-D", _BUILD_BRANCH], cwd, check=False)
        _run(["git", "checkout", "--orphan", _BUILD_BRANCH], worktree)
        _run(["git", "rm", "-rf", "--cached", "."], worktree, check=False)

        for entry in worktree.iterdir():
            if entry.name == ".git":
                continue
            if entry.is_dir():
                import shutil

                shutil.rmtree(entry)
            else:
                entry.unlink()

        import shutil

        shutil.copytree(site_dir, worktree, dirs_exist_ok=True)

        _run(["git", "add", "-A"], worktree)
        status = _run(["git", "status", "--porcelain"], worktree, check=False)
        if not status.stdout.strip():
            log.info("no changes to publish")
            return True

        env_name = os.environ.get("GIT_AUTHOR_NAME", "newsx-bot")
        env_email = os.environ.get("GIT_AUTHOR_EMAIL", "newsx-bot@users.noreply.github.com")
        _run(["git", "-c", f"user.name={env_name}", "-c", f"user.email={env_email}",
              "commit", "-m", f"site: {stamp}"], worktree)

        # --force is REQUIRED and intentional: it discards the previous
        # published commit, which is exactly the retention guarantee.
        _run(["git", "push", "--force", "origin", f"HEAD:{PAGES_BRANCH}"], worktree)
        log.info("published to %s (%s)", PAGES_BRANCH, stamp)
        return True

    except subprocess.CalledProcessError as e:
        log.error("publish failed: %s\n%s", e, e.stderr)
        return False
    finally:
        # Remove the worktree first — the build branch cannot be deleted while
        # it is checked out anywhere.
        _run(["git", "worktree", "remove", "--force", str(worktree)], cwd, check=False)
        _run(["git", "branch", "-D", _BUILD_BRANCH], cwd, check=False)
