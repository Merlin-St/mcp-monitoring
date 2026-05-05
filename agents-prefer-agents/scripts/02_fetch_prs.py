"""Fetch PR events for every repo in data/repos.json.

Uses GitHub GraphQL API to bulk-load PR metadata, commits, reviews,
review-thread comments, issue comments, and merge/close timeline events in
a single round trip per 50-PR page. This reduces the ~5 REST endpoints per
PR to ~1 GraphQL call per 50 PRs.

Window: [2025-04-01, 2026-03-31] by PR updated_at (inclusive).
Cap: 10000 PRs per repo, subsampled uniformly by week if more.

Checkpointing: writes one ``data/prs/{owner}__{repo}.jsonl`` per repo, plus
``data/prs/_done.json`` tracking which repos are complete. Re-running skips
done repos. Rows already written are not re-written within a repo (we resume
per-repo from scratch to avoid half-written state).

Output schema (one JSON object per PR):
    {
      "repo": "owner/name",
      "number": 123,
      "title": "...",
      "body": "...",
      "author_login": "...",
      "author_type": "Bot"|"User"|"Mannequin",
      "created_at": "...",
      "updated_at": "...",
      "closed_at": "...",
      "merged_at": "...",
      "merged": bool,
      "merged_by_login": "...",
      "merged_by_type": "Bot"|"User",
      "additions": int,
      "deletions": int,
      "changed_files": int,
      "commits": [{"oid":..., "message":..., "author_login":..., "committer_login":..., "authored_at":...}],
      "reviews": [{"author_login":..., "state":..., "body":..., "submitted_at":...}],
      "review_comments": [{"author_login":..., "body":..., "created_at":...}],
      "issue_comments": [{"author_login":..., "body":..., "created_at":...}],
      "timeline_events": [{"type":..., "actor_login":..., "created_at":...}]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

utils = import_module("99_utils")
get_github_token = utils.get_github_token
get_logger = utils.get_logger
parse_datetime_safe = utils.parse_datetime_safe
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

PRS_DIR = DATA_DIR / "prs"
PRS_DIR.mkdir(parents=True, exist_ok=True)
DONE_FILE = PRS_DIR / "_done.json"
COUNTS_FILE = PRS_DIR / "_repo_pr_counts.json"

WINDOW_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)

GRAPHQL_URL = "https://api.github.com/graphql"

logger = get_logger("02_fetch_prs")


# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------
PR_LIST_QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:30, orderBy:{field:CREATED_AT, direction:DESC}, after:$cursor, states:[OPEN, CLOSED, MERGED]) {
      pageInfo { hasNextPage endCursor }
      totalCount
      nodes {
        number
        title
        body
        createdAt
        updatedAt
        closedAt
        mergedAt
        merged
        additions
        deletions
        changedFiles
        author { login __typename }
        mergedBy { login __typename }
        commits(first:30) {
          totalCount
          nodes {
            commit {
              oid
              message
              authoredDate
              author { user { login } name email }
              committer { user { login } name email }
            }
          }
        }
        reviews(first:8) {
          totalCount
          nodes {
            author { login __typename }
            state
            body
            submittedAt
          }
        }
        reviewThreads(first:4) {
          nodes {
            comments(first:3) {
              nodes {
                author { login __typename }
                body
                createdAt
              }
            }
          }
        }
        comments(first:10) {
          totalCount
          nodes {
            author { login __typename }
            body
            createdAt
          }
        }
        timelineItems(first:8, itemTypes:[MERGED_EVENT, CLOSED_EVENT]) {
          nodes {
            __typename
            ... on MergedEvent { actor{login __typename} createdAt }
            ... on ClosedEvent { actor{login __typename} createdAt }
          }
        }
      }
    }
  }
  rateLimit { remaining resetAt cost used }
}
"""


def _login(obj):
    if not obj:
        return ""
    return obj.get("login") or ""


def _type(obj):
    if not obj:
        return ""
    return obj.get("__typename") or ""


def flatten_pr(node: dict, repo_full: str) -> dict:
    """Turn a GraphQL PR node into the flat row format we store."""
    commits = []
    for c in (node.get("commits") or {}).get("nodes", []):
        commit = c.get("commit") or {}
        author = commit.get("author") or {}
        committer = commit.get("committer") or {}
        commits.append(
            {
                "oid": commit.get("oid", ""),
                "message": commit.get("message", "") or "",
                "authored_at": commit.get("authoredDate", ""),
                "author_login": _login(author.get("user") or {}),
                "author_name": author.get("name", ""),
                "author_email": author.get("email", ""),
                "committer_login": _login(committer.get("user") or {}),
                "committer_name": committer.get("name", ""),
                "committer_email": committer.get("email", ""),
            }
        )

    reviews = []
    for r in (node.get("reviews") or {}).get("nodes", []):
        reviews.append(
            {
                "author_login": _login(r.get("author") or {}),
                "author_type": _type(r.get("author") or {}),
                "state": r.get("state", ""),
                "body": r.get("body", "") or "",
                "submitted_at": r.get("submittedAt", ""),
            }
        )

    rc = []
    for thread in (node.get("reviewThreads") or {}).get("nodes", []):
        for c in (thread.get("comments") or {}).get("nodes", []):
            rc.append(
                {
                    "author_login": _login(c.get("author") or {}),
                    "author_type": _type(c.get("author") or {}),
                    "body": c.get("body", "") or "",
                    "created_at": c.get("createdAt", ""),
                }
            )

    ic = []
    for c in (node.get("comments") or {}).get("nodes", []):
        ic.append(
            {
                "author_login": _login(c.get("author") or {}),
                "author_type": _type(c.get("author") or {}),
                "body": c.get("body", "") or "",
                "created_at": c.get("createdAt", ""),
            }
        )

    tl = []
    for ev in (node.get("timelineItems") or {}).get("nodes", []):
        tl.append(
            {
                "type": ev.get("__typename", ""),
                "actor_login": _login(ev.get("actor") or {}),
                "actor_type": _type(ev.get("actor") or {}),
                "created_at": ev.get("createdAt", ""),
            }
        )

    return {
        "repo": repo_full,
        "number": node.get("number"),
        "title": node.get("title", "") or "",
        "body": node.get("body", "") or "",
        "author_login": _login(node.get("author") or {}),
        "author_type": _type(node.get("author") or {}),
        "created_at": node.get("createdAt", ""),
        "updated_at": node.get("updatedAt", ""),
        "closed_at": node.get("closedAt", ""),
        "merged_at": node.get("mergedAt", ""),
        "merged": bool(node.get("merged", False)),
        "merged_by_login": _login(node.get("mergedBy") or {}),
        "merged_by_type": _type(node.get("mergedBy") or {}),
        "additions": node.get("additions", 0),
        "deletions": node.get("deletions", 0),
        "changed_files": node.get("changedFiles", 0),
        "commits": commits,
        "reviews": reviews,
        "review_comments": rc,
        "issue_comments": ic,
        "timeline_events": tl,
        "commits_total": (node.get("commits") or {}).get("totalCount", len(commits)),
        "reviews_total": (node.get("reviews") or {}).get("totalCount", len(reviews)),
        "comments_total": (node.get("comments") or {}).get("totalCount", len(ic)),
    }


# ---------------------------------------------------------------------------
# GraphQL client with rate-limit awareness
# ---------------------------------------------------------------------------
class GraphQLClient:
    def __init__(self, token: str, max_concurrent: int = 6, min_interval_sec: float = 1.0):
        self.token = token
        self.sem = asyncio.Semaphore(max_concurrent)
        self._session: aiohttp.ClientSession | None = None
        self.remaining = 5000
        self.reset_at_ts = 0
        self.request_count = 0
        self.log = logger
        # Global pacing: minimum seconds between request starts, across all workers.
        # Protects against secondary rate limits (GitHub recommends <80 req/min).
        self.min_interval_sec = min_interval_sec
        self._next_allowed_ts = 0.0
        self._pacing_lock = asyncio.Lock()

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "User-Agent": "agents-prefer-agents",
                },
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _pace(self):
        """Enforce a minimum interval between successive request starts."""
        async with self._pacing_lock:
            now = time.time()
            wait = max(0.0, self._next_allowed_ts - now)
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.time()
            self._next_allowed_ts = now + self.min_interval_sec

    async def query(self, q: str, variables: dict, max_retries: int = 4) -> dict | None:
        async with self.sem:
            await self._pace()
            session = await self._session_get()
            body = json.dumps({"query": q, "variables": variables})
            for attempt in range(max_retries):
                try:
                    async with session.post(GRAPHQL_URL, data=body) as r:
                        self.request_count += 1
                        if r.status == 200:
                            data = await r.json()
                            rl = (data.get("data") or {}).get("rateLimit") or {}
                            if rl:
                                self.remaining = rl.get("remaining", self.remaining)
                                ra = rl.get("resetAt", "")
                                dt = parse_datetime_safe(ra)
                                if dt:
                                    self.reset_at_ts = int(dt.timestamp())
                                if self.remaining < 50:
                                    wait = max(self.reset_at_ts - time.time(), 0) + 2
                                    self.log.warning(
                                        "GraphQL rate low (%d). Sleeping %.0fs.",
                                        self.remaining,
                                        wait,
                                    )
                                    await asyncio.sleep(wait)
                            if "errors" in data and data["errors"]:
                                # Some errors are partial (like TIMEOUT on subfields) and
                                # we still want the data we got. Only fail hard on AUTH etc.
                                types = [e.get("type", "") for e in data["errors"]]
                                if "UNAUTHORIZED" in types or "FORBIDDEN" in types:
                                    self.log.error(
                                        "GraphQL auth error: %s", data["errors"][:2]
                                    )
                                    return None
                                self.log.warning(
                                    "GraphQL partial errors (attempt %d): %s",
                                    attempt + 1,
                                    types[:3],
                                )
                            return data
                        if r.status in (502, 503, 504):
                            self.log.warning("GraphQL transient %d; retrying", r.status)
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if r.status == 403:
                            txt = await r.text()
                            # Honour Retry-After if present; otherwise long backoff.
                            retry_after = r.headers.get("Retry-After")
                            wait_s = 60
                            if retry_after:
                                try:
                                    wait_s = max(int(retry_after), 30)
                                except ValueError:
                                    pass
                            # Exponential attempt backoff on top of base wait.
                            wait_s += 15 * attempt
                            self.log.warning(
                                "GraphQL 403 (secondary rate limit). Sleeping %ds. body=%s",
                                wait_s,
                                txt[:120],
                            )
                            # Also shift global pacing window — no new requests for at
                            # least wait_s seconds.
                            async with self._pacing_lock:
                                self._next_allowed_ts = max(
                                    self._next_allowed_ts, time.time() + wait_s
                                )
                            await asyncio.sleep(wait_s)
                            continue
                        self.log.error(
                            "GraphQL status %d body %s", r.status, (await r.text())[:300]
                        )
                        return None
                except asyncio.TimeoutError:
                    self.log.warning("GraphQL timeout (attempt %d)", attempt + 1)
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    self.log.warning("GraphQL err (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)
            return None


# ---------------------------------------------------------------------------
# Per-repo fetch + filter
# ---------------------------------------------------------------------------
async def fetch_repo_prs(
    client: GraphQLClient,
    owner: str,
    name: str,
    max_prs_per_repo: int,
) -> tuple[list[dict], int]:
    """Page through all PRs updated in [WINDOW_START, WINDOW_END].

    Returns ``(rows, total_in_window_before_cap)``: ``rows`` is the (possibly
    subsampled) flat PR dicts (see flatten_pr); the second element is the
    total in-window PR count before any per-repo cap was applied. Callers
    can use the second element to compute "share of repos with > cap PRs"
    and to characterise the PR-volume distribution of the universe.

    We walk UPDATED_AT desc; as soon as we see updatedAt < WINDOW_START, we stop.
    PRs with updatedAt > WINDOW_END are still written if they were created/merged
    within window, but we use updatedAt > WINDOW_END to mean "keep going" since
    we're walking desc: the first few pages may be entirely out-of-window new.
    """
    rows: list[dict] = []
    cursor = None
    pages = 0
    first_kept_page = 0  # index of first page where at least one PR was kept
    empty_pages_streak = 0
    MAX_EMPTY_PAGES_STREAK = 25  # allow walking through post-window newest PRs (up to 750 post-window PRs)
    MAX_PAGES = 350  # hard cap per repo (~10500 raw PRs); early-exit on out-of-window keeps cost zero for normal repos
    LOW_YIELD_PAGES = 20  # after this many *post-first-kept* pages, require density
    LOW_YIELD_MIN_RATIO = 0.20
    while pages < MAX_PAGES:
        pages += 1
        data = await client.query(PR_LIST_QUERY, {"owner": owner, "name": name, "cursor": cursor})
        if not data or not data.get("data"):
            logger.warning("No data for %s/%s on page %d", owner, name, pages)
            break
        repo_obj = (data["data"] or {}).get("repository")
        if not repo_obj:
            logger.warning("Repo %s/%s not accessible (deleted/renamed?)", owner, name)
            break
        conn = repo_obj.get("pullRequests") or {}
        nodes = conn.get("nodes") or []
        if not nodes:
            break

        page_in_window = 0
        any_older_than_window = False
        for node in nodes:
            created = parse_datetime_safe(node.get("createdAt", ""))
            if not created:
                continue
            # Sorted by createdAt DESC: once created < WINDOW_START, everything
            # after is older; stop paging.
            if created < WINDOW_START:
                any_older_than_window = True
                continue
            # PR is too new (created after our window end); skip but keep paging —
            # earlier pages may contain PRs whose createdAt is in-window.
            if created > WINDOW_END:
                continue
            row = flatten_pr(node, f"{owner}/{name}")
            rows.append(row)
            page_in_window += 1

        pinfo = conn.get("pageInfo") or {}
        has_next = pinfo.get("hasNextPage", False)
        cursor = pinfo.get("endCursor")
        if page_in_window == 0:
            empty_pages_streak += 1
        else:
            empty_pages_streak = 0
            if first_kept_page == 0:
                first_kept_page = pages
        if pages % 5 == 0 or page_in_window > 0 or any_older_than_window:
            logger.info(
                "  %s/%s page %d: kept %d/%d  (cumulative=%d, empty_streak=%d, older_tail=%s)",
                owner,
                name,
                pages,
                page_in_window,
                len(nodes),
                len(rows),
                empty_pages_streak,
                any_older_than_window,
            )
        if not has_next:
            break
        if any_older_than_window:
            break
        if empty_pages_streak >= MAX_EMPTY_PAGES_STREAK:
            logger.info(
                "  %s/%s: %d consecutive empty pages; giving up pagination (likely all PRs post-window).",
                owner,
                name,
                empty_pages_streak,
            )
            break
        # Low-yield cutoff: after we've walked LOW_YIELD_PAGES post-first-keep,
        # require density. This tolerates walking through many pages of post-
        # window newest PRs before hitting in-window ones.
        pages_since_first = (pages - first_kept_page) if first_kept_page else 0
        if first_kept_page and pages_since_first >= LOW_YIELD_PAGES:
            pages_ratio = len(rows) / max(pages_since_first * 30, 1)
            if pages_ratio < LOW_YIELD_MIN_RATIO:
                logger.info(
                    "  %s/%s: low yield post-kept (%d rows in %d pages, ratio=%.3f); stopping.",
                    owner,
                    name,
                    len(rows),
                    pages_since_first,
                    pages_ratio,
                )
                break
        # IMPORTANT: we deliberately do NOT stop early on `len(rows) >= max_prs_per_repo`,
        # because CREATED_AT DESC gives newest-created first. Stopping at cap would
        # bias the sample to the most-recent months. Instead we walk to MAX_PAGES
        # (or an earlier stop trigger) and subsample uniformly by week after.

    # Record the true in-window PR count BEFORE any subsampling. Used by
    # downstream stats (share of repos exceeding the per-repo cap, etc.).
    total_in_window_before_cap = len(rows)

    # If we somehow overshot, subsample uniformly by week.
    if len(rows) > max_prs_per_repo:
        # Group by ISO week of created_at, then sample evenly.
        from collections import defaultdict

        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            dt = parse_datetime_safe(r["created_at"])
            if dt:
                iso = dt.isocalendar()
                key = f"{iso.year}-W{iso.week:02d}"
            else:
                key = "unknown"
            buckets[key].append(r)
        per_bucket = max(1, max_prs_per_repo // max(len(buckets), 1))
        sampled: list[dict] = []
        for k, bs in sorted(buckets.items()):
            bs.sort(key=lambda x: x.get("number", 0))
            stride = max(1, len(bs) // per_bucket)
            sampled.extend(bs[::stride][:per_bucket])
        rows = sampled[:max_prs_per_repo]
        logger.info(
            "  %s/%s subsampled to %d via weekly uniform sampling.", owner, name, len(rows)
        )
    return rows, total_in_window_before_cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def load_done() -> set[str]:
    if DONE_FILE.exists():
        return set(json.loads(DONE_FILE.read_text()))
    return set()


def save_done(done: set[str]):
    DONE_FILE.write_text(json.dumps(sorted(done)))


def load_counts() -> dict:
    if COUNTS_FILE.exists():
        return json.loads(COUNTS_FILE.read_text())
    return {}


def save_counts(counts: dict):
    COUNTS_FILE.write_text(json.dumps(counts, indent=2, sort_keys=True))


async def main_async(args):
    token = get_github_token()
    if not token:
        logger.error("No GitHub token.")
        sys.exit(1)

    repos_file = DATA_DIR / "repos.json"
    if not repos_file.exists():
        logger.error("data/repos.json not found; run 01_build_repo_list.py first.")
        sys.exit(1)
    repos = json.loads(repos_file.read_text())
    if args.limit:
        repos = repos[: args.limit]
    done = load_done()
    counts = load_counts()
    todo = [r for r in repos if r["full_name"] not in done]
    logger.info(
        "Loaded %d repos; %d already done; %d to process.",
        len(repos),
        len(done),
        len(todo),
    )

    client = GraphQLClient(token, max_concurrent=args.concurrency, min_interval_sec=args.min_interval)
    t0 = time.time()
    total_prs_holder = [0]  # mutable so the coroutine can update

    # Process repos in parallel. Each worker pulls from the queue until empty.
    repo_queue: asyncio.Queue = asyncio.Queue()
    for i, repo in enumerate(todo):
        repo_queue.put_nowait((i, repo))
    n_total = repo_queue.qsize()

    async def worker(worker_id: int):
        while True:
            try:
                i, repo = repo_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            owner, name = repo["owner"], repo["repo"]
            full = f"{owner}/{name}"
            out_path = PRS_DIR / f"{owner}__{name}.jsonl"
            logger.info(
                "[%d/%d][w%d] Fetching %s (stars=%d)",
                i + 1,
                n_total,
                worker_id,
                full,
                repo.get("stars", 0),
            )
            try:
                rows, total_in_window = await fetch_repo_prs(client, owner, name, args.max_prs_per_repo)
            except Exception as e:
                logger.error("Fetch failed for %s: %s", full, e)
                continue
            with open(out_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_prs_holder[0] += len(rows)
            counts[full] = {
                "total_in_window": total_in_window,
                "written": len(rows),
                "saturated_at_cap": total_in_window > args.max_prs_per_repo,
            }
            done.add(full)
            # Periodic checkpoint from one worker.
            if len(done) % 10 == 0 and worker_id == 0:
                save_done(done)
                save_counts(counts)
                logger.info(
                    "Checkpoint: %d/%d repos done, %d PRs total, GraphQL remaining=%d, elapsed=%.0fs",
                    len(done),
                    n_total,
                    total_prs_holder[0],
                    client.remaining,
                    time.time() - t0,
                )

    try:
        workers = [asyncio.create_task(worker(k)) for k in range(args.repo_concurrency)]
        await asyncio.gather(*workers)
    finally:
        save_done(done)
        save_counts(counts)
        await client.close()
    total_prs = total_prs_holder[0]

    # Final stats
    stats = {
        "generated_at": utils.now_iso(),
        "repos_done": len(done),
        "repos_attempted": len(todo),
        "total_prs": total_prs,
        "graphql_requests": client.request_count,
        "graphql_remaining_end": client.remaining,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    logger.info("=== PHASE 2 DONE ===")
    logger.info(json.dumps(stats, indent=2))
    with open(RESULTS_DIR / "phase2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process only first N repos (debug).")
    parser.add_argument("--max-prs-per-repo", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent GraphQL HTTP requests.")
    parser.add_argument("--repo-concurrency", type=int, default=3, help="Repos processed in parallel.")
    parser.add_argument("--min-interval", type=float, default=0.9,
                        help="Minimum seconds between successive GraphQL requests (all workers). "
                             "Protects against secondary rate limits (~80 req/min). Default 0.9.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
