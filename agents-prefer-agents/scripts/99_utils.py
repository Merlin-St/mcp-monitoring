"""Shared utilities for the agents-prefer-agents pipeline.

- GitHub token resolver (env var → `gh auth token` fallback).
- Rate-limited async REST client (ported from detect_ai_created.py).
- Rate-limited synchronous REST client (for simple scripts / Phase 1).
- Logging setup that writes to both stdout and ``agents-prefer-agents/logs/{name}.log``.
- Small helpers (``iso_week``, ``extract_owner_repo``, ``parse_datetime_safe``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp
import requests

SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = SUBPROJECT_ROOT / "logs"
DATA_DIR = SUBPROJECT_ROOT / "data"
RESULTS_DIR = SUBPROJECT_ROOT / "results"
for p in (LOG_DIR, DATA_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to logs/{name}.log AND stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_DIR / f"{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------
def get_github_token() -> str:
    """Resolve GitHub token: GH_TOKEN → GITHUB_TOKEN → `gh auth token`."""
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(var, "")
        if v:
            return v
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


# ---------------------------------------------------------------------------
# Sync REST client (for Phase 1 repo list + anything not worth asyncifying)
# ---------------------------------------------------------------------------
class GitHubSyncClient:
    """Simple synchronous GH REST client with rate-limit awareness."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "agents-prefer-agents",
            }
        )
        self.core_remaining = 5000
        self.core_reset = 0
        self.search_remaining = 30
        self.search_reset = 0
        self.request_count = 0
        self.log = get_logger("gh_sync")

    def _handle_rate_headers(self, r: requests.Response, is_search: bool = False):
        rem = r.headers.get("X-RateLimit-Remaining")
        rst = r.headers.get("X-RateLimit-Reset")
        if rem is not None:
            if is_search:
                self.search_remaining = int(rem)
            else:
                self.core_remaining = int(rem)
        if rst is not None:
            if is_search:
                self.search_reset = int(rst)
            else:
                self.core_reset = int(rst)

        # If we are under 10 on core or 3 on search, pause until reset.
        remaining = self.search_remaining if is_search else self.core_remaining
        reset_ts = self.search_reset if is_search else self.core_reset
        threshold = 3 if is_search else 50
        if remaining < threshold:
            wait = max(reset_ts - time.time(), 0) + 2
            self.log.warning(
                "Rate limit low (remaining=%d, is_search=%s). Sleeping %.0fs.",
                remaining,
                is_search,
                wait,
            )
            time.sleep(wait)

    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        is_search: bool = False,
        max_retries: int = 3,
    ) -> tuple[int, Any]:
        url = f"{self.BASE_URL}{path}" if path.startswith("/") else path
        for attempt in range(max_retries):
            try:
                r = self.session.get(url, params=params, timeout=30)
                self.request_count += 1
                self._handle_rate_headers(r, is_search=is_search)
                if r.status_code == 200:
                    return r.status_code, r.json()
                if r.status_code == 403:
                    body = r.text
                    if "rate limit" in body.lower() or "abuse" in body.lower():
                        reset_ts = self.search_reset if is_search else self.core_reset
                        wait = max(reset_ts - time.time(), 0) + 5
                        self.log.warning("403 rate-limited. Sleeping %.0fs.", wait)
                        time.sleep(wait)
                        continue
                    return r.status_code, None
                if r.status_code in (404, 422, 451):
                    return r.status_code, None
                self.log.debug("Unexpected status %d for %s", r.status_code, path)
                return r.status_code, None
            except requests.exceptions.Timeout:
                self.log.warning("Timeout for %s (attempt %d)", path, attempt + 1)
                time.sleep(2 ** attempt)
            except Exception as e:
                self.log.warning("Error for %s: %s", path, e)
                time.sleep(2 ** attempt)
        return 0, None

    def search(self, path: str, params: dict) -> tuple[int, Any]:
        return self.get(path, params=params, is_search=True)


# ---------------------------------------------------------------------------
# Async REST client (for Phase 2 PR collection — ported from detect_ai_created.py)
# ---------------------------------------------------------------------------
class GitHubAsyncClient:
    """Async GH REST client with rate-limit handling and bounded concurrency."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, max_concurrent: int = 8):
        self.token = token
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.core_remaining = 5000
        self.core_reset = 0
        self.request_count = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self.log = get_logger("gh_async")

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "agents-prefer-agents",
                },
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _handle_rate_limit(self, response: aiohttp.ClientResponse):
        rem = response.headers.get("X-RateLimit-Remaining")
        rst = response.headers.get("X-RateLimit-Reset")
        if rem is not None:
            self.core_remaining = int(rem)
        if rst is not None:
            self.core_reset = int(rst)
        if self.core_remaining < 50:
            wait = max(self.core_reset - time.time(), 0) + 2
            self.log.warning(
                "Rate low (%d remaining). Sleeping %.0fs.", self.core_remaining, wait
            )
            await asyncio.sleep(wait)

    async def get(
        self, path: str, params: Optional[dict] = None, max_retries: int = 3
    ) -> tuple[int, Any]:
        async with self.semaphore:
            url = f"{self.BASE_URL}{path}" if path.startswith("/") else path
            session = await self._session_get()
            for attempt in range(max_retries):
                try:
                    async with session.get(url, params=params) as r:
                        self.request_count += 1
                        await self._handle_rate_limit(r)
                        if r.status == 200:
                            return r.status, await r.json()
                        if r.status == 403:
                            body = await r.text()
                            if "rate limit" in body.lower() or "abuse" in body.lower():
                                wait = max(self.core_reset - time.time(), 0) + 5
                                self.log.warning(
                                    "403 rate-limited. Sleeping %.0fs.", wait
                                )
                                await asyncio.sleep(wait)
                                continue
                            return r.status, None
                        if r.status in (404, 422, 451):
                            return r.status, None
                        return r.status, None
                except asyncio.TimeoutError:
                    self.log.warning("Timeout for %s (attempt %d)", path, attempt + 1)
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    self.log.warning("Error for %s: %s", path, e)
                    await asyncio.sleep(2 ** attempt)
            return 0, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_owner_repo(github_url: str) -> Optional[tuple[str, str]]:
    """Extract (owner, repo) from a github URL or 'owner/repo' id."""
    if not github_url:
        return None
    s = github_url.rstrip("/")
    if s.endswith(".git"):
        s = s[:-4]
    m = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", s)
    if m:
        return m.group(1), m.group(2)
    # bare owner/repo
    m = re.match(r"^([^/]+)/([^/]+)$", s)
    if m:
        return m.group(1), m.group(2)
    return None


def parse_datetime_safe(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def iso_week(dt: datetime) -> str:
    """Return 'YYYY-Www' ISO-week label for a datetime."""
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
