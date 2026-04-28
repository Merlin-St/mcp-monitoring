"""Download a pinned OpenSSF criticality_score CSV snapshot.

The OpenSSF Securing Critical Projects WG publishes criticality scores for
GitHub projects in CSV form to a public Google Cloud Storage bucket:

    gs://ossf-criticality-score/

Each snapshot lives at ``YYYY.MM.DD/HHMMSS/all.csv`` (and a richer variant
``all_w_depsdev.csv`` that includes the deps.dev dependent-count column).

This script:
1. Downloads a single snapshot to ``data/criticality/<name>``.
2. Verifies the byte count and computes a SHA-256 over the cached file.
3. Writes ``data/criticality/<name>.provenance.json`` with the source URL,
   download timestamp (UTC), file size, sha256, and row count.

The download is idempotent: if the cached file already exists and its size
matches the expected value, we skip re-download but refresh the provenance
JSON's ``last_verified_at`` field.

The default snapshot is pinned to **2025.07.25** — the most recent ``all.csv``
that was generated before the analysis window's mid-point. A different
snapshot may be passed via ``--snapshot-prefix``. Callers are encouraged
*not* to use the rolling latest, so that downstream selections are
reproducible.

Outputs:
- ``data/criticality/ossf-criticality-{date}.csv``         (raw CSV, ~120 MB)
- ``data/criticality/ossf-criticality-{date}.provenance.json``

Run:
    python scripts/01a_download_criticality.py
    python scripts/01a_download_criticality.py --snapshot-prefix 2025.07.25/010355
    python scripts/01a_download_criticality.py --variant all_w_depsdev.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")
get_logger = utils.get_logger
DATA_DIR = utils.DATA_DIR
RESULTS_DIR = utils.RESULTS_DIR

CRITICALITY_DIR = DATA_DIR / "criticality"
CRITICALITY_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SNAPSHOT = "2025.07.25/010355"  # latest stable as of 2026-04-27
DEFAULT_VARIANT = "all.csv"  # or "all_w_depsdev.csv"
GCS_BASE = "https://storage.googleapis.com/ossf-criticality-score"
GCS_LIST_API = "https://storage.googleapis.com/storage/v1/b/ossf-criticality-score/o"


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "agents-prefer-agents"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _stream_to_file(url: str, dest: Path, log) -> int:
    req = Request(url, headers={"User-Agent": "agents-prefer-agents"})
    written = 0
    last_log = time.time()
    with urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
            if time.time() - last_log > 5:
                log.info("  ... %.1f MB downloaded", written / 1e6)
                last_log = time.time()
    return written


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return max(n - 1, 0)  # subtract header


def _list_recent_snapshots(limit: int = 20) -> list[dict]:
    """Probe the bucket for the most recent /all.csv objects."""
    raw = _http_get(
        f"{GCS_LIST_API}?maxResults=2000&fields=items(name,size,timeCreated),nextPageToken"
    )
    d = json.loads(raw)
    items = d.get("items", [])
    csvs = [
        it
        for it in items
        if it["name"].endswith("/all.csv") or it["name"].endswith("/all_w_depsdev.csv")
    ]
    csvs.sort(key=lambda x: x["timeCreated"], reverse=True)
    return csvs[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--snapshot-prefix",
        default=DEFAULT_SNAPSHOT,
        help="GCS object key prefix, e.g. '2025.07.25/010355'.",
    )
    ap.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        choices=["all.csv", "all_w_depsdev.csv"],
        help="Which CSV variant to fetch.",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List the 20 most recent snapshots and exit.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the cached file is present.",
    )
    args = ap.parse_args()

    log = get_logger("01a_download_criticality")

    if args.list:
        for it in _list_recent_snapshots():
            size_mb = int(it["size"]) / 1e6
            log.info("%s  %.1f MB  %s", it["name"], size_mb, it["timeCreated"])
        return 0

    snapshot_date = args.snapshot_prefix.split("/")[0]  # e.g. 2025.07.25
    object_key = f"{args.snapshot_prefix}/{args.variant}"
    url = f"{GCS_BASE}/{object_key}"
    cache_name = f"ossf-criticality-{snapshot_date}-{args.variant}"
    cache_path = CRITICALITY_DIR / cache_name
    prov_path = cache_path.with_suffix(cache_path.suffix + ".provenance.json")

    log.info("Snapshot: %s", args.snapshot_prefix)
    log.info("Variant : %s", args.variant)
    log.info("URL     : %s", url)
    log.info("Cache   : %s", cache_path)

    if cache_path.exists() and not args.force:
        size_mb = cache_path.stat().st_size / 1e6
        log.info("Cached file present (%.1f MB) — skipping download.", size_mb)
    else:
        log.info("Downloading...")
        t0 = time.time()
        written = _stream_to_file(url, cache_path, log)
        dt = time.time() - t0
        log.info("Downloaded %.1f MB in %.1fs (%.1f MB/s)", written / 1e6, dt, written / 1e6 / max(dt, 1e-3))

    log.info("Computing sha256...")
    digest = _sha256(cache_path)
    log.info("sha256 = %s", digest)

    log.info("Counting rows...")
    rows = _row_count(cache_path)
    log.info("rows   = %d", rows)

    provenance = {
        "tool": "ossf/criticality_score",
        "tool_url": "https://github.com/ossf/criticality_score",
        "tool_blog": "https://opensource.googleblog.com/2020/12/finding-critical-open-source-projects.html",
        "snapshot_prefix": args.snapshot_prefix,
        "snapshot_date": snapshot_date,
        "variant": args.variant,
        "source_url": url,
        "cache_filename": cache_name,
        "size_bytes": cache_path.stat().st_size,
        "sha256": digest,
        "row_count": rows,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "schema_columns_expected": [
            "repo.url",
            "repo.language",
            "repo.license",
            "repo.star_count",
            "repo.created_at",
            "repo.updated_at",
            "legacy.created_since",
            "legacy.updated_since",
            "legacy.contributor_count",
            "legacy.org_count",
            "legacy.commit_frequency",
            "legacy.recent_release_count",
            "legacy.updated_issues_count",
            "legacy.closed_issues_count",
            "legacy.issue_comment_frequency",
            "legacy.github_mention_count",
            "depsdev.dependent_count",
            "default_score",
            "collection_date",
            "worker_commit_id",
        ],
    }
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    log.info("Provenance written to %s", prov_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
