#!/usr/bin/env python3
"""99_test_access.py — Pilot test of external access the autonomous agent needs.

Run:
    /home/ubuntu/mcp-monitoring/.venv/bin/python agents-prefer-agents/99_test_access.py

Exits 0 if all BLOCKERs pass, 1 otherwise. WARNs are surfaced but non-blocking.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Literal

Status = Literal["PASS", "FAIL", "WARN", "SKIP"]
results: list[tuple[str, Status, str]] = []


def record(name: str, status: Status, detail: str = "") -> None:
    results.append((name, status, detail))
    tag = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[status]
    print(f"{tag} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. GitHub token (mirrors scripts/data-classification-aicreatedmcp/gh-claude-activity.py)
# ---------------------------------------------------------------------------
print("\n=== GitHub API access ===")

gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
gh_token_source = ""
if os.environ.get("GITHUB_TOKEN"):
    gh_token_source = "GITHUB_TOKEN env var"
elif os.environ.get("GH_TOKEN"):
    gh_token_source = "GH_TOKEN env var"
else:
    # Fall back to gh CLI — useful for the pilot, but the autonomous scripts
    # read env vars directly, so flag this as WARN regardless of the result.
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0 and out.stdout.strip():
            gh_token = out.stdout.strip()
            gh_token_source = "gh CLI (env var NOT set — autonomous scripts will fail)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

if not gh_token:
    record(
        "GitHub token present",
        "FAIL",
        "neither GITHUB_TOKEN nor GH_TOKEN set, and gh CLI unavailable",
    )
else:
    is_env = gh_token_source.endswith("env var")
    record(
        "GitHub token present",
        "PASS" if is_env else "WARN",
        gh_token_source,
    )

    # Hit /rate_limit — the instruction's Phase 0.4 check.
    try:
        import requests

        resp = requests.get(
            "https://api.github.com/rate_limit",
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Authorization": f"Bearer {gh_token}",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            core = resp.json()["resources"]["core"]
            search = resp.json()["resources"]["search"]
            record(
                "GitHub /rate_limit reachable",
                "PASS",
                f"core={core['remaining']}/{core['limit']}, search={search['remaining']}/{search['limit']}",
            )
            if core["limit"] < 5000:
                record(
                    "GitHub token has authenticated quota",
                    "FAIL",
                    f"core limit={core['limit']} (expected 5000; token may be invalid)",
                )
            else:
                record("GitHub token has authenticated quota", "PASS", "")
        else:
            record(
                "GitHub /rate_limit reachable",
                "FAIL",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as e:
        record("GitHub /rate_limit reachable", "FAIL", f"{type(e).__name__}: {e}")

    # Tiny commit-search pilot mirroring gh-claude-activity.py
    try:
        import requests

        resp = requests.get(
            "https://api.github.com/search/commits",
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Authorization": f"Bearer {gh_token}",
            },
            params={
                "q": '"Co-authored-by: Claude" committer-date:2026-03-01',
                "per_page": 1,
            },
            timeout=20,
        )
        if resp.status_code == 200:
            total = resp.json().get("total_count", 0)
            record(
                "GitHub commit search API",
                "PASS",
                f"sample query returned total_count={total}",
            )
        else:
            record(
                "GitHub commit search API",
                "FAIL",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as e:
        record("GitHub commit search API", "FAIL", f"{type(e).__name__}: {e}")

    # GraphQL too — Phase 1/2 use it for bulk PR metadata.
    try:
        import requests

        resp = requests.post(
            "https://api.github.com/graphql",
            headers={"Authorization": f"Bearer {gh_token}"},
            json={"query": "{ viewer { login } }"},
            timeout=15,
        )
        if resp.status_code == 200 and "data" in resp.json():
            login = resp.json()["data"]["viewer"]["login"]
            record("GitHub GraphQL API", "PASS", f"viewer.login={login}")
        else:
            record(
                "GitHub GraphQL API",
                "FAIL",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as e:
        record("GitHub GraphQL API", "FAIL", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 2. Anthropic / LLM access (mirrors scripts/data-classification-servers/clservers_2_inspect.py)
# ---------------------------------------------------------------------------
print("\n=== LLM API access ===")

anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "")
if not anthropic_key:
    record("ANTHROPIC_API_KEY set", "FAIL", "env var missing")
else:
    record(
        "ANTHROPIC_API_KEY set",
        "PASS",
        f"len={len(anthropic_key)}"
        + (f", base_url={anthropic_base}" if anthropic_base else ", default base_url"),
    )

    # The AISI proxy expects the ARN to be resolved via aisitools before use
    # when calling the SDK directly (inspect_ai does this automatically).
    try:
        import anthropic

        resolved_key = anthropic_key
        if anthropic_key.startswith("aws-secretsmanager://"):
            from aisitools.api_key import get_api_key_for_proxy

            resolved_key = get_api_key_for_proxy(anthropic_key)

        client = anthropic.Anthropic(api_key=resolved_key)
        msg = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
        if text:
            record(
                "Anthropic messages API (claude-sonnet-4-5)",
                "PASS",
                f'reply="{text[:40]}" input_tokens={msg.usage.input_tokens} output_tokens={msg.usage.output_tokens}',
            )
        else:
            record(
                "Anthropic messages API (claude-sonnet-4-5)",
                "FAIL",
                "empty response content",
            )
    except Exception as e:
        record(
            "Anthropic messages API (claude-sonnet-4-5)",
            "FAIL",
            f"{type(e).__name__}: {str(e)[:300]}",
        )

# OpenAI — only used if the autonomous agent decides to mirror cltools
# (gpt-4o-mini). Non-blocking; report as WARN if absent, PASS if working.
openai_key = os.environ.get("OPENAI_API_KEY", "")
if not openai_key:
    record("OPENAI_API_KEY set (optional fallback LLM)", "WARN", "env var missing")
else:
    try:
        import openai

        resolved_oai = openai_key
        if openai_key.startswith("aws-secretsmanager://"):
            from aisitools.api_key import get_api_key_for_proxy

            resolved_oai = get_api_key_for_proxy(openai_key)

        client = openai.OpenAI(api_key=resolved_oai)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = (resp.choices[0].message.content or "").strip()
        record(
            "OpenAI chat completions (gpt-4o-mini, optional)",
            "PASS" if text else "WARN",
            f'reply="{text[:40]}"',
        )
    except Exception as e:
        record(
            "OpenAI chat completions (gpt-4o-mini, optional)",
            "WARN",
            f"{type(e).__name__}: {str(e)[:200]}",
        )


# ---------------------------------------------------------------------------
# 3. Tooling the autonomous agent will need (Phase 0 of 99_instruction.md)
# ---------------------------------------------------------------------------
print("\n=== Toolchain ===")

# LaTeX — Phase 6 writes paper.tex and Phase 0.5 compiles a hello world.
for bin_name in ["pdflatex", "latexmk"]:
    path = shutil.which(bin_name)
    if path:
        record(f"{bin_name} available", "PASS", path)
    else:
        record(
            f"{bin_name} available",
            "FAIL",
            f"not on PATH — install texlive (needed for paper.pdf)",
        )

# Poppler — §0.9 notes it may be needed to render the reference PDF.
for bin_name in ["pdftotext", "pdfinfo"]:
    path = shutil.which(bin_name)
    if path:
        record(f"{bin_name} available", "PASS", path)
    else:
        record(
            f"{bin_name} available",
            "WARN",
            "not installed; pypdf is a fallback but OCR-quality text may suffer",
        )

# Repo venv has the deps the autonomous scripts import.
print("\n=== Python deps (repo venv) ===")
required = {
    "requests": "HTTP client",
    "aiohttp": "concurrent GH API fetch",
    "pandas": "dataframes",
    "pyarrow": "parquet writes",
    "numpy": "math",
    "scipy": "Wilson CIs / bootstrap",
    "matplotlib": "figure rendering",
    "anthropic": "LLM API",
    "inspect_ai": "Inspect framework (optional LLM classification path)",
    "pypdf": "reference PDF reading",
    "tqdm": "progress bars",
    "dateutil": "ISO week bucketing",
}
for mod, purpose in required.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        record(f"import {mod}", "PASS", f"{ver} — {purpose}")
    except ImportError as e:
        record(f"import {mod}", "FAIL", f"missing ({purpose}) — {e}")


# ---------------------------------------------------------------------------
# 4. Outbound internet for non-GitHub sources (ICML template, OpenReview, arXiv)
# ---------------------------------------------------------------------------
print("\n=== Outbound HTTP (non-GitHub) ===")
try:
    import requests

    for label, url in [
        ("icml.cc", "https://icml.cc/"),
        ("openreview.net", "https://openreview.net/"),
        ("arxiv.org", "https://arxiv.org/abs/2502.12743"),
    ]:
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            if r.status_code < 400:
                record(f"reach {label}", "PASS", f"HTTP {r.status_code}")
            else:
                record(f"reach {label}", "WARN", f"HTTP {r.status_code}")
        except Exception as e:
            record(f"reach {label}", "WARN", f"{type(e).__name__}: {e}")
except Exception as e:
    record("outbound HTTP", "FAIL", str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
counts = {s: 0 for s in ("PASS", "FAIL", "WARN", "SKIP")}
for _, s, _ in results:
    counts[s] += 1
print(
    f"PASS={counts['PASS']}  FAIL={counts['FAIL']}  WARN={counts['WARN']}  SKIP={counts['SKIP']}"
)

fails = [(n, d) for n, s, d in results if s == "FAIL"]
warns = [(n, d) for n, s, d in results if s == "WARN"]
if fails:
    print("\nBLOCKERS (must fix before autonomous run):")
    for n, d in fails:
        print(f"  - {n}: {d}")
if warns:
    print("\nWARNINGS (non-blocking, flag to user):")
    for n, d in warns:
        print(f"  - {n}: {d}")

sys.exit(1 if fails else 0)
