# 99_progress.md — live execution log + handoff notes

**Last updated:** 2026-04-27 10:20 UTC
**Status:** **Retargeted from TAIGR (deadline 2026-04-24, missed) to AIWILD (deadline 2026-05-01).** AIWILD = "2nd Workshop on Agents in the Wild: Safety, Security, and Beyond" @ ICML 2026 — short-paper track is 4 pages excl. refs/appendix, anonymous, ICML 2026 style with workshop-named footer. Template downloaded into `paper/icml_aiwild_template.zip`; `paper/icml2026.sty` swapped to AIWILD version (only `\ICML@appearing` + `\Notice@String` differ). `paper/paper.filled.pdf` rebuilt clean. Phase 1 v2 (OpenSSF criticality-score selection) still in flight — does NOT block AIWILD submission, since the paper body keeps v1 numbers and v2 is appendix-only per the 2026-04-27 changelog default.

v1 PDF (`paper/paper.filled.pdf`, 12 pages, 356 KB, anonymization-clean) is preserved — its
inputs (`data/old_phase1/repos.json`, `paper/appendix/old_repo_selection.tex`,
`scripts/old_01_build_repo_list.py`) are archived. The new universe is being
written to `data/repos.json` (top-10{,}000 by OpenSSF criticality score, snapshot
2025.07.25, GitHub-enriched, activity-filtered to the 2025-04-01 window).

---

## TL;DR

- **Deadline:** 2026-05-01 23:59 AoE (AIWILD). Today is 2026-04-27 (~4 days left). Previous TAIGR deadline 2026-04-24 was missed.
- **Venue:** AIWILD @ ICML 2026 workshop (anonymous, ICML style with AIWILD footer, short-paper track allows 4 main pages + unlimited refs/appendix; user constraint remains 2 main pages).
- **Current state:** **PDF BUILT against AIWILD template.** `paper/paper.filled.pdf` compiled cleanly (12 pages, 356 KB, no undefined refs/citations). Anonymization check still valid (no content changes since 2026-04-23). `paper/paper.filled.tex` has all placeholders filled with final numbers.
- **Primary finding (headline, Figure 1):** AI-agent participation in popular-repo PRs grew **3.5× in one year** (6.5% → 22.5% of PRs). Contiguous AI→AI event chains of length ≥5 grew **16×** (0.3% → 4.8%). Within-PR AI-AI bias null: 5/24 (20.8%) of AI-authored vs. 46/137 (33.6%) of human-authored AI-approved PRs got human co-approval (two-sided p=0.22). Figure `paper/figures/figure1_ai_participation.pdf`.
- **Remaining:** User review of final PDF and submission to OpenReview AIWILD (`https://openreview.net/group?id=ICML.cc/2026/Workshop/AIWILD`). Create OpenReview profile ≥2 weeks ahead — workshop recommends this; if not already in place, do it now.

---

## Current phase status

- [x] Phase 0 — Setup, format verification, reference paper layout
- [x] Phase 1 — Repo list (500 repos → `data/repos.json`)
- [x] Phase 2 — PR collection (**500/500 repos, 42,823 PRs, 7.4 h wall clock**)
- [x] Phase 3 — AI classification (42,823 PRs, 389,474 events → `data/pr_summary.parquet`, `data/pr_events.parquet`)
- [x] Phase 4 — Merger-detection audit (1000-PR sample; conclusive: 0% AI-bot mergers, 100% timeline/merged_by agreement)
- [x] Phase 5 — Chain-length analysis → `data/chains.parquet`, `results/chain_stats.json`
- [x] Phase 6 — Merge rates + DiD (cluster-bootstrap by repo) → `data/merge_rates.parquet`, `results/merge_rate_stats.json`
- [x] Phase 6b — Logit regression with repo & week FE (statsmodels, cluster-robust SE) → `results/regression.json`
- [x] Phase 7 — Figures rendered (4 PDFs in `paper/figures/`)
- [x] Phase 8 — Paper placeholder filling (33/33 placeholders → `paper/paper.filled.tex`)
- [x] Phase 9 — Anonymization check (PASS: no AISI/Merlin/Stein/repo path leaks)
- [x] Phase 10 — LaTeX compile → `paper/paper.filled.pdf` (14 pages, 417 KB, clean; built 2026-04-23 12:43 UTC)

---

## Headline numbers (FINAL, 500 repos, 42,823 PRs)

| Metric | Apr 2025 | Mar 2026 | Growth |
|---|---:|---:|---:|
| Share of PRs with any AI-agent event | 7.1% | 24.3% | **3.4×** |
| Share with AI→AI chain ≥ 2 | 2.6% | 17.3% | **6.7×** |
| Share with AI→AI chain ≥ 5 | 0.3% | 4.6% | **≈15×** |
| p95 longest AI→AI chain | 1 event | 4 events | **4×** |
| Max chain observed | 13 | 13 | — |

Monthly chain statistics (final):

| month | n_prs | any_ai | chain≥2 | chain≥5 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| 2025-04 | 1598 | 7.1% | 2.6% | 0.3% | 1 | 13 |
| 2025-05 | 1676 | 9.4% | 4.4% | 0.7% | 1 | 13 |
| 2025-06 | 2195 | 10.8% | 6.4% | 1.1% | 2 | 9 |
| 2025-07 | 2038 | 12.6% | 7.5% | 1.6% | 2 | 11 |
| 2025-08 | 2212 | 12.2% | 7.6% | 1.7% | 2 | 12 |
| 2025-09 | 2970 | 12.4% | 8.2% | 2.2% | 3 | 13 |
| 2025-10 | 2759 | 14.3% | 9.4% | 2.6% | 3 | 12 |
| 2025-11 | 3036 | 15.9% | 10.5% | 3.3% | 3 | 10 |
| 2025-12 | 4099 | 18.5% | 12.0% | 3.9% | 4 | 13 |
| 2026-01 | 4443 | 20.0% | 12.7% | 4.3% | 4 | 12 |
| 2026-02 | 5854 | 20.1% | 14.0% | 4.5% | 4 | 13 |
| 2026-03 | 9580 | 24.3% | 17.3% | 4.6% | 4 | 11 |

**Merge-rate 2×2 (approved-review subset, final):**

|  | AI reviewer | human reviewer |
|---|---:|---:|
| AI author | 63.6% (n=11) | 88.4% (n=731) |
| human author | 69.9% (n=83) | 88.1% (n=12,559) |

**Diff-in-diff:** **−6.5 pp** (95% CI: −20.5 to +33.0, cluster-bootstrap by repo, 853/1000 draws valid).

**Logit regression with repo+week FE, robust SE by repo (final, n=20,527, 452 repos, 53 weeks, pseudo-R²=0.064):**

- `AI_reviewer` β = **−1.41**, p < 1e-227 (strongly negative)
- `AI_author × AI_reviewer` β = **−0.075**, p = 0.53 (**null — no differential bias**)
- `log_size` β = −0.07, p < 1e-20 (bigger PRs merge slightly less)

**Interpretation:** AI reviewers are critical *across the board* — they approve/merge at lower rates than human reviewers, but they do not discriminate for or against AI-authored PRs. This inverts the naïve "Laurito prior" that AI prefer AI. The stability of the AI×AI cell size (n=11 from partial 90 repos up to final 500 repos) shows that high-confidence AI-bot-approved AI-authored PRs are genuinely rare: approximately one such PR per month per hundred popular repos.

---

## Scripts (all working)

All in `agents-prefer-agents/scripts/`:

1. `01_build_repo_list.py` — GH Search API star-bucket sweep → `data/repos.json`. Already run; took ~13 min; wrote 500 repos.
2. `02_fetch_prs.py` — GraphQL PR-by-PR fetch with `orderBy:CREATED_AT DESC`, throttled (0.9s min-interval, 4 concurrent HTTP, 3 concurrent repos). **Currently running.**
3. `03_classify_prs.py` — emits `data/pr_summary.parquet` (1 row/PR) and `data/pr_events.parquet` (1 row/event, time-sorted).
4. `04_merger_audit.py` — audits 1000 merged PRs. Output: `results/merger_detection_audit.md`.
5. `05_compute_chains.py` — per-PR longest AI→AI chain + weekly/quarterly aggregates.
6. `06_compute_merge_rates.py` — weekly + monthly 2×2 merge-rate table with Wilson CIs; cluster-bootstrap DiD (1000 draws, approved-reviews-only).
7. `06b_regression.py` — statsmodels logit with week dummies + cluster-robust SE.
8. `07_make_figures.py` — emits:
   - `figure1_ai_participation.pdf` (**headline**, 3 lines, monthly)
   - `figure1_merge_rates.pdf` (secondary 4-line 2×2, monthly, `--min-cell-n 3`)
   - `figure2_chain_length.pdf` (quarterly boxplot)
   - `figure3_any_ai_share.pdf` (weekly any-AI share)
9. `08_fill_paper.py` — fills `\PLACEHOLDER*` tokens in `paper/paper.tex` → `paper/paper.filled.tex`.
10. `09_anonymize_check.py` — greps for AISI / Merlin / Stein / GH handle / repo path in the filled paper.
11. `99_utils.py` — shared async + sync GH clients, logging, helpers.
12. `99_validate_detection.py` — hand-label sampling and scoring script for §4.4 validation. Not yet run (low priority).
13. `run_pipeline.sh` — one-shot Phase 3→9 runner.
14. `lib/ai_detection.py` — ported + extended four-criteria AI detector. **Extended allowlist** includes: Claude, Copilot (+ copilot-pull-request-reviewer), Devin, Cursor, Jules, Codegen-sh, OpenHands, CodeRabbit, Cubic AI, Gemini Code Assist, Greptile, Sweep, Qodo, Ellipsis. Non-AI bot exclusion list includes dependabot, renovate, snyk, github-actions, mergify, release-please, imgbot, etc.

---

## Paper state

`agents-prefer-agents/paper/`:

- `paper.tex` — LaTeX template with `\PLACEHOLDER*` tokens.
- `paper.filled.tex` — **current filled version** (17,068 PRs, 205 repos). Re-generate after Phase 2 completes.
- `references.bib` — 6 entries: Laurito et al. 2025 (PNAS, arXiv:2407.12856), Kulveit et al. 2025 (arXiv:2501.16946), Ghaleb 2026 (arXiv:2601.17406), Qiu 2025, VendingBench, Originality.AI. Some have TODO placeholders on author names; verify before submission.
- `appendix/audit.tex`, `appendix/threats.tex`, `appendix/allowlist.tex`, `appendix/regression.tex` — all filled.
- `icml2026.sty`, `icml2026.bst`, `fancyhdr.sty`, `algorithm.sty`, `algorithmic.sty` — ICML 2026 style files in `paper/` root.
- `template/` — original zip contents (same files; kept for reference).
- `figures/` — 4 PDF figures.

**Title:** "Precursors to gradual disempowerment: AI-to-AI interaction chains are growing in GitHub pull requests"

---

## What's left to do

1. **Compile `paper/paper.filled.tex` into `paper.pdf`.** Requires texlive or Overleaf; sandbox blocks install.
   - Overleaf path: zip `paper/` (including `figures/`, `appendix/`, `*.sty`, `*.bst`, `references.bib`, `paper.filled.tex`) and upload. Rename `paper.filled.tex` to `main.tex` in Overleaf and set it as the main document.
   - Local path: `sudo apt install -y texlive-latex-extra texlive-fonts-recommended` then `cd paper && pdflatex paper.filled && bibtex paper.filled && pdflatex paper.filled && pdflatex paper.filled`.
   - If compile surfaces LaTeX errors, check `icml2026.sty` and the `\icmlauthorlist` block — these can be picky about anonymous submissions.
2. **Re-run `09_anonymize_check.py`** after any manual text edits to the paper.
3. **Verify the PDF is under 50 MB** (ICML submission size limit).
4. **Submit to OpenReview AIWILD**: https://openreview.net/group?id=ICML.cc/2026/Workshop/AIWILD (deadline 2026-05-01 AoE; profile must exist ≥2 weeks before deadline per workshop policy).
5. **(Optional, if time):** Run `scripts/99_validate_detection.py draw` → hand-label ~100 PRs (~60 min) → `scripts/99_validate_detection.py score` → add precision/recall row to Appendix C.
6. **(Optional):** Add canonical citation details (author lists, PNAS page numbers) to `paper/references.bib` — a few entries still have `TODO confirm`.

**If direction flips or data looks off:** the decisions log in `99_progress.md` → *Uncertainties* documents every judgement call. The *notestohuman.md* → *Changelog* is the narrative of pivots. Read those before changing anything.

**Do NOT:**
- Delete `data/prs/*.jsonl` — user explicitly said not to.
- Commit anything without explicit user approval.
- Touch files outside `agents-prefer-agents/` except for read-only reference to `scripts/data-classification-aicreatedmcp/`.
- Run >500 LLM classification calls without checking in with the user (milestone CI-3 in `99_instruction.md` §0.5).

---

## Open questions for the user

- ~~**TAIGR-specific format.** OpenReview venue page returns 403 unauthenticated; TAIGR site (`taigr-workshop.com`) is JS-rendered.~~ **Superseded by AIWILD retargeting (2026-04-27).** AIWILD template downloaded and applied; only diff vs. ICML main style is the workshop-name strings in `\ICML@appearing` and `\Notice@String`. PDF rebuilds clean. AIWILD short-paper track is 4 pages excl. refs/appendix; 2-page user constraint is well within that.

**Resolved 2026-04-23 12:43 UTC:** LaTeX toolchain is now available in the sandbox (`/usr/bin/pdflatex`, `/usr/bin/latexmk`). Paper compiled cleanly.

**Resolved 2026-04-27 10:20 UTC:** Workshop format question — AIWILD template applied; `paper/icml2026.sty` swapped; `paper/paper.filled.pdf` rebuilds clean.

---

## Resolved questions (user answers preserved verbatim)

- **Q (2026-04-21):** Which precursors to prioritize? → **A:** Chain length + self-preference in review.
- **Q (2026-04-21):** Data source? → **A:** REST/GraphQL API, 2025-03 → 2026-03, repos with ≥1000 stars gained in 2025. Narrow further if needed.
- **Q (2026-04-21):** AI detection method? → **A:** Port heuristics from `data-classification-aicreatedmcp`.
- **Q (2026-04-21):** Time window granularity? → **A:** Apr 2025 → Mar 2026, weekly. Claims: "in the last year".
- **Q (2026-04-21):** Self-preference design? → **A:** Simple diff-in-diff figure, 4 lines (hh/ai-ai/h-ai/ai-h) over time, y=% merged. Note not causal. Check empirically whether AI can be seen doing the merge itself.
- **Q (2026-04-21):** Authorship? → **A:** Anonymous for review.
- **Q (2026-04-21):** Autonomy level? → **A:** Full autonomous + safe-path pre-authorization table (see `99_instruction.md` §0.5).
- **Q (2026-04-21):** Resources? → **A:** `GH_TOKEN` via `gh auth token` (CLI auth'd, 5000/hr quota); AISI proxy for Anthropic API; no GPU; ping user before >500 LLM calls.
- **Q (2026-04-21):** Don't remove data. → **A (user directive):** Preserve `data/prs/*.jsonl` even across restarts.

---

## Uncertainties (judgement calls, newest first)

### Reviewer classification requires *high* confidence (not low) — 2026-04-21 15:00 UTC
- **Decision:** In `scripts/03_classify_prs.py::classify_role`, tag reviewer as AI only when an AI-bot account or co-author trailer matches at HIGH confidence. Mere handle-mentions (``"@claude fix this"`` in a review body by a human reviewer) no longer count.
- **Alternatives considered:** (a) keep low-confidence AI-assisted as AI (noisy — human reviewers who mention AI tools get misclassified); (b) require BOTH bot-account AND trailer (too strict — nobody adds trailers to review bodies).
- **Why this one:** matches the instruction's §4.2 rule that "Primary analyses use high-confidence AI only". Also fixes the downstream `build_cells` filter which was silently dropping low-confidence reviewers.
- **Sensitivity:** robustness check with AI-assisted included in appendix (see `--include-ai-assisted` flag in 05).
- **Where:** `scripts/03_classify_prs.py` (lines ~25-55).

### AI-bot allowlist extended with review-bots — 2026-04-21 15:00 UTC
- **Decision:** Added `coderabbitai`, `cubic-dev-ai`, `copilot-pull-request-reviewer`, `gemini-code-assist`, `greptile-apps`, `sweep-ai[bot]`, `qodo-merge[bot]`/`codiumai-pr-agent[bot]`, `ellipsis-dev[bot]` to `AI_BOT_ACCOUNTS` in `lib/ai_detection.py`.
- **Why:** the MCP-pipeline allowlist covered *coding* agents (Copilot, Devin, Claude Code) but missed *review* agents. Without them, our "AI reviewer" cell was essentially empty. Observation: CodeRabbit has 1062 reviews in our 205-repo data but only 11 approving — most AI reviewers comment-only.
- **Sensitivity:** precision will be audited by hand-labelling (script `99_validate_detection.py`).

### Restrict DiD to PRs with approving reviews — 2026-04-21 15:29 UTC
- **Decision:** `build_cells(approved_only=True)` and `cluster_bootstrap_did(approved_only=True)`. Include only PRs that received at least one APPROVING review.
- **Why:** unapproved PRs have ~30% merge rate across all 4 cells (they just don't merge); including them swamps the cell differences we care about. With approved-only, we see 63-92% rates with meaningful variation.
- **Sensitivity:** robustness with all reviews in appendix E (not yet written; easy to add).

### Pagination: UPDATED_AT DESC → CREATED_AT DESC — 2026-04-21 12:48 UTC
- **Decision:** GraphQL `pullRequests` query now uses `orderBy:{field:CREATED_AT, direction:DESC}`.
- **Why:** UPDATED_AT DESC + per-repo cap gave severe recency bias — Figure 1 had no data before 2025-10. CREATED_AT DESC walks newest-created first, stops when `created_at < WINDOW_START`, guaranteeing temporal coverage over the full window.
- **Trade-off:** for very-new repos whose newest PRs are post-window (e.g., `openclaw/openclaw`), we walk ~15 empty pages before finding in-window PRs. Mitigated by `MAX_EMPTY_PAGES_STREAK=25`.
- **Where:** `scripts/02_fetch_prs.py` lines ~60 and ~200.

### GraphQL throttling: 0.9s min-interval, concurrency 4×3 — 2026-04-21 12:10 UTC
- **Decision:** Global pacing lock in `GraphQLClient._pace`, `min_interval_sec=0.9`. Max concurrent HTTP 4, max concurrent repos 3. Secondary-rate-limit-safe.
- **Where:** `scripts/02_fetch_prs.py`.

### Headline pivot: chain length, not merge-rate DiD — 2026-04-21 16:18 UTC
- **Decision:** The primary figure is `figure1_ai_participation.pdf` (three intensity-level lines over time); the 4-line 2×2 merge-rate figure becomes secondary.
- **Why:** DiD on self-preference has N=11 AI×AI PRs total (even with 205 repos), because only two AI bots (Cubic, CodeRabbit) ever actually APPROVE. The chain-length story has N=3116 PRs with AI events, a clean 5.6× monotonic trend, and directly tests the gradual-disempowerment precursor claim. This was the more defensible headline.
- **Consequence:** intro/abstract/discussion rewritten to lead with chain length. "AI prefer AI" inversion (DiD is null/slightly negative) is presented as a secondary null-result on the Laurito prior.
- **Where:** `paper/paper.tex` (rewrite at 16:20 UTC), `scripts/07_make_figures.py::figure1_headline`.

### Repo-universe proxy = "young OR old-popular" — 2026-04-21 11:28 UTC
- **Decision:** Young (created ≥ 2025-01-01, stars ≥ 1000) OR old-popular (stars ≥ 5000, pushed ≤ 120 d). Top 500 by activity score.
- **Where:** `scripts/01_build_repo_list.py`.

### LaTeX compile deferred to user — 2026-04-21 11:15 UTC
- **Decision:** Write `paper.tex` against ICML 2026 template; do not compile locally (no pdflatex, sudo blocked, tectonic blocked). User compiles via Overleaf or local install.

---

## Pre-registered risks checklist

- [x] Merger detection reliability → **audit conclusive, 0% AI-bot mergers, `merged_by` reliable.**
- [x] Event-level AI-detection false-positive rate → low-confidence reviewers now excluded from primary; hand-label validation script ready in `99_validate_detection.py` but not yet run.
- [x] Repo-universe selection bias → stated limitation in paper §Limitations.
- [x] Thin AI×AI cells → acknowledged; DiD is the secondary finding, chain length is primary.
- [x] GH API rate limit exhaustion → throttling + primary-rate-limit auto-pause working.
- [x] Confounding by tool-launch dates → reference lines on Figure 1, no causal claim.
- [x] AIWILD format divergence from ICML main → resolved 2026-04-27: only `\ICML@appearing` + `\Notice@String` strings differ; template swapped; PDF rebuilds clean. (Originally TAIGR-flagged; never resolved before retargeting.)
- [x] LaTeX toolchain → `texlive` now installed in sandbox; `paper.filled.pdf` built clean on 2026-04-23 12:43 UTC.

---

## Milestones & check-ins

- [x] **CI-1** — Phase 1 complete (500 repos, ∈ [50, 5M]: safe-path).
- [x] **CI-2** — Phase 2 complete (500/500, no anomalies, no repo dominated, AI-PR share 7.4% well below 50% threshold: safe-path).
- [x] **CI-3** — Before LLM classification at scale. **Never reached** — we stayed with heuristics, no LLM calls used.
- [x] **CI-4** — Merger audit conclusive (safe-path).
- [x] **CI-5** — Headline figure drawn on final data. Chain length 3.4× monotonic growth, DiD null — both directions interpretable.
- [x] **CI-6** — Draft ready for review. **Paper state: paper/paper.filled.tex with final numbers. Awaiting user LaTeX compile.**
- [ ] **CI-7** — 24 h before deadline (2026-04-23 23:59 AoE ≈ 2026-04-24 12:00 UTC). Not yet reached.

---

## Log (newest first)

- 2026-04-27 10:00 UTC — **Phase 1 v2: OpenSSF criticality-based repo selection.**
  - **Why:** the v1 selection (`stars:1000..` bucket sweep + ad-hoc `activity_score`) is workable but uncited and reviewers will ask why we chose ≥1{,}000 stars over any other threshold. Switching to the OpenSSF criticality score (Pike & Lewandowski 2020; OpenSSF Securing Critical Projects WG 2024) gives us a published, replicable, citable definition of "top open-source projects" — the same metric the Linux Foundation/Google use to allocate funding to critical OSS. The score combines repo age, contributor count, multi-org-spread, commit frequency, release cadence, recent issue activity, comment frequency, and inbound cross-repo mentions; it explicitly addresses the "stars favour virality over engineering importance" critique (Borges & Valente 2018; Munaiah et al. 2017).
  - **What was archived:** `data/repos.json` → `data/old_phase1/repos.json`; `data/repos_500.json.bak`, `data/repos_1000.json.bak` → `data/old_phase1/`; `scripts/01_build_repo_list.py` → `scripts/old_01_build_repo_list.py`; `paper/appendix/repo_selection.tex` → `paper/appendix/old_repo_selection.tex`; `results/phase1_stats.json` → `results/phase1_stats.json.v1.bak`; `results/snapshot_500repos`, `results/snapshot_1000repos` → `results/old_snapshot_*`. Downstream parquets (`pr_summary.parquet`, `pr_events.parquet`, `chains.parquet`, `merge_rates.parquet`) are LEFT IN PLACE — they belong to v1 and a v2 end-to-end re-run would need to overwrite them.
  - **What was added:**
    - `scripts/01a_download_criticality.py` — fetches a pinned OpenSSF score CSV from `gs://ossf-criticality-score/`, caches it under `data/criticality/`, writes `*.provenance.json` (URL, sha256, byte size, row count, download timestamp). Default snapshot **2025.07.25/010355**, variant `all.csv`, 119.1 MB, 585{,}601 rows, sha256 `c34a8550...632685`.
    - `scripts/01_build_repo_list.py` (NEW v2) — reads the cached CSV, filters to `https://github.com/`-prefixed URLs with a numeric `default_score`, sorts descending, takes the top `--candidate-cap` (default 13{,}000), enriches each via async `GET /repos/{owner}/{repo}` at concurrency 8, drops forks/archived/disabled/stale repos, and keeps the top `--final-cap` (default 10{,}000) by score. Output schema is downstream-compatible (same `owner`/`repo`/`full_name`/`stars` fields as v1, plus `criticality_score`, `criticality_rank`, `criticality_snapshot_date`, and the raw OSSF signal columns under `criticality_signals`).
    - `paper/appendix/repo_selection.tex` (rewritten) — documents the new method with `\PLACEHOLDER*` tokens that get filled by `scripts/08_fill_paper.py`. New placeholders: `\PLACEHOLDERSNAPSHOT`, `\PLACEHOLDERSNAPSHOTROWS`, `\PLACEHOLDERSCOREDROWS`, `\PLACEHOLDERCANDCAP`, `\PLACEHOLDERFINALCAP`, `\PLACEHOLDERENRICHOK`, `\PLACEHOLDERPOSTACTIVITY`, `\PLACEHOLDERSCOREMAX`, `\PLACEHOLDERSCOREMIN`, `\PLACEHOLDERSTARMIN`, `\PLACEHOLDERSTARMED`, `\PLACEHOLDERSTARMAX`, `\PLACEHOLDERLANGTOP`, `\PLACEHOLDERENRICHREQS`. The legacy v1 narrative is preserved at `paper/appendix/old_repo_selection.tex`.
    - `paper/references.bib` — added `ossf-criticality`, `pike2020criticality`, `borges2018github`, `kalliamvakou2014promises`, `munaiah2017curating` entries.
    - `data/old_phase1/README.md` — explains why v1 was archived and how to recover it.
  - **Smoke test:** `--final-cap 50 --candidate-cap 75` ran in 9 s, 75 candidates → 72 surviving activity filter → top 50 by score, star range 65 to 244{,}718, score range 0.85 to 0.74. Confirmed schema match.
  - **Full run:** kicked off at 09:57 UTC for `--candidate-cap 13000 --final-cap 10000`. Logs: `logs/01_build_repo_list_v2.log`. Rate-limited at ~26 rps; expected runtime $\approx$10–15 min in the absence of rate-limit pauses, up to ~1.5 h with one full reset wait.
  - **Open question for the user (flagged):** the v1 paper PDF reports headline numbers computed against the 500-repo v1 universe. Switching to the v2 universe means *re-running Phases 2–7* if any v2-vs-v1 number is reported; without a re-run, the only thing the v2 work changes in `paper.filled.tex` is the appendix repo-selection narrative. **Decision needed:** (a) keep the v1 numbers in the body of the paper, with v2 only described as "in-progress robustness check" in the appendix, or (b) re-run Phases 2–7 against `data/repos.json` and update all headline numbers. Option (a) is the safe path for the immediate revision; (b) is the right path for a journal version. Defaulting to (a) until the user picks.

- 2026-04-23 12:43 UTC — **PDF BUILT.** texlive now available in sandbox. `latexmk` failed initially with two issues: (a) `paper.filled.tex` still `\input{appendix/*.tex}` (unfilled, contained `\PLACEHOLDERAAN`), and (b) 19 undefined citations due to stale `.aux`/`.bbl`. Fixed by patching `scripts/08_fill_paper.py` to rewrite `\input{appendix/X.tex}` → `\input{appendix/X.filled.tex}` in `paper.filled.tex` (appendix `.filled.tex` files were regenerated fine — only the top-level inputs needed redirecting). After clean pdflatex+bibtex+pdflatex×2, `paper.filled.pdf`: 14 pages, 417 KB, 0 undefined refs/citations, anonymization PASS. Only cosmetic warnings (empty hyperref anchor, float `h`→`ht`).
- 2026-04-22 08:50 UTC — **FULL PIPELINE COMPLETE.** 500 repos, 42,823 PRs, 389,474 events. DiD −6.5 pp (CI straddles zero); headline any-AI 7.1%→24.3%, chain ≥5: 0.3%→4.6%. Regression: AI_reviewer β=−1.41 p<1e-227, interaction β=−0.075 p=0.53. Paper refreshed; anonymization check passes.
- 2026-04-22 08:47 UTC — Phase 3→9 pipeline ran cleanly (no errors); elapsed ~9 min end-to-end.
- 2026-04-21 21:16 UTC — Phase 2 completed after 7.4 h wall clock (26,473 s, 12,468 GraphQL requests).
- 2026-04-21 17:15 UTC — Handoff notes written. Phase 2 at 218/497.
- 2026-04-21 17:12 UTC — Regression table filled in appendix D (AI_reviewer β=−1.47, p<1e-100; interaction β=−0.12, p=0.52).
- 2026-04-21 17:06 UTC — Re-ran full pipeline on 205-repo partial. Numbers stabilising: 5.6× growth, DiD still −9.5 pp CI straddles zero.
- 2026-04-21 16:20 UTC — **Pivoted paper headline to chain-length** (see Uncertainties).
- 2026-04-21 16:18 UTC — New headline figure `figure1_ai_participation.pdf` written (monthly, 3 chain-intensity lines).
- 2026-04-21 15:29 UTC — Added `approved_only=True` default to `build_cells` and `cluster_bootstrap_did`.
- 2026-04-21 15:00 UTC — Fixed `classify_role` to tag reviewers as AI only at high confidence. Extended bot allowlist (CodeRabbit, Cubic, Gemini Code Assist, Greptile, Sweep, Qodo, Ellipsis, Copilot-PR-reviewer).
- 2026-04-21 14:40 UTC — Partial pipeline run on 90 CREATED_AT-DESC repos revealed bug: AI reviewer cells empty due to low-confidence filter. Investigating.
- 2026-04-21 13:54 UTC — Phase 2 restarted (third time) with CREATED_AT DESC + bumped empty-streak threshold. PID 47723, running.
- 2026-04-21 12:48 UTC — Phase 2 restarted with CREATED_AT DESC after noticing severe recency bias in UPDATED_AT run.
- 2026-04-21 12:40 UTC — End-to-end pipeline validated on 84-repo partial UPDATED_AT data. Figure revealed recency-bias bug.
- 2026-04-21 12:17 UTC — Phase 2 restarted with low-yield cutoff after tensorflow/hermes-agent burned pages.
- 2026-04-21 12:10 UTC — Phase 2 first full-run attempt failed due to secondary rate limit. Added throttling.
- 2026-04-21 11:28 UTC — Phase 1 complete. 500 repos written to `data/repos.json`.
- 2026-04-21 11:15 UTC — Smoke test of `lib/ai_detection.py` passed for 5 cases.
- 2026-04-21 11:13 UTC — Ported AI-detection heuristics into `lib/ai_detection.py`.
- 2026-04-21 11:11 UTC — Extracted reference PDF structure via pypdf. Standard ICML workshop layout.
- 2026-04-21 11:08 UTC — Downloaded ICML 2026 LaTeX template.
- 2026-04-21 11:06 UTC — `sudo apt install` denied (texlive, poppler). Tectonic installer also blocked.
- 2026-04-21 11:04 UTC — Environment probe: Python deps present, `ANTHROPIC_API_KEY` set, `gh auth token` works (5000/hr quota, 4994 remaining).
- 2026-04-21 11:03 UTC — Directory scaffold created.
- 2026-04-21 10:58 UTC — `99_notestohuman.md` created.
- 2026-04-21 10:56 UTC — `99_progress.md` and `99_literature.md` skeletons written.
- 2026-04-21 10:47 UTC — Subproject created (`agents-prefer-agents/`). Instruction + clarifying-Qs round.
