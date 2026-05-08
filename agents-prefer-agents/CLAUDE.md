# CLAUDE.md — agents-prefer-agents subproject

Paper-subproject-specific notes. For repo-wide environment / linting / venv conventions, see the parent [CLAUDE.md](../CLAUDE.md).

## Dual-file rule (non-negotiable)

The paper has two parallel copies of every TeX file:

| Authoritative                       | Rendered                                       |
|-------------------------------------|------------------------------------------------|
| `paper/paper.tex`                   | `paper/paper.filled.tex`                       |
| `paper/appendix/{name}.tex`         | `paper/appendix/{name}.filled.tex`             |

The `.filled.tex` copies are produced by `scripts/08_fill_paper.py`, which substitutes `\PLACEHOLDER…` macros with computed numerical values. **Every text edit must be made in both copies.** Edits made only to `.filled.tex` are silently overwritten the next time the fill script runs.

Quick parity check:

```bash
diff <(grep -v PLACEHOLDER paper.tex) <(grep -v PLACEHOLDER paper.filled.tex)
```

Body content should match — only the `.tex` ↔ `.filled.tex` `\input` filename suffix and lines that contain placeholder strings should differ.

## Pipeline stages (`scripts/`)

| Stage | Script                        | Reads                                          | Writes                                                         |
|-------|-------------------------------|------------------------------------------------|----------------------------------------------------------------|
| 1a    | `01a_download_criticality.py` | OpenSSF criticality CSV (public GCS bucket)    | `data/criticality/ossf-criticality-{snap}-all.csv(+.provenance.json)`; `logs/01a_download_criticality.log` |
| 1     | `01_build_repo_list.py`       | criticality CSV                                | `data/repos.json` (final 10k), `data/repos.top10k.json`, `results/phase1_stats.json`, `logs/01_build_repo_list.log` |
| 1-old | `old_01_build_repo_list.py`   | (legacy v1: star-bucket sweep)                 | `data/old_phase1/repos.json`                                   |
| 2     | `02_fetch_prs.py`             | `data/repos.json`, GH GraphQL                  | `data/prs/*.jsonl` (one file per repo), `data/prs/_repo_pr_counts.json` |
| 3     | `03_classify_prs.py`          | `data/prs/*.jsonl`, `lib/allowlist.py`         | `data/pr_events.parquet`, `data/pr_summary.parquet`             |
| 4     | `04_merger_audit.py`          | `data/pr_events.parquet`                       | `results/merger_audit.json`                                     |
| 5     | `05_compute_chains.py`        | `data/pr_events.parquet`                       | `data/chains.parquet`                                           |
| 6     | `06_*.py` family              | parquets                                       | `data/merge_rates.parquet`, `results/regression*.json`          |
| 6c    | `06c_within_pr.py`            | parquets                                       | within-PR DiD numbers (used by 07/08)                           |
| 7     | `07_make_figures.py`          | parquets                                       | `paper/figures/figure*.pdf`                                     |
| 8     | `08_fill_paper.py`            | results JSONs + `paper.tex` / `appendix/*.tex` | `paper.filled.tex` / `appendix/*.filled.tex`                    |
| 11    | `11_build_pdf.py`             | `paper.filled.tex`                             | `paper/paper.filled.pdf`                                        |

Audit / test harnesses:
- `scripts/ai_verdict_parser.py` — per-bot regex catalogue used by Stage 3 to extract `APPROVED`/`CHANGES_REQUESTED` from `COMMENTED` review bodies.
- `paper/test_ai_verdict_regex.py` — reproduces per-bot match counts and a sample of matches/unmatches for hand-verification.
- `paper/test_did_v2.py` — DiD sanity-check harness.

## S3 source-of-truth (heavy artefacts not in git)

| Path                                                                               | Size  | Recovery                                                                  |
|------------------------------------------------------------------------------------|-------|---------------------------------------------------------------------------|
| `s3://aisi-data-eu-west-2-prod/users/merlin-stein/agents-prefer-agents/data/prs/`  | 24G (10,915 files)  | `aws s3 sync s3://…/data/prs/ data/prs/` — needed for Stages 3, 4         |
| `s3://aisi-data-eu-west-2-prod/users/merlin-stein/agents-prefer-agents/data/old_phase1/` | 1.8G  | only re-fetch if you need the v1 universe                                 |

After a VM migration the typical missing-file pattern is exactly `data/prs/` + `data/old_phase1/` (both in `.gitignore`, so neither follows the repo). Resync is the recovery path; no data is actually lost.

Downstream parquets (`pr_events.parquet`, `pr_summary.parquet`, `chains.parquet`, `merge_rates.parquet`) are committed locally, so RQ1–RQ3 figures and tables can be regenerated without re-running Stages 2–4.

## Appendix-letter mapping

Order is set by `\input{...}` calls in `paper.tex`:

| Letter | File                                       | Title                                                  |
|--------|--------------------------------------------|--------------------------------------------------------|
| A      | `appendix/allowlist.tex`                   | AI identification                                      |
| B      | `appendix/ai_verdict_regex.tex`            | AI verdict regex catalogue                             |
| C      | `appendix/repo_selection.tex`              | Sample-selection pipeline                              |
| D      | `appendix/extra_figures.tex`               | AI→AI chain-length distribution by quarter             |
| E      | `appendix/methods.tex`                     | Data and methods (full description)                    |
| F      | `appendix/anchoring_robustness.tex`        | Robustness: does AI's review anchor humans?            |

`appendix/threats.tex`, `appendix/examples.tex`, `appendix/audit.tex`, `appendix/old_repo_selection.tex`, and `appendix/regression.tex` exist on disk but are NOT `\input`'d — they're dormant and won't render. (`examples.tex` and `threats.tex` were dropped during the May 2026 annex review: `examples.tex` because its v1 numbers — 42,823-PR sample, max chain 13, n=11 AI×AI cell — no longer match the v2 universe; `threats.tex` because the surviving paragraphs were already covered by §Limitations in the main paper.)

## Build commands

```bash
# 1. Regenerate figures (requires parquets to exist)
python scripts/07_make_figures.py

# 2. Fill placeholders into paper.filled.tex / appendix/*.filled.tex
python scripts/08_fill_paper.py

# 3. Build the PDF (requires a TeX toolchain — see note below)
python scripts/11_build_pdf.py
# or directly:
cd paper && pdflatex paper.filled && bibtex paper.filled && pdflatex paper.filled && pdflatex paper.filled
```

**Local PDF build works on this VM.** `pdflatex` and `bibtex` are installed at `/usr/bin/` (TeX Live 2023/Debian). The Overleaf workflow (`paper/overleaf.zip`, regenerated by `scripts/10_make_overleaf_zip.py`) remains a fallback if the local toolchain breaks.

Figure-only iteration: re-run `07_make_figures.py`, then rebuild via pdflatex or Overleaf. No need to re-run `08_fill_paper.py` unless results JSON values changed.

## Reviewer-facing rule

The submitted paper is anonymous and reviewers cannot inspect this codebase. Do **not** add references to specific script paths, data filenames, log paths, or `data/.../*.parquet|*.jsonl` artefacts inside `paper.tex` or any `appendix/*.tex`. Discuss the pipeline in prose; release scripts and intermediate artefacts come later via a separate replication artefact. (This file is the canonical place to keep the inventory of those internal pointers.)

## Terminology conventions in the paper

- **reviewer** (the role): use this everywhere when talking about the actor *doing* a review of a PR. The paper does **not** call this actor a "judge".
- **judgement** / **judgment** (the cognitive act): keep these intact when they refer to taste, judgement-under-uncertainty, or AI judgements as a *concept*. Do not rewrite them to "review".
- The split is enforced by `grep -E "\bjudge\b" paper*.tex appendix/*.tex` returning empty (matches inside `judgement`/`judgment` are excluded by the word boundaries).

## Gotchas / lessons

- **Stale comment headers in appendix files.** Each `appendix/*.tex` opens with a `% Appendix X — …` comment that no longer matches the actual letter the appendix gets at render time (e.g. `repo_selection.tex` says `% Appendix E` but renders as Appendix D). Treat the comment as flavour; the authoritative letter mapping is the table above.
- **`08_fill_paper.py` overwrites `.filled.tex` files in place.** Any hand edit you make only to a `.filled.tex` file evaporates the next time the fill script runs. The dual-file rule is not a style preference, it's a correctness rule.
- **`audit.tex`, `old_repo_selection.tex`, `regression.tex`** exist but are NOT `\input`'d. Don't waste time editing them when sweeping the appendix — they don't render.
- **Manual user edits land in `paper.filled.tex` first.** When the user iterates on phrasing in the IDE, they tend to edit the rendered file (because that's what they preview), and expect the unfilled twin to be backported. Always diff `paper.tex` ↔ `paper.filled.tex` at session start to find drift before adding new edits on top.
- **Known typo to flag (do not silently fix).** A missing space at line 69 of both `paper.tex` and `paper.filled.tex` (`themselves.Chatbot`). Surfaced for the user; left in place to avoid silent fixes during a backport.
- **PDF build sanity-check on a VM.** Run `which pdflatex` before promising a local PDF rebuild on a fresh VM — historically this VM lacked it, but TeX Live is now installed and works.
