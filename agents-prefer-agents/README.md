# agents-prefer-agents

Measuring precursors to gradual disempowerment in open-source code review.
Target submission: AIWILD @ ICML 2026 workshop (2026-05-01 AoE).
Previously targeted TAIGR @ ICML 2026 (2026-04-24 AoE, missed); historical log
entries in `99_progress.md`/`99_notestohuman.md` reference the original venue.

See `99_instruction.md` for the full plan, `99_progress.md` for the live
execution log, `99_notestohuman.md` for decisions/open questions, and
`99_literature.md` for raw source extracts.

## Layout

```
99_instruction.md       the plan (must-read)
99_progress.md          live log, uncertainties, check-in milestones
99_notestohuman.md      decisions, open questions, changelog per phase
99_literature.md        raw extracts from every source read
2502.15212v1.pdf        reference paper (structural example only)

lib/ai_detection.py     four-criteria AI attribution (Co-Authored-By, config
                        files, bot allowlist, handle mentions) ported from
                        scripts/data-classification-aicreatedmcp/

scripts/99_utils.py     rate-limited GH REST clients (sync + async), logging,
                        token resolver, ISO-week helper
scripts/01_*.py         Phase 1 — build repo universe
scripts/02_*.py         Phase 2 — GraphQL PR fetch (throttled)
scripts/03_*.py         Phase 3 — per-event AI classification
scripts/04_*.py         Phase 4 — merger-detection audit
scripts/05_*.py         Phase 5 — AI→AI chain lengths
scripts/06_*.py         Phase 6 — weekly 2×2 merge rates + cluster-bootstrap DiD
scripts/07_*.py         Phase 6 — figures (headline + quarterly box)
scripts/08_*.py         Phase 6 — fill paper.tex \PLACEHOLDER* tokens
scripts/09_*.py         Phase 7 — anonymization grep
scripts/run_pipeline.sh one-shot Phase 3→6 runner

data/repos.json         Phase 1 output: 500 curated repos
data/prs/*.jsonl        Phase 2 output: one file per repo
data/pr_summary.parquet Phase 3 output: one row per PR
data/pr_events.parquet  Phase 3 output: one row per event, time-sorted
data/chains.parquet     Phase 5 output
data/merge_rates.parquet Phase 6 output

results/*.json          small summary stats for each phase
results/merger_detection_audit.md  Phase 4 narrative report

paper/paper.tex         LaTeX template with \PLACEHOLDER* tokens
paper/paper.filled.tex  paper.tex with results filled in (run 08_fill_paper.py)
paper/template/         ICML 2026 style files
paper/appendix/         audit + threats + allowlist + regression
paper/figures/          pdf figures from 07_make_figures.py

logs/                   one log per script
```

## How to run

```bash
source /home/ubuntu/mcp-monitoring/.venv/bin/activate

python scripts/01_build_repo_list.py --cap 500                 # ~10 min
python scripts/02_fetch_prs.py --max-prs-per-repo 100 \
    --concurrency 4 --repo-concurrency 3 --min-interval 0.9    # ~2 h
bash scripts/run_pipeline.sh                                    # ~10 min (03→07+08+09)

# Compile the PDF (requires texlive; not available in this sandbox)
cd paper && pdflatex paper.filled && bibtex paper && \
    pdflatex paper.filled && pdflatex paper.filled
```

## Bulk PR data (not in git)

The Phase 2 PR dump under `data/prs/` (~4.4 GB of per-repo `.jsonl` files) is
not committed. An archival copy lives in S3:

```
s3://aisi-data-eu-west-2-prod/users/merlin-stein/agents-prefer-agents/data/prs/
```

Mounted at `/mnt/s3/users/merlin-stein/agents-prefer-agents/data/prs/` on
AISI dev VMs. To restore locally: `aws s3 sync <s3-uri> data/prs/`.

## Isolation

This subproject lives entirely under `agents-prefer-agents/`. It does not
modify any parent-repo files. The only code imported from outside is the
original four-criteria detector at
`scripts/data-classification-aicreatedmcp/detect_ai_created.py`, which was
ported (not imported) into `lib/ai_detection.py` so the subproject can be
extracted to its own repo later.
