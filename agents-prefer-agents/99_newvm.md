# 99_newvm.md — Resume on a new VM (handoff written 2026-04-28 22:50 UTC)

## TL;DR — restore data from S3 first

Data is too big for git. Before running anything, pull the bulk artifacts:

```bash
cd /home/ubuntu/mcp-monitoring/agents-prefer-agents
S3="s3://aisi-data-eu-west-2-prod/users/merlin-stein/agents-prefer-agents"

# Per-repo PR JSONLs (3,840 files, ~7.4 GB) — REQUIRED for Phase 3+
aws s3 sync "$S3/data/prs/" data/prs/

# v1 baseline (3,000-repo) preserved for comparison appendix
aws s3 sync "$S3/data/old_phase1/" data/old_phase1/

# Criticality CSV snapshot (120 MB) — needed only if extending Phase 2 beyond
# the current top-10k enrichment, or for re-deriving the universe from scratch
aws s3 sync "$S3/data/criticality/" data/criticality/

# Small but not-in-git artifacts: top-10k enrichment + result/parquet snapshots
aws s3 cp "$S3/data/repos.top10k.json" data/repos.top10k.json
aws s3 cp "$S3/data/repos.json" data/repos.json
aws s3 sync "$S3/results/" results/  # phase[123]_stats.json
aws s3 sync "$S3/paper/" paper/      # paper.filled.pdf, .tex, overleaf.zip
```

Whole restore is ~9.5 GB; takes a few minutes on AISI dev VMs (the same
bucket is also mounted at `/mnt/s3/users/merlin-stein/agents-prefer-agents/`
so you can also `cp -r` from there if `aws` isn't available).

After restore, the workflow below assumes everything is on local disk again.

---


The previous VM was about to shut down with Phase 3 of the v2.5 (3,840-repo)
pipeline at ~35% (1,350 / 3,840 JSONLs classified). The user wants the new
agent to:

1. **Pick up Phase 3+ on the same 3,840-repo snapshot** that's already on
   disk (no need to re-fetch — `data/prs/` has 3,840 complete JSONLs and
   the bookkeeping files are clean and consistent).
2. **In parallel, extend Phase 2 to ~6,000 or 10,000 repos** so a richer
   universe is ready next time. The fetcher's `_done.json` will skip the
   3,840 already-fetched repos.

This file tells you exactly what to do and what *not* to do. Read in order.

---

## 0. Sanity check the environment

```bash
cd /home/ubuntu/mcp-monitoring/agents-prefer-agents

# venv
source /home/ubuntu/mcp-monitoring/.venv/bin/activate
python -c "import orjson, pandas, statsmodels; print('deps OK')"

# GH auth (Phase 2 needs it)
gh auth status || { echo "auth missing — gh auth login or set GH_TOKEN"; }

# tmux
which tmux
```

If `orjson` is missing run `cd /home/ubuntu/mcp-monitoring && uv sync` (orjson
is in `pyproject.toml`).

---

## 1. Verify the inherited state is intact

The new VM should have the same `agents-prefer-agents/` directory tree as
the old VM. Verify the key files:

```bash
cd /home/ubuntu/mcp-monitoring/agents-prefer-agents

# Phase 1 outputs (criticality-based selection, top 3,840 by rank among done)
python -c "
import json
r = json.load(open('data/repos.json'))
print(f'data/repos.json: {len(r)} rows  (expect 3840)')
print(f'  criticality_rank range : {r[0][\"criticality_rank\"]}..{r[-1][\"criticality_rank\"]}')
print(f'  score range            : {r[0][\"criticality_score\"]:.4f} -- {r[-1][\"criticality_score\"]:.4f}')
"

# Phase 2 PR data (3,840 JSONLs)
python -c "
import json, os
done = json.load(open('data/prs/_done.json'))
counts = json.load(open('data/prs/_repo_pr_counts.json'))
disk = [f for f in os.listdir('data/prs') if f.endswith('.jsonl')]
print(f'_done.json       : {len(done)}  (expect 3840)')
print(f'_repo_pr_counts  : {len(counts)}  (expect 3840)')
print(f'on-disk jsonls   : {len(disk)}  (expect 3840)')
done_set = set(done)
orphans = [f for f in disk if f.replace(\"__\",\"/\").rsplit(\".jsonl\",1)[0] not in done_set]
print(f'orphans          : {len(orphans)}  (expect 0)')
"

# Should ALL match: 3840, 3840, 3840, 0
```

If any mismatch, **stop and ask the user before proceeding** — do not
attempt a recovery without confirmation. There is also
`data/repos.top10k.json` (full top-10k enrichment) that you'll need for
extending Phase 2.

---

## 2. Backup the v1 paper PDF (one-off, safe)

```bash
# Don't lose the v1 submission PDF — it'll be overwritten by the pipeline.
cp -n paper/paper.filled.pdf paper/paper.filled.v1.pdf
```

`-n` = no-clobber, so this is idempotent and safe to run repeatedly.

---

## 3. Kick off the two parallel jobs in tmux

### 3a. Phase 3+ on the existing 3,840-repo snapshot

Use the existing `scripts/run_pipeline.sh`. It runs Phases 3 → 4 → 5 →
6abc → 7 → 8 → 10 → 11 → 9 (PDF build + anonymise included).

```bash
tmux new-session -d -s phase3_v25 \
  -c /home/ubuntu/mcp-monitoring/agents-prefer-agents \
  'bash scripts/run_pipeline.sh 2>&1 | tee logs/run_pipeline_v25.log; \
   echo "--- run_pipeline exited $? ---"; exec bash'
```

**Expected runtime: ~25-35 min**, dominated by Phase 3 (parallel + orjson —
~80 files/min on 8 cores). Phase 5 is now vectorised and runs in <2 s.

### 3b. Phase 2 extension to 6,000 (or 10,000)

You can't naively re-use `scripts/run_phase2_v2.sh` because it would re-trim
`data/repos.json` to a smaller cap. Instead, do this directly:

```bash
# Choose a target. 6000 is conservative (~24h wall clock at 0.46 min/repo).
# 10000 would take ~2-3 days. Confirm with user if uncertain.
TARGET_N=6000

# Expand data/repos.json to top TARGET_N from the preserved 10k enrichment.
python - <<PY
import json
from pathlib import Path
src = json.loads(Path("data/repos.top10k.json").read_text())
src.sort(key=lambda r: r["criticality_rank"])
top = src[:$TARGET_N]
Path("data/repos.json").write_text(json.dumps(top, indent=2) + "\n")
print(f"data/repos.json: {len(top)} rows  (rank 1..{top[-1]['criticality_rank']})")
PY

# Resume the fetcher in tmux. It will read 6000 from repos.json, see the
# 3840 already in _done.json, and process the new 2,160 repos.
tmux new-session -d -s phase2_extend_6k \
  -c /home/ubuntu/mcp-monitoring/agents-prefer-agents \
  'source /home/ubuntu/mcp-monitoring/.venv/bin/activate && \
   python scripts/02_fetch_prs.py --max-prs-per-repo 1000 \
   2>&1 | tee logs/02_fetch_prs_extend_6k.log; \
   echo "--- fetch exited $? ---"; exec bash'
```

The fetcher will print
`Loaded 6000 repos; 3840 already done; 2160 to process.`
within ~10 s of starting — **verify** that line appears, then `tmux detach`.

**Expected runtime: ~16-18h** at the observed 0.46 min/repo rate.

---

## 4. Important: trim repos.json BACK before re-running Phase 3

The Phase 3 run in 3a operates on `data/repos.json` only via Phase 3's
implicit reading of `data/prs/*.jsonl` — it does not actually read
repos.json. **But Phase 8 (`08_fill_paper.py`) does read
`results/phase1_stats.json` for sample-characteristics.** That stats file
was refreshed for 3,840 before the kill, so the appendix will say "3,840
repositories selected". If you've expanded `data/repos.json` to 6,000 in
step 3b BEFORE running 3a, you must also refresh `phase1_stats.json` to
reflect the actual analytical sample (which is whatever's in `_done.json`).

Recommended order: **start step 3a FIRST, then expand for 3b.** That way
phase1_stats stays at 3,840 while Phase 3 runs, and you can refresh it
later when the extension completes.

If you accidentally reverse the order, fix `phase1_stats.json` like this:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

stats = json.loads(Path("results/phase1_stats.json").read_text())
# Use the count of repos that actually have data on disk for the analysis universe.
done = json.loads(Path("data/prs/_done.json").read_text())
counts = json.loads(Path("data/prs/_repo_pr_counts.json").read_text())

# Reconstruct the analytical repos.json from data/repos.top10k.json filtered to done.
src = json.loads(Path("data/repos.top10k.json").read_text())
src.sort(key=lambda r: r["criticality_rank"])
done_set = set(done)
analytical = [r for r in src if r["full_name"] in done_set]

stats["final_cap"] = len(analytical)
stats["final_repo_count"] = len(analytical)
stats["score_max"] = max(r["criticality_score"] for r in analytical)
stats["score_min"] = min(r["criticality_score"] for r in analytical)
stars = sorted(r["stars"] for r in analytical)
stats["star_min"] = stars[0]
stats["star_median"] = stars[len(stars)//2]
stats["star_max"] = stars[-1]
stats["language_top10"] = Counter(r.get("language") for r in analytical).most_common(10)
stats["pr_active_repo_count"] = sum(1 for v in counts.values() if int(v.get("total_in_window", 0)) > 0)
stats["pr_inactive_repo_count"] = sum(1 for v in counts.values() if int(v.get("total_in_window", 0)) == 0)
Path("results/phase1_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
print(f"refreshed phase1_stats: final_repo_count={stats['final_repo_count']} pr_active={stats['pr_active_repo_count']}")
PY
```

---

## 5. Watching progress without sitting on the terminal

```bash
tmux ls
tmux attach -t phase3_v25       # detach with Ctrl-b d
tmux attach -t phase2_extend_6k # detach with Ctrl-b d

# Phase 3 log — milestones every 50 files
tail -F logs/run_pipeline_v25.log

# Phase 2 extension log — checkpoints every 10 repos
grep "Checkpoint:" logs/02_fetch_prs_extend_6k.log | tail -5
```

Set up monitors with the Monitor tool so you get notifications at phase
transitions and on errors. Patterns I used last time:

```
# Phase 3 transitions + PDF build
tail -F logs/run_pipeline_v25.log | grep -E --line-buffered \
  "=== Phase|prs_so_far=.*\\[(2[05]0|5[05]0|10[05]0|15[05]0|20[05]0|25[05]0|30[05]0|3500|3840)/|run_pipeline exited|paper\\.filled\\.pdf|Anonymization|Traceback|FATAL"

# Phase 2 extension milestones every ~250 repos + completion
tail -F logs/02_fetch_prs_extend_6k.log | grep -E --line-buffered \
  "Loaded \\d+ repos|Checkpoint: (4[0-9]50|5[0-9]50|5[0-9]90|6000)/|PHASE 2 DONE|fetch exited|Traceback|FATAL"
```

---

## 6. After everything finishes — full re-run on the wider sample

When `phase2_extend_6k` completes (or whenever you decide to stop it
again), repeat the trim + refresh + re-run cycle:

```bash
cd /home/ubuntu/mcp-monitoring/agents-prefer-agents
source /home/ubuntu/mcp-monitoring/.venv/bin/activate

# 1. Trim data/repos.json to whatever's actually in _done.json
python - <<'PY'
import json
from pathlib import Path
done = set(json.loads(Path("data/prs/_done.json").read_text()))
src = json.loads(Path("data/repos.top10k.json").read_text())
src.sort(key=lambda r: r["criticality_rank"])
kept = [r for r in src if r["full_name"] in done]
Path("data/repos.json").write_text(json.dumps(kept, indent=2) + "\n")
print(f"data/repos.json: {len(kept)} rows")
PY

# 2. Refresh phase1_stats.json (script in section 4 above).

# 3. Backup current PDF, then re-run pipeline.
cp paper/paper.filled.pdf paper/paper.filled.v25.pdf  # rename for the v2.5 archive
bash scripts/run_pipeline.sh 2>&1 | tee logs/run_pipeline_v3.log
```

---

## Background context the previous agent built up

(Read these if you need motivation / why-we-did-it; not needed to operate
the pipeline.)

- `99_progress.md` — phase-by-phase log, including the v1→v2 → v2.5
  reselection on 2026-04-27.
- `99_notestohuman.md` — design decisions + open questions to the user.
  v2 changelog explains why criticality replaced star-buckets.
- `99_causalvalidity.md` — author's separate notes on chain
  identification (open in editor; user is iterating on this).
- `99_newdataset.md` — author's separate notes (open in editor).
- `paper/appendix/repo_selection.tex` — replicable methodology section
  describing the OpenSSF-based selection. Pinned snapshot
  **2025.07.25/010355**.

## Code changes made on this VM (already in place; don't redo)

These are uncommitted but on disk:

- `scripts/01a_download_criticality.py` — NEW, downloads OSSF CSV.
- `scripts/01_build_repo_list.py` — NEW (criticality-based; legacy at
  `scripts/old_01_build_repo_list.py`).
- `scripts/02_fetch_prs.py` — now writes `data/prs/_repo_pr_counts.json`.
- `scripts/03_classify_prs.py` — `multiprocessing.Pool` (default
  cpu_count-1 workers) + orjson. Falls back to stdlib json if orjson
  unavailable. Backward-compatible serial via `--workers 1`.
- `scripts/05_compute_chains.py` — vectorised `longest_chain_per_group`
  (numpy single pass) + `groupby.sum` for counts. ~20× faster.
- `scripts/08_fill_paper.py` — fills new criticality + PR-volume
  placeholders.
- `scripts/run_phase2_v2.sh` — parametric `TARGET_N` env var.
- `tests/` — 38 pytest tests, all passing.
- `pyproject.toml` — `orjson` added.

Run `python -m pytest tests/ -q` to verify the test suite passes (~1 s).
**If tests fail on the new VM, do not run the pipeline — investigate.**

## What NOT to do

- Don't re-run `01_build_repo_list.py` — it would re-enrich and overwrite
  `data/repos.json` and `data/repos.top10k.json`. The current files are
  the canonical universe.
- Don't run `01a_download_criticality.py` again — the cached CSV is in
  `data/criticality/` already.
- Don't `git add`/`commit`/`push` without explicit user instruction.
- Don't move/delete `data/old_phase1/` — that's the v1 universe preserved
  for the comparison paragraph in the appendix.
- Don't move/delete `data/prs/*.jsonl` — that's the live Phase 2 output
  the new pipeline reads. The user has been explicit about not deleting
  PR JSONLs across restarts.

## Snapshot of state at handoff (2026-04-28 22:45 UTC)

```
data/repos.json              3,840 rows (criticality rank 1..4240, score 0.85..0.55)
data/repos.top10k.json       10,000 rows (full enrichment, sensitivity reserve)
data/prs/                    3,840 jsonls + _done.json + _repo_pr_counts.json
data/old_phase1/             v1 (3000 repo) data preserved
results/phase1_stats.json    final_repo_count=3840, pr_active=2929, pr_inactive=911
paper/paper.filled.pdf       v2 (1,219-active) PDF — RENAME before re-run
data/pr_summary.parquet      v2 outputs (will be overwritten by next pipeline)
data/pr_events.parquet       v2 outputs (will be overwritten)
data/chains.parquet          v2 outputs (will be overwritten)
```

If anything looks wrong: stop, ask the user, do not improvise.
