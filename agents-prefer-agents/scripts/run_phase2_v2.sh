#!/usr/bin/env bash
# Auto-pipeline: wait for Phase 1 v2 to finish, then run trim → fill → archive → Phase 2.
# Designed to run inside a long-lived tmux session so it survives laptop disconnect.
#
# Idempotent: every step checks whether it already happened and skips if so.
# Re-running this script is safe.
#
# Usage:
#   bash scripts/run_phase2_v2.sh
#
# Exits non-zero on hard errors. On rate-limit / transient errors during Phase 2,
# the underlying 02_fetch_prs.py is itself resume-friendly (per-repo _done.json),
# so re-running this script after a kill will pick up where it left off.

set -euo pipefail

cd "$(dirname "$0")/.."   # cd into agents-prefer-agents/

# Activate the parent venv. (Subproject uses parent's .venv per CLAUDE.md.)
# shellcheck disable=SC1091
source /home/ubuntu/mcp-monitoring/.venv/bin/activate

LOGFILE="logs/run_phase2_v2.sh.log"
mkdir -p logs

# Tee everything to a log file in addition to the tmux pane.
exec > >(tee -a "$LOGFILE") 2>&1

ts() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }
say() { echo "[$(ts)] $*"; }

PHASE1_LOG="logs/01_build_repo_list_v2.log"
PHASE1_DONE_MARKER='Wrote stats to'
REPOS_JSON="data/repos.json"
REPOS_TOP10K_JSON="data/repos.top10k.json"
PHASE1_STATS="results/phase1_stats.json"

# ---------------------------------------------------------------------------
# Step 1 — Wait for Phase 1 v2 enrichment to finish.
# ---------------------------------------------------------------------------
say "=== STEP 1: wait for Phase 1 v2 enrichment ==="
if [[ -f "$PHASE1_STATS" ]] && grep -q "$PHASE1_DONE_MARKER" "$PHASE1_LOG" 2>/dev/null; then
  say "Phase 1 already complete (found $PHASE1_STATS and done marker in log). Skipping wait."
else
  say "Polling $PHASE1_LOG for completion marker '$PHASE1_DONE_MARKER'..."
  while true; do
    if [[ -f "$PHASE1_STATS" ]] && grep -q "$PHASE1_DONE_MARKER" "$PHASE1_LOG" 2>/dev/null; then
      say "Phase 1 marker found."
      break
    fi
    if [[ -f "$PHASE1_LOG" ]] && grep -qE "Traceback|FATAL|sys\.exit\(2\)" "$PHASE1_LOG"; then
      say "ERROR: Phase 1 log shows a fatal error. Aborting."
      tail -30 "$PHASE1_LOG"
      exit 1
    fi
    sleep 30
  done
fi

# ---------------------------------------------------------------------------
# Step 2 — Trim to top N by criticality_rank; archive 10k as repos.top10k.json.
# ---------------------------------------------------------------------------
TARGET_N=${TARGET_N:-2000}
say "=== STEP 2: trim repos.json to top $TARGET_N ==="
if [[ -f "$REPOS_TOP10K_JSON" ]]; then
  # Already trimmed in a prior run. If the current data/repos.json has the
  # right size, skip; otherwise re-trim from the preserved 10k.
  CURR_N=$(python -c "import json; print(len(json.load(open('data/repos.json'))))")
  if [[ "$CURR_N" == "$TARGET_N" ]]; then
    say "Trim already done ($REPOS_TOP10K_JSON exists, repos.json has $CURR_N rows). Skipping."
  else
    say "data/repos.json has $CURR_N rows, want $TARGET_N. Re-slicing from $REPOS_TOP10K_JSON."
    python - <<PY
import json
from pathlib import Path
src = json.loads(Path("data/repos.top10k.json").read_text())
src.sort(key=lambda r: r["criticality_rank"])
top = src[:$TARGET_N]
Path("data/repos.json").write_text(json.dumps(top, indent=2) + "\n")
print(f"  wrote data/repos.json (top {len(top)})")
PY
  fi
else
  python - <<PY
import json
from pathlib import Path
src = Path("data/repos.json")
all_rows = json.loads(src.read_text())
print(f"  source rows: {len(all_rows)}")
Path("data/repos.top10k.json").write_text(json.dumps(all_rows, indent=2) + "\n")
all_rows.sort(key=lambda r: r["criticality_rank"])
top = all_rows[:$TARGET_N]
src.write_text(json.dumps(top, indent=2) + "\n")
print(f"  wrote data/repos.top10k.json ({len(all_rows)} rows)")
print(f"  wrote data/repos.json (top {len(top)} by criticality)")
PY
fi

# ---------------------------------------------------------------------------
# Step 3 — Refresh phase1_stats.json so the appendix reflects the cap.
# ---------------------------------------------------------------------------
say "=== STEP 3: refresh phase1_stats.json for the trimmed sample ==="
python - <<PY
import json
from collections import Counter
from pathlib import Path

stats_path = Path("results/phase1_stats.json")
stats = json.loads(stats_path.read_text())

# Recompute the sample-characteristics fields from the trimmed repos.json.
repos = json.loads(Path("data/repos.json").read_text())
stats["final_cap"] = $TARGET_N
stats["final_repo_count"] = len(repos)
stats["score_max"] = max(r["criticality_score"] for r in repos)
stats["score_min"] = min(r["criticality_score"] for r in repos)
stars = sorted(r.get("stars", 0) for r in repos)
stats["star_min"] = stars[0]
stats["star_median"] = stars[len(stars) // 2]
stats["star_max"] = stars[-1]
langs = Counter(r.get("language") for r in repos)
stats["language_top10"] = langs.most_common(10)

stats_path.write_text(json.dumps(stats, indent=2) + "\n")
print(f"  refreshed: final_repo_count={stats['final_repo_count']}, "
      f"score=[{stats['score_min']:.4f},{stats['score_max']:.4f}], "
      f"stars=[{stats['star_min']},{stats['star_median']},{stats['star_max']}]")
PY

# ---------------------------------------------------------------------------
# Step 4 — Fill the appendix placeholders.
# ---------------------------------------------------------------------------
say "=== STEP 4: fill paper appendix placeholders ==="
python scripts/08_fill_paper.py
unfilled=$(grep -oE '\\PLACEHOLDER[A-Z0-9]+' paper/appendix/repo_selection.filled.tex 2>/dev/null | sort -u || true)
if [[ -n "$unfilled" ]]; then
  say "Note: these placeholders are still unfilled (expected: PR-volume placeholders fill after Phase 2):"
  echo "$unfilled" | sed 's/^/    /'
fi

# ---------------------------------------------------------------------------
# Step 5 — Archive v1 downstream parquets to data/old_phase1/.
# ---------------------------------------------------------------------------
say "=== STEP 5: archive v1 downstream parquets ==="
mkdir -p data/old_phase1
moved_any=0
for f in data/pr_summary.parquet data/pr_events.parquet data/chains.parquet data/merge_rates.parquet; do
  if [[ -f "$f" ]]; then
    mv "$f" "data/old_phase1/$(basename "$f")"
    say "  moved $f"
    moved_any=1
  fi
done
if [[ -d data/prs && ! -d data/old_phase1/prs ]]; then
  mv data/prs data/old_phase1/prs
  say "  moved data/prs/ (v1 PR JSONLs) → data/old_phase1/prs/"
  moved_any=1
fi
mkdir -p data/prs   # fresh empty dir for Phase 2 v2.
if [[ "$moved_any" -eq 0 ]]; then
  say "  (nothing to archive — v1 parquets already moved)"
fi

# ---------------------------------------------------------------------------
# Step 6 — Phase 2 v2: fetch PRs for top-1,000 repos with cap=1,000.
# ---------------------------------------------------------------------------
say "=== STEP 6: Phase 2 v2 — fetch PRs (max 1,000 per repo) ==="
say "Logging to logs/02_fetch_prs_v2.log."
python scripts/02_fetch_prs.py --max-prs-per-repo 1000 \
    > logs/02_fetch_prs_v2.log 2>&1 &
PHASE2_PID=$!
say "Phase 2 PID: $PHASE2_PID"
wait $PHASE2_PID
PHASE2_RC=$?

if [[ $PHASE2_RC -ne 0 ]]; then
  say "Phase 2 exited non-zero ($PHASE2_RC). The script is resume-friendly; re-run scripts/run_phase2_v2.sh to pick up where it left off."
  exit $PHASE2_RC
fi

say "=== ALL DONE — Phase 2 v2 complete. ==="
say "Next manual step (when you're back): run 'bash scripts/run_pipeline.sh' for Phases 3-7+8+9, then re-fill the paper appendix."
