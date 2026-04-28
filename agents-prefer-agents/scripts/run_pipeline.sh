#!/usr/bin/env bash
# One-shot runner for Phases 3 → 6 (assumes Phase 2 has already populated
# data/prs/). Phase 7 (LaTeX compile + anonymize) is manual because the
# sandbox has no LaTeX installed.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/ubuntu/mcp-monitoring/.venv/bin/activate

echo "=== Phase 3: classify PRs ==="
python scripts/03_classify_prs.py

echo "=== Phase 4: merger-detection audit ==="
python scripts/04_merger_audit.py

echo "=== Phase 5a: chains ==="
python scripts/05_compute_chains.py

echo "=== Phase 6a: within-PR approval comparison (two-sided z-test) ==="
python scripts/06c_within_pr.py

echo "=== Phase 6b: figures ==="
python scripts/07_make_figures.py --all --granularity month --min-cell-n 3

echo "=== Phase 6c: fill paper placeholders ==="
python scripts/08_fill_paper.py

echo "=== Phase 6d: bundle Overleaf zip ==="
python scripts/10_make_overleaf_zip.py

echo "=== Phase 6e: build PDF (pdflatex -> bibtex -> pdflatex -> pdflatex) ==="
python scripts/11_build_pdf.py

echo "=== Phase 7: anonymization check ==="
python scripts/09_anonymize_check.py
echo
echo "Artifacts:"
echo "  paper/paper.filled.pdf (compiled PDF)"
echo "  paper/overleaf.zip (upload to Overleaf)"
