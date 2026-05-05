"""Run the full LaTeX build loop on paper/paper.filled.tex.

Executes: pdflatex -> bibtex -> pdflatex -> pdflatex (the standard cycle
needed for citations and cross-references to resolve).

Fails with a non-zero exit if any pass fails. Intended to be the last step
of the paper pipeline, right after ``10_make_overleaf_zip.py``.
"""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT
get_logger = utils.get_logger

PAPER_DIR = SUBPROJECT_ROOT / "paper"
PAPER_STEM = "paper.filled"

logger = get_logger("11_build_pdf")


def _run(cmd: list[str], *, fatal: bool = True) -> int:
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=PAPER_DIR, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        level = logger.error if fatal else logger.warning
        level("Command exit %d: %s", proc.returncode, " ".join(cmd))
        tail = (proc.stdout or "") + (proc.stderr or "")
        level("Last 40 lines:\n%s", "\n".join(tail.splitlines()[-40:]))
        if fatal:
            raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> None:
    if not (PAPER_DIR / f"{PAPER_STEM}.tex").exists():
        raise SystemExit(
            f"{PAPER_DIR / (PAPER_STEM + '.tex')} not found — run 08_fill_paper.py first."
        )
    # First pdflatex pass may legitimately exit non-zero before .bbl is fresh
    # (e.g., when references.bib gained new keys). Don't abort — bibtex still
    # needs to run so the citations get resolved on the next pass.
    _run(["pdflatex", "-interaction=nonstopmode", f"{PAPER_STEM}.tex"], fatal=False)
    _run(["bibtex", PAPER_STEM])
    _run(["pdflatex", "-interaction=nonstopmode", f"{PAPER_STEM}.tex"], fatal=False)
    _run(["pdflatex", "-interaction=nonstopmode", f"{PAPER_STEM}.tex"])

    log_path = PAPER_DIR / f"{PAPER_STEM}.log"
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        undef_cites = [
            line for line in log_text.splitlines()
            if "Citation" in line and "undefined" in line
        ]
        undef_refs = [
            line for line in log_text.splitlines()
            if "Reference" in line and "undefined" in line
        ]
        if undef_cites:
            logger.warning("Undefined citations remain (%d):", len(undef_cites))
            for line in undef_cites[:10]:
                logger.warning("  %s", line.strip())
        if undef_refs:
            logger.warning("Undefined references remain (%d):", len(undef_refs))
            for line in undef_refs[:10]:
                logger.warning("  %s", line.strip())

    pdf = PAPER_DIR / f"{PAPER_STEM}.pdf"
    if not pdf.exists():
        raise SystemExit(f"Expected {pdf} to exist after the build loop.")
    size_kb = pdf.stat().st_size / 1024
    logger.info("Built %s (%.1f KB).", pdf.relative_to(SUBPROJECT_ROOT), size_kb)


if __name__ == "__main__":
    main()
