"""Bundle the filled paper + all LaTeX dependencies into paper/overleaf.zip.

The zip is structured so it can be dragged into Overleaf and compiled directly
with no further setup. Inside the zip the main document is named ``paper.tex``
(built from ``paper.filled.tex``); appendix ``.filled.tex`` files are shipped
as-is because the updated ``08_fill_paper.py`` rewrites the main document's
``\\input`` directives to reference the filled versions directly.

Run after ``08_fill_paper.py`` (which produces the ``.filled.tex`` files).
"""

from __future__ import annotations

import sys
import zipfile
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
utils = import_module("99_utils")
SUBPROJECT_ROOT = utils.SUBPROJECT_ROOT
get_logger = utils.get_logger

PAPER_DIR = SUBPROJECT_ROOT / "paper"
ZIP_PATH = PAPER_DIR / "overleaf.zip"

logger = get_logger("10_make_overleaf_zip")


def _iter_entries() -> list[tuple[Path, str]]:
    """Return (source_path_on_disk, arcname_inside_zip) pairs."""
    entries: list[tuple[Path, str]] = []

    paper_filled = PAPER_DIR / "paper.filled.tex"
    if not paper_filled.exists():
        raise FileNotFoundError(
            f"{paper_filled} not found — run 08_fill_paper.py first."
        )
    entries.append((paper_filled, "paper.tex"))

    for name in ("references.bib", "icml2026.sty", "icml2026.bst",
                 "algorithm.sty", "algorithmic.sty", "fancyhdr.sty"):
        src = PAPER_DIR / name
        if src.exists():
            entries.append((src, name))
        else:
            logger.warning("Missing %s — skipping.", name)

    appendix_dir = PAPER_DIR / "appendix"
    if appendix_dir.is_dir():
        filled_stems = {
            p.name.replace(".filled.tex", "")
            for p in appendix_dir.glob("*.filled.tex")
        }
        for src in sorted(appendix_dir.glob("*.tex")):
            if src.name.endswith(".filled.tex"):
                # Ship the filled version under its full name; the main
                # document's \input directives reference appendix/xxx.filled.tex.
                entries.append((src, f"appendix/{src.name}"))
            elif src.stem not in filled_stems:
                entries.append((src, f"appendix/{src.name}"))

    figures_dir = PAPER_DIR / "figures"
    if figures_dir.is_dir():
        for src in sorted(figures_dir.iterdir()):
            if src.is_file():
                entries.append((src, f"figures/{src.name}"))

    return entries


def main() -> None:
    entries = _iter_entries()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            zf.write(src, arcname=arc)
    size_kb = ZIP_PATH.stat().st_size / 1024
    logger.info(
        "Wrote %s (%d files, %.1f KB). Main document inside: paper.tex",
        ZIP_PATH.relative_to(SUBPROJECT_ROOT), len(entries), size_kb,
    )


if __name__ == "__main__":
    main()
