"""Phase 7 anonymization check.

Grep the final paper artefacts for identifiers that must not appear in the
blind submission PDF: "AISI", "Merlin", "Stein", "UK AI Safety", common GH
handles, "mcp-monitoring", the subproject path, and the author's email.

Fails (exits 1) if any are found. Run this before submitting the PDF.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]

# Files to scan. Add more as the submission bundle grows.
# .tex files only — references.bib legitimately contains co-author names in
# citation entries (these render as "[Smith et al.]" in the PDF bibliography,
# which is expected in double-blind venues). What must NOT leak is the paper
# author block, running title, or body prose.
FILES = [
    SUBPROJECT_ROOT / "paper" / "paper.tex",
    SUBPROJECT_ROOT / "paper" / "paper.filled.tex",
    SUBPROJECT_ROOT / "paper" / "appendix" / "audit.tex",
    SUBPROJECT_ROOT / "paper" / "appendix" / "threats.tex",
    SUBPROJECT_ROOT / "paper" / "appendix" / "allowlist.tex",
    SUBPROJECT_ROOT / "paper" / "appendix" / "regression.tex",
    SUBPROJECT_ROOT / "paper" / "appendix" / "repo_selection.tex",
]

PATTERNS = [
    (r"\bAISI\b", "AISI"),
    (r"\bMerlin\b", "Merlin"),
    (r"\bStein\b", "Stein (author surname)"),
    (r"UK AI Safety", "UK AISI institution"),
    (r"Safety Institute", "AI Safety Institute"),
    (r"mcp-monitoring", "repo-identifying path"),
    (r"agents-prefer-agents", "subproject-identifying path"),
    (r"@bsg\.ox\.ac\.uk", "Oxford email"),
    (r"\bMerlin-St\b", "GitHub handle"),
]

# Whitelist substrings that might contain a pattern but are fine:
# e.g., "UK AI" standalone doesn't trigger; "Safety Institute" doesn't trigger
# in a context we explicitly allow. Keep this empty by default.
ALLOWLIST_SUBSTRINGS: list[str] = []


def main() -> int:
    problems: list[tuple[Path, int, str, str]] = []
    for f in FILES:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            # Skip LaTeX comment lines.
            if line.lstrip().startswith("%"):
                continue
            for pat, desc in PATTERNS:
                if re.search(pat, line, flags=re.IGNORECASE):
                    if any(s in line for s in ALLOWLIST_SUBSTRINGS):
                        continue
                    problems.append((f.relative_to(SUBPROJECT_ROOT), i, desc, line.strip()))
    if problems:
        print("ANONYMIZATION ISSUES FOUND:")
        for f, i, desc, line in problems:
            print(f"  {f}:{i}  [{desc}]  {line[:140]}")
        return 1
    print("Anonymization check: OK (no identifying patterns found).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
