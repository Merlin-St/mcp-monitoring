#!/usr/bin/env python3
"""
Print all LLM prompts used in the MCP monitoring pipeline.

This script imports prompt definitions from the various classification and filtering
scripts to provide visibility into the exact prompts used for LLM-based analysis.

Usage:
    python scripts/llm_prompts.py
"""

import sys
from pathlib import Path

# Add project directories to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data-classification-servers"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data-cleaning-readmes"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data-classification-tools"))


def print_separator(title: str):
    """Print a formatted section separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_prompt(name: str, prompt: str):
    """Print a prompt with a clear label"""
    print(f"--- {name} ---")
    print(prompt)
    print()


def main():
    """Import and display all LLM prompts"""

    # ========================================================================
    # CLServers Step 2 Inspect - Finance Identification & NAICS Classification
    # ========================================================================
    print_separator("CLServers Step 2: Finance Identification & NAICS Classification")

    from clservers_2_inspect import (
        FINANCE_IDENTIFICATION_SYSTEM_PROMPT,
        NAICS_CLASSIFICATION_SYSTEM_PROMPT
    )
    from naics_3digit_data import format_naics_for_prompt

    print_prompt(
        "1. Finance Identification System Prompt",
        FINANCE_IDENTIFICATION_SYSTEM_PROMPT
    )

    # Format the NAICS prompt with actual NAICS list
    naics_list = format_naics_for_prompt()
    naics_prompt = NAICS_CLASSIFICATION_SYSTEM_PROMPT.format(naics_list=naics_list)
    print_prompt(
        "2. NAICS Classification System Prompt (with NAICS codes)",
        naics_prompt
    )

    # ========================================================================
    # README Content Filtering
    # ========================================================================
    print_separator("README Content Filtering")

    from data_readme_filter_inspect import README_FILTER_SYSTEM_PROMPT

    print_prompt(
        "3. README Filter System Prompt",
        README_FILTER_SYSTEM_PROMPT
    )

    # ========================================================================
    # CLTools - O*NET Task Classification & Functionality
    # ========================================================================
    print_separator("CLTools: O*NET Task Classification & Functionality")

    from cltools_main import FUNCTIONALITY_PROMPT

    print_prompt(
        "4. Functionality Classification Prompt",
        FUNCTIONALITY_PROMPT
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Summary")
    print("Total prompts displayed: 4")
    print()
    print("Prompt Sources:")
    print("  1. scripts/data-classification-servers/clservers_2_inspect.py")
    print("  2. scripts/data-classification-servers/clservers_2_inspect.py")
    print("  3. scripts/data-cleaning-readmes/data_readme_filter_inspect.py")
    print("  4. scripts/data-classification-tools/cltools_main.py")
    print()


if __name__ == "__main__":
    main()
