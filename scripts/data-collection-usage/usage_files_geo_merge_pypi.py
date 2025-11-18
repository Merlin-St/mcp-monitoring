#!/usr/bin/env python3
"""
PyPI Geographic Usage Data Merger

Merges multiple JSONL files from BigQuery export into a single consolidated JSONL file.
Each input file contains PyPI download statistics broken down by package, month, and country.

Features:
- Reads multiple mcp_table_*.jsonl files from specified folder
- Concatenates them into single JSONL output file
- Memory-efficient line-by-line processing
- Progress logging and validation
- Handles large files (~10GB total)

Usage:
    python usage_files_geo_merge_pypi.py --folder data/external-usage/99_geousagedataparts
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
import glob


DEFAULT_OUTPUT = "data/external-usage/usage_bigquery_webresults_pypi.json"


def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/usage_files_geo_merge_pypi.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_input_files(folder: str, logger) -> List[Path]:
    """
    Find all mcp_table_*.jsonl files in the specified folder.

    Returns sorted list of Path objects.
    """
    folder_path = Path(folder)

    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder}")
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder_path.is_dir():
        logger.error(f"Path is not a directory: {folder}")
        raise NotADirectoryError(f"Not a directory: {folder}")

    # Find all mcp_table_*.jsonl files
    pattern = str(folder_path / "mcp_table_*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.error(f"No mcp_table_*.jsonl files found in {folder}")
        raise FileNotFoundError(f"No mcp_table_*.jsonl files in {folder}")

    logger.info(f"Found {len(files)} input files in {folder}")

    return [Path(f) for f in files]


def merge_jsonl_files(input_files: List[Path], output_file: str, logger):
    """
    Merge multiple JSONL files into a single output file.

    Processes line-by-line for memory efficiency.
    """
    total_lines = 0
    total_bytes = 0
    malformed_lines = 0
    progress_interval = 50000  # Log every 50K lines

    logger.info(f"Starting merge of {len(input_files)} files")
    logger.info(f"Output file: {output_file}")

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output file for writing
    with open(output_file, 'w') as outf:
        # Process each input file
        for idx, input_file in enumerate(input_files, 1):
            file_lines = 0
            file_bytes = input_file.stat().st_size

            logger.info(f"Processing file {idx}/{len(input_files)}: {input_file.name} ({file_bytes / 1024 / 1024:.1f} MB)")

            with open(input_file, 'r') as inf:
                for line_num, line in enumerate(inf, 1):
                    line = line.strip()

                    if not line:
                        continue

                    # Validate JSON format
                    try:
                        json.loads(line)
                        outf.write(line + '\n')
                        file_lines += 1
                        total_lines += 1

                        # Progress logging
                        if total_lines % progress_interval == 0:
                            logger.info(f"  Progress: {total_lines:,} lines written")

                    except json.JSONDecodeError as e:
                        malformed_lines += 1
                        logger.warning(f"Skipping malformed JSON at {input_file.name}:{line_num}: {e}")

            total_bytes += file_bytes
            logger.info(f"  Completed {input_file.name}: {file_lines:,} lines")

    # Verify output file
    output_size = output_path.stat().st_size

    logger.info("=== MERGE COMPLETE ===")
    logger.info(f"Input files processed: {len(input_files)}")
    logger.info(f"Total lines written: {total_lines:,}")
    logger.info(f"Malformed lines skipped: {malformed_lines}")
    logger.info(f"Total input size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    logger.info(f"Output file size: {output_size / 1024 / 1024 / 1024:.2f} GB")
    logger.info(f"Output file: {output_file}")

    return total_lines, malformed_lines


def validate_output(output_file: str, expected_lines: int, logger):
    """
    Validate the output file by counting lines and checking format.
    """
    logger.info("Validating output file...")

    output_path = Path(output_file)

    if not output_path.exists():
        logger.error(f"Output file not found: {output_file}")
        raise FileNotFoundError(f"Output file missing: {output_file}")

    # Count lines and validate first/last records
    line_count = 0
    first_record = None
    last_record = None

    with open(output_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if line_num == 1:
                        first_record = record
                    last_record = record
                    line_count += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at output line {line_num}: {e}")

    logger.info(f"Validation complete:")
    logger.info(f"  Lines in output: {line_count:,}")
    logger.info(f"  Expected lines: {expected_lines:,}")
    logger.info(f"  Match: {'✓' if line_count == expected_lines else '✗'}")

    if first_record:
        logger.info(f"  First record: {first_record.get('name', 'N/A')} ({first_record.get('month', 'N/A')}, {first_record.get('country_code', 'N/A')})")

    if last_record:
        logger.info(f"  Last record: {last_record.get('name', 'N/A')} ({last_record.get('month', 'N/A')}, {last_record.get('country_code', 'N/A')})")

    if line_count != expected_lines:
        logger.warning(f"Line count mismatch! Expected {expected_lines:,}, got {line_count:,}")

    return line_count == expected_lines


def main():
    parser = argparse.ArgumentParser(
        description="Merge PyPI geographic usage data files into single JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing mcp_table_*.jsonl files (required)"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("=== PyPI Geographic Usage Data Merger ===")
    logger.info(f"Input folder: {args.folder}")
    logger.info(f"Output file: {args.output}")

    try:
        # Find input files
        input_files = find_input_files(args.folder, logger)

        # Merge files
        total_lines, malformed_lines = merge_jsonl_files(input_files, args.output, logger)

        # Validate output
        validation_success = validate_output(args.output, total_lines, logger)

        if validation_success:
            logger.info("SUCCESS: Merge completed and validated")
        else:
            logger.warning("WARNING: Validation found discrepancies")

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        raise


if __name__ == "__main__":
    main()
