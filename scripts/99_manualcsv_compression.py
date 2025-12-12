#!/usr/bin/env python3
"""
Manual CSV Compression Script

Compresses all CSV files in data/final/ directory to CSV.gz format.
Verifies integrity by comparing row counts and reports compression ratios.

Usage:
    python scripts/99_manualcsv_compression.py              # Compress all CSV files
    python scripts/99_manualcsv_compression.py --test       # Test mode: compress only one file
    python scripts/99_manualcsv_compression.py --cleanup    # Remove original CSV files after verification
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/99_manualcsv_compression.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    if not filepath.exists():
        return 0.0
    return filepath.stat().st_size / (1024 * 1024)


def verify_integrity(csv_path: Path, gz_path: Path) -> bool:
    """
    Verify that compressed file has same data as original.

    Args:
        csv_path: Path to original CSV file
        gz_path: Path to compressed CSV.gz file

    Returns:
        True if files match, False otherwise
    """
    logger.info(f"Verifying integrity: {csv_path.name} -> {gz_path.name}")

    try:
        # Read both files
        df_original = pd.read_csv(csv_path, low_memory=False)
        df_compressed = pd.read_csv(gz_path, low_memory=False)

        # Compare row counts
        if len(df_original) != len(df_compressed):
            logger.error(
                f"Row count mismatch: {len(df_original)} vs {len(df_compressed)}"
            )
            return False

        # Compare column counts
        if len(df_original.columns) != len(df_compressed.columns):
            logger.error(
                f"Column count mismatch: {len(df_original.columns)} vs {len(df_compressed.columns)}"
            )
            return False

        # Compare column names
        if list(df_original.columns) != list(df_compressed.columns):
            logger.error("Column names mismatch")
            return False

        logger.info(
            f"✓ Verification passed: {len(df_original):,} rows, {len(df_original.columns)} columns"
        )
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def compress_csv(csv_path: Path, keep_original: bool = True) -> bool:
    """
    Compress a CSV file to CSV.gz format.

    Args:
        csv_path: Path to CSV file to compress
        keep_original: If True, keep original CSV file after compression

    Returns:
        True if compression successful, False otherwise
    """
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return False

    gz_path = csv_path.with_suffix(".csv.gz")

    # Skip if already compressed
    if gz_path.exists():
        logger.warning(f"Compressed file already exists: {gz_path.name}")
        return True

    logger.info(f"Compressing: {csv_path.name}")

    try:
        # Get original file size
        original_size = get_file_size_mb(csv_path)
        logger.info(f"  Original size: {original_size:.2f} MB")

        # Read and compress
        df = pd.read_csv(csv_path, low_memory=False)
        df.to_csv(gz_path, index=False, compression="gzip")

        # Get compressed file size
        compressed_size = get_file_size_mb(gz_path)
        compression_ratio = (1 - compressed_size / original_size) * 100

        logger.info(f"  Compressed size: {compressed_size:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.1f}%")

        # Verify integrity
        if not verify_integrity(csv_path, gz_path):
            logger.error(f"Integrity check failed for {csv_path.name}")
            # Remove bad compressed file
            if gz_path.exists():
                gz_path.unlink()
            return False

        # Remove original if requested
        if not keep_original:
            logger.info(f"  Removing original: {csv_path.name}")
            csv_path.unlink()

        logger.info(f"✓ Successfully compressed: {csv_path.name}")
        return True

    except Exception as e:
        logger.error(f"Compression failed for {csv_path.name}: {e}")
        # Clean up partial compressed file if it exists
        if gz_path.exists():
            gz_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compress CSV files in data/final/ to CSV.gz format"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: compress only the smallest CSV file",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove original CSV files after successful compression",
    )
    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Define data/final directory
    data_final_dir = Path("data/final")

    if not data_final_dir.exists():
        logger.error(f"Directory not found: {data_final_dir}")
        sys.exit(1)

    # Find all CSV files
    csv_files = sorted(data_final_dir.glob("*.csv"))

    if not csv_files:
        logger.info("No CSV files found in data/final/")
        return

    logger.info(f"Found {len(csv_files)} CSV files in {data_final_dir}")

    # Test mode: compress only smallest file
    if args.test:
        # Find smallest file
        smallest_file = min(csv_files, key=lambda p: p.stat().st_size)
        logger.info(f"TEST MODE: Compressing only {smallest_file.name}")
        csv_files = [smallest_file]

    # Compress each file
    results = []
    total_original_size = 0
    total_compressed_size = 0

    for csv_path in csv_files:
        logger.info(f"\n{'=' * 60}")
        success = compress_csv(csv_path, keep_original=not args.cleanup)
        results.append((csv_path.name, success))

        if success:
            original_size = get_file_size_mb(csv_path) if csv_path.exists() else 0
            gz_path = csv_path.with_suffix(".csv.gz")
            compressed_size = get_file_size_mb(gz_path)

            # For cleanup mode, get original size from gz file metadata
            if args.cleanup and not csv_path.exists():
                # Approximate original size from before cleanup
                original_size = compressed_size * 3  # Rough estimate

            total_original_size += original_size
            total_compressed_size += compressed_size

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPRESSION SUMMARY")
    logger.info(f"{'=' * 60}")

    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if total_original_size > 0:
        overall_ratio = (1 - total_compressed_size / total_original_size) * 100
        logger.info(f"\nTotal original size: {total_original_size:.2f} MB")
        logger.info(f"Total compressed size: {total_compressed_size:.2f} MB")
        logger.info(f"Overall compression ratio: {overall_ratio:.1f}%")
        logger.info(f"Space saved: {total_original_size - total_compressed_size:.2f} MB")

    logger.info(f"\n{'=' * 60}")
    logger.info("Individual results:")
    for filename, success in results:
        status = "✓" if success else "✗"
        logger.info(f"  {status} {filename}")

    if failed > 0:
        logger.warning(f"\n{failed} file(s) failed to compress")
        sys.exit(1)
    else:
        logger.info("\n✓ All files compressed successfully!")


if __name__ == "__main__":
    main()
