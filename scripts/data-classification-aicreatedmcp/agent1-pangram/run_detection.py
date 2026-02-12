"""
Orchestrator for Agent 1 (Pangram/AI Detection) pipeline.

Runs the AI text detection pipeline in priority order:
1. Pangram API (if PANGRAM_API_KEY is set and pangram-sdk installed)
2. Binoculars (if GPU available and binoculars package installed)
3. Heuristic fingerprint detection (always available, zero dependencies)

The heuristic detector always runs as it provides complementary signals.
Results from the best available ML method are combined with heuristics
into a final merged output.

Usage:
    python run_detection.py                    # Auto-detect best method
    python run_detection.py --method pangram   # Force Pangram API
    python run_detection.py --method binoculars # Force Binoculars
    python run_detection.py --method heuristic  # Heuristic only
    python run_detection.py --method all        # Run all available methods
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "agent1-pangram"
MERGED_OUTPUT = OUTPUT_DIR / "merged_results.json"
MERGED_SUMMARY = OUTPUT_DIR / "merged_summary.json"
LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent1_run_detection.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_detection")


def check_pangram_available() -> bool:
    """Check if Pangram API is available."""
    api_key = os.environ.get("pangram_key")
    if not api_key:
        logger.info("pangram_key not set – Pangram API unavailable")
        return False
    try:
        import pangram  # noqa: F401
        logger.info("pangram-sdk found and API key set – Pangram API available")
        return True
    except ImportError:
        # SDK not installed but we have REST fallback
        logger.info("pangram-sdk not installed, but API key set – REST API available")
        return True


def check_binoculars_available() -> bool:
    """Check if Binoculars can run."""
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("No GPU available – Binoculars not practical")
            return False
    except ImportError:
        logger.info("PyTorch not installed – Binoculars unavailable")
        return False

    try:
        import binoculars  # noqa: F401
        logger.info("Binoculars package found with GPU – available")
        return True
    except ImportError:
        logger.info("Binoculars package not installed")
        return False


def run_pangram():
    """Run Pangram detection."""
    from pangram_detect import main as pangram_main
    pangram_main()


def run_binoculars():
    """Run Binoculars detection."""
    from binoculars_detect import main as binoculars_main
    binoculars_main()


def run_heuristic():
    """Run heuristic detection."""
    from detect_heuristic import main as heuristic_main
    heuristic_main()


def merge_results() -> None:
    """
    Merge results from all available detection methods into a single output.
    Priority: Pangram > Binoculars > Heuristic for the primary score.
    Heuristic signals are always included as supplementary data.
    """
    pangram_file = OUTPUT_DIR / "pangram_results.json"
    binoculars_file = OUTPUT_DIR / "binoculars_results.json"
    heuristic_file = OUTPUT_DIR / "heuristic_results.json"

    pangram_results = {}
    binoculars_results = {}
    heuristic_results = {}

    if pangram_file.exists():
        data = json.loads(pangram_file.read_text())
        pangram_results = {r["id"]: r for r in data}
        logger.info("Loaded %d Pangram results", len(pangram_results))

    if binoculars_file.exists():
        data = json.loads(binoculars_file.read_text())
        binoculars_results = {r["id"]: r for r in data}
        logger.info("Loaded %d Binoculars results", len(binoculars_results))

    if heuristic_file.exists():
        data = json.loads(heuristic_file.read_text())
        heuristic_results = {r["id"]: r for r in data}
        logger.info("Loaded %d Heuristic results", len(heuristic_results))

    if not heuristic_results:
        logger.error("No heuristic results found – cannot merge")
        return

    # Use heuristic IDs as the base set
    all_ids = list(heuristic_results.keys())
    merged = []

    for server_id in all_ids:
        h = heuristic_results.get(server_id, {})
        p = pangram_results.get(server_id, {})
        b = binoculars_results.get(server_id, {})

        # Determine primary method and score
        primary_method = None
        primary_score = None

        if p and p.get("ai_generated_probability") is not None:
            primary_method = p["detection_method"]
            primary_score = p["ai_generated_probability"]
        elif b and b.get("ai_generated_probability") is not None:
            primary_method = b["detection_method"]
            primary_score = b["ai_generated_probability"]

        heuristic_score = h.get("ai_generated_probability")

        # Compute ensemble score (weighted average if multiple methods available)
        if primary_score is not None and heuristic_score is not None:
            # ML method gets 70% weight, heuristic 30%
            ensemble_score = round(0.7 * primary_score + 0.3 * heuristic_score, 4)
            ensemble_method = f"ensemble({primary_method}+heuristic)"
        elif primary_score is not None:
            ensemble_score = primary_score
            ensemble_method = primary_method
        elif heuristic_score is not None:
            ensemble_score = heuristic_score
            ensemble_method = "heuristic_fingerprint"
        else:
            ensemble_score = None
            ensemble_method = "none"

        # Classification
        if ensemble_score is not None:
            if ensemble_score >= 0.7:
                classification = "likely_ai_generated"
            elif ensemble_score >= 0.4:
                classification = "mixed_or_uncertain"
            else:
                classification = "likely_human_written"
        else:
            classification = "insufficient_data"

        # Determine likely AI agent (from any source)
        likely_agent = (
            p.get("likely_ai_agent")
            or b.get("likely_ai_agent")
            or h.get("likely_ai_agent")
        )

        merged.append({
            "id": server_id,
            "name": h.get("name", ""),
            "github_url": h.get("github_url", ""),
            "ai_generated_probability": ensemble_score,
            "detection_method": ensemble_method,
            "classification": classification,
            "likely_ai_agent": likely_agent,
            "method_scores": {
                "pangram": p.get("ai_generated_probability") if p else None,
                "binoculars": b.get("ai_generated_probability") if b else None,
                "heuristic": heuristic_score,
            },
            "analysis_details": {
                "heuristic_signals": h.get("analysis_details", {}).get("signal_scores", {}),
                "pangram_details": p.get("analysis_details", {}) if p else None,
                "binoculars_details": b.get("analysis_details", {}) if b else None,
            },
        })

    # Save merged results
    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Merged results saved to %s", MERGED_OUTPUT)

    # Generate summary
    from collections import Counter
    valid = [r for r in merged if r["ai_generated_probability"] is not None]
    if valid:
        probs = [r["ai_generated_probability"] for r in valid]
        classifications = Counter(r["classification"] for r in valid)
        methods_used = Counter(r["detection_method"] for r in valid)

        summary = {
            "total_servers": len(merged),
            "successfully_analyzed": len(valid),
            "failed_or_skipped": len(merged) - len(valid),
            "methods_used": dict(methods_used),
            "classification_distribution": dict(classifications),
            "likely_ai_generated_count": classifications.get("likely_ai_generated", 0),
            "mixed_or_uncertain_count": classifications.get("mixed_or_uncertain", 0),
            "likely_human_written_count": classifications.get("likely_human_written", 0),
            "likely_ai_generated_pct": round(
                100 * classifications.get("likely_ai_generated", 0) / len(valid), 2
            ),
            "mixed_or_uncertain_pct": round(
                100 * classifications.get("mixed_or_uncertain", 0) / len(valid), 2
            ),
            "likely_human_written_pct": round(
                100 * classifications.get("likely_human_written", 0) / len(valid), 2
            ),
            "avg_ai_probability": round(sum(probs) / len(probs), 4),
            "median_ai_probability": round(sorted(probs)[len(probs) // 2], 4),
            "p10_ai_probability": round(sorted(probs)[int(len(probs) * 0.1)], 4),
            "p25_ai_probability": round(sorted(probs)[int(len(probs) * 0.25)], 4),
            "p75_ai_probability": round(sorted(probs)[int(len(probs) * 0.75)], 4),
            "p90_ai_probability": round(sorted(probs)[int(len(probs) * 0.9)], 4),
            "agents_detected": dict(Counter(
                r["likely_ai_agent"]
                for r in valid
                if r["likely_ai_agent"] is not None
            )),
            "pangram_available": bool(pangram_results),
            "binoculars_available": bool(binoculars_results),
        }
    else:
        summary = {
            "total_servers": len(merged),
            "successfully_analyzed": 0,
            "error": "No valid results obtained",
        }

    with open(MERGED_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Merged summary:\n%s", json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run AI text detection pipeline")
    parser.add_argument(
        "--method",
        choices=["auto", "pangram", "binoculars", "heuristic", "all"],
        default="auto",
        help="Detection method to use (default: auto-detect best available)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Agent 1: AI Text Detection Pipeline")
    logger.info("=" * 60)

    start_time = time.time()

    # Always run heuristic (fast, no dependencies)
    logger.info("--- Running heuristic detection (always runs) ---")
    try:
        # Add script dir to path for imports
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        run_heuristic()
    except Exception as e:
        logger.error("Heuristic detection failed: %s", e)
        raise

    # Run ML-based method based on selection
    if args.method == "auto":
        if check_pangram_available():
            logger.info("--- Running Pangram API detection ---")
            try:
                run_pangram()
            except Exception as e:
                logger.error("Pangram detection failed: %s", e)
        elif check_binoculars_available():
            logger.info("--- Running Binoculars detection ---")
            try:
                run_binoculars()
            except Exception as e:
                logger.error("Binoculars detection failed: %s", e)
        else:
            logger.info("No ML-based detection available – using heuristic only")

    elif args.method == "pangram":
        logger.info("--- Running Pangram API detection ---")
        run_pangram()

    elif args.method == "binoculars":
        logger.info("--- Running Binoculars detection ---")
        run_binoculars()

    elif args.method == "all":
        if check_pangram_available():
            logger.info("--- Running Pangram API detection ---")
            try:
                run_pangram()
            except Exception as e:
                logger.error("Pangram detection failed: %s", e)
        if check_binoculars_available():
            logger.info("--- Running Binoculars detection ---")
            try:
                run_binoculars()
            except Exception as e:
                logger.error("Binoculars detection failed: %s", e)

    # elif args.method == "heuristic": already ran above

    # Merge all available results
    logger.info("--- Merging results from all methods ---")
    try:
        merge_results()
    except Exception as e:
        logger.error("Merge failed: %s", e)

    elapsed = time.time() - start_time
    logger.info("Pipeline complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
