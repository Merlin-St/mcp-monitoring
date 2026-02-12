"""
Binoculars-based AI text detection for MCP server READMEs.

Uses the Binoculars method (ICML 2024, arXiv 2401.12070) for zero-shot
LLM-generated text detection. This approach computes a perplexity ratio
between two closely related language models (by default, Falcon-7B and
Falcon-7B-Instruct) to detect machine-generated text.

Requirements:
    pip install binoculars  (or clone https://github.com/ahans30/Binoculars)
    pip install torch transformers accelerate

Hardware requirements:
    - GPU with >= 16GB VRAM (recommended) for Falcon-7B models
    - CPU mode available but very slow (~2-5 min per sample)
    - ~32GB system RAM for CPU-only mode

This script is designed as a high-quality fallback when:
1. Pangram API key is not available
2. More rigorous ML-based detection is needed beyond heuristics
3. GPU resources are available for running transformer models
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "data_unified_filtered_subset.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "agent1-pangram"
OUTPUT_FILE = OUTPUT_DIR / "binoculars_results.json"
SUMMARY_FILE = OUTPUT_DIR / "binoculars_summary.json"
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
        logging.FileHandler(LOG_DIR / "agent1_binoculars_detect.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("binoculars_detect")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Maximum characters to send to Binoculars (truncate very long READMEs)
MAX_TEXT_LENGTH = 8000  # Falcon-7B context is ~2048 tokens, roughly 8K chars
# Batch size for Binoculars inference
BATCH_SIZE = 4
# Binoculars thresholds (from the paper)
BINOCULARS_THRESHOLD_LOW = 0.9015  # Below this = AI-generated (low FPR mode)
BINOCULARS_THRESHOLD_HIGH = 0.8536  # Below this = AI-generated (high accuracy mode)


def clean_readme_for_detection(text: str) -> str:
    """
    Clean README markdown for better detection.
    Remove code blocks, links, images, HTML that would confuse the model.
    Keep the prose content which is what we want to detect.
    """
    if not text:
        return ""

    # Remove code blocks (triple backtick)
    cleaned = re.sub(r'```[\s\S]*?```', ' [code block] ', text)
    # Remove inline code
    cleaned = re.sub(r'`[^`]+`', ' [code] ', cleaned)
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Remove image markdown
    cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)
    # Simplify links to just text
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
    # Remove badge markdown
    cleaned = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', cleaned)
    # Remove markdown heading markers but keep text
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)

    # Truncate to max length
    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH]

    return cleaned.strip()


def detect_with_binoculars(entries: list[dict]) -> list[dict]:
    """
    Run Binoculars detection on all entries.
    Requires: pip install binoculars torch transformers
    """
    try:
        from binoculars import Binoculars
    except ImportError:
        logger.error(
            "Binoculars package not installed. Install with:\n"
            "  git clone https://github.com/ahans30/Binoculars.git\n"
            "  cd Binoculars && pip install -e .\n"
            "Also requires: pip install torch transformers accelerate"
        )
        raise

    # Check for GPU
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning(
                "No GPU detected. Binoculars will run on CPU which is very slow "
                "(~2-5 min per sample). Consider using heuristic detection instead."
            )
    except ImportError:
        device = "cpu"
        logger.warning("PyTorch not available for device check, defaulting to CPU")

    logger.info("Initializing Binoculars model (this may take a few minutes to download)...")
    start_time = time.time()

    # Initialize Binoculars (downloads Falcon-7B and Falcon-7B-Instruct if needed)
    try:
        bino = Binoculars()
    except Exception as e:
        logger.error("Failed to initialize Binoculars: %s", e)
        raise

    init_time = time.time() - start_time
    logger.info("Binoculars initialized in %.1f seconds on %s", init_time, device)

    results = []
    batch_texts = []
    batch_entries = []

    for i, entry in enumerate(entries):
        readme = entry.get("readme_content", "") or ""
        cleaned = clean_readme_for_detection(readme)

        if not cleaned or len(cleaned.strip()) < 100:
            logger.warning("Skipping %s – cleaned README too short (%d chars)", entry["id"], len(cleaned))
            results.append(_empty_result(entry, reason="README too short after cleaning"))
            continue

        batch_texts.append(cleaned)
        batch_entries.append(entry)

        # Process in batches
        if len(batch_texts) >= BATCH_SIZE:
            batch_results = _process_batch(bino, batch_entries, batch_texts)
            results.extend(batch_results)
            batch_texts = []
            batch_entries = []

            if (i + 1) % 20 == 0:
                logger.info("Progress: %d / %d servers processed", i + 1, len(entries))
                _save_checkpoint(results)

    # Process remaining batch
    if batch_texts:
        batch_results = _process_batch(bino, batch_entries, batch_texts)
        results.extend(batch_results)

    return results


def _process_batch(bino, entries: list[dict], texts: list[str]) -> list[dict]:
    """Process a batch of texts through Binoculars."""
    results = []

    try:
        # Binoculars supports batch inference
        scores = bino.compute_score(texts)
        predictions = bino.predict(texts)

        # Handle single item (not returned as list)
        if not isinstance(scores, list):
            scores = [scores]
        if not isinstance(predictions, list):
            predictions = [predictions]

        for entry, score, prediction in zip(entries, scores, predictions):
            # Convert Binoculars score to AI probability
            # Binoculars score: lower = more likely AI-generated
            # We need to invert: higher score = more likely AI
            ai_probability = _binoculars_score_to_probability(float(score))

            results.append({
                "id": entry["id"],
                "name": entry.get("name", ""),
                "github_url": entry.get("github_url", ""),
                "ai_generated_probability": round(ai_probability, 4),
                "detection_method": "binoculars",
                "likely_ai_agent": None,  # Binoculars cannot identify which AI
                "analysis_details": {
                    "binoculars_raw_score": round(float(score), 6),
                    "binoculars_prediction": str(prediction),
                    "threshold_low_fpr": BINOCULARS_THRESHOLD_LOW,
                    "threshold_high_acc": BINOCULARS_THRESHOLD_HIGH,
                    "below_threshold_low": float(score) < BINOCULARS_THRESHOLD_LOW,
                    "below_threshold_high": float(score) < BINOCULARS_THRESHOLD_HIGH,
                    "readme_length_chars": len(entry.get("readme_content", "") or ""),
                },
            })

    except Exception as e:
        logger.error("Batch processing error: %s", e)
        for entry in entries:
            results.append(_empty_result(entry, reason=f"Batch error: {e}"))

    return results


def _binoculars_score_to_probability(score: float) -> float:
    """
    Convert Binoculars raw score to an AI-generation probability [0, 1].

    Binoculars score interpretation:
    - Lower scores indicate more likely AI-generated
    - Score < 0.8536 (high accuracy threshold): very likely AI
    - Score < 0.9015 (low FPR threshold): likely AI
    - Score > 1.0: very likely human

    We map this to a probability where:
    - 0.0 = definitely human
    - 1.0 = definitely AI
    """
    import math

    # Use sigmoid mapping centered around the threshold
    # score of 0.85 → ~0.85 probability of AI
    # score of 0.90 → ~0.60 probability of AI
    # score of 1.00 → ~0.20 probability of AI
    # score of 1.10 → ~0.05 probability of AI

    center = 0.92  # midpoint between thresholds
    steepness = 12  # how sharp the transition is

    probability = 1.0 / (1.0 + math.exp(steepness * (score - center)))
    return max(0.0, min(1.0, probability))


def _empty_result(entry: dict, reason: str = "unknown") -> dict:
    """Return empty result for skipped/failed entries."""
    return {
        "id": entry["id"],
        "name": entry.get("name", ""),
        "github_url": entry.get("github_url", ""),
        "ai_generated_probability": None,
        "detection_method": "binoculars",
        "likely_ai_agent": None,
        "analysis_details": {
            "error": reason,
            "readme_length_chars": len(entry.get("readme_content", "") or ""),
        },
    }


def _save_checkpoint(results: list[dict]) -> None:
    """Save intermediate checkpoint."""
    checkpoint_file = OUTPUT_DIR / "binoculars_results_checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Checkpoint saved: %d results", len(results))


def load_input_data() -> list[dict]:
    """Load the 500-server subset."""
    logger.info("Loading input data from %s", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d entries", len(data))
    return data


def save_results(results: list[dict]) -> None:
    """Save final results and summary."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", OUTPUT_FILE)

    valid = [r for r in results if r["ai_generated_probability"] is not None]
    if valid:
        probs = [r["ai_generated_probability"] for r in valid]
        ai_count = sum(1 for p in probs if p >= 0.7)
        mixed_count = sum(1 for p in probs if 0.3 <= p < 0.7)
        human_count = sum(1 for p in probs if p < 0.3)

        # Count predictions from Binoculars
        from collections import Counter
        prediction_dist = Counter(
            r["analysis_details"].get("binoculars_prediction", "unknown")
            for r in valid
        )

        summary = {
            "total_servers": len(results),
            "successfully_analyzed": len(valid),
            "failed_or_skipped": len(results) - len(valid),
            "detection_method": "binoculars",
            "ai_generated_count": ai_count,
            "mixed_count": mixed_count,
            "human_written_count": human_count,
            "ai_generated_pct": round(100 * ai_count / len(valid), 2),
            "mixed_pct": round(100 * mixed_count / len(valid), 2),
            "human_written_pct": round(100 * human_count / len(valid), 2),
            "avg_ai_probability": round(sum(probs) / len(probs), 4),
            "median_ai_probability": round(sorted(probs)[len(probs) // 2], 4),
            "binoculars_prediction_distribution": dict(prediction_dist),
        }
    else:
        summary = {
            "total_servers": len(results),
            "successfully_analyzed": 0,
            "error": "No results obtained – check GPU/model availability",
        }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", SUMMARY_FILE)
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


def main():
    """Run Binoculars-based AI detection."""
    logger.info("=" * 60)
    logger.info("Starting Binoculars AI detection for MCP server READMEs")
    logger.info("=" * 60)

    # Pre-flight checks
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning(
                "No GPU available. Binoculars on CPU will be extremely slow.\n"
                "For 500 servers, expect ~12-24 hours on CPU.\n"
                "Consider using --limit N to test on a subset first,\n"
                "or use detect_heuristic.py for a fast alternative."
            )
            response = os.environ.get("BINOCULARS_FORCE_CPU", "")
            if response.lower() != "true":
                logger.error(
                    "Set BINOCULARS_FORCE_CPU=true to proceed on CPU, "
                    "or use detect_heuristic.py instead."
                )
                sys.exit(1)
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    # Parse optional --limit argument
    limit = None
    if "--limit" in sys.argv:
        try:
            limit_idx = sys.argv.index("--limit")
            limit = int(sys.argv[limit_idx + 1])
            logger.info("Limiting to first %d servers", limit)
        except (IndexError, ValueError):
            logger.warning("Invalid --limit argument, processing all servers")

    entries = load_input_data()
    if limit:
        entries = entries[:limit]

    results = detect_with_binoculars(entries)
    save_results(results)

    logger.info("Binoculars detection complete.")


if __name__ == "__main__":
    main()
