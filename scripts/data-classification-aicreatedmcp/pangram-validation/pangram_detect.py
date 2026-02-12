"""
Pangram API-based AI text detection for MCP server READMEs.

Uses the Pangram v3 REST API to detect AI-generated content in README files.
Authentication via `x-api-key` header per Pangram docs:
  https://pangram.readthedocs.io/en/stable/api/rest.html

Requires `pangram_key` environment variable (lowercase).

Output: JSON with ai_generated_probability, detection_method, and analysis_details per server.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # mcp-monitoring/
INPUT_FILE = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "data_unified_filtered_subset.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external-aicreatedmcp" / "agent1-pangram"
OUTPUT_FILE = OUTPUT_DIR / "pangram_results.json"
SUMMARY_FILE = OUTPUT_DIR / "pangram_summary.json"
CHECKPOINT_FILE = OUTPUT_DIR / "pangram_results_checkpoint.json"
LOG_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Environment variable name for the Pangram API key
# ---------------------------------------------------------------------------
PANGRAM_KEY_ENV = "pangram_key"
# Fallback: read key from file if env var not set
PANGRAM_KEY_FILE = Path.home() / ".pangram_key"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent1_pangram_detect.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pangram_detect")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Pangram v3 REST API endpoint
# Docs: https://pangram.readthedocs.io/en/stable/api/rest.html
# Auth: x-api-key header (NOT Bearer token)
PANGRAM_V3_URL = "https://text.api.pangramlabs.com/v3"

# Pangram predict analyses ~400 words per window by default.
# For longer documents the v3 endpoint automatically applies windowed analysis.
MAX_TEXT_LENGTH = 15000  # characters - truncate very long READMEs
RATE_LIMIT_DELAY = 0.6  # seconds between requests to respect rate limits
RETRY_MAX = 5
RETRY_BACKOFF = 2.0  # exponential back-off multiplier
CHECKPOINT_INTERVAL = 25  # save checkpoint every N servers


def _get_api_key() -> str:
    """
    Get the Pangram API key.
    Checks in order: env var 'pangram_key', file ~/.pangram_key, --key-file arg.
    """
    # 1. Environment variable
    api_key = os.environ.get(PANGRAM_KEY_ENV)
    if api_key:
        return api_key.strip()

    # 2. Key file at ~/.pangram_key
    if PANGRAM_KEY_FILE.exists():
        api_key = PANGRAM_KEY_FILE.read_text().strip()
        if api_key:
            logger.info("Read API key from %s", PANGRAM_KEY_FILE)
            return api_key

    # 3. --key-file command line argument
    if "--key-file" in sys.argv:
        try:
            idx = sys.argv.index("--key-file")
            key_path = Path(sys.argv[idx + 1])
            if key_path.exists():
                api_key = key_path.read_text().strip()
                if api_key:
                    logger.info("Read API key from %s", key_path)
                    return api_key
        except (IndexError, OSError) as e:
            logger.warning("Could not read --key-file: %s", e)

    raise ValueError(
        f"Pangram API key not found. Provide it via one of:\n"
        f"  1. export {PANGRAM_KEY_ENV}=<your-key>\n"
        f"  2. echo '<your-key>' > {PANGRAM_KEY_FILE}\n"
        f"  3. python pangram_detect.py --key-file /path/to/keyfile"
    )


def load_input_data() -> list[dict]:
    """Load the 500-server subset JSON."""
    logger.info("Loading input data from %s", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d entries", len(data))
    return data


def truncate_readme(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Truncate README to max_len characters, keeping the beginning."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len]


def _load_checkpoint() -> dict[str, dict]:
    """Load previously checkpointed results, keyed by server id."""
    if not CHECKPOINT_FILE.exists():
        return {}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        checkpoint = {r["id"]: r for r in data}
        logger.info("Loaded checkpoint with %d already-processed servers", len(checkpoint))
        return checkpoint
    except Exception as e:
        logger.warning("Could not load checkpoint: %s - starting fresh", e)
        return {}


def _parse_v3_response(response: dict, entry: dict) -> dict:
    """
    Parse a Pangram v3 API response into our standard result format.

    V3 response fields (per docs at pangram.readthedocs.io):
      - headline: str  ("Fully Human Written", "AI Assisted", "AI Detected")
      - prediction: str  (detailed explanation)
      - short_prediction: str  ("AI", "Human", "Mixed")
      - fraction_human: float  (fraction of doc classified as human-written)
      - fraction_ai_assisted: float  (fraction classified as AI-assisted)
      - fraction_ai: float  (fraction classified as AI-generated)
      - fraction_ai_content: float  (fraction of windows classified as AI)
      - windows: list[dict]  (per-window classifications)
      - avg_ai_likelihood: float  (average AI likelihood across windows)
      - max_ai_likelihood: float  (max AI likelihood across windows)
      - ai_likelihood: float  (legacy single-prediction score, 0-1)
      - dashboard_link: str  (optional link to results dashboard)
    """
    # V3 fraction fields (sum to 1.0)
    fraction_ai = response.get("fraction_ai")
    fraction_ai_assisted = response.get("fraction_ai_assisted")
    fraction_human = response.get("fraction_human")
    fraction_ai_content = response.get("fraction_ai_content")

    # Headline and prediction strings
    headline = response.get("headline", "")
    prediction = response.get("prediction", "")
    short_prediction = response.get("short_prediction", "")

    # Legacy and aggregate likelihood scores
    ai_likelihood = response.get("ai_likelihood")
    avg_ai_likelihood = response.get("avg_ai_likelihood")
    max_ai_likelihood = response.get("max_ai_likelihood")

    # Calculate composite AI probability from best available fields
    ai_prob = None
    if fraction_ai is not None:
        # V3 fractions: combine full-AI + weighted AI-assisted
        ai_prob = fraction_ai + 0.5 * (fraction_ai_assisted or 0)
    elif fraction_ai_content is not None:
        # fraction_ai_content = fraction of windows classified as AI
        ai_prob = fraction_ai_content
    elif avg_ai_likelihood is not None:
        ai_prob = avg_ai_likelihood
    elif ai_likelihood is not None:
        ai_prob = ai_likelihood

    # Window-level analysis
    windows = response.get("windows", [])
    window_details = []
    for w in windows[:20]:  # cap at 20 windows for output size
        window_details.append({
            "label": w.get("label", ""),
            "ai_assistance_score": w.get("ai_assistance_score", w.get("ai_likelihood")),
            "confidence": w.get("confidence", ""),
            "word_count": w.get("word_count"),
        })

    return {
        "id": entry["id"],
        "name": entry.get("name", ""),
        "github_url": entry.get("github_url", ""),
        "ai_generated_probability": round(ai_prob, 4) if ai_prob is not None else None,
        "detection_method": "pangram_api_v3",
        "likely_ai_agent": None,
        "analysis_details": {
            "headline": headline,
            "prediction": prediction,
            "short_prediction": short_prediction,
            "fraction_ai": fraction_ai,
            "fraction_ai_assisted": fraction_ai_assisted,
            "fraction_human": fraction_human,
            "fraction_ai_content": fraction_ai_content,
            "ai_likelihood": ai_likelihood,
            "avg_ai_likelihood": avg_ai_likelihood,
            "max_ai_likelihood": max_ai_likelihood,
            "num_windows": len(windows),
            "window_classifications": window_details,
            "readme_length_chars": len(entry.get("readme_content", "") or ""),
            "readme_truncated": len(entry.get("readme_content", "") or "") > MAX_TEXT_LENGTH,
        },
    }


# ---------------------------------------------------------------------------
# REST API detection (primary approach - no SDK dependency)
# ---------------------------------------------------------------------------
def detect_with_rest(entries: list[dict]) -> list[dict]:
    """
    Use the Pangram v3 REST API directly via requests.

    Authentication: x-api-key header (per Pangram docs).
    Endpoint: POST https://text.api.pangramlabs.com/v3
    Body: {"text": "<readme content>"}
    """
    api_key = _get_api_key()

    # Pangram uses x-api-key header for authentication (NOT Bearer token)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    # Validate the API key with a quick test call
    logger.info("Validating API key with a test request...")
    test_ok = _validate_api_key(headers)
    if not test_ok:
        logger.error("API key validation failed. Check your pangram_key.")
        sys.exit(1)
    logger.info("API key validated successfully.")

    # Load any previously checkpointed results for resume
    checkpoint = _load_checkpoint()
    results = []
    skipped_from_checkpoint = 0

    for i, entry in enumerate(entries):
        # Resume: skip already-processed servers
        if entry["id"] in checkpoint:
            results.append(checkpoint[entry["id"]])
            skipped_from_checkpoint += 1
            continue

        readme = truncate_readme(entry.get("readme_content", "") or "")
        if not readme or len(readme.strip()) < 50:
            logger.warning("Skipping %s - README too short (%d chars)", entry["id"], len(readme.strip()))
            results.append(_empty_result(entry, reason="README too short"))
            continue

        result = _call_pangram_rest(headers, entry, readme, attempt=0)
        results.append(result)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            logger.info(
                "Progress: %d / %d servers processed (%d from checkpoint)",
                i + 1, len(entries), skipped_from_checkpoint,
            )
            _save_checkpoint(results)

        time.sleep(RATE_LIMIT_DELAY)

    if skipped_from_checkpoint > 0:
        logger.info("Resumed %d servers from checkpoint", skipped_from_checkpoint)

    return results


def _validate_api_key(headers: dict) -> bool:
    """Send a short test text to verify the API key works."""
    try:
        resp = requests.post(
            PANGRAM_V3_URL,
            headers=headers,
            json={"text": "This is a short test sentence to validate the API key."},
            timeout=30,
        )
        if resp.status_code == 200:
            return True
        logger.error(
            "API key validation returned status %d: %s",
            resp.status_code, resp.text[:500],
        )
        return False
    except Exception as e:
        logger.error("API key validation request failed: %s", e)
        return False


def _call_pangram_rest(headers: dict, entry: dict, readme: str, attempt: int) -> dict:
    """Call Pangram v3 REST API with retries."""
    try:
        resp = requests.post(
            PANGRAM_V3_URL,
            headers=headers,
            json={"text": readme},
            timeout=60,
        )

        # Handle specific HTTP errors
        if resp.status_code == 429:
            # Rate limited - wait longer and retry
            if attempt < RETRY_MAX:
                wait = max(5.0, RETRY_BACKOFF ** (attempt + 1))
                logger.warning(
                    "Rate limited for %s (attempt %d/%d) - waiting %.1fs",
                    entry["id"], attempt + 1, RETRY_MAX, wait,
                )
                time.sleep(wait)
                return _call_pangram_rest(headers, entry, readme, attempt + 1)

        if resp.status_code == 402:
            # Out of credits
            logger.error("Pangram API credits exhausted (HTTP 402). Stopping.")
            return _empty_result(entry, reason="API credits exhausted (HTTP 402)")

        resp.raise_for_status()
        response = resp.json()

        # Log the raw response keys on first successful call for debugging
        if attempt == 0:
            logger.debug("Response keys for %s: %s", entry["id"], list(response.keys()))

        return _parse_v3_response(response, entry)

    except requests.exceptions.HTTPError as e:
        if attempt < RETRY_MAX:
            wait = RETRY_BACKOFF ** (attempt + 1)
            logger.warning(
                "HTTP error for %s (attempt %d/%d): %s - retrying in %.1fs",
                entry["id"], attempt + 1, RETRY_MAX, e, wait,
            )
            time.sleep(wait)
            return _call_pangram_rest(headers, entry, readme, attempt + 1)
        logger.error("REST API failed for %s after %d retries: %s", entry["id"], RETRY_MAX, e)
        return _empty_result(entry, reason=f"HTTP error: {e}")

    except requests.exceptions.Timeout:
        if attempt < RETRY_MAX:
            wait = RETRY_BACKOFF ** (attempt + 1)
            logger.warning(
                "Timeout for %s (attempt %d/%d) - retrying in %.1fs",
                entry["id"], attempt + 1, RETRY_MAX, wait,
            )
            time.sleep(wait)
            return _call_pangram_rest(headers, entry, readme, attempt + 1)
        logger.error("REST API timed out for %s after %d retries", entry["id"], RETRY_MAX)
        return _empty_result(entry, reason="Timeout after retries")

    except Exception as e:
        if attempt < RETRY_MAX:
            wait = RETRY_BACKOFF ** (attempt + 1)
            logger.warning(
                "REST API error for %s (attempt %d/%d): %s - retrying in %.1fs",
                entry["id"], attempt + 1, RETRY_MAX, e, wait,
            )
            time.sleep(wait)
            return _call_pangram_rest(headers, entry, readme, attempt + 1)
        logger.error("REST API failed for %s after %d retries: %s", entry["id"], RETRY_MAX, e)
        return _empty_result(entry, reason=f"REST API error: {e}")


# ---------------------------------------------------------------------------
# SDK-based detection (if pangram-sdk is installed)
# ---------------------------------------------------------------------------
def detect_with_sdk(entries: list[dict]) -> list[dict]:
    """
    Use the official pangram-sdk to classify each README.
    Requires: pip install pangram-sdk
    """
    try:
        from pangram import Pangram
    except ImportError:
        logger.error("pangram-sdk not installed. Install with: pip install pangram-sdk")
        raise

    api_key = _get_api_key()
    client = Pangram(api_key=api_key)

    # Load any previously checkpointed results for resume
    checkpoint = _load_checkpoint()
    results = []
    skipped_from_checkpoint = 0

    for i, entry in enumerate(entries):
        # Resume: skip already-processed servers
        if entry["id"] in checkpoint:
            results.append(checkpoint[entry["id"]])
            skipped_from_checkpoint += 1
            continue

        readme = truncate_readme(entry.get("readme_content", "") or "")
        if not readme or len(readme.strip()) < 50:
            logger.warning("Skipping %s - README too short (%d chars)", entry["id"], len(readme))
            results.append(_empty_result(entry, reason="README too short"))
            continue

        result = _call_pangram_sdk(client, entry, readme, attempt=0)
        results.append(result)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            logger.info(
                "Progress: %d / %d servers processed (%d from checkpoint)",
                i + 1, len(entries), skipped_from_checkpoint,
            )
            _save_checkpoint(results)

        time.sleep(RATE_LIMIT_DELAY)

    if skipped_from_checkpoint > 0:
        logger.info("Resumed %d servers from checkpoint", skipped_from_checkpoint)

    return results


def _call_pangram_sdk(client, entry: dict, readme: str, attempt: int) -> dict:
    """Call Pangram SDK with retries."""
    try:
        response = client.predict(readme)
        return _parse_v3_response(response, entry)
    except Exception as e:
        if attempt < RETRY_MAX:
            wait = RETRY_BACKOFF ** (attempt + 1)
            logger.warning(
                "Pangram SDK error for %s (attempt %d/%d): %s - retrying in %.1fs",
                entry["id"], attempt + 1, RETRY_MAX, e, wait,
            )
            time.sleep(wait)
            return _call_pangram_sdk(client, entry, readme, attempt + 1)
        logger.error("Pangram SDK failed for %s after %d retries: %s", entry["id"], RETRY_MAX, e)
        return _empty_result(entry, reason=f"SDK error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _empty_result(entry: dict, reason: str = "unknown") -> dict:
    """Return a result with null probability and an error reason."""
    return {
        "id": entry["id"],
        "name": entry.get("name", ""),
        "github_url": entry.get("github_url", ""),
        "ai_generated_probability": None,
        "detection_method": "pangram_api_v3",
        "likely_ai_agent": None,
        "analysis_details": {
            "error": reason,
            "readme_length_chars": len(entry.get("readme_content", "") or ""),
        },
    }


def _save_checkpoint(results: list[dict]) -> None:
    """Save intermediate results as checkpoint for resume capability."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Checkpoint saved: %d results to %s", len(results), CHECKPOINT_FILE)


def save_results(results: list[dict]) -> None:
    """Save final results and summary."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", OUTPUT_FILE)

    # Summary statistics
    valid = [r for r in results if r["ai_generated_probability"] is not None]
    if valid:
        probs = [r["ai_generated_probability"] for r in valid]
        ai_count = sum(1 for p in probs if p >= 0.7)
        mixed_count = sum(1 for p in probs if 0.3 <= p < 0.7)
        human_count = sum(1 for p in probs if p < 0.3)

        summary = {
            "total_servers": len(results),
            "successfully_analyzed": len(valid),
            "failed_or_skipped": len(results) - len(valid),
            "detection_method": "pangram_api_v3",
            "ai_generated_count": ai_count,
            "mixed_count": mixed_count,
            "human_written_count": human_count,
            "ai_generated_pct": round(100 * ai_count / len(valid), 2),
            "mixed_pct": round(100 * mixed_count / len(valid), 2),
            "human_written_pct": round(100 * human_count / len(valid), 2),
            "avg_ai_probability": round(sum(probs) / len(probs), 4),
            "median_ai_probability": round(sorted(probs)[len(probs) // 2], 4),
        }
    else:
        summary = {
            "total_servers": len(results),
            "successfully_analyzed": 0,
            "error": "No results obtained",
        }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", SUMMARY_FILE)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


def main():
    """Run Pangram API detection."""
    logger.info("=" * 60)
    logger.info("Starting Pangram API AI detection for MCP server READMEs")
    logger.info("=" * 60)

    # Check API key
    try:
        api_key = _get_api_key()
        logger.info("API key found (env var '%s', length=%d)", PANGRAM_KEY_ENV, len(api_key))
    except ValueError as e:
        logger.error(
            "%s\nGet a key at https://pangram.com (free researcher keys available).\n"
            "Falling back to heuristic detection - run detect_heuristic.py instead.",
            e,
        )
        sys.exit(1)

    entries = load_input_data()

    # Try SDK first, fall back to REST
    try:
        from pangram import Pangram  # noqa: F401
        logger.info("Using pangram-sdk for detection")
        results = detect_with_sdk(entries)
    except ImportError:
        logger.info("pangram-sdk not installed - using direct REST API (x-api-key auth)")
        results = detect_with_rest(entries)

    save_results(results)
    logger.info("Pangram detection complete.")


if __name__ == "__main__":
    main()
