#!/usr/bin/env python3
"""
Publish MCP Classification Study to RU Study Service API

This script publishes the generated study configuration to the RU Study Service.
"""

import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RU Study Service API endpoints
API_BASE_URL = "https://study-service.i.apps.ai-safety-institute.org.uk"
STUDIES_ENDPOINT = f"{API_BASE_URL}/studies"


def load_study_config(config_path: str) -> Dict[str, Any]:
    """Load study configuration from JSON file."""
    logger.info(f"Loading study config from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_existing_studies() -> Optional[Dict[str, Any]]:
    """Fetch list of existing studies from the API."""
    try:
        logger.info("Fetching existing studies...")
        response = requests.get(STUDIES_ENDPOINT)
        response.raise_for_status()
        studies = response.json()
        logger.info(f"Found {len(studies.get('items', []))} existing studies")
        return studies
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching studies: {e}")
        return None


def check_study_exists(study_name: str) -> Optional[str]:
    """Check if a study with the given name already exists."""
    studies = get_existing_studies()
    if studies:
        for study in studies.get('items', []):
            if study.get('name') == study_name:
                return study.get('id')
    return None


def create_study(study_config: Dict[str, Any]) -> Optional[str]:
    """
    Create a new study via POST request.
    Returns the study ID if successful.
    """
    try:
        logger.info("Creating new study...")

        # Prepare the payload according to PostStudyDto schema
        payload = {
            "name": study_config.get("name"),
            "studyContentsConfig": study_config.get("studyContentsConfig"),
            "model": study_config.get("model"),
            "maxTokens": study_config.get("maxTokens"),
            "providerOptions": study_config.get("providerOptions"),
            "provider": study_config.get("provider"),
            "responseMode": study_config.get("responseMode"),
            "props": study_config.get("props", {})
        }

        # Make POST request
        response = requests.post(
            STUDIES_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()
        study_id = result.get('id')

        logger.info(f"✅ Study created successfully!")
        logger.info(f"   Study ID: {study_id}")
        logger.info(f"   Study Name: {payload['name']}")

        return study_id

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error creating study: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"   Response: {e.response.text}")
        return None


def update_study(study_id: str, study_config: Dict[str, Any]) -> bool:
    """
    Update an existing study via PUT request.
    Returns True if successful.
    """
    try:
        logger.info(f"Updating study {study_id}...")

        # Prepare the payload according to PutStudyDto schema
        payload = {
            "name": study_config.get("name"),
            "studyContentsConfig": study_config.get("studyContentsConfig"),
            "model": study_config.get("model"),
            "maxTokens": study_config.get("maxTokens"),
            "providerOptions": study_config.get("providerOptions"),
            "provider": study_config.get("provider"),
            "responseMode": study_config.get("responseMode"),
            "props": study_config.get("props", {})
        }

        # Make PUT request
        response = requests.put(
            f"{STUDIES_ENDPOINT}/{study_id}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        logger.info(f"✅ Study updated successfully!")
        logger.info(f"   Study ID: {study_id}")
        logger.info(f"   Study Name: {payload['name']}")

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error updating study: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"   Response: {e.response.text}")
        return False


def get_study(study_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a specific study by ID."""
    try:
        logger.info(f"Fetching study {study_id}...")
        response = requests.get(f"{STUDIES_ENDPOINT}/{study_id}")
        response.raise_for_status()
        study = response.json()
        logger.info(f"✅ Study retrieved: {study.get('name')}")
        return study
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching study: {e}")
        return None


def publish_study(
    config_path: str,
    update_if_exists: bool = True,
    custom_study_id: Optional[str] = None
) -> Optional[str]:
    """
    Main function to publish or update a study.

    Args:
        config_path: Path to study config JSON file
        update_if_exists: If True, update existing study with same name
        custom_study_id: Optional custom study ID (for updates or custom creation)

    Returns:
        Study ID if successful, None otherwise
    """
    # Load config
    study_config = load_study_config(config_path)
    study_name = study_config.get("name")

    logger.info(f"Publishing study: {study_name}")

    # Check if study exists
    if custom_study_id:
        existing_study = get_study(custom_study_id)
        if existing_study:
            logger.info(f"Study with ID {custom_study_id} exists")
            if update_if_exists:
                success = update_study(custom_study_id, study_config)
                return custom_study_id if success else None
            else:
                logger.warning("Study exists but update_if_exists=False. Aborting.")
                return None
        else:
            # Create with custom ID
            logger.info(f"Creating study with custom ID: {custom_study_id}")
            # Note: May need to use PUT with custom ID depending on API
            return create_study(study_config)
    else:
        # Check by name
        existing_id = check_study_exists(study_name)
        if existing_id:
            logger.info(f"Study '{study_name}' already exists with ID: {existing_id}")
            if update_if_exists:
                success = update_study(existing_id, study_config)
                return existing_id if success else None
            else:
                logger.warning("Study exists but update_if_exists=False. Aborting.")
                return existing_id
        else:
            # Create new study
            return create_study(study_config)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Publish MCP Classification Study to RU Study Service"
    )
    parser.add_argument(
        "--config",
        default="99_study_config_mcp_classification.json",
        help="Path to study config JSON file"
    )
    parser.add_argument(
        "--api-url",
        help="Custom API base URL (overrides default)"
    )
    parser.add_argument(
        "--study-id",
        help="Custom study ID for update or creation"
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update if study already exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all existing studies and exit"
    )
    parser.add_argument(
        "--get",
        metavar="STUDY_ID",
        help="Fetch and display a specific study by ID"
    )

    args = parser.parse_args()

    # Update API URL if provided
    global API_BASE_URL, STUDIES_ENDPOINT
    if args.api_url:
        API_BASE_URL = args.api_url
        STUDIES_ENDPOINT = f"{API_BASE_URL}/studies"
        logger.info(f"Using custom API URL: {API_BASE_URL}")

    # Handle list command
    if args.list:
        studies = get_existing_studies()
        if studies:
            print("\n=== Existing Studies ===")
            for study in studies.get('items', []):
                print(f"  ID: {study.get('id')}")
                print(f"  Name: {study.get('name')}")
                print(f"  ---")
        return

    # Handle get command
    if args.get:
        study = get_study(args.get)
        if study:
            print(json.dumps(study, indent=2))
        return

    # Publish study
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    study_id = publish_study(
        config_path=str(config_path),
        update_if_exists=not args.no_update,
        custom_study_id=args.study_id
    )

    if study_id:
        logger.info("\n" + "="*60)
        logger.info("🎉 Study published successfully!")
        logger.info(f"   Study ID: {study_id}")
        logger.info(f"   API URL: {STUDIES_ENDPOINT}/{study_id}")
        logger.info("="*60)


if __name__ == "__main__":
    main()
