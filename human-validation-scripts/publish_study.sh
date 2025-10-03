#!/bin/bash
# Publish MCP Classification Study to RU Study Service
# Usage: ./publish_study.sh

API_URL="https://study-service.i.apps.ai-safety-institute.org.uk/studies"
CONFIG_FILE="99_study_config_mcp_classification.json"

echo "=========================================="
echo "Publishing MCP Classification Study"
echo "=========================================="
echo ""
echo "API Endpoint: $API_URL"
echo "Config File: $CONFIG_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Sending POST request..."
echo ""

# Make POST request and save response
RESPONSE=$(curl -s -X POST "$API_URL" \
    -H 'accept: */*' \
    -H 'Content-Type: application/json' \
    -d @"$CONFIG_FILE")

# Check if response is valid JSON
if echo "$RESPONSE" | jq . > /dev/null 2>&1; then
    echo "✅ Study published successfully!"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | jq .
    echo ""

    # Extract and display study ID
    STUDY_ID=$(echo "$RESPONSE" | jq -r '.id // empty')
    if [ -n "$STUDY_ID" ]; then
        echo "=========================================="
        echo "🎉 Study ID: $STUDY_ID"
        echo "=========================================="
        echo ""
        echo "View study at:"
        echo "  $API_URL/$STUDY_ID"
        echo ""

        # Save study ID to file
        echo "$STUDY_ID" > study_id.txt
        echo "Study ID saved to: study_id.txt"
    fi
else
    echo "❌ Error: Invalid response from API"
    echo ""
    echo "Response:"
    echo "$RESPONSE"
    exit 1
fi
