# Preview Study in Storybook

## Prerequisites

You need access to the RU Study Composer repository.

## Option 1: Local Storybook

### Setup Steps

1. **Clone the RU Study repository** (if not already):
   ```bash
   cd /home/ubuntu
   git clone <ru-study-composer-repo-url>
   cd <ru-study-composer-directory>
   ```

2. **Install dependencies**:
   ```bash
   yarn install
   ```

3. **Start Storybook**:
   ```bash
   yarn storybook
   ```

4. **Preview your study**:
   - Navigate to "Study-UI Components/Study-As-Code/Study Composer"
   - Find the "EditableConfig" story
   - Copy the contents of `99_study_config_mcp_classification.json`
   - Paste into the Storybook config editor
   - Preview the study interactively

## Option 2: Config Validator

1. In Storybook, navigate to:
   - "Study-UI Components/Study-As-Code/ConfigValidator"

2. Paste your config to validate:
   - Copy from `99_study_config_mcp_classification.json`
   - Check for any validation errors
   - View schema reference at bottom

## Your Study Config Location

```
/home/ubuntu/mcp-monitoring/human-validation-scripts/99_study_config_mcp_classification.json
```

## What to Check in Storybook

### Server Classification Pages (2 pages)
- ✅ Server name, description, README summary displayed
- ✅ All tools listed with names and descriptions
- ✅ 3 questions: Industry Generality, Environment Generality, Payment Autonomy

### Tool Classification Pages (2 pages)
- ✅ Tool name, description, input schema displayed
- ✅ Server name shown
- ✅ Autonomy classification with conditional subcategories
- ✅ O*NET classification with 2-level conditional hierarchy

### Navigation
- ✅ Back button enabled
- ✅ Progress bar showing
- ✅ Page numbers displayed
- ✅ Elapsed time showing

## Common Issues

### If Storybook won't start:
```bash
# Clear cache and reinstall
yarn cache clean
rm -rf node_modules
yarn install
yarn storybook
```

### If config validation fails:
- Check JSON syntax (should be valid - we already validated)
- Ensure all question IDs are unique (they are)
- Verify conditional logic references exist (they do)

## Alternative: Direct API Testing

If you can't run Storybook locally, you can POST directly to the API:

```bash
curl -X POST \
  'https://study-service.i.apps.ai-safety-institute.org.uk/studies' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d @/home/ubuntu/mcp-monitoring/human-validation-scripts/99_study_config_mcp_classification.json
```

Then view the study at the returned URL.
