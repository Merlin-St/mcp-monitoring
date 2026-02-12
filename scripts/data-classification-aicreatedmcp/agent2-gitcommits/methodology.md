# Agent 2: AI-Created MCP Server Detection via Git History Mining

## Approach

This agent detects AI-created MCP servers by mining four categories of evidence from each
repository's GitHub metadata: Co-Authored-By commit trailers, AI configuration files,
AI bot contributors, and AI tool handle mentions. Evidence is aggregated into a binary
classification (`ai_authored = yes/no`).

## Key Design Decision: Binary Classification

The previous version used a weighted composite score in [0, 1], which introduced arbitrary
thresholds and made results difficult to interpret. The updated approach uses a binary
`ai_authored` field set to `"yes"` if ANY of four criteria is met, and `"no"` otherwise.
Each triggered criterion is recorded in `ai_authored_reasons` for transparency.

## Full Commit History Scanning

The previous version fetched only 1 page of commits (max 100) per repository. This update
paginates through the full commit history using the GitHub REST API's `page` parameter,
fetching up to 10,000 commits (100 pages of 100 commits). Pagination stops when an empty
page is returned or the 10,000-commit cap is reached. This ensures that AI tool evidence
in older commits is not missed, which is particularly important for long-lived repositories
where AI tools may have been adopted partway through development.

## Data Sources

For each server's GitHub repository, we query the GitHub REST API for:

1. **Commits** -- full paginated history, up to 10,000 per repo (100 per page, up to 100 pages)
2. **Pull requests** -- up to 30 most recent (all states)
3. **Repository file tree** -- recursive listing via Git Trees API

## Binary Classification Criteria

A server is classified as `ai_authored = "yes"` if ANY of the following conditions is true.

### Criterion 1: Co-Authored-By Lines

At least one `Co-Authored-By` trailer in any commit message or PR body that references
a known AI tool. AI coding agents frequently add these trailers automatically:

- Claude Code: `Co-Authored-By: Claude <noreply@anthropic.com>`
- GitHub Copilot: `Co-Authored-By: Copilot <noreply@github.com>`
- ChatGPT/OpenAI, Devin, Codex, Aider, Cline, Roo Code, Augment, Continue.dev,
  Gemini, Windsurf

**Rationale**: Co-Authored-By headers are the strongest single indicator because they are
machine-generated (not typed by humans) and explicitly attribute code to an AI tool.
The MSR 2026 "Fingerprinting AI Coding Agents" paper (arXiv 2601.17406) confirms that
Co-Authored-By patterns are explicitly and consistently used by Claude Code.

### Criterion 2: AI Configuration Files

At least one AI tool configuration file is present in the repository file tree:

| Tool | Config Files |
|------|-------------|
| Claude | `CLAUDE.md`, `.claude/`, `.claude/settings.json`, `.claude/settings.local.json` |
| Cursor | `.cursor/`, `.cursorrules`, `.cursor/rules`, `.cursorignore` |
| Copilot | `.github/copilot-instructions.md` |
| Aider | `.aider.conf.yml`, `.aider/`, `.aiderignore` |
| Codeium | `.codeium/` |
| Windsurf | `.windsurfrules` |
| Cline | `.clinerules`, `.cline/` |
| Roo Code | `.roo/`, `.roorules`, `.roomodes` |
| Codex | `AGENTS.md`, `codex.md` |
| Augment | `.augment/`, `.augment-guidelines` |
| Continue.dev | `.continue/`, `.continuerules` |

**Rationale**: These files are created by or for specific AI coding tools and their presence
indicates active use of that tool during development. This is checked via a single API call
to the Git Trees endpoint (recursive listing).

### Criterion 3: AI Bot Contributors

At least one commit author or PR author matches a known AI bot account:

- `devin-ai-integration[bot]` (Devin)
- `copilot[bot]`, `github-copilot[bot]` (GitHub Copilot)
- `claude[bot]`, `anthropic-ai[bot]` (Claude)

**Exclusions**: Dependency management bots (dependabot, renovate, snyk, greenkeeper,
allcontributors) are explicitly excluded because they represent automated dependency
management, not AI-assisted code creation.

**Rationale**: Bot accounts authoring commits directly indicate automated AI-driven
development, not just AI assistance.

### Criterion 4: AI Tool Handle Mentions >= 1

At least one total mention of AI tool handles or names across all commit messages and PR
text. Patterns searched include `@claude`, `@copilot`, `@chatgpt`, `@cursor`, `@devin`,
`@codex`, `@aider`, `@cline`, `@windsurf`, `@gemini`, `@roo`, `@augment`, `@cody`,
`@replit`, and tool-specific compound names (e.g., `claude code`, `github copilot`,
`devin-ai`, `cursor ai`, `roo code`, `augment code`, `sourcegraph cody`, `replit agent`,
`continue.dev`, `v0.dev`, `bolt.new`, `bolt.diy`, `lovable.dev`).

The threshold is set at >=1. All four criteria use the same minimum threshold of 1,
providing a consistent and transparent classification rule: any single piece of evidence
from any criterion is sufficient to flag AI involvement.

**Rationale**: Developers using AI tools often reference them in commit messages, especially
when documenting changes or in PR descriptions explaining the approach used.

## Likely AI Agent Determination

The `likely_ai_agent` field reports which AI tool accumulated the highest weighted score
across all evidence types. Weights per occurrence:

- Config file match: 10
- Bot contributor: 5
- Co-Authored-By match: 3
- Handle/name mention: 1

This weighting reflects the relative reliability of each signal type. Config files and
bot accounts are definitive indicators of a specific tool, while mentions are more
ambiguous.

## Output Format

Each server produces a result object:

```json
{
  "id": "owner/repo-name",
  "name": "repo-name",
  "github_url": "https://github.com/owner/repo-name",
  "ai_authored": "yes",
  "ai_authored_reasons": ["co_authored_by", "config_files"],
  "likely_ai_agent": "claude",
  "total_commits_scanned": 347,
  "co_author_count": 42,
  "ai_config_files_found": ["CLAUDE.md", ".claude"],
  "bot_contributors": [],
  "multiline_commit_ratio": 0.72,
  "ai_mention_count": 5,
  "ai_mention_details": {"claude": 4, "copilot": 1},
  "commit_evidence": [...],
  "tool_scores": {"claude": 156, "copilot": 1},
  "error": "",
  "processed_at": "2026-02-10T..."
}
```

A summary JSON is also produced with:
- Total counts and percentages for `ai_authored = yes/no`
- Breakdown by triggered criteria
- Tool distribution among ai_authored=yes servers
- Commit scanning statistics (total commits, repos with >100 commits)
- Co-author and AI mention statistics

## Rate Limiting

- The script uses the `GH_TOKEN` (or `gh auth token`) for 5,000 requests/hour.
- Each repository now requires variable API calls: 1 (tree) + 1 (PRs) + N (commit pages).
  Most repos have <100 commits (1 page); repos with more commits require additional pages.
- The client tracks `X-RateLimit-Remaining` and sleeps automatically when remaining
  requests drop below 50.
- Processing uses async IO with configurable concurrency (default: 5 concurrent requests).
- Batches of 10 servers are processed at a time with 1s pauses between batches.
- Checkpoints are saved every 5 batches for crash recovery.
- Rate limit is re-checked every 10 batches.

## Limitations

1. **Commit history cap at 10,000**: Repositories with more than 10,000 commits will have
   their oldest commits unchecked. In practice, very few MCP servers exceed this threshold.

2. **False positives from tool names**: Words like "aider", "cline", "bolt", and "windsurf"
   can produce false matches in commit messages. All four criteria use a >=1 threshold,
   so a single false match on any criterion will produce a false positive. Criteria 1-3
   are highly specific; criterion 4 (handle mentions) carries the most false-positive risk.

3. **False negatives**: Developers may use AI tools without leaving detectable traces
   (e.g., copying from ChatGPT web interface, using AI without attribution). This approach
   can only detect explicit evidence in the git history and repository structure.

4. **Squashed/force-pushed history**: Squash merges and force pushes lose individual commit
   messages, potentially hiding Co-Authored-By trailers and AI tool mentions.

5. **Config files added retroactively**: A `.cursorrules` file might be added after initial
   development, indicating current AI tool usage but not necessarily that the server was
   originally created by AI.

7. **Multiline commit ratio**: The multiline commit ratio is tracked in output data for
   reference but is not used as a classification criterion, as it produced too many false
   positives (some human developers write detailed multiline commits).

8. **PR body limitations**: The GitHub API returns PR body text but not review comments or
   review bodies. AI mentions in review discussions are not captured.

## References

- "Fingerprinting AI Coding Agents in Open-Source Repositories" (MSR 2026, arXiv 2601.17406)
- GitHub REST API documentation: commits, pulls, git/trees endpoints
