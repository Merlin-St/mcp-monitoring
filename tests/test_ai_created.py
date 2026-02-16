"""Tests for detect_ai_created.py — likely_creators_details invariants."""

import sys
from pathlib import Path

import pytest

# Add the script directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data-classification-aicreatedmcp"))
from detect_ai_created import ServerResult, _compute_creators_details


# ---------------------------------------------------------------------------
# Invariant: ai_authored="no" → likely_creators_details == {"human": 100}
# ---------------------------------------------------------------------------

class TestCreatorsDetailsNoAI:
    """Every ai_authored='no' result must have likely_creators_details={'human': 100}."""

    def test_no_evidence(self):
        r = ServerResult(id="t1", name="t1", github_url="", ai_authored="no")
        assert _compute_creators_details(r) == {"human": 100}

    def test_no_evidence_with_commits(self):
        r = ServerResult(id="t2", name="t2", github_url="", ai_authored="no",
                         total_commits_scanned=500)
        assert _compute_creators_details(r) == {"human": 100}

    def test_no_evidence_empty_tool_scores(self):
        r = ServerResult(id="t3", name="t3", github_url="", ai_authored="no",
                         tool_scores={})
        assert _compute_creators_details(r) == {"human": 100}

    def test_no_evidence_despite_config_files_field(self):
        """Even if config files were found, if ai_authored is 'no' we trust the label."""
        r = ServerResult(id="t4", name="t4", github_url="", ai_authored="no",
                         ai_config_files_found=[".cursorrules"])
        assert _compute_creators_details(r) == {"human": 100}


# ---------------------------------------------------------------------------
# Invariant: all results must sum to 100
# ---------------------------------------------------------------------------

class TestCreatorsDetailsSumsTo100:
    """Every result must have percentages summing to exactly 100."""

    @pytest.mark.parametrize("criteria,co_auth,commits,mentions,configs,bots,scores", [
        (["co_authored_by"], 5, 100, 0, [], [], {"claude": 15}),
        (["config_files"], 0, 200, 0, [".cursorrules"], [], {"cursor": 10}),
        (["ai_handle_mentions"], 0, 50, 2, [], [], {"aider": 2}),
        (["bot_contributors"], 0, 30, 0, [], ["devin-ai-integration"], {"devin": 5}),
        (["co_authored_by", "config_files"], 20, 100, 0, ["CLAUDE.md"], [], {"claude": 70, "copilot": 10}),
        (["co_authored_by", "config_files", "bot_contributors", "ai_handle_mentions"],
         25, 30, 10, [".cursor", ".cursorrules"], ["devin-ai-integration"],
         {"cursor": 30, "devin": 25, "claude": 5}),
    ])
    def test_sums_to_100(self, criteria, co_auth, commits, mentions, configs, bots, scores):
        r = ServerResult(
            id="test", name="test", github_url="", ai_authored="yes",
            ai_authored_reasons=criteria,
            co_author_count=co_auth,
            total_commits_scanned=commits,
            ai_mention_count=mentions,
            ai_config_files_found=configs,
            bot_contributors=bots,
            tool_scores=scores,
        )
        details = _compute_creators_details(r)
        assert sum(details.values()) == 100, f"Sum is {sum(details.values())}: {details}"
        assert all(v >= 0 for v in details.values())


# ---------------------------------------------------------------------------
# Behaviour: ai_authored="yes" → human < 100
# ---------------------------------------------------------------------------

class TestCreatorsDetailsWithAI:
    """When ai_authored='yes', at least one AI tool should appear."""

    def test_single_tool(self):
        r = ServerResult(id="t", name="t", github_url="", ai_authored="yes",
                         ai_authored_reasons=["config_files"],
                         ai_config_files_found=["CLAUDE.md"],
                         total_commits_scanned=100,
                         tool_scores={"claude": 10})
        details = _compute_creators_details(r)
        assert details["human"] < 100
        assert "claude" in details
        assert details["claude"] > 0

    def test_multiple_tools(self):
        r = ServerResult(id="t", name="t", github_url="", ai_authored="yes",
                         ai_authored_reasons=["co_authored_by", "config_files"],
                         co_author_count=30, total_commits_scanned=100,
                         ai_config_files_found=["CLAUDE.md", ".cursorrules"],
                         tool_scores={"claude": 60, "cursor": 20})
        details = _compute_creators_details(r)
        assert "claude" in details
        assert "cursor" in details
        assert details["claude"] > details["cursor"]

    def test_weak_mentions_capped(self):
        """Just 2 handle mentions should give a modest AI share."""
        r = ServerResult(id="t", name="t", github_url="", ai_authored="yes",
                         ai_authored_reasons=["ai_handle_mentions"],
                         ai_mention_count=2, total_commits_scanned=50,
                         tool_scores={"cursor": 2})
        details = _compute_creators_details(r)
        assert details["human"] >= 85, f"Expected human >= 85, got {details}"

    def test_heavy_ai_low_human(self):
        """All 4 criteria with many co-authored commits should give high AI share."""
        r = ServerResult(id="t", name="t", github_url="", ai_authored="yes",
                         ai_authored_reasons=["co_authored_by", "config_files",
                                              "bot_contributors", "ai_handle_mentions"],
                         co_author_count=80, total_commits_scanned=100,
                         ai_config_files_found=[".cursor"], bot_contributors=["devin-ai-integration"],
                         ai_mention_count=15,
                         tool_scores={"cursor": 50, "devin": 30})
        details = _compute_creators_details(r)
        assert details.get("human", 0) <= 25, f"Expected human <= 25 for heavy AI, got {details}"

    def test_fully_bot_authored(self):
        """A fully bot-authored repo can have 0% human."""
        r = ServerResult(id="t", name="t", github_url="", ai_authored="yes",
                         ai_authored_reasons=["co_authored_by", "config_files",
                                              "bot_contributors", "ai_handle_mentions"],
                         co_author_count=100, total_commits_scanned=100,
                         ai_config_files_found=["AGENTS.md"],
                         bot_contributors=["devin-ai-integration"],
                         ai_mention_count=50,
                         tool_scores={"devin": 100})
        details = _compute_creators_details(r)
        assert "human" not in details or details["human"] == 0, f"Expected no human for fully bot repo, got {details}"
        assert sum(details.values()) == 100
