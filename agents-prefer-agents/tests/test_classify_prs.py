"""Tests for scripts/03_classify_prs.py — classify_role, classify_pr, flatten_events."""
from __future__ import annotations

import importlib

# 03_classify_prs.py starts with a digit so it can't be imported normally.
m = importlib.import_module("03_classify_prs")
classify_role = m.classify_role
classify_pr = m.classify_pr
flatten_events = m.flatten_events


# ---------------------------------------------------------------------------
# Synthetic PR fixtures
# ---------------------------------------------------------------------------
def _empty_pr(**overrides) -> dict:
    base = {
        "repo": "alice/example",
        "number": 1,
        "title": "Fix bug",
        "body": "",
        "author_login": "alice",
        "author_type": "User",
        "created_at": "2025-04-15T10:00:00Z",
        "updated_at": "2025-04-15T10:00:00Z",
        "closed_at": None,
        "merged_at": None,
        "merged": False,
        "merged_by_login": "",
        "merged_by_type": "",
        "additions": 5,
        "deletions": 2,
        "changed_files": 1,
        "commits": [],
        "reviews": [],
        "review_comments": [],
        "issue_comments": [],
        "timeline_events": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# classify_role
# ---------------------------------------------------------------------------
class TestClassifyRole:
    def test_empty_returns_none(self):
        t, fam, conf = classify_role([])
        assert t == "none"
        assert conf == "none"

    def test_only_humans(self):
        t, fam, conf = classify_role([("alice", ""), ("bob", "")])
        assert t == "human"

    def test_only_non_ai_bot_returns_none(self):
        # Non-AI bots are stripped before tallying.
        t, fam, conf = classify_role([("dependabot[bot]", "bump")])
        assert t == "none"

    def test_pure_ai_high_confidence(self):
        t, fam, conf = classify_role([("claude[bot]", "")])
        assert t == "AI"
        assert conf == "high"

    def test_handle_mention_only_does_not_promote_to_ai(self):
        # A human reviewer who writes "@claude fix" must stay human.
        t, fam, conf = classify_role([("alice", "@claude please fix")])
        assert t == "human"


# ---------------------------------------------------------------------------
# classify_pr
# ---------------------------------------------------------------------------
class TestClassifyPR:
    def test_pure_human_pr(self):
        s = classify_pr(_empty_pr())
        assert s["author_type"] == "human"
        assert s["reviewer_type"] == "none"

    def test_ai_bot_author(self):
        pr = _empty_pr(author_login="claude[bot]")
        s = classify_pr(pr)
        assert s["author_type"] == "AI"
        assert s["author_confidence"] == "high"

    def test_human_author_with_coauthor_commit(self):
        # Co-Authored-By trailer in an at-open commit message should mark
        # the PR as AI authored. Authored-at must be within the
        # AT_OPEN_SKEW_SECONDS window of created_at, otherwise the commit is
        # treated as review-cycle and ignored for authorship.
        pr = _empty_pr(
            commits=[
                {
                    "oid": "abc",
                    "message": "fix something\n\nCo-Authored-By: Claude <ai@x>\n",
                    "author_login": "alice",
                    "committer_login": "alice",
                    "authored_at": "2025-04-15T10:00:00Z",
                }
            ],
        )
        s = classify_pr(pr)
        assert s["author_type"] == "AI"
        assert s["author_confidence"] == "high"

    def test_ai_reviewer_high_confidence(self):
        pr = _empty_pr(
            reviews=[
                {
                    "author_login": "coderabbitai",
                    "state": "APPROVED",
                    "body": "looks good",
                    "submitted_at": "2025-04-15T12:00:00Z",
                }
            ]
        )
        s = classify_pr(pr)
        assert s["reviewer_type"] == "AI"


# ---------------------------------------------------------------------------
# flatten_events
# ---------------------------------------------------------------------------
class TestFlattenEvents:
    def test_minimal_pr_emits_open_event(self):
        rows = flatten_events(_empty_pr())
        assert len(rows) == 1
        assert rows[0]["event_type"] == "pr_open"
        assert rows[0]["event_idx"] == 0

    def test_events_are_time_sorted(self):
        pr = _empty_pr(
            commits=[
                {"oid": "a", "message": "later", "author_login": "alice",
                 "committer_login": "alice", "authored_at": "2025-04-16T10:00:00Z"},
                {"oid": "b", "message": "earlier", "author_login": "alice",
                 "committer_login": "alice", "authored_at": "2025-04-15T10:00:00Z"},
            ],
            created_at="2025-04-14T10:00:00Z",
        )
        rows = flatten_events(pr)
        # event_idx increases by timestamp.
        timestamps = [r["timestamp"] for r in rows]
        assert timestamps == sorted(timestamps)
        for i, r in enumerate(rows):
            assert r["event_idx"] == i

    def test_emits_one_row_per_event_type(self):
        pr = _empty_pr(
            commits=[{
                "oid": "a", "message": "hi", "author_login": "alice",
                "committer_login": "alice", "authored_at": "2025-04-15T11:00:00Z",
            }],
            reviews=[{"author_login": "bob", "state": "APPROVED",
                      "body": "lgtm", "submitted_at": "2025-04-15T12:00:00Z"}],
            review_comments=[{"author_login": "bob", "body": "nit",
                              "created_at": "2025-04-15T13:00:00Z"}],
            issue_comments=[{"author_login": "carol", "body": "thanks",
                             "created_at": "2025-04-15T14:00:00Z"}],
            timeline_events=[{"type": "merged", "actor_login": "carol",
                              "created_at": "2025-04-15T15:00:00Z"}],
        )
        rows = flatten_events(pr)
        types = sorted([r["event_type"] for r in rows])
        assert "pr_open" in types
        assert "commit" in types
        assert any(t.startswith("review_") for t in types)
        assert "review_comment" in types
        assert "issue_comment" in types
        assert any(t.startswith("tl_") for t in types)
