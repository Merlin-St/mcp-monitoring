"""Tests for lib/ai_detection.py — login + event classification."""
from __future__ import annotations

from lib.ai_detection import (
    classify_event,
    classify_login,
    count_ai_handle_mentions,
    detect_coauthor_ai,
)


# ---------------------------------------------------------------------------
# classify_login
# ---------------------------------------------------------------------------
class TestClassifyLogin:
    def test_empty_login_is_human(self):
        assert classify_login("") == ("human", "")

    def test_known_ai_bot(self):
        # claude[bot] / coderabbitai are canonical AI bots in the allowlist.
        actor, fam = classify_login("claude[bot]")
        assert actor == "AI-bot"
        assert fam == "claude"

    def test_known_ai_bot_case_insensitive(self):
        actor1, _ = classify_login("Coderabbitai")
        actor2, _ = classify_login("CODERABBITAI")
        actor3, _ = classify_login("coderabbitai")
        assert actor1 == actor2 == actor3 == "AI-bot"

    def test_non_ai_bot_dependabot(self):
        actor, fam = classify_login("dependabot[bot]")
        assert actor == "non_ai_bot"

    def test_unknown_bot_account_is_non_ai_bot(self):
        # Any [bot] account not on either list is conservatively non_ai_bot.
        actor, fam = classify_login("some-unknown-bot[bot]")
        assert actor == "non_ai_bot"

    def test_human_login(self):
        actor, fam = classify_login("alice")
        assert actor == "human"


# ---------------------------------------------------------------------------
# detect_coauthor_ai / count_ai_handle_mentions
# ---------------------------------------------------------------------------
class TestCoauthorAndHandle:
    def test_coauthor_trailer_detected(self):
        text = (
            "Some commit message body.\n"
            "\n"
            "Co-Authored-By: Claude <noreply@anthropic.com>\n"
        )
        hits = detect_coauthor_ai(text)
        # Some 'claude'-family hit should be present.
        assert any("claude" in k.lower() for k in hits.keys()) or hits

    def test_handle_mention(self):
        text = "@claude please fix this bug"
        hits = count_ai_handle_mentions(text)
        # Should detect at least one mention.
        assert sum(hits.values()) >= 1

    def test_no_signals(self):
        text = "Plain text with no AI signals at all."
        assert detect_coauthor_ai(text) == {}
        assert count_ai_handle_mentions(text) == {}


# ---------------------------------------------------------------------------
# classify_event — full event-level decision
# ---------------------------------------------------------------------------
class TestClassifyEvent:
    def test_human_with_no_text(self):
        c = classify_event("alice", "")
        assert c.actor_type == "human"
        assert c.confidence == "none"

    def test_ai_bot_account_is_high_confidence(self):
        c = classify_event("claude[bot]", "")
        assert c.actor_type == "AI-bot"
        assert c.confidence == "high"

    def test_non_ai_bot_short_circuits(self):
        c = classify_event("dependabot[bot]", "Bumps lodash from 4.17.20 to 4.17.21")
        assert c.actor_type == "non_ai_bot"

    def test_human_with_coauthor_trailer_is_ai_assisted(self):
        # Human author whose commit carries a Co-Authored-By: Claude trailer.
        c = classify_event(
            "alice",
            "fix bug\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n",
        )
        # Should mark as AI-assisted (high confidence, since trailer is high).
        assert c.actor_type == "AI-assisted"
        assert c.confidence == "high"

    def test_human_with_handle_mention_only_is_low_confidence(self):
        # Mention-only events should NOT promote to high-confidence AI.
        c = classify_event("alice", "@claude please review this")
        # Either AI-assisted with low confidence, or human — both acceptable
        # under the conservative §4.2 rule. The KEY invariant is: if it's AI,
        # confidence must NOT be "high" (because no bot account, no trailer).
        if c.actor_type in ("AI-bot", "AI-assisted"):
            assert c.confidence != "high"
