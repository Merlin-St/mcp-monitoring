"""Tests for scripts/05_compute_chains.py — longest_chain logic."""
from __future__ import annotations

import importlib

import numpy as np

m = importlib.import_module("05_compute_chains")
longest_chain = m.longest_chain
longest_chain_per_group = m.longest_chain_per_group


class TestLongestChain:
    def test_empty(self):
        assert longest_chain([], {"AI-bot"}) == 0

    def test_no_ai(self):
        assert longest_chain(["human", "human", "human"], {"AI-bot"}) == 0

    def test_all_ai(self):
        assert longest_chain(["AI-bot", "AI-bot", "AI-bot"], {"AI-bot"}) == 3

    def test_single_run(self):
        assert longest_chain(
            ["human", "AI-bot", "AI-bot", "AI-bot", "human"], {"AI-bot"}
        ) == 3

    def test_multiple_runs_returns_longest(self):
        assert longest_chain(
            ["AI-bot", "human", "AI-bot", "AI-bot", "AI-bot", "human", "AI-bot"],
            {"AI-bot"},
        ) == 3

    def test_multiple_ai_types(self):
        # Loose set includes AI-powered.
        types = ["AI-bot", "AI-powered", "AI-bot", "human", "AI-bot", "AI-bot"]
        assert longest_chain(types, {"AI-bot", "AI-powered"}) == 3
        assert longest_chain(types, {"AI-bot"}) == 2

    def test_non_ai_bot_breaks_chain(self):
        # non_ai_bot is NOT in ai_set, so it should break a chain.
        types = ["AI-bot", "non_ai_bot", "AI-bot", "AI-bot"]
        assert longest_chain(types, {"AI-bot"}) == 2


class TestLongestChainPerGroup:
    """The vectorised version is what the production pipeline calls.
    It must match the per-group output of the Python-loop version.
    """

    def test_empty(self):
        out = longest_chain_per_group(np.array([], dtype=bool), np.array([], dtype=bool))
        assert out.shape == (0,)

    def test_single_group_no_ai(self):
        is_ai = np.array([False, False, False])
        gc = np.array([True, False, False])
        out = longest_chain_per_group(is_ai, gc)
        assert out.tolist() == [0]

    def test_single_group_all_ai(self):
        is_ai = np.array([True, True, True])
        gc = np.array([True, False, False])
        out = longest_chain_per_group(is_ai, gc)
        assert out.tolist() == [3]

    def test_two_groups(self):
        # group A: [T T F T] -> longest 2
        # group B: [T F T T T] -> longest 3
        is_ai = np.array([True, True, False, True, True, False, True, True, True])
        gc = np.array([True, False, False, False, True, False, False, False, False])
        out = longest_chain_per_group(is_ai, gc)
        assert out.tolist() == [2, 3]

    def test_matches_python_loop_against_random_input(self):
        # Random correctness check vs the reference Python loop.
        rng = np.random.default_rng(42)
        n_groups = 50
        sizes = rng.integers(1, 30, size=n_groups)
        # Build is_ai randomly, group_change at group boundaries.
        is_ai_chunks = [rng.random(s) < 0.5 for s in sizes]
        is_ai = np.concatenate(is_ai_chunks)
        gc = np.zeros(is_ai.shape[0], dtype=bool)
        idx = 0
        for s in sizes:
            gc[idx] = True
            idx += s
        out_vec = longest_chain_per_group(is_ai, gc)
        out_ref = np.array([
            longest_chain([("AI-bot" if x else "human") for x in chunk], {"AI-bot"})
            for chunk in is_ai_chunks
        ])
        assert out_vec.tolist() == out_ref.tolist()
