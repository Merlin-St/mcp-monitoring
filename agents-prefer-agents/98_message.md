Yes — I searched hard. Here's the honest overlap/gap map for your paper's three contributions:

## (i) Precursor taxonomy for gradual disempowerment
- **Closest prior work:** [Alignment Forum — Gradual Disempowerment: Concrete Research Projects](https://www.alignmentforum.org/posts/GAv4DRGyDHe2orvwB/gradual-disempowerment-concrete-research-projects) proposes economy/culture/politics metrics, but at macro level. Your nine PR-event-level precursors are more operational.
- **Novelty:** Original framing; no published peer-reviewed taxonomy at this granularity.

## (ii) Event-level classification pipeline (AI / AI-assisted / AI-bot)
- **Closest prior work:**
  - [AIDev — Studying AI Coding Agents on GitHub](https://arxiv.org/abs/2602.09185) / [The Rise of AI Teammates in SE 3.0](https://arxiv.org/abs/2507.15003) — 932K Agentic-PRs, 5 agents, but classifies **PRs**, not **events within PRs**.
  - [Fingerprinting AI Coding Agents](https://arxiv.org/abs/2601.17406) — classifies **whole repos** (you cite this).
- **Novelty:** Event-level is the differentiator. Squad's post explicitly calls out the per-event gap.

## (iii-a) Longitudinal AI participation growth
- **Heavily overlapping:**
  - [LogicStar "Agents in the Wild"](https://insights.logicstar.ai/) — ongoing live tracker (~4.7 AI PRs/mo in >500-star repos; test inclusion 31%→52% Jan–Jul 2025).
  - [AIDev](https://arxiv.org/abs/2602.09185), [On the Use of Agentic Coding](https://arxiv.org/abs/2509.14745), [The Rise of AI Teammates](https://arxiv.org/abs/2507.15003) — all report growth.
- **Your differentiator:** the 3.4× multiplier headline is new, but the *fact* of growth is well-established. Worth positioning modestly.

## (iii-b) AI→AI event chain length
- **Closest prior work:** [GitHub Blog — Squad](https://github.blog/ai-and-ml/github-copilot/how-squad-runs-coordinated-ai-agents-inside-your-repository/) reports 34.1% of revision commits co-authored by Claude Code, but no chain-length distribution. [Human-AI Synergy in Agentic Code Review](https://arxiv.org/abs/2603.15911) studies conversation rounds, not agent-author→agent-reviewer chains.
- **Novelty:** Chain-length ≥5 as a precursor metric appears genuinely novel.

## (iii-c) Within-PR AI-vs-human reviewer approval (AI-AI bias in the wild)
- **This is the cleanest novelty.** I specifically looked for it.
- Related but not the same:
  - [Laurito et al. — AI-AI bias](https://arxiv.org/abs/2407.12856) — single-turn choice, not PRs (you cite).
  - [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076) — self-preference in LLM-judge benchmarks, not GitHub.
  - [More Code, Less Reuse](https://arxiv.org/html/2601.21276) — reviewer sentiment toward AI PRs, but human reviewers only.
  - [From Industry Claims to Empirical Reality](https://arxiv.org/abs/2604.03196) — CRA-only vs human-only PRs (19,450 / 3,109), but **different** PRs, not within-PR.
  - [Human-AI Synergy](https://arxiv.org/abs/2603.15911) — 278K conversations across 300 repos; does not split by author×reviewer type.
- **Verdict:** No published paper does the within-PR AI-reviewer × author-type cross-tab. Your null result is a real contribution even given the small n (3,347 PRs).

## Recommendation
Your paper is **not redundant**, but you should pre-emptively position against AIDev, "Rise of AI Teammates," LogicStar, and [From Industry Claims to Empirical Reality] in related work — they'll be the first things reviewers compare against. The within-PR diff-in-diff and event-chain metrics are the places to plant your flag.

Sources:
- [Alignment Forum — Concrete Research Projects](https://www.alignmentforum.org/posts/GAv4DRGyDHe2orvwB/gradual-disempowerment-concrete-research-projects)
- [AIDev: Studying AI Coding Agents on GitHub (arXiv 2602.09185)](https://arxiv.org/abs/2602.09185)
- [The Rise of AI Teammates in SE 3.0 (arXiv 2507.15003)](https://arxiv.org/abs/2507.15003)
- [Fingerprinting AI Coding Agents (arXiv 2601.17406)](https://arxiv.org/abs/2601.17406)
- [Agents in the Wild — LogicStar](https://insights.logicstar.ai/)
- [On the Use of Agentic Coding (arXiv 2509.14745)](https://arxiv.org/abs/2509.14745)
- [Squad — coordinated AI agents (GitHub Blog)](https://github.blog/ai-and-ml/github-copilot/how-squad-runs-coordinated-ai-agents-inside-your-repository/)
- [Human-AI Synergy in Agentic Code Review (arXiv 2603.15911)](https://arxiv.org/abs/2603.15911)
- [Laurito et al. — AI-AI bias (arXiv 2407.12856)](https://arxiv.org/abs/2407.12856)
- [LLM Evaluators Favor Their Own Generations (arXiv 2404.13076)](https://arxiv.org/abs/2404.13076)
- [More Code, Less Reuse (arXiv 2601.21276)](https://arxiv.org/html/2601.21276)
- [From Industry Claims to Empirical Reality (arXiv 2604.03196)](https://arxiv.org/abs/2604.03196)
