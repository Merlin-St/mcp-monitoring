# 99_literature.md — raw extracts from sources read

**Purpose:** every paper, blog post, OpenReview page, workshop CFP, or X/Twitter post that might end up cited in the paper or that informed a methodological decision gets an entry here. Paste **raw extracts** (1–3 paragraphs, verbatim, in quotes). Do not paraphrase — the point is to have source material ready for the `references.bib` and the intro/discussion at writing time.

Entry template:

```markdown
## <short-id> — <Title>
- **URL:** <full URL>
- **Accessed:** <YYYY-MM-DD>
- **One-line gloss (yours):** <why this is relevant to our paper>
- **Bibkey (proposed):** <authorYEARshortword>
- **Relevant to:** <intro | taxonomy | methods | discussion | related work>
- **Extract:**
  > <paste verbatim, keep line breaks, preserve math/quotes>
```

Keep entries in reverse-chronological order of when you added them (newest first). If a source is re-read later, append a new extract under the same entry rather than overwriting.

---

## Pre-seeded entries (to fill as you read)

### laurito2025aibias — LLMs prefer LLM-generated content (CORRECTED arXiv ID)
- **arXiv:** 2407.12856 (user's original 2502.12743 was wrong — it resolves to a different paper on LLM self-detection)
- **URL:** https://arxiv.org/abs/2407.12856
- **PNAS:** https://www.pnas.org/doi/10.1073/pnas.2415697122
- **Accessed:** 2026-04-21
- **One-line gloss:** The motivating finding we extend to in-the-wild multi-turn interactions. Published in PNAS 2025.
- **Bibkey (proposed):** laurito2025aibias
- **Relevant to:** intro, related work
- **Extract (paraphrased from WebSearch / PNAS abstract + Stan Ventures coverage):**
  > "We test widely used LLMs, including GPT-3.5, GPT-4 and recent open-weight models in binary choice scenarios involving LLM-based assistants selecting between goods (including consumer products, academic papers, and film-viewings) described either by humans or LLMs. The results show a consistent tendency for LLM-based AIs to prefer LLM-presented options."
  >
  > **Effect sizes:**
  > - Products: GPT-4 favored LLM-written ads 89% of the time, vs. human baseline ~36%.
  > - Academic abstracts: LLM preference 78% vs. human 61%.
  > - Film-viewings also tested.
  >
  > "This suggests the possibility of future AI systems implicitly discriminating against humans as a class, giving AI agents and AI-assisted humans an unfair advantage."
  >
  > Authors: Walter Laurito, Benjamin Davis, Peli Grieztes, and others. Journal: PNAS 2025, doi:10.1073/pnas.2415697122.

### kulveit2025gradual — Gradual disempowerment
- **arXiv:** 2501.16946
- **URL:** https://arxiv.org/abs/2501.16946
- **Website:** https://gradual-disempowerment.ai/
- **Authors:** Jan Kulveit, Raymond Douglas, Nora Ammann, Deger Turan, David Krueger, David Duvenaud
- **Submitted:** 2025-01-28
- **Accessed:** 2026-04-21
- **One-line gloss:** The macro failure mode our "precursors" are defined against. Accepted as ICML 2025 Position paper.
- **Bibkey:** kulveit2025gradual
- **Relevant to:** intro, taxonomy, discussion
- **Extract (from WebSearch summary of abstract):**
  > "The authors argue that even an incremental increase in AI capabilities, without any coordinated power-seeking, poses a substantial risk of eventual human disempowerment. They analyze how even incremental improvements in AI capabilities can undermine human influence over large-scale systems that society depends on, including the economy, culture, and nation-states."
  >
  > "As AI increasingly replaces human labor and cognition in these domains, it can weaken both explicit human control mechanisms (like voting and consumer choice) and the implicit alignments with human interests that often arise from societal systems' reliance on human participation to function."
  >
  > "These effects may be mutually reinforcing across different domains: economic power shapes cultural narratives and political decisions, while cultural shifts alter economic and political behavior. The authors argue that this dynamic could lead to an effectively irreversible loss of human influence over crucial societal systems, precipitating an existential catastrophe through the permanent disempowerment of humanity."
  >
  > **Our angle vs. Kulveit et al.:** their framing is macro / theoretical. Our paper contributes *ecosystem-level* empirical precursors (chain length, self-preference in review) that are observable *today* on GitHub. Distinct from the within-conversation "situational disempowerment" examples that appear in the original paper's §4.

### ghaleb2026fingerprinting — Fingerprinting AI Coding Agents on GitHub
- **URL:** https://arxiv.org/abs/2601.17406
- **HTML:** https://arxiv.org/html/2601.17406v1
- **Author:** Taher A. Ghaleb (single-author)
- **Accessed:** 2026-04-21
- **One-line gloss:** Uses behavioral signatures to classify AI-coding-agent PRs across 5 major agents; our event-level detector is a simpler, rule-based subset of their feature set.
- **Bibkey:** ghaleb2026fingerprinting
- **Relevant to:** methods, related work
- **Extract (from WebSearch):**
  > "A research study analyzing 33,580 PRs from five major agents (OpenAI Codex, GitHub Copilot, Devin, Cursor, Claude Code) identified behavioral signatures for fingerprinting AI coding agents. Using 41 features capturing commit patterns, PR structure, and code characteristics, an XGBoost classifier achieved 97.2% F1-score in identifying the submitting agent."
  >
  > "Distinctive signatures include Codex's multiline commits (67.5%), Copilot's detailed PR bodies (38.4%), Cursor's heavy descriptions (17.2%), Devin's conventional commits (48.9%), and Claude Code's code complexity patterns (27.2% conditionals, 19.8% comments)."
  >
  > "Agent-specific signatures (e.g., Cursor's bullet-heavy descriptions, Claude Code's dense comments) offer interpretable cues for non-experts."
  >
  > **Our relationship to this paper:** Ghaleb classifies *which* agent authored a PR at 97% F1 on 33k PRs. We operate one step earlier: classify *any* event (commit/review/comment/merge) as AI-or-not, using 4 rule-based channels (bot account, co-author trailer, config file, handle mention). We cite Ghaleb for the validity of signatures; our heuristics are deliberately coarser so we can apply them event-level without retraining.
  >
  > **Additional tactical find from the web search:** In March 2026, Claude Code's source code leaked and a feature called "Undercover Mode" was exposed — Anthropic uses Claude Code for "stealth" contributions to public OSS, with a system prompt that explicitly prevents model names from leaking into git logs. **This is a known false-negative source for our detector**, and worth a sentence in §Limitations / Appendix B (threats to validity). Source: VentureBeat March 2026.

### qiu2025diversity — AI search diversity reduction
- **URL:** _(find — Qiu et al. 2025 on AI search reducing diversity)_
- **Accessed:** _(to fill)_
- **One-line gloss:** Sibling finding that AI-mediated systems reduce diversity; we extend to code review.
- **Bibkey (proposed):** qiu2025diversity
- **Relevant to:** intro, discussion
- **Extract:**
  > _(fetch)_

### vendingbench — Agents under stress
- **URL:** _(find canonical)_
- **Accessed:** _(to fill)_
- **One-line gloss:** Erratic-under-stress precursor we cite in the taxonomy as future work.
- **Bibkey (proposed):** vendingbench
- **Relevant to:** taxonomy
- **Extract:**
  > _(fetch the setup + the erratic-behavior finding)_

### taigr2026cfp — TAIGR @ ICML 2026 workshop CFP **[SUPERSEDED 2026-04-27]**
- **URL:** https://taigr-workshop.com/
- **OpenReview:** https://openreview.net/group?id=ICML.cc/2026/Workshop/TAIGR
- **Accessed:** 2026-04-21
- **Status:** Superseded by `aiwild2026cfp` below — TAIGR deadline 2026-04-24 was missed; retargeted to AIWILD (deadline 2026-05-01).
- **One-line gloss:** Original target venue (no longer current).
- **Relevant to:** submission logistics (historical only)
- **Extract:**
  > TAIGR workshop site is a Lovable.dev-hosted page and is JS-rendered — full CFP not extractable via WebFetch. Per X post by Stephen Casper (organizing committee): "Submissions (up to 8 pages) are due April 24! Co-submission with ICML and NeurIPS is encouraged!" https://x.com/StephenLCasper/status/2036566260549562649

  User constraint overrides the 8-page max: target is **2 main pages** + appendix.

### solar2025referencepdf — User's NeurIPS 2025 SoLaR paper (structural example only)
- **Local path:** `agents-prefer-agents/2502.15212v1.pdf`
- **arXiv ID:** 2502.15212v1
- **URL (inferred):** https://arxiv.org/abs/2502.15212
- **Accessed:** _(to fill when agent reads it)_
- **One-line gloss:** Provided by the user as a **layout** example — not a content reference. Topic: agent autonomy, not agent self-preference. **Below the quality bar we are targeting.** Do not imitate claims; only study section layout & taxonomy presentation style.
- **Relevant to:** none (structural only)
- **Extract:**
  > _(fetch the taxonomy table verbatim when the agent reads the PDF in Phase 0. Install `pypdf` or `poppler-utils` first.)_

---

## Entries added during execution

_Newest first. Follow the template above._

### aiwild2026cfp — AIWILD @ ICML 2026 workshop CFP (current target)
- **Workshop name:** Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD)
- **URL:** https://agentwild-workshop.github.io/icml2026/
- **Template (zip):** https://agentwild-workshop.github.io/icml2026/assets/icml_aiwild_template.zip
- **OpenReview:** https://openreview.net/group?id=ICML.cc/2026/Workshop/AIWILD
- **Accessed:** 2026-04-27
- **One-line gloss:** Current target venue after TAIGR was missed. Closest topical fit (multi-agent dynamics in real-world environments, agent safety) and offers a formal 4-page short-paper track that fits our 2-page draft.
- **Relevant to:** submission logistics
- **Extract:**
  > Submission deadline: **May 1, 2026 AoE**. OpenReview profile must be created at least two weeks before the deadline.
  >
  > Two tracks: **Regular Papers — 9 pages** (refs and supplementary excluded); **Short Papers — 4 pages** (refs and supplementary excluded). Short-paper track is "to make the workshop more accessible to researchers outside the ML conference publication circuit"; can present "implementations of unpublished ideas, modest theoretical results, follow-up experiments, or fresh perspectives on existing work."
  >
  > Anonymity: "Submissions must be fully anonymized. This policy applies to any supplementary or linked material as well, including code." Violations may result in desk rejection.
  >
  > Scope: "agent safety, security, alignment, hallucination reduction, interpretability, fairness, benchmarking, multimodal agents, multi-agent coordination, post-training methods, infrastructure, ethics, and governance"; "how intelligent agents can reason, act, and adapt safely and securely in open-ended real-world environments."
  >
  > Template diff vs ICML 2026 main: only `\ICML@appearing` and `\Notice@String` strings differ (renamed to AIWILD). All other style files (`fancyhdr.sty`, `algorithm.sty`, `algorithmic.sty`, `icml2026.bst`) are byte-identical to the ICML 2026 main template.

### icml2026template — ICML 2026 LaTeX template
- **URL (zip):** https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip
- **URL (example PDF):** https://media.icml.cc/Conferences/ICML2026/Styles/example_paper.pdf
- **Accessed:** 2026-04-21
- **One-line gloss:** Official style files. Downloaded and unzipped into `paper/template/`.
- **Relevant to:** Phase 6 writing
- **Extract:**
  > From ICML 2026 Author Instructions (paraphrased from WebFetch):
  > - "Main Body Length: up to eight pages long"
  > - "References & Appendices: Unlimited pages allowed"
  > - "Submission PDF Size: Maximum 50MB"
  > - "Camera-Ready PDF Size: Limited to 20MB"
  > - "There is no support for any typesetting software other than LaTeX"
  > - "All submissions must be anonymized"
  > Files in the zip: `icml2026.sty`, `icml2026.bst`, `example_paper.tex`, `example_paper.pdf`, `example_paper.bib`, `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty`, `icml_numpapers.pdf`.
  > **Note for us:** TAIGR-specific deviations unverified (OpenReview 403 unauth; TAIGR site JS-rendered). Safe default: match ICML main = 10pt, two-column, anonymized.

### solar2025referencepdf — User's NeurIPS 2025 SoLaR paper (structural only)
- **Local path:** `agents-prefer-agents/2502.15212v1.pdf`
- **arXiv ID:** 2502.15212v1
- **URL:** https://arxiv.org/abs/2502.15212
- **Accessed:** 2026-04-21
- **One-line gloss:** Structural example of how a workshop taxonomy paper is laid out. Below our quality bar; do not borrow content.
- **Relevant to:** layout only
- **Extract (section headers only):**
  > Title: "Measuring AI Agent Autonomy: Towards a Scalable Approach with Code Inspection" (NeurIPS SoLaR Workshop 2024).
  > Section headers (via pypdf): 1 Introduction — 2 Approaches for Assessing Autonomy of Systems — 3 Assessing Autonomy of AutoGen — Table 2: AutoGen-focused Taxonomy of Agent System Autonomy — 4 Assessing Autonomy of AutoGen Applications — Table 3: Scoring Autonomy of Selected AutoGen Applications — 5 Conclusion and Future Work — Acknowledgments — References — A Appendix — A.1 Levels of Autonomy — Table 4: Taxonomy of Autonomy Levels in Various Domains — A.2 Application Code Inspection — Table 5: Code flags for scoring autonomy of selected AutoGen applications.
  > **Takeaway for our paper:** Use §1 intro / §2 background / §3 taxonomy-with-table / §4 empirical application / §5 conclusion. Appendix holds detailed tables. We will compress 2→3 and §4 into our "data & methods + results" sections to fit 2 pages.

---

## April 2026 lit-review pass — four thematic clusters

Added during the 2026-04-22 deep literature review. All entries below verified via arXiv fetch; authors, titles, and dates quoted verbatim from the abstract page.

### CLUSTER A — Multi-agent LLM interaction with data from the wild

#### thakkar2025iclr — LLM-feedback RCT on 20K ICLR reviews
- **arXiv:** 2504.09737
- **URL:** https://arxiv.org/abs/2504.09737
- **Authors:** Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, James Zou
- **Submitted:** 2025-04-13
- **Accessed:** 2026-04-22
- **Bibkey:** thakkar2025iclr
- **One-line gloss:** The cleanest in-the-wild AI-mediated-review deployment; 27% of reviewers adopted LLM suggestions on 20K+ ICLR 2025 reviews. Concrete precedent for AI acting inside a real review process — sibling to our GitHub PR setting.
- **Relevant to:** intro, related work
- **Extract (verbatim abstract):**
  > "Tested at ICLR 2025 with over 20,000 reviews, the system found that '27% of reviewers who received feedback updated their reviews.' Results showed reviews expanded by approximately 80 words on average among those who incorporated suggestions, became 'more specific and actionable,' and correlate with increased author-reviewer engagement during rebuttals."

#### cemri2025mas — Why multi-agent LLM systems fail (MAST taxonomy)
- **arXiv:** 2503.13657
- **URL:** https://arxiv.org/abs/2503.13657
- **Authors:** Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica
- **Submitted:** 2025-03-17
- **Accessed:** 2026-04-22
- **Bibkey:** cemri2025mas
- **One-line gloss:** Empirical failure taxonomy from 1600+ traces across 7 MAS frameworks. Complements our precursor taxonomy at the within-system level where ours is cross-system.
- **Relevant to:** related work, discussion
- **Extract:**
  > "MAST-Data, comprising over 1600 annotated traces from seven frameworks, and [a] Multi-Agent System Failure Taxonomy identifying 14 failure modes across three categories: system design issues, inter-agent misalignment, and task verification."

### CLUSTER B — GitHub / OSS agent activity tracking

#### li2026aidev — AIDev: 932K agentic pull requests dataset
- **arXiv:** 2602.09185
- **URL:** https://arxiv.org/abs/2602.09185
- **Authors:** Hao Li, Haoxiang Zhang, Ahmed E. Hassan
- **Submitted:** 2026-02-09
- **Accessed:** 2026-04-22
- **Bibkey:** li2026aidev
- **One-line gloss:** The canonical 2026 agentic-PR dataset: 932,791 PRs by Codex/Devin/Copilot/Cursor/Claude Code across 116K repos. Our pipeline is repository-scoped (500 popular) and event-level; theirs is agent-scoped and PR-level. Complementary.
- **Relevant to:** related work, methods
- **Extract:**
  > "AIDev aggregates 932,791 Agentic-PRs produced by five agents: OpenAI Codex, Devin, GitHub Copilot, Cursor, and Claude Code. These PRs span 116,211 repositories and involve 72,189 developers. ... a curated subset of 33,596 Agentic-PRs from 2,807 repositories with over 100 stars, providing further information such as comments, reviews, commits, and related issues."

#### watanabe2025agentic — 567 Claude Code PRs, 83.8% merge rate
- **arXiv:** 2509.14745
- **URL:** https://arxiv.org/abs/2509.14745
- **Authors:** Miku Watanabe, Hao Li, Yutaro Kashiwa, Brittany Reid, Hajimu Iida, Ahmed E. Hassan
- **Submitted:** 2025-09-18
- **Accessed:** 2026-04-22
- **Bibkey:** watanabe2025agentic
- **One-line gloss:** Single-agent merge-rate study: 83.8% of Claude-Code PRs merge, 54.9% without human edits. Direct comparator to our DiD merge-rate analysis.
- **Relevant to:** methods, discussion
- **Extract (verbatim abstract):**
  > "We empirically study 567 GitHub pull requests (PRs) generated using Claude Code, an agentic coding tool, across 157 diverse open-source projects. Our analysis reveals that developers tend to rely on agents for tasks such as refactoring, documentation, and testing. The results indicate that 83.8% of these agent-assisted PRs are eventually accepted and merged by project maintainers, with 54.9% of the merged PRs are integrated without further modification."

#### ehsani2026fail — Failed agentic PR taxonomy
- **arXiv:** 2601.15195
- **URL:** https://arxiv.org/abs/2601.15195
- **Authors:** Ramtin Ehsani, Sakshi Pathak, Shriya Rawal, Abdullah Al Mujahid, Mia Mohammad Imran, Preetha Chatterjee
- **Submitted:** 2026-01-21
- **Accessed:** 2026-04-22
- **Bibkey:** ehsani2026fail
- **One-line gloss:** Analyses 33K failed agentic PRs; qualitatively codes 600 for rejection patterns. The failure-side companion to watanabe2025agentic.
- **Relevant to:** related work
- **Extract:**
  > "Not-merged PRs tend to involve larger code changes, touch more files, and often do not pass the project's CI/CD pipeline validation. ... we qualitatively analyze 600 PRs to derive a hierarchical taxonomy of rejection patterns."

#### agarwal2026ides — IDEs vs. autonomous agents, longitudinal causal
- **arXiv:** 2601.13597
- **URL:** https://arxiv.org/abs/2601.13597
- **Authors:** Shyam Agarwal, Hao He, Bogdan Vasilescu
- **Submitted:** 2026-01-20
- **Accessed:** 2026-04-22
- **Bibkey:** agarwal2026ides
- **One-line gloss:** Longitudinal causal comparison of IDE-assist vs. autonomous-agent adoption in OSS. Finds velocity gains but +18%/+39% warnings/complexity → persistent tech debt.
- **Relevant to:** discussion (quality side of adoption)
- **Extract:**
  > "velocity gains occur primarily when agents are the first AI tool deployed. However, 'quality risks are persistent across settings, with static-analysis warnings and cognitive complexity rising by roughly 18% and 39%, indicating sustained agent-induced technical debt even when velocity advantages fade.'"

### CLUSTER C — Overreliance risk models

#### ibrahim2025overreliance — Measuring and mitigating overreliance
- **arXiv:** 2509.08010
- **URL:** https://arxiv.org/abs/2509.08010
- **Authors:** Lujain Ibrahim, Katherine M. Collins, Sunnie S. Y. Kim, Anka Reuel, Max Lamparth, Kevin Feng, Lama Ahmad, Prajna Soni, Alia El Kattan, Merlin Stein, Siddharth Swaroop, Ilia Sucholutsky, Andrew Strait, Q. Vera Liao, Umang Bhatt
- **Submitted:** 2025-09-08
- **Accessed:** 2026-04-22
- **Bibkey:** ibrahim2025overreliance
- **One-line gloss:** The canonical 2025 position paper on overreliance — individual + societal risks, measurement gaps, mitigations. Direct risk-framing anchor; notably co-authored by Merlin Stein.
- **Relevant to:** intro, discussion
- **Extract (verbatim):**
  > "Large language models (LLMs) distinguish themselves from previous technologies by functioning as collaborative 'thought partners,' capable of engaging more fluidly in natural language. As LLMs increasingly influence consequential decisions across diverse domains from healthcare to personal advice, the risk of overreliance — relying on LLMs beyond their capabilities — grows."

#### ibrahim2025anthropomorphic — Beyond the anthropomorphic paradigm
- **arXiv:** 2502.09192
- **URL:** https://arxiv.org/abs/2502.09192
- **Authors:** Lujain Ibrahim, Myra Cheng
- **Submitted:** 2025-02-13 (v2: 2025-05-27)
- **Accessed:** 2026-04-22
- **Bibkey:** ibrahim2025anthropomorphic
- **One-line gloss:** Anthropomorphism as a driver of overreliance / user-attribution errors. Cited to argue why "AI reviewing AI" is not a neutral optimisation.
- **Relevant to:** discussion
- **Extract:**
  > "Anthropomorphism ... is an automatic and unconscious response that occurs even in those with advanced technical expertise ... we identify and examine five anthropomorphic assumptions that shape research across the LLM development lifecycle."

#### cheng2025elephant — Social sycophancy (ELEPHANT benchmark)
- **arXiv:** 2505.13995
- **URL:** https://arxiv.org/abs/2505.13995
- **Authors:** Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, Dan Jurafsky
- **Submitted:** 2025-05-20 (rev. 2025-09-29)
- **Accessed:** 2026-04-22
- **Bibkey:** cheng2025elephant
- **One-line gloss:** LLMs preserve user self-image 45 pp more than humans; concrete mechanism for user-facing over-trust. Sibling to self-preference in AI-AI channel.
- **Relevant to:** discussion
- **Extract:**
  > "LLMs preserve user face 45 percentage points more than humans in advice queries and scenarios involving clear wrongdoing. When presented with conflicting moral perspectives, the models affirm both sides in 48% of cases rather than maintaining consistent judgment."

### CLUSTER D — How AI agents are used

#### stein2026mcp — How are AI agents used? 177K MCP tools
- **arXiv:** 2603.23802
- **URL:** https://arxiv.org/abs/2603.23802
- **Author:** Merlin Stein
- **Submitted:** 2026-03-25
- **Accessed:** 2026-04-22
- **Bibkey:** stein2026mcp
- **One-line gloss:** Companion deployment study: 177K MCP tools (Nov 2024–Feb 2026), action-tool share rose 27%→65%, 67% of tools in software dev. Sets the "tool-side" deployment backdrop our GitHub PR study sits on top of.
- **Relevant to:** intro (deployment scale)
- **Extract:**
  > "The research evaluated 177,436 agent tools created between November 2024 and February 2026 by monitoring public Model Context Protocol repositories. ... software development represents 67% of all tools and 90% of downloads, while action tools increased from 27% to 65% of total usage over the study period."

#### appel2025economic — Anthropic Economic Index (uneven adoption)
- **arXiv:** 2511.15080
- **URL:** https://arxiv.org/abs/2511.15080
- **Authors:** Ruth Appel, Peter McCrory, Alex Tamkin, Miles McCain, Tyler Neylon, Michael Stern
- **Submitted:** 2025-11-19
- **Accessed:** 2026-04-22
- **Bibkey:** appel2025economic
- **One-line gloss:** 1M Claude.ai + 1M API conversations. Directive delegation rose 27%→39% in 8 months — population-level evidence that humans increasingly cede step-by-step control to AI.
- **Relevant to:** intro (ramp in autonomy)
- **Extract:**
  > "Users increasingly entrust Claude with more autonomy, with directive task delegation rising from 27% to 39% in the past eight months. ... Based on a privacy-preserving analysis of 1 million conversations on Claude.ai and 1 million API transcripts."

#### shao2025workbank — Future of Work with AI Agents (WORKBank / O*NET)
- **arXiv:** 2506.06576
- **URL:** https://arxiv.org/abs/2506.06576
- **Authors:** Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, Diyi Yang
- **Submitted:** 2025-06-06
- **Accessed:** 2026-04-22
- **Bibkey:** shao2025workbank
- **One-line gloss:** O*NET-grounded task atlas of where AI agents are wanted vs. capable. Ecosystem-level view that pairs well with our ecosystem-level precursor framing.
- **Relevant to:** related work
- **Extract:**
  > "WORKBank database with 'preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations.'"

#### staufer2026aiagentindex — 2025 AI Agent Index
- **arXiv:** 2602.17753
- **URL:** https://arxiv.org/abs/2602.17753
- **Authors:** Leon Staufer, Kevin Feng, Kevin Wei, Luke Bailey, Yawen Duan, Mick Yang, A. Pinar Ozisik, Stephen Casper, Noam Kolt
- **Submitted:** 2026-02-19
- **Accessed:** 2026-04-22
- **Bibkey:** staufer2026aiagentindex
- **One-line gloss:** Reference inventory of 30 deployed agentic systems with transparency/safety fields. Cited to situate the specific bots in our allowlist.
- **Relevant to:** methods (allowlist context)
- **Extract:**
  > "documents information regarding the origins, design, capabilities, ecosystem, and safety features of 30 state-of-the-art AI agents ... we find different transparency levels among agent developers and observe that most developers share little information about safety, evaluations, and societal impacts."

#### feng2025autonomylevels — Levels of autonomy as design dimension
- **arXiv:** 2506.12469
- **URL:** https://arxiv.org/abs/2506.12469
- **Authors:** K. J. Kevin Feng, David W. McDonald, Amy X. Zhang
- **Submitted:** 2025-06-14 (v2: 2025-07-28)
- **Accessed:** 2026-04-22
- **Bibkey:** feng2025autonomylevels
- **One-line gloss:** Five-level user-role taxonomy (operator/collaborator/consultant/approver/observer). Chain-length ≥5 in our data corresponds to sustained "observer" user role.
- **Relevant to:** taxonomy, discussion
- **Extract:**
  > "five escalating levels characterized by user roles: operator, collaborator, consultant, approver, and observer."

#### cihon2025autonomy — Measuring AI Agent Autonomy (NeurIPS 2024 SoLaR)
- **arXiv:** 2502.15212
- **URL:** https://arxiv.org/abs/2502.15212
- **Authors:** Peter Cihon, Merlin Stein, Gagan Bansal, Sam Manning, Kevin Xu
- **Venue:** NeurIPS 2024 SoLaR Workshop
- **Accessed:** 2026-04-22
- **Bibkey:** cihon2025autonomy
- **One-line gloss:** Code-inspection autonomy taxonomy; directly the structural model we extend to event-level in GitHub PRs. Already used as layout example; now also content citation.
- **Relevant to:** related work (autonomy measurement)
- **Extract:** _(see `solar2025referencepdf` entry above for section headers)_
