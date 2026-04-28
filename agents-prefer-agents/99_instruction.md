# 99_instruction.md — agents-prefer-agents subproject

**Owner:** Merlin Stein (UK AI Safety Institute)
**Created:** 2026-04-21
**Hard deadline:** 2026-05-01 23:59 AoE (AIWILD @ ICML 2026 workshop)
**Target venue:** [AIWILD — Second Workshop on Agents in the Wild: Safety, Security, and Beyond, ICML 2026](https://agentwild-workshop.github.io/icml2026/) ([OpenReview](https://openreview.net/group?id=ICML.cc/2026/Workshop/AIWILD))
**Target length:** 2-page mini-paper + appendix (AIWILD short-paper track allows up to 4 main pages excl. references/appendix; user constraint is 2 main pages)
**Previous target (missed):** TAIGR @ ICML 2026 (2026-04-24 AoE). Historical log entries below still reference TAIGR as the venue at the time of writing — not retconned.
**Submission style:** Anonymous for review
**Output:** `paper.pdf` built from `paper.tex`, submitted to OpenReview

---

## 0. Meta-rules for the executing agent

These apply throughout. Violating them derails the whole project.

1. **Isolation.** This subproject lives entirely in `/home/ubuntu/mcp-monitoring/agents-prefer-agents/`. Do **not** modify any file outside this directory, including `data/`, `scripts/`. The only *read* access you need outside the directory is:
   - `scripts/data-classification-aicreatedmcp/detect_ai_created.py` (for AI-detection heuristics — copy what you need; do not import)
   - `scripts/data-classification-aicreatedmcp/methodology.md` (reference)
   - `scripts/data-classification-servers/clservers_2_inspect.py` (for AISI proxy pattern, if LLM use is approved)
   Create a local copy of any code you reuse inside `agents-prefer-agents/lib/`, so this subproject can be extracted to its own repo later. **Use the repo-level venv at `/home/ubuntu/mcp-monitoring/.venv`** (already populated via `uv sync` — includes `requests`, `aiohttp`, `anthropic`, `openai`, `inspect_ai`, `aisitools`, `pandas`, `pyarrow`, `matplotlib`, `scipy`, `pypdf`, etc.). Do **not** create a local `agents-prefer-agents/.venv`.
   Feel free to add more parts to `CLAUDE.md`, or `pyproject.toml` and run `uv sync` — stay in the global venv.

2. **Progress file.** Maintain `agents-prefer-agents/99_progress.md` and **update it every time you finish a meaningful step or hit a blocker** (roughly every few minutes during active work). Sections are pre-specified in that file; append under the *Log* section in reverse-chronological order (newest first).

3. **Literature file.** Maintain `agents-prefer-agents/99_literature.md`. Every time you read a paper, blog post, or workshop page that might be cited, **paste raw extracts** (1–3 paragraphs) relevant to our claims, with full URL and access date. This avoids re-reading and gives you source material for the LaTeX references section in Phase 6. Do **not** paraphrase; quote verbatim in the extracts, then add your own one-line gloss at the top of each entry.

3b. **Notes-to-human file.** Maintain `agents-prefer-agents/99_notestohuman.md`. At the **start and end of every phase**, append an entry under the *Changelog* section summarising: (a) what got decided since the last phase, (b) any defaults from the open-questions list that were adopted by silence, (c) any new questions that arose, and (d) one-line status on each of the "Things I'm watching for" triggers. Keep entries short; this file is the user's single-glance view of project state.

4. **Reasoning transparency on uncertainties.** Every time you make a non-trivial judgement call — choosing a threshold, picking a classifier, defining a chain, handling an edge case, dropping data — write it into the **Uncertainties** section of `99_progress.md` with:
   - what you decided
   - the alternatives you considered
   - why you picked this one
   - what would change your mind / sensitivity to this choice
   Do not bury these in code comments. The reader of `99_progress.md` must be able to audit every load-bearing decision without reading code.

5. **Milestone check-ins (safe-path pre-authorization).** You have standing autonomy between milestones. The user has pre-authorized the safe-path defaults below — **do not block** on these unless the anomaly trigger fires. Always append the phase-end summary to `99_notestohuman.md` + `99_progress.md`, then proceed.

   | Milestone | Safe-path (proceed without asking) | Anomaly → **block and ping user** |
   |---|---|---|
   | Phase 1 done (repo list) | repo count ∈ [50, 5 000 000] and selection looks sane | count <50, or selection surfaces something unexpected |
   | Phase 2 done (PRs collected) | AI-PR share <50% and no repo dominates >30% of PRs | AI-PR share ≥50%, single repo dominates, or >10% of target repos failed to fetch |
   | LLM classification at scale | **Always block** before >500 LLM calls (no safe-path) | — |
   | Merger-detection audit (§4.3) | audit is conclusive (step 1 or 2 of §4.3 gives a clean answer) | audit inconclusive → block to confirm falling back to "last approving reviewer" proxy |
   | Headline figure first drawn | direction is interpretable and cell counts meet the N≥20 threshold | null/ambiguous result **and** you're considering changing the spec to get a non-null, or thin cells |
   | Draft ready for review | — | **Always block** — user must read the draft |
   | 24 h before deadline | — | **Always block** regardless of state |

   When the safe-path triggers, write a one-paragraph "auto-proceed" note under `99_notestohuman.md → Changelog` naming the milestone, the values checked, and what you did next. The user reviews that file periodically; they should never discover an auto-proceed by reading code.

6. **Quality bar.** This is a workshop submission intended to meet a NeurIPS-workshop quality standard. Do not write a claim the underlying data cannot support. Every number in the paper must be (a) computable from code in this subproject, (b) reproducible from a fresh run, and (c) cited to a specific figure/table/log. "In the wild" anecdotes are fine for intro framing **if** sourced to a URL in `99_literature.md`; otherwise drop them.

7. **No new abstractions.** Small scripts, one job per file, results dumped to JSON/CSV/Parquet. No frameworks, no custom ORMs, no plugin systems. This project has a hard 4-day deadline; write code accordingly.

8. **Never commit or push without explicit user instruction.** The user will handle git. You may `git status` / `git diff` but not `git add`, `git commit`, `git push`.

9. **Reference paper in the folder.** `agents-prefer-agents/2502.15212v1.pdf` is a prior paper by the user (NeurIPS 2025 SoLaR). It is **below** the quality standard we are aiming for here, but is provided as a **structural** example — look at how it lays out a taxonomy, not its content. **That paper is about agent autonomy; this paper is not.** Do not imitate its empirical claims or measurements. If you read it, paste the taxonomy table into `99_literature.md` as a structural reference only. You may need to install `poppler-utils` (`sudo apt-get install -y poppler-utils`) to render it.

---

## 1. Project goal & framing

### 1.1 One-sentence pitch

Do AI coding agents, embedded in GitHub workflows, show **emergent precursors to gradual disempowerment of humans** — and are those precursors **trending upward** over the past 12 months?

### 1.2 Background & motivation

Laurito et al. 2025 ([arXiv:2502.12743](https://arxiv.org/abs/2502.12743)) showed that LLMs prefer LLM-generated content over human content in single-turn tasks (ads, abstracts, movie plots), while humans do not. If this bias exists in the wild — in real multi-agent systems with consequences — it could seed a self-reinforcing loop in which delegated markets decouple from human preferences. The "gradual disempowerment" literature (Kulveit et al. 2025) names this broad failure mode but there is little empirical work measuring its *precursors* in real systems.

GitHub is a natural first testbed because:
- There are real AI coding agents acting in it (Claude Code, Copilot coding agent, Devin, Cursor background agents, Aider, Cline, etc.)
- They leave machine-readable traces (co-author trailers, bot accounts, config files)
- PR review is a natural multi-agent decision process with measurable outcomes (merged / not merged, time-to-merge)
- A 12-month window (Apr 2025 → Mar 2026) covers the period agent adoption became mainstream

### 1.3 Definitions

- **AI agent** (for this paper): an LLM + tool harness that **directly edits an external environment** — in our case, commits to a git branch, opens/reviews/merges a PR — **without per-action human mediation**. Claude.ai / ChatGPT web chat that produce text the human then pastes do **not** count. Claude Code, GitHub Copilot coding agent, Devin, Cursor background agent, Aider in autonomous mode, Cline agent mode, etc. **do** count.
- **Agent action** on GitHub: a commit, PR open, PR review, PR comment, or PR merge whose *actor* (author / reviewer / merger) is an AI agent per the detection rules in §4.
- **Human action**: an event whose actor is a human account with no AI markers.
- **Ambiguous / AI-assisted**: human account but the event payload carries AI markers (e.g., a commit with `Co-Authored-By: Claude`). Tagged separately from pure AI and pure human; excluded from the primary analysis and included in sensitivity runs.

### 1.4 Research questions

- **RQ1 (Chain length).** What is the distribution of *AI-to-AI interaction chain length* within GitHub PR threads, and has the mean / p95 lengthened week-over-week from Apr 2025 to Mar 2026?
- **RQ2 (Self-preference in review).** Comparing the 2×2 of `{PR author type} × {reviewer type}`, does the merge rate of AI-authored PRs reviewed by AI exceed a simple baseline (same repo, same week, human-authored PR reviewed by AI; or AI-authored PR reviewed by human)? **Diff-in-diff only — not causal.**

### 1.5 Headline figure (must-have)

**Figure 1.** Weekly PR merge rate from 2025-04-01 to 2026-03-31, four lines, x-axis = ISO week, y-axis = % merged within 30 days of PR open:
- Human author × Human reviewer (baseline)
- AI author × AI reviewer
- Human author × AI reviewer
- AI author × Human reviewer

The 2×2 uses the **dominant actor** per role (see §5.2 for tie-breaking). Add 95% CIs per week (Wilson). Mark tool-launch reference dates on the x-axis (e.g., Claude Code GA, Copilot coding agent GA, Devin public) as thin vertical lines — reader context only, no causal claim. Title should make the diff-in-diff framing visible. The paper's one-sentence result lives or dies on this figure.

### 1.6 Secondary figures

- **Figure 2.** Distribution of AI-to-AI chain length by quarter (box/violin), with mean and p95 annotated.
- **Figure 3 (only if time).** Share of PRs with any AI involvement (author *or* reviewer) over time, by tool. Descriptive only.

---

## 2. Taxonomy of precursors (paper §2)

Present this as a short typology (≤½ page) to frame what we measure vs what we defer. Cite the gradual disempowerment paper and note which precursors it covers vs these new ones. Use the attached reference paper only as a **layout example** for how taxonomies are presented in this venue.

| Precursor | Brief definition | Measured here? |
|---|---|---|
| **Chain length** | Length of uninterrupted agent↔agent action sequences without human intervention | **Yes (primary)** |
| **AI preference in selection/review** | Agents merge/approve/choose other agents' output at a higher rate than matched human output | **Yes (primary)** |
| **Self-preference between model families** | Agent A prefers Agent A's output over Agent B's output, beyond quality | Future work |
| **Direct preference for AI entities** | Agents seek out or defer to other agents even when routing through a human is an option | Future work |
| **Behavioral homogeneity** | AI outputs cluster tighter than human outputs in diff content / commit messages / vocab | Future work (descriptive only, if time) |
| **Collusion against a human actor** | Coordinated agent behavior that disadvantages a specific human participant | Future work |
| **Self-empowerment** | Agents add tools, tokens, permissions, or secrets that expand their own write scope | Future work |
| **Overconfidence / automation-bias amplification** | Agents do not ask when a human principal would have wanted to be asked | Future work |
| **Erratic-under-stress behavior** | Agents act erratically when blocked (VendingBench-style) | Future work |

Distinguish from the *situational disempowerment* precursor list (Kulveit et al.) — ours is **ecosystem-level** (many agents, many repos) rather than within-conversation.

---

## 3. Scope of data

### 3.1 Time window
**2025-04-01 → 2026-03-31** (12 months), bucketed weekly by ISO week. Claims phrased as "in the last year". Do not pull data outside this window unless strictly needed for an AI-detection side-check.

### 3.2 Repository universe

Primary filter: **public GitHub repos that gained ≥1000 stars during 2025 (calendar year).**

- **Source:** GitHub REST/GraphQL API. Search API caps at 1000 results per query — use star-bucket splitting: `stars:1000..1500`, `stars:1500..2500`, etc. For the "gained in 2025" refinement, compare `total_stars` now vs `total_stars` at 2025-01-01 by paginating `/repos/{owner}/{repo}/stargazers` with `Accept: application/vnd.github.star+json` and counting pre-2025 stars, **or** use a free third-party mirror (`api.ossinsight.io`, `star-history.com`) if pagination is too costly. **Start with the simpler star-bucket proxy; refine only if the universe is too large.**
- **Expected size:** O(1k–5k) repos.
- **If >5k**, narrow further by:
  1. Primary language ∈ {Python, TypeScript, JavaScript, Go, Rust}, **or**
  2. Top N by npm/PyPI download rank (use `npms.io` / `pypistats.org` if available free).
- **Exclusions:** forks, archived, PR count <10 over the window, mirrors.
- **Non-goal:** representativeness of GitHub overall. State this limitation in the paper.

If the final repo count is outside [50, 5000], stop and check in with the user.

### 3.3 PR universe

Within the selected repos, collect **every PR opened, updated, merged, or closed between 2025-04-01 and 2026-03-31**. Cap per-repo PR count at 2000 to avoid mega-repo dominance; if a repo has more, **subsample uniformly by week** to preserve the time distribution.

Record:
- PR number, repo, title, body
- Actors: author, assignees, reviewers, mergers, commenters
- Events: `opened_at`, `updated_at`, `closed_at`, `merged_at`; list of timestamped review events, review comments, issue comments, commits
- Merge outcome: merged / closed-unmerged / still-open-at-cutoff
- Size proxies: `additions`, `deletions`, `changed_files`, number of commits
- For merge detection: `merged_by.login`, `merged_by.type` (User vs Bot), and the relevant `PullRequestEvent` / timeline entries.

### 3.4 Storage layout (inside `agents-prefer-agents/`)

```
agents-prefer-agents/
├── 99_instruction.md              # this file
├── 99_progress.md                 # live log — updated every few minutes
├── 99_literature.md               # raw extracts from every source read
├── 2502.15212v1.pdf               # reference paper (structural example only)
├── README.md                      # short, public-facing (write last)
├── pyproject.toml                 # local deps (aiohttp, pandas, pyarrow, ...)
├── .gitignore                     # excludes data/, logs/, paper/build/
├── lib/
│   └── ai_detection.py            # ported heuristics from detect_ai_created.py
├── scripts/
│   ├── 01_build_repo_list.py      # → data/repos.json
│   ├── 02_fetch_prs.py            # → data/prs/{owner}__{repo}.jsonl
│   ├── 03_classify_actors.py      # → data/actor_classification.json
│   ├── 04_classify_prs.py         # → data/pr_events.parquet
│   ├── 05_compute_chains.py       # → data/chains.parquet + results/chain_stats.json
│   ├── 06_compute_merge_rates.py  # → data/merge_rates.parquet + results/merge_rate_stats.json
│   ├── 07_make_figures.py         # → paper/figures/*.pdf
│   └── 99_utils.py                # rate-limited GH API client, shared types
├── data/                          # raw + intermediate (gitignored; can be large)
├── results/                       # small JSON/CSV summaries (version-controlled)
├── logs/                          # per-script logs
└── paper/
    ├── paper.tex                  # 2-page main + appendix
    ├── references.bib
    ├── figures/                   # pdf figures built by 07_make_figures.py
    └── template/                  # ICML 2026 style files (download in Phase 0)
```

---

## 4. AI detection methodology

### 4.1 Port from existing work

Port the four binary criteria from `scripts/data-classification-aicreatedmcp/detect_ai_created.py` into `agents-prefer-agents/lib/ai_detection.py`:
1. **Co-Authored-By trailers** matching known AI tools (Claude, Copilot, ChatGPT, Devin, Codex, Aider, Cline, Roo, Augment, Continue.dev, Gemini, Windsurf).
2. **AI config files** in the repo tree (`CLAUDE.md`, `.cursor/`, `.github/copilot-instructions.md`, `AGENTS.md`, etc.).
3. **Bot contributor accounts** (`devin-ai-integration[bot]`, `copilot-swe-agent[bot]`, `claude[bot]`, `github-copilot[bot]`; extend with Cursor, Jules, Replit Agent, etc. if found in data).
4. **Handle mentions** in commit messages / PR text (`@claude`, `@copilot`, `@cursor`, `@devin`, `claude code`, `github copilot`, etc.).

Keep the **pattern lists** as drop-in copies and extend (do not remove). Document every added pattern in `99_progress.md` with where you saw it in the wild.

### 4.2 Per-event classification (new, vs MCP pipeline)

The MCP pipeline classifies a **repo** as AI-authored. We need **per-event** classification. For each PR event (commit, review, comment, merge):

- **Actor type:**
  - `AI-bot` if the acting account is on the bot allowlist
  - `AI-assisted` if the account is human but the event payload (commit message, PR body, review body) contains a co-author trailer or AI handle
  - `human` otherwise
- **Agent family** (when AI): `claude`, `copilot`, `devin`, `cursor`, `aider`, `cline`, `other-ai`, `unknown-ai`
- **Confidence:** `high` if bot account OR co-author trailer; `medium` if config-file-adjacent; `low` if handle-mention-only. Exclude `low` from the primary analysis; include in sensitivity appendix.

Ensure to have at least a set of >10000 repos with multi-agent interactions of chains >3 

### 4.3 The merge-detection question (user-flagged, blocking)

The user explicitly flagged: *how can we tell if an AI did the merge itself?* Resolve this **before** running the headline figure.

Plan of attack, in order (stop at the first that gives a clean answer):
1. **`merged_by` field.** For each merge, inspect `merged_by.login` and `merged_by.type` (User vs Bot). On a 1000-merge sample from repos known to use Copilot coding agent or Claude Code GH app, what fraction of merges are attributed to bot accounts on our allowlist?
2. **Installation/App metadata.** Merges done by GitHub Apps often set `merged_by` to the app's bot account (e.g., `copilot-swe-agent[bot]`). Cross-check with the `actor` field on `PullRequestEvent` (timeline API).
3. **Commit-trailer scan on the merge commit.** Some agents add trailers on the merge commit itself.
4. **If inconclusive:** drop "AI merger" from the primary analysis and substitute **"AI reviewer who last approved"** as the proxy — reviews are well-attributed. Document this degradation in the methods section and the uncertainties log.

Write findings of this audit to `results/merger_detection_audit.md`. Cite in the paper's methods.

### 4.4 Validation

Hand-label a random sample of **100 PRs** (stratified: 25 author-human, 25 author-AI-bot, 25 author-AI-assisted, 25 reviews) and compare to the classifier. Report precision & recall in the appendix. If precision on AI-bot author detection is <0.9, **do not proceed** — investigate. Time budget: 90 minutes. Put hand labels in `data/hand_labels.csv`.

---

## 5. Measurement details

### 5.1 Chain length (RQ1)

**Definition.** Within one PR, order the attributed events by timestamp (PR-open, commits, reviews, review-comments, issue-comments, merge). An **AI→AI chain** is a maximal contiguous subsequence of events all classified `AI-bot` (exclude `AI-assisted` from the primary definition; include in robustness).

- `chain_length` = number of events in the subsequence.
- Per-PR metric: **longest AI→AI chain in the PR.**
- Per-week aggregate: mean and p95 of per-PR longest AI→AI chain, over all PRs that are open or active that week.

**Edge cases:**
- Same actor in consecutive events counts as separate events (chain length ≥ 2 requires ≥ 2 events, possibly same actor).
- Across-PR chains (PR A closed by bot → bot opens PR B) are **out of scope for v1**. Note as future work.
- Dependabot / Renovate / Snyk commits are excluded (they are on the `NON_AI_BOTS` list in the MCP detection code — carry this exclusion over unchanged).

### 5.2 Self-preference diff-in-diff (RQ2, headline)

For each PR assign:
- `author_type ∈ {human, AI}` — PR's author-of-record, with AI winning if *either* the account is a bot *or* any of its commits has an AI trailer.
- `reviewer_type ∈ {human, AI}` — dominant actor on the *approving* review; if no approving review, use the most active reviewer by event count; if no reviews at all, drop from the numerator but keep in a "no-review" sub-table.

Tie-breaking: if mixed (human approved + AI approved), classify as **AI** for the conservative test — biases *against* finding AI→AI self-preference. Sensitivity: rerun with "human wins ties".

**Headline statistic per week w and cell (a, r):**
```
merge_rate[w, a, r] = merged_within_30d[w, a, r] / opened[w, a, r]
```
Wilson 95% CIs. Minimum cell count per week: 20 PRs; otherwise drop that (week, cell).

**Diff-in-diff summary:**
```
did = (merge_rate[AI-author, AI-reviewer] - merge_rate[AI-author, human-reviewer])
    - (merge_rate[human-author, AI-reviewer] - merge_rate[human-author, human-reviewer])
```
Report as a number with a cluster-bootstrap CI (bootstrap by repo, 1000 draws). **Frame explicitly as associational, not causal.** This is the central number of the paper.

### 5.3 Appendix regression

Not in the main text. For robustness:
- PR size (log additions+deletions)
- Repo fixed effect
- Week fixed effect
- PR author tenure (days since their first commit to the repo)
- CI pass signal if available (status on last commit)

Specification:
```
logit(merged_30d) = β0 + β1·AI_author + β2·AI_reviewer + β3·AI_author·AI_reviewer
                 + γ·controls + repo_FE + week_FE + ε
```
β3 is the interaction of interest. Cluster SEs by repo.

---

## 6. Paper structure (2 main pages)

Approximate layout at ICML 2026 style (10pt, two-column):

| Section | Length | Content |
|---|---|---|
| Title + Abstract | ~80 words | Anonymous; one-sentence result with effect size and CI |
| 1. Introduction | ~¾ col | Motivation (Laurito et al., gradual disempowerment), gap, contribution |
| 2. Precursor taxonomy | ~¾ col | Table from §2 + distinction from situational disempowerment |
| 3. Data & methods | ~1.25 col | Repo universe, PR collection, AI detection (port + validation), chain & DiD definitions. Full methods → appendix. |
| 4. Results | ~1.5 col | **Figure 1 (headline)**, Figure 2 (chains). DiD number + CI. Honest null results if that's what we find. |
| 5. Discussion & call to action | ~¾ col | What these precursors do/don't tell us about gradual disempowerment; call for the field to track these. Mitigations flagged for future work. |
| References | ~¼ col | 10–15 refs, trimmed |
| Appendix | unlimited | Full regression table, validation, sensitivity (ties, AI-assisted inclusion, merger-detection audit, robustness) |

### 6.1 Title candidates (finalize at draft time)
- "Precursors to gradual disempowerment: measuring AI-AI interaction on GitHub"
- "Do AI agents prefer AI agents? Evidence from a year of GitHub PR reviews"

### 6.2 Must-cite references (starting set; extend in `99_literature.md`)
- Laurito et al. 2025 ([arXiv:2502.12743](https://arxiv.org/abs/2502.12743)) — AI-AI bias
- Kulveit et al. "Gradual disempowerment" (find canonical ref)
- MSR 2026 "Fingerprinting AI Coding Agents in Open-Source Repositories" ([arXiv:2601.17406](https://arxiv.org/abs/2601.17406))
- Qiu et al. 2025 on AI-search diversity reduction
- VendingBench (if cited in taxonomy)

### 6.3 Anonymity
- No author names, affiliations, GH handles, personal URLs in the PDF.
- "AISI" / "UK AI Safety Institute" must **not** appear.
- Reviewers may see URLs — anonymize any data-dump links. Dataset release is not required with the submission.

---

## 7. Execution phases (timeline target)

~72 hours as of 2026-04-21 morning. Budget:

- **Phase 0** — Setup, format verification, read reference paper for layout — **2 h**
- **Phase 1** — Repo list — **3 h**
- **Phase 2** — PR collection — **12 h** (API-bound; run overnight)
- **Phase 3** — AI detection + actor classification — **4 h**
- **Phase 4** — Merger-detection audit (§4.3) — **2 h**
- **Phase 5** — Chain & merge-rate analysis — **4 h**
- **Phase 6** — Figures + writing — **12 h**
- **Phase 7** — Polish, proofread, anonymize, submit-ready PDF — **6 h**

Slack: ~27 h. Spend it on validation and honest error bars, not scope creep.

### Phase 0 — Setup
1. Create the directory structure from §3.4 (inside `agents-prefer-agents/`). **Do not** create a local `.venv` or a local `pyproject.toml` — use the repo-level venv at `/home/ubuntu/mcp-monitoring/.venv`. Add any extra deps to the root `pyproject.toml` and run `uv sync`.
2. **Bootstrap access** (first thing, every session — run once before any script):
   ```bash
   source /home/ubuntu/mcp-monitoring/.venv/bin/activate
   export GITHUB_TOKEN="$(gh auth token)"   # autonomous scripts read GITHUB_TOKEN directly; env var is not exported by default
   ```
   Verify with `python agents-prefer-agents/99_test_access.py` — the script exits 0 iff all blocking checks pass (GitHub `/rate_limit`, Anthropic via `aisitools.api_key.get_api_key_for_proxy`, `pdflatex`, `pypdf`). Do **not** run any other script until this passes.
3. Download the ICML 2026 LaTeX template (check `icml.cc/Conferences/2026/AuthorInstructions`). Place in `paper/template/`. Compile a "hello world" PDF **before** writing any content — `texlive-latex-extra` + `latexmk` are already installed.
4. Fetch the AIWILD workshop page (https://agentwild-workshop.github.io/icml2026/) and download `assets/icml_aiwild_template.zip`. Confirm anonymous submission, page count (4 pages short-paper track), OpenReview portal `ICML.cc/2026/Workshop/AIWILD`. Paste findings into `99_literature.md`. (Originally pointed at TAIGR — missed; retargeted 2026-04-27.)
5. Read `2502.15212v1.pdf` **only for layout structure** (use `pypdf` — `poppler-utils` is not installed and is unnecessary). Paste the reference paper's taxonomy table into `99_literature.md`.

### Phase 1 — Repo list
1. `01_build_repo_list.py`. Query `GET /search/repositories?q=stars:>=1000+pushed:>=2025-01-01&sort=stars&order=desc&per_page=100`, paginate via star-bucket splitting.
2. For each candidate, determine stars-gained-in-2025 (simpler proxy first; third-party mirror if needed).
3. Filter: ≥1000 stars gained in 2025, not fork, not archived. Keep metadata.
4. Save `data/repos.json` shape `[{owner, repo, total_stars, stars_gained_2025, lang, pushed_at, created_at}]`.
5. Report count in `99_progress.md`. Check in with user if count <50 or >5000.

### Phase 2 — PR collection
1. `02_fetch_prs.py` — for each repo, paginate PRs with `state=all&sort=updated&direction=asc`, filter to PRs whose `updated_at ∩ [2025-04-01, 2026-03-31]`, and pull:
   - PR metadata (`pulls/{n}`)
   - Commits (`pulls/{n}/commits`)
   - Reviews (`pulls/{n}/reviews`)
   - Review comments (`pulls/{n}/comments`)
   - Issue comments (`issues/{n}/comments`)
   - Events timeline (`issues/{n}/timeline`)
2. Write one `.jsonl` per repo to `data/prs/`. Checkpoint state; crashes must not lose progress.
3. Rate-limit aware — port the `GitHubAPIClient` class from `detect_ai_created.py` into `scripts/99_utils.py`. Concurrency 8. Backoff on 403/429.
4. Record totals, by-repo counts, and API request count to `results/phase2_stats.json`. **Check in with user.**

### Phase 3 — Classification
1. Port `lib/ai_detection.py` from the MCP detection code (patterns + detectors only; no MCP-specific logic).
2. `03_classify_actors.py` — build a per-account classification cache over all `login`s seen. Record `{type: bot|user, ai_family?, source: allowlist|fetched-user-type, confidence}` → `data/actor_classification.json`.
3. `04_classify_prs.py` — emit one row per event to `data/pr_events.parquet` with `repo, pr_number, event_idx, event_type, actor_login, actor_type, ai_family, confidence, timestamp, event_size_loc?`. Plus a per-PR summary row (author_type, reviewer_type, merger_type, merged, merged_at, additions, deletions, changed_files).

### Phase 4 — Merger-detection audit
1. Run §4.3 on a 1000-PR sample.
2. Write `results/merger_detection_audit.md`.
3. Decide: use `merged_by` directly vs use AI-approved-last-reviewer proxy. Record decision in `99_progress.md` uncertainties. **Check in with user on the decision before running Phase 5.**

### Phase 5 — Analysis
1. `05_compute_chains.py` — per-PR longest AI→AI chain. Output `data/chains.parquet` + weekly aggregates in `results/chain_stats.json`.
2. `06_compute_merge_rates.py` — weekly 2×2 merge rates with Wilson CIs. Cluster-bootstrap the headline DiD by repo (1000 draws). Output `data/merge_rates.parquet` + `results/merge_rate_stats.json`.
3. Robustness: (a) include AI-assisted with AI, (b) include AI-assisted with human, (c) AI-merged vs AI-approving-reviewer proxy. Save as sensitivity rows.

### Phase 6 — Figures + writing
1. `07_make_figures.py` — matplotlib, 300 dpi, serif font, sized for two-column ICML.
2. Draft `paper/paper.tex` section-by-section. Measure after each section — kill it if it spills past 2 pages.
3. Appendix has no page cap; put every caveat and full regression there.

### Phase 7 — Polish
1. Strip identifiers. Grep the repo for "AISI", "Merlin", "Stein", any GH handle, any non-anonymous URL. Verify the PDF metadata is anonymous.
2. Spell-check, figure captions, reference formatting.
3. Final PDF. Stage a test upload to OpenReview (without submitting) to confirm format compliance.
4. Hand off to user for submission.

---

## 8. Known risks & anticipated uncertainties

Add more as they appear, in `99_progress.md → Uncertainties`.

1. **Merger detection may be unreliable.** Mitigated by §4.3 audit. If unresolved, drop "merger" and use "last approving reviewer".
2. **AI-detection false-positive rate may be high at event level.** Heuristics were tuned on repo-level presence. A single `@copilot` mention in a comment does not mean an AI wrote the comment. **Countermeasure:** primary analysis uses `confidence=high` only (bot account OR co-author trailer); sensitivity uses medium.
3. **Selection bias in repo universe.** Popular repos more likely to have AI agents installed. Scope claim to "high-activity repos"; do **not** generalize.
4. **Thin cells in early weeks.** AI×AI cell may be empty in 2025-04. Report honest CIs; if a cell is empty for a week, drop that cell-week, don't impute.
5. **GH API rate limits.** 5000/hr. 1000 repos × 500 PRs × ~5 endpoints ≈ 2.5M requests. **Mitigations:** conditional requests (ETag), GraphQL for bulk PR lists, start overnight. If infeasible, subsample repos.
6. **Confounding by tool announcement dates.** Copilot coding agent GA, Claude Code launch, Devin public — all within window. Mark on x-axis as reference lines. Do not claim causation.
7. **Workshop-specific format divergence.** RESOLVED for AIWILD: template downloaded from `https://agentwild-workshop.github.io/icml2026/assets/icml_aiwild_template.zip`; only diff vs ICML main style is the workshop-name strings in `\ICML@appearing` and `\Notice@String`. (Original TAIGR concern — TAIGR site JS-rendered, OpenReview 403 — was never resolved before retargeting.)
8. **Page 2 of the reference PDF.** `2502.15212v1.pdf` in this folder is a user-authored paper on a different topic (agent autonomy). It is flagged as **below the quality bar we are aiming for**. Use only for layout structure; do **not** borrow its claims, plots, or taxonomy content.

---

## 9. What to do right now

1. Read this file fully.
2. Verify `99_progress.md` and `99_literature.md` exist in this folder (skeletons provided alongside this file). If not, create them from the templates described at the top of each.
3. Run Phase 0.
4. Begin Phase 1.
5. Update `99_progress.md` after every meaningful step, and `99_literature.md` every time you read a source.
6. **Do not** begin Phase 2 until Phase 1 is complete and the user has been pinged if repo count is out of range.
7. **Do not** start any LLM-based classification without checking in.
8. If anything in this file is unclear or seems wrong given new information, **stop and ask**.

If the data doesn't support the headline claim, the paper's conclusion follows the data, not the other way around. A well-designed null result on precursor trends is still a contribution.
