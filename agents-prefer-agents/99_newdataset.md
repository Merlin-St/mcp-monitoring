# 99_newdataset.md — Paper updates for v2 (criticality-defined) sample

## STATUS: APPLIED on 2026-04-28
## ADDENDUM STATUS: APPLIED on 2026-04-28 (RQ3 main stat → §8.7 clean DiD; see bottom of file for tickoff)

All edits in this doc have been written to the source `.tex` files in this repo. `paper.filled.pdf` was rebuilt against `results/phase1_stats.json` (snapshot 2025.07.25, 1,219 PR-active repos, median stars 5,256) and `results/within_pr_stats.json` (DiD = −13.47 pp on n=8,076 stable doubly-engaged PRs). Future runs of `python scripts/06c_within_pr.py && python scripts/08_fill_paper.py && python scripts/11_build_pdf.py` will continue to auto-fill the placeholders from `results/*.json` — so re-running the data pipeline updates the paper text automatically. **Do not re-apply this doc** — keep it as a record of the v1→v2 framing transition.

## Why this doc exists

The paper currently has two sample-selection regimes living in different files:

- **v2 (current pipeline, what the data actually is):** sample is the **1,219 PR-active repositories** drawn from a top-by-criticality enrichment of the **OpenSSF `criticality_score`** (default_score, snapshot **2025.07.25**). This is the analytic universe for every chain-length, participation, and AI-AI-bias statistic in the paper. Implemented in `scripts/01a_download_criticality.py` + `scripts/01_build_repo_list.py` + `scripts/run_phase2_v2.sh`. `paper/appendix/repo_selection.tex` is already written against this pipeline and documents the full universe-vs-active funnel; **no body text should mention the 2,000-repo universe — only the 1,219 active sample.**
- **v1 (legacy, what the paper body still says):** sample built from a star-bucket sweep — "≥1,000 stars in 2025", capped at 3,000, median stars 53,583, "young-popular OR old-active-popular". This framing leaks into `paper.tex`, `paper/appendix/methods.tex`, and `paper/appendix/threats.tex`. **It must be rewritten to v2.**

Stats to reference (from `results/phase1_stats.json`):

| Metric | v1 wording (what's in the paper now) | v2 value (what it should be) |
|---|---|---|
| Sample size in body / abstract | 3,000 | **1,219** (`\PLACEHOLDERPRACTIVE`) |
| Selection method | "≥1,000 stars in 2025" star-bucket | **OpenSSF criticality_score** (snapshot 2025.07.25) |
| Median stars | 53,583 | **5,256** (`\PLACEHOLDERSTARMED`) |
| Min stars | 18,755 | **25** (`\PLACEHOLDERSTARMIN`) |
| Max stars | n/a | **443,674** (`\PLACEHOLDERSTARMAX`) |
| Score range | n/a | **0.5875–0.8487** (`\PLACEHOLDERSCOREMIN`–`\PLACEHOLDERSCOREMAX`) |

`08_fill_paper.py` already populates every placeholder used below — no script changes needed. **Just edit the .tex files.** Then run `python scripts/08_fill_paper.py` and `python scripts/11_build_pdf.py`.

---

## Naming convention to use

Throughout the body and abstract, the sample is **"critical OS software repositories"**. Not "popular", not "high-activity", not "OpenSSF-critical", not "critical infrastructure". Just **critical OS software repositories**.

When citing the source mechanism (only in methods/limitations/appendix, not the abstract): "selected by the OpenSSF `criticality_score`" with `\citep{ossf-criticality, pike2020criticality}`.

**Do not introduce the 2,000-repo universe figure into the body, abstract, captions, methods, or limitations.** The full universe-vs-active funnel is already documented in `paper/appendix/repo_selection.tex`; the body simply reports the 1,219 active sample.

---

## File-by-file changes

### 1. `paper/paper.tex` ✅ APPLIED

Note: line numbers below refer to the current state of `paper.tex` as of 2026-04-28. Confirm by reading the file before editing — the rest of the paper has been actively edited.

#### 1a. Line 55 — Abstract: "from N popular repositories" ✅

**Find:**
```
We review \PLACEHOLDERNPRS{} GitHub pull requests from \PLACEHOLDERNREPOS{} popular repositories (April~2025--March~2026).
```
**Replace with:**
```
We review \PLACEHOLDERNPRS{} GitHub pull requests from \PLACEHOLDERPRACTIVE{} critical OS software repositories (April~2025--March~2026).
```

#### 1b. Line 122 — §"Data and methods" first sentence ✅

**Find:**
```
We analyse all \PLACEHOLDERNPRS{} PRs \emph{created} between 2025-04-01 and 2026-03-31 in \PLACEHOLDERNREPOS{} high-activity public GitHub repositories, fetched via GitHub GraphQL.
```
**Replace with:**
```
We analyse all \PLACEHOLDERNPRS{} PRs \emph{created} between 2025-04-01 and 2026-03-31 in \PLACEHOLDERPRACTIVE{} critical OS software repositories selected by OpenSSF \texttt{criticality\_score} (snapshot \PLACEHOLDERSNAPSHOT{}; see Appendix~D), fetched via GitHub GraphQL.
```

#### 1c. Line 124 — Limitations (i) "Selection" ✅

**Find:**
```
(i)~\emph{Selection.} Our \PLACEHOLDERNREPOS{} repositories are the top of a high-activity slice of GitHub (median stars 53{,}583, see Appendix~D). We do not claim GitHub-wide representativeness.
```
**Replace with:**
```
(i)~\emph{Selection.} Our \PLACEHOLDERPRACTIVE{} repositories are critical OS software projects ranked by OpenSSF \texttt{criticality\_score}~\citep{ossf-criticality,pike2020criticality} (median stars \PLACEHOLDERSTARMED{}, see Appendix~D). The criticality score favours load-bearing, multi-org, heavily-cross-referenced infrastructure rather than viral-popularity projects. We do not claim GitHub-wide representativeness.
```

#### 1d. Line 137 — Figure 1 caption ✅

**Find:**
```
\caption{Monthly AI-agent participation in GitHub PRs across \PLACEHOLDERNREPOS{} popular repositories (Apr 2025--Mar 2026).
```
**Replace with:**
```
\caption{Monthly AI-agent participation in GitHub PRs across \PLACEHOLDERPRACTIVE{} critical OS software repositories (Apr 2025--Mar 2026).
```

#### 1e. Line 141 — RQ1 paragraph last sentence ✅

**Find:**
```
A single high-activity slice of GitHub cannot establish that AI-agent activity is accelerating ecosystem-wide, but within this slice the participation accelerator is clearly measurable and moving.
```
**Replace with:**
```
A single criticality-defined slice of GitHub cannot establish that AI-agent activity is accelerating ecosystem-wide, but within this slice the participation accelerator is clearly measurable and moving.
```

---

### 2. `paper/appendix/methods.tex` ✅ APPLIED

#### 2a. Line 3 — "Repository universe" paragraph ✅

**Find:**
```
\paragraph{Repository universe.} Public GitHub repositories plausibly gaining $\geq$1{,}000 stars in 2025 and actively pushed within our analysis window, capped at the top \PLACEHOLDERNREPOS{} by a recency-weighted activity score. Forks and archives are excluded. Non-goal: GitHub-wide representativeness. Full selection procedure in Appendix~D.
```
**Replace with:**
```
\paragraph{Repository universe.} \PLACEHOLDERPRACTIVE{} critical OS software repositories selected by OpenSSF \texttt{criticality\_score}~\citep{ossf-criticality,pike2020criticality} (default\_score, snapshot \PLACEHOLDERSNAPSHOT{}), enriched via the GitHub REST API and filtered to drop forks, archived, disabled, and repos not pushed within the analysis window. Non-goal: GitHub-wide representativeness. Full selection procedure in Appendix~D.
```

---

### 3. `paper/appendix/threats.tex` ✅ APPLIED

#### 3a. Line 6 — Selection-bias paragraph ✅

**Find:**
```
We restrict to \PLACEHOLDERNREPOS{} repositories passing the ``young-popular OR old-active-popular'' filter (Appendix~D). The resulting sample has median stars 53{,}583 and minimum stars 18{,}755 --- it is effectively the top most-active highly-popular repositories on GitHub, not a uniform sample. Effects we estimate apply to ``high-activity open-source repositories''. We do not generalise to GitHub as a whole. In particular, chain-length growth in mid-popularity repositories (100--10{,}000 stars) may be smaller or larger than what we report, and we do not have data to say which.
```
**Replace with:**
```
We restrict to \PLACEHOLDERPRACTIVE{} critical OS software repositories ranked by OpenSSF \texttt{criticality\_score} (Appendix~D). The resulting sample has median stars \PLACEHOLDERSTARMED{} (range \PLACEHOLDERSTARMIN{}--\PLACEHOLDERSTARMAX{}) and a criticality-score range of \PLACEHOLDERSCOREMIN{}--\PLACEHOLDERSCOREMAX{} --- it skews toward load-bearing, multi-org, heavily-cross-referenced open-source infrastructure (language toolchains, runtimes, foundation projects) rather than uniform popularity. Effects we estimate apply to ``critical OS software repositories''. We do not generalise to GitHub as a whole. In particular, chain-length growth in lower-criticality repositories may be smaller or larger than what we report, and we do not have data to say which.
```

---

### 4. `paper/appendix/repo_selection.tex` — already v2; verify only ✅ verified

This file was already rewritten for v2 and uses the correct placeholders. **No changes needed.** It legitimately documents the universe-vs-active funnel (which is where the 2,000-vs-1,219 distinction lives); body/abstract/limitations should not duplicate this. After running `08_fill_paper.py`, spot-check `repo_selection.filled.tex`: PR-active 1,219; median stars 5,256.

---

### 5. Things NOT to touch

- `paper/appendix/old_repo_selection.tex` — v1 record kept on purpose; do not edit.
- `scripts/old_01_build_repo_list.py` — v1 selection, kept for replication only.
- `data/old_phase1/` — archived v1 artefacts.
- The five accelerators / theoretical-framework / abstract framing — unrelated to the dataset switch and edited in recent rounds; leave them.

---

## After editing

```bash
source ~/mcp-monitoring/.venv/bin/activate
python scripts/08_fill_paper.py
python scripts/11_build_pdf.py
```

Then sanity-check:

1. `grep -in "popular\|high-activity\|young-popular\|53{,}583\|18{,}755\|3{,}000" paper/paper.tex paper/appendix/methods.tex paper/appendix/threats.tex` should return **no results** (the v1-vs-v2 comparison paragraph in `repo_selection.tex` is the only legitimate remaining "popular" mention).
2. `grep -in "critical OS software\|criticality\|PRACTIVE\|OpenSSF" paper/paper.tex paper/appendix/methods.tex paper/appendix/threats.tex` should show the new framing in all three files.
3. Open `paper/paper.filled.pdf` and confirm:
   - Abstract reads "from 1,219 critical OS software repositories"
   - Limitations (i) reads "Our 1,219 repositories are critical OS software projects … median stars 5,256"
   - Figure 1 caption reads "across 1,219 critical OS software repositories"
   - **No mention of the 2,000-repo universe** anywhere in the body, abstract, captions, methods, or limitations.

---

# ADDENDUM (2026-04-28): Replace RQ3 main stat with the §8.7 clean DiD

## Status
**Not yet applied.** This addendum is a brief for the next agent. The dataset migration above is unchanged. This addendum is a *separate* task: rewire RQ3 in the paper to use the new clean DiD from `99_causalvalidity.md` §8.7.

## What's changing and why

The current RQ3 (Option A in `99_causalvalidity.md` §1) conditions on `AI bot APPROVED` and compares human co-approval rates by author. It has six confounders (collider on AI-approval, quality-selection, anchoring, AI-reviewer concentration, authorship-error, repo/time) and is statistically dead at $n_{\text{AI-auth}}=8$ on the strict cohort.

The new main stat is **`99_causalvalidity.md` §8.7**: a difference-in-differences on the §8.5(b) **stable doubly-engaged** cohort. In words:

$$
\widehat{\rbt^A - \rbt^H} = \underbrace{[P(\text{AI app}\mid\text{AI-auth}) - P(\text{H app}\mid\text{AI-auth})]}_{\text{AI prefers AI more than humans do}} - \underbrace{[P(\text{AI app}\mid\text{H-auth}) - P(\text{H app}\mid\text{H-auth})]}_{\text{AI prefers humans less than humans do}}
$$

with the AI-side approval pool expanded from 415 native explicit votes to ~30k PRs via the regex parser in `paper/test_ai_verdict_regex.py`, restricted to PRs where no commits were pushed between the AI's first verdict and the human's first explicit review.

**Read this before starting:** `99_causalvalidity.md` §§7–8 contain the full design, regex catalogue, hand-audit results, and the verified v2 numbers. **The rest of this addendum tells you what files to change to wire it into the paper.**

## v2 numbers to be reproduced by the new pipeline

From `paper/test_did_v2.py` (run on v2 = 1,219 critical OS software repos, 487,446 PRs):

| Quantity | Value |
|---|---:|
| AI opinion events (native + parsed) | 44,093 |
| Unique PRs with $\geq 1$ AI opinion | 30,560 |
| Doubly-engaged cohort (any state, all) | 15,712 |
| **Stable doubly-engaged cohort (no commits between AI verdict and human review)** | **8,076** |
|   AI-author / H-author | 782 / 7,167 |
| AI bot approval rate × AI-author | 16.50% (129/782) |
| AI bot approval rate × H-author | 31.67% (2,270/7,167) |
| Human approval rate × AI-author | 90.66% (709/782) |
| Human approval rate × H-author | 92.37% (6,620/7,167) |
| Bracket 1: $P(\text{AI app}\mid\text{AI-auth}) - P(\text{H app}\mid\text{AI-auth})$ | **−74.17 pp** |
| Bracket 2: $P(\text{AI app}\mid\text{H-auth}) - P(\text{H app}\mid\text{H-auth})$ | **−60.69 pp** |
| **DiD = Bracket 1 − Bracket 2 = $\widehat{\rbt^A - \rbt^H}$** | **−13.47 pp** |
| Logit interaction $\delta$ (HC0 SEs) | $-0.6329$, $SE = 0.164$, $p \approx 1.2 \times 10^{-4}$ |
| % AI-bot reviewed PRs without any human explicit review (motivation, not headline) | 43.7% (20,155 of 46,092) |

**Direction is anti-self-preference:** AI bots are *less* enthusiastic about AI-authored code than humans are, by a margin substantially larger than their analogous shortfall on human-authored code. This reverses the lab-bench AI-AI bias result of Laurito et al. 2025 in the wild on real PR review.

## Files to change (in order)

### A. Pipeline scripts ✅ APPLIED

#### A1. New: `scripts/ai_verdict_parser.py` ✅
Promote `paper/test_ai_verdict_regex.py` into a reusable pipeline module. Same `BOT_PATTERNS` and `classify()` function (verbatim — do not modify the regex set without re-auditing). Strip the `__main__` argparse harness; expose `classify(login, body)` and `parse_ai_opinions(prs_jsonl_dir, repos: set[str]) -> pandas.DataFrame` (columns: `repo, number, t, kind ∈ {approve_native, reject_native, approve_parsed, reject_parsed}, bot`).

`paper/test_ai_verdict_regex.py` and `paper/test_did_v2.py` should be left in place as standalone test artefacts (they're documented in `99_causalvalidity.md` §7–8) but updated to import from `scripts/ai_verdict_parser.py` rather than carry their own copy.

#### A2. New or replace: `scripts/06c_within_pr.py` ✅
Currently emits `results/within_pr_stats.json` for the legacy 282-PR cohort. Replace its contents (or rename old to `06c_old_within_pr.py` and create `06c_within_pr.py` fresh) so it computes:

1. The §8.7 **stable doubly-engaged DiD**:
   - Build AI-opinion table via `parse_ai_opinions()` over `data/prs/*.jsonl` filtered to v2 repos.
   - Compute the four cells (reviewer × author) on the stable cohort (PRs where no commit is pushed strictly between `min(t_AI_verdict, t_human_review)` and `max(...)`).
   - Output: `did.cells`, `did.bracket1_pp`, `did.bracket2_pp`, `did.did_pp`, `did.logit_delta`, `did.logit_se`, `did.logit_p`, `did.n`, `did.n_AI_auth`, `did.n_H_auth`.
   - Use HC0 SEs via `statsmodels.api.Logit(...).fit(cov_type='HC0')` on the long-format `(reviewer, author, approved)` design, with `interaction = author_AI × reviewer_AI`.
2. The §8.4 **cross-stratum (anchoring) DiD** as a robustness check (see `99_causalvalidity.md` §8.4):
   - Stratum A: no AI-bot review event AT ALL + ≥1 human explicit review + no commits between PR open (`pr_summary.created_at`) and first human explicit review.
   - Stratum B: AI bot's first comment/review event preceded any human engagement + ≥1 human explicit review.
   - Output: `anchoring.A_n`, `anchoring.B_n`, `anchoring.A_gap_pp`, `anchoring.B_gap_pp`, `anchoring.cross_stratum_did_pp` (= B_gap − A_gap).
3. **Context stat** for the abstract / methods motivation:
   - `ai_only.ai_reviewed_n`, `ai_only.no_human_explicit_n`, `ai_only.no_human_explicit_pct`.

Output JSON path: keep `results/within_pr_stats.json` but reorganise its top-level keys to `{ "did": {...}, "anchoring": {...}, "ai_only": {...}, "ai_opinion_pool": {n_native_app, n_native_rej, n_parsed_app, n_parsed_rej, n_unique_prs} }`. Update `08_fill_paper.py` accordingly (see §A4).

`paper/test_did_v2.py` is the spec — port its three blocks (build AI opinions; doubly-explicit-plus DiD with stable filter; cross-stratum anchoring DiD; AI-only review %) into `06c_within_pr.py` with the same numerical outputs.

#### A3. Update: `scripts/07_make_figures.py` `figure2_withinpr()` ✅
Replace the current 2-bar within-PR plot. New design: **2-panel figure** on the stable doubly-engaged cohort.

- **Left panel (4 bars).** Reviewer × author cells: `AI×AI`, `AI×H`, `Human×AI`, `Human×H`. Wilson 95% CIs. Colour: AI-reviewer bars dark grey/red; Human-reviewer bars blue. Annotate cell rates inside the bars.
- **Right panel (2 bars or a single annotated arrow).** The two brackets:
  - `P(AI|AI-auth) − P(H|AI-auth) = -74.17 pp` labelled "AI prefers AI more than humans do" (the sign tells you AI does the *opposite*).
  - `P(AI|H-auth) − P(H|H-auth) = -60.69 pp` labelled "AI prefers humans less than humans do".
  - Difference annotated: $\widehat{\rbt^A - \rbt^H} = -13.47$ pp ($\delta = -0.633$, $p < 10^{-3}$).

Output paths unchanged: `paper/figures/figure2_withinpr.pdf`. Keep the existing `figure2_chain_length.pdf` untouched (it's used elsewhere).

#### A4. Update: `scripts/08_fill_paper.py` ✅
Replace the `\PLACEHOLDERWITHIN*` block (currently lines ~296–360, reading from the old `conditional_on_ai_approval` schema) with new placeholders matching the new `within_pr_stats.json` schema. Add at minimum:

| Placeholder | Source field | Format | Example value |
|---|---|---|---|
| `\PLACEHOLDERDIDN` | `did.n` | int with thousands separator | `8,076` |
| `\PLACEHOLDERDIDNAI` | `did.n_AI_auth` | int | `782` |
| `\PLACEHOLDERDIDNH` | `did.n_H_auth` | int | `7,167` |
| `\PLACEHOLDERDIDB1` | `did.bracket1_pp` | `%+.2f`~pp | `-74.17` |
| `\PLACEHOLDERDIDB2` | `did.bracket2_pp` | `%+.2f`~pp | `-60.69` |
| `\PLACEHOLDERDIDPP` | `did.did_pp` | `%+.2f`~pp | `-13.47` |
| `\PLACEHOLDERDIDDELTA` | `did.logit_delta` | `%+.3f` | `-0.633` |
| `\PLACEHOLDERDIDSE` | `did.logit_se` | `%.3f` | `0.164` |
| `\PLACEHOLDERDIDPVAL` | `did.logit_p` | scientific notation | `1.2 \times 10^{-4}` |
| `\PLACEHOLDERAIAPPAIRATE` | `did.cells['AI_x_AI'].rate` | `%.2f\%` | `16.50` |
| `\PLACEHOLDERAIAPPHRATE` | `did.cells['AI_x_H'].rate` | `%.2f\%` | `31.67` |
| `\PLACEHOLDERHAPPAIRATE` | `did.cells['H_x_AI'].rate` | `%.2f\%` | `90.66` |
| `\PLACEHOLDERHAPPHRATE` | `did.cells['H_x_H'].rate` | `%.2f\%` | `92.37` |
| `\PLACEHOLDERAIONLYPCT` | `ai_only.no_human_explicit_pct` | `%.1f\%` | `43.7` |
| `\PLACEHOLDERAIONLYN` | `ai_only.no_human_explicit_n` | int | `20,155` |
| `\PLACEHOLDERAIOPPRS` | `ai_opinion_pool.n_unique_prs` | int | `30,560` |
| `\PLACEHOLDERANCHORADID` | `anchoring.cross_stratum_did_pp` | `%+.2f`~pp | `+0.75` |
| `\PLACEHOLDERANCHORAN` | `anchoring.A_n` | int | `199,354` |
| `\PLACEHOLDERANCHORBN` | `anchoring.B_n` | int | `5,940` |

Keep the old `\PLACEHOLDERWITHIN*` placeholders defined and pointing to `--` so any remaining body text still compiles during transition; remove them in a second pass after confirming the abstract no longer references them.

### B. Paper text ✅ APPLIED

#### B1. `paper/paper.tex` — abstract third finding (currently around line 55) ✅

**Find:**
```
Third, there is no strong evidence for AI-AI bias: of \PLACEHOLDERWITHINN{} PRs receiving an AI-bot \emph{approving} review (humans need not have reviewed), a human reviewer also approved \PLACEHOLDERWITHINAIHR{} of \PLACEHOLDERWITHINAIN{} AI-authored vs.\ \PLACEHOLDERWITHINHUHR{} of \PLACEHOLDERWITHINHUN{} human-authored (\PLACEHOLDERWITHINP{}). This is very preliminary work, not yet adjusted for code quality.
```

**Replace with:**
```
Third, on the \PLACEHOLDERDIDN{} PRs reviewed by both AI bots and humans (with no commits between the AI verdict and the human review), AI bots are less enthusiastic about AI-authored code than humans are by a wider margin than they are on human-authored code: $\widehat{\rbt^A - \rbt^H} = \PLACEHOLDERDIDPP{}$~pp ($\delta=\PLACEHOLDERDIDDELTA{}$, $p<10^{-3}$). The direction is opposite of self-preference.
```

#### B2. `paper/paper.tex` — RQ3 results paragraph (currently around line 141) ✅

**Find:** the entire `\paragraph{RQ3 (accelerator~5) --- ...}` paragraph (one paragraph long).

**Replace with:**
```
\paragraph{RQ3 (accelerator~5) --- AI bots dislike AI-authored code more than humans do.} A difference-in-differences on the doubly-engaged cohort (PRs reviewed by both an AI bot and a human reviewer, with no commits between the AI verdict and the human review) tests $\widehat{\rbt^A - \rbt^H}$ directly. Decomposing the estimand by author: AI bots approve AI-authored PRs \PLACEHOLDERAIAPPAIRATE{} of the time vs.\ humans \PLACEHOLDERHAPPAIRATE{} (gap \PLACEHOLDERDIDB1{}~pp); on human-authored PRs the corresponding gap is \PLACEHOLDERDIDB2{}~pp. The DiD $\PLACEHOLDERDIDPP{}$~pp (logit $\delta=\PLACEHOLDERDIDDELTA{}$, $SE=\PLACEHOLDERDIDSE{}$, $p<10^{-3}$) is in the \emph{opposite} direction from the self-preference predicted by single-turn lab studies~\citep{laurito2025aibias}: AI bots are less enthusiastic about AI-authored code than humans are, by a margin substantially larger than their analogous shortfall on human-authored code. The AI-side approval pool was expanded from \PLACEHOLDERAIOPRSNATIVE{} native explicit votes to \PLACEHOLDERAIOPPRS{} PRs by parsing structured verdict lines from each bot's \texttt{COMMENTED} review bodies (Appendix~F). We exclude PRs whose code changed between the AI's verdict and the human's review to ensure both reviewers evaluated the same commit graph; this drops the cohort from \PLACEHOLDERDIDNFULL{} to \PLACEHOLDERDIDN{} (a \PLACEHOLDERDIDDROPPCT{}\% drop, asymmetric by author: \PLACEHOLDERDIDDROPAIPCT{}\% of AI-authored and \PLACEHOLDERDIDDROPHPCT{}\% of human-authored excluded).
```

You will need to add `\PLACEHOLDERDIDNFULL`, `\PLACEHOLDERDIDDROPPCT`, `\PLACEHOLDERDIDDROPAIPCT`, `\PLACEHOLDERDIDDROPHPCT`, `\PLACEHOLDERAIOPRSNATIVE` to `08_fill_paper.py`. The "asymmetric drop" statistic itself is substantively interesting — flag it as an in-text observation rather than a footnote.

#### B3. `paper/paper.tex` — Figure 2 caption (currently around line 145) ✅

**Find:** the existing Figure 2 caption.

**Replace with:**
```
\caption{Within-PR AI-AI bias test on the \PLACEHOLDERDIDN{} stable doubly-engaged PRs (AI bot reviewed \emph{and} human reviewed, with no commits between the AI's verdict and the human's review). \textbf{Left:} approval rates by reviewer-type ($\{$AI, Human$\}$) $\times$ author-type ($\{$AI, Human$\}$); error bars are Wilson 95\% CIs. \textbf{Right:} the two brackets in $\widehat{\rbt^A - \rbt^H}$ — ``AI prefers AI more than humans do'' (\PLACEHOLDERDIDB1{}~pp on AI-authored PRs) and ``AI prefers humans less than humans do'' (\PLACEHOLDERDIDB2{}~pp on human-authored PRs). Their difference is the DiD: $\widehat{\rbt^A - \rbt^H} = \PLACEHOLDERDIDPP{}$~pp (logit $\delta=\PLACEHOLDERDIDDELTA{}$, $p<10^{-3}$). Both brackets are negative because AI bots approve everything less than humans do; the bracket on AI-authored PRs is \emph{more} negative, so the DiD is anti-self-preference.}
```

#### B4. `paper/paper.tex` — Limitation (iv), currently mentions "planned analyses" (around line 124) ✅

**Find:**
```
A cleaner read on the human side decomposes the cohort into PRs with no AI-bot reviewer ($n=72{,}946$, anchoring-free) versus PRs where the AI expressed an explicit verdict before any human engagement ($n=67$ strict, anchoring-prone); the no-AI-reviewer stratum on its own is the cleanest single-difference read of human preference for AI-authored code. Within-repo twin-matching of AI- and human-authored PRs ($n\approx 4{,}000$ pairs available in our data) supports a difference-in-differences across reviewer type that purges shared code-quality confounders. We treat these as planned analyses; the present paper reports the descriptive within-PR comparison.
```

**Replace with:** (remove the "planned analyses" sentence; the DiD is now the headline. Replace with a real-limitation paragraph.)
```
The DiD identification rests on the homogeneity assumption that PR quality shifts AI-bot and human approval probabilities on the same scale. Humans approve nearly all PRs they review (\PLACEHOLDERHAPPHRATE{} baseline); AI bots have a much wider acceptance range. A robustness check using stratification on AI-comment timing (Appendix~G) shows that anchoring of humans on AI's verdict moves the human-side gap by only \PLACEHOLDERANCHORADID{}~pp, ruling out anchoring as the main driver of the result.
```

#### B5. `paper/appendix/methods.tex` — Add a new paragraph documenting the AI-verdict regex parser ✅

After the existing "Repository universe" paragraph, add:
```
\paragraph{AI verdict extraction.} AI bots issue explicit \texttt{APPROVED}/\texttt{REQUEST\_CHANGES} review states on only \PLACEHOLDERAIOPRSNATIVE{} PRs in our window. The remaining ~70{,}000 AI-bot reviews carry the \texttt{COMMENTED} state but embed a structured verdict line written by the bot itself (e.g., CodeRabbit's \texttt{Actionable comments posted: N}, Cubic's \texttt{N issues found}, Cursor Bugbot's \texttt{found N potential issues}, Sourcery's intro line). We extract verdicts via per-bot regex on the structured headers — never on free-text sentiment. The full pattern catalogue and corpus-wide yield (\PLACEHOLDERAIOPPRS{} unique PRs with $\geq 1$ AI opinion after parsing, vs.\ \PLACEHOLDERAIOPRSNATIVE{} with native explicit votes only) is in Appendix~F. Gemini Code Assist reviews are free-prose without a stable structured header and are left unclassified (~10\% of AI-bot review events).
```

#### B6. New: `paper/appendix/ai_verdict_regex.tex` (Appendix F) ✅

Add a new appendix titled "AI-bot verdict regex catalogue" containing the regex table and worked examples from `99_causalvalidity.md` §7. Reference it from the body via `\input{appendix/ai_verdict_regex.tex}` after the existing appendices in `paper/paper.tex`.

#### B7. New: `paper/appendix/anchoring_robustness.tex` (Appendix G — robustness annex) ✅

Add a new appendix titled "Robustness: does AI's review anchor humans?" implementing the §8.4 cross-stratum DiD as a robustness check. Content:

```
\paragraph{Design.} A potential confound for the within-PR DiD (\S\ref{sec:results}, RQ3) is anchoring: humans who see an AI verdict before deciding may be nudged toward agreement, conflating preference with anchoring. We test this by comparing the human-side preference gap across two strata:

\begin{itemize}
\item \textbf{Stratum A (anchoring-free):} \PLACEHOLDERANCHORAN{} PRs with no AI-bot review event of any kind, ≥1 explicit human review, and no commits between PR open and human review. Human-side gap (AI-author − H-author approval rate): \PLACEHOLDERANCHORAGAP{}~pp.
\item \textbf{Stratum B (anchoring-prone):} \PLACEHOLDERANCHORBN{} PRs where the AI bot's first comment/review event preceded any human engagement, ≥1 explicit human review. Human-side gap: \PLACEHOLDERANCHORBGAP{}~pp.
\end{itemize}

\paragraph{Result.} Cross-stratum DiD (B − A) = \PLACEHOLDERANCHORADID{}~pp. AI's prior review nudges humans toward approving AI-authored PRs by less than 1 percentage point relative to the no-AI-reviewer baseline. The within-PR DiD reported in \S\ref{sec:results} is not driven by anchoring on the human side.
```

Add `\PLACEHOLDERANCHORAGAP` and `\PLACEHOLDERANCHORBGAP` to `08_fill_paper.py`.

Reference from body: replace the placeholder mention in the new Limitation (iv) text (B4 above) with `Appendix~\ref{app:anchoring}`.

### C. Re-run pipeline ✅ run on 2026-04-28; outputs verified

```bash
source ~/mcp-monitoring/.venv/bin/activate
python scripts/06c_within_pr.py     # produces results/within_pr_stats.json (new schema)
python scripts/07_make_figures.py   # rebuilds figure2_withinpr.pdf with the 2-panel design
python scripts/08_fill_paper.py     # fills new placeholders
python scripts/11_build_pdf.py      # rebuilds paper.filled.pdf
```

### D. Sanity checks ✅ all pass

The grep for "no strong evidence for AI-AI bias|22.6|27.9|7/31|70/251" returned zero hits in `paper.tex` (old Option A wording is gone). The grep for `PLACEHOLDERDID|stable doubly|rbt^A - rbt^H` shows the new framework in abstract, RQ3, Figure 2 caption, methods, and Appendices F/G. The two D1 hits for "popular" in `paper.tex`/`threats.tex` are legitimate (the new text *contrasts* "viral-popularity" with criticality-based ranking).

Verified outputs:
- `results/within_pr_stats.json`: `did.did_pp = -13.47`, `did.logit_delta = -0.6329`, `did.logit_p ≈ 1.18e-4`, `did.n = 8076` (AI-auth 782, H-auth 7167); `anchoring.cross_stratum_did_pp = +0.75`; `ai_only.no_human_explicit_pct = 43.7`.
- `paper/paper.filled.pdf`: 15 pages, 392 KB. Abstract third finding, RQ3 paragraph, Figure 2 caption, Limitations (iii)+(iv) all reflect the new DiD; appendices F+G compile.



1. `grep -in "no strong evidence for AI-AI bias\|22.6\|27.9\|7/31\|70/251" paper/paper.tex` should return no results — the old Option A wording is gone.
2. `grep -in "PLACEHOLDERDID\|rbt\^A - \\\\rbt\^H\|stable doubly-engaged" paper/paper.tex paper/appendix/*.tex` should show the new framework in abstract, RQ3 paragraph, Figure 2 caption, methods, and the new appendices F and G.
3. Open `paper.filled.pdf` and confirm:
   - Abstract third finding describes the DiD with $-13.47$ pp (or whatever the v2 value is at re-run time).
   - Figure 2 is the 2-panel design (4-bar reviewer×author + 2-bar bracket decomposition).
   - Limitation (iv) cites Appendix~G for the anchoring robustness check.
   - Appendix F documents the regex parser; Appendix G documents the cross-stratum DiD.

### E. Things NOT to touch

- `paper/test_ai_verdict_regex.py` and `paper/test_did_v2.py` — keep as standalone reproducibility artefacts, but update them to import `BOT_PATTERNS`/`classify` from `scripts/ai_verdict_parser.py` so there's only one source of truth.
- `99_causalvalidity.md` — design doc; do not edit unless the regex set or cohort definitions change.
- The five accelerators / theoretical-framework / Stratum-A single-difference text in `99_causalvalidity.md` §8.2 — that section is a supporting analysis, not the headline; do not promote it into the body.

## Sample-size sanity bound

Before running the pipeline, the next agent should expect:
- `did.n` in the range 7,500–8,500 (depends on data refresh).
- `did.did_pp` in the range −10 to −16 pp.
- `did.logit_p` $< 0.01$.

If the DiD comes back **positive** (AI-AI bias) on v2, stop and reconcile against `99_causalvalidity.md` §8.7 — something has changed in the data or pipeline.
