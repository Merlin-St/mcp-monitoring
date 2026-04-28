# Causal validity of the within-PR AI–AI bias test (RQ3)

This note evaluates how to answer **RQ3 (is there implicit AI–AI bias in PR review?)** with minimal confounding using only data already in `data/old_phase1/pr_events.parquet` and `data/old_phase1/pr_summary.parquet`. The aim is to make Figure 2 a defensible *causal* read on $\rbt^A - \rbt^H$, not a descriptive comparison.

---

## 1. Current Figure 2 (Option A, status quo)

**Cohort.** 282 PRs that received an `APPROVED` review from an AI bot.
**Estimand.** Among AI-approved PRs, share also approved by a human, AI-author vs human-author.
**Result.** 22.6% (7/31) vs 27.9% (70/251); gap = −5.3 pp; $p\!\approx\!0.53$.

This is a conditional association, not a causal estimate of $\rbt^A - \rbt^H$.

---

## 2. Confounders

1. **Quality-selection.** AI-authored PRs are not exchangeable with human-authored PRs in unobserved quality. They tend to be smaller, narrower-scope, more bot-friendly. If their *intrinsic* quality differs, the human-approval gap reflects quality, not preference. **This is the hardest confounder.**
2. **Collider on AI-approval.** "AI bot approved" is downstream of both authorship and quality. Conditioning on it selects a non-random subset of each authorship cell.
3. **Anchoring.** AI bots review first ${\sim}90\%$ of the time; humans who see `APPROVED` before deciding may anchor. "Human co-approval" is partly downstream of AI review.
4. **AI-reviewer concentration.** Two bots (Cubic, CodeRabbit) supply >95% of approvals.
5. **Authorship-classification error.** Silent AI use puts true-AI PRs in the human-author cell — biases toward zero.
6. **Repo and time confounding.** AI-authored PRs cluster in certain repos and months.

A causal interpretation requires assuming all six are negligible. 1, 2, 4, 6 are not.

---

## 3. Available cohorts (verified)

From `pr_events.parquet` + `pr_summary.parquet`:

| Cohort | $n$ | $n_{\text{AI-auth}}$ | $n_{\text{H-auth}}$ |
|---|---:|---:|---:|
| All PRs in window | 225,655 | 12,561 | 184,720 |
| AI bot reviewed (any state) | 24,506 | — | — |
| **Both AI-bot and human reviewed (any state)** | **13,651** | **2,163** | **11,402** |
| AI bot APPROVED | 282 | 31 | 251 |
| **Doubly explicit: AI explicit vote AND human explicit vote** | **119** | **8** | **109** |
| AI-author PRs with $\geq 1$ candidate human twin (same repo, ±60d, ±50% LOC, ±1 file) | 7,481 | 7,481 | — |
| AI-author PRs with $\geq 1$ strict human twin (±60d, ±25% LOC, ±1 file) | 5,849 | 5,849 | — |

Two facts shape the design space:

- AI bots overwhelmingly emit `COMMENTED` reviews (152k events) rather than explicit votes (`APPROVED` 96k, `REQUEST_CHANGES` 10k events). Restricting to *doubly explicit* votes is conceptually clean but cuts the sample to $n=119$ with only 8 AI-author PRs.
- A within-repo matched pool of ~5–7k AI-author PRs with size-and-time-matched human twins is available.

---

## 4. Options ranked

### Option A — Status quo
$n=282$ ($31$ vs $251$). Confounders 1–6 all live. Keep as a robustness row, not the headline.

### Option B — Difference-in-differences across reviewer type, doubly-explicit ⭐ recommended primary

**Universe.** PRs where at least one AI bot AND at least one human reviewer issued an explicit vote (`APPROVED` or `REQUEST_CHANGES`). $n=119$, $n_{\text{AI-auth}}=8$, $n_{\text{H-auth}}=109$.

**Two binary outcomes per PR.**
- $\text{AI}_\text{approve}\in\{0,1\}$: AI bot's explicit vote was `APPROVED`.
- $\text{Hum}_\text{approve}\in\{0,1\}$: human reviewer's explicit vote was `APPROVED`.

**Estimand.**

$$
\widehat{\rbt^A - \rbt^H} = \big[\Pr(\text{AI}_{\text{approve}}\mid \text{AI-auth}) - \Pr(\text{AI}_{\text{approve}}\mid \text{H-auth})\big] - \big[\Pr(\text{Hum}_{\text{approve}}\mid \text{AI-auth}) - \Pr(\text{Hum}_{\text{approve}}\mid \text{H-auth})\big].
$$

**In words.** Compute four numbers on the same set of PRs: AI's approval rate on AI-authored PRs, AI's approval rate on human-authored PRs, the human's approval rate on AI-authored PRs, and the human's approval rate on human-authored PRs. Take AI's preference gap (first minus second) and subtract the human's preference gap (third minus fourth). The remainder is the *differential* preference for AI authors — what the AI rates higher than the human does. Any factor that changes both reviewer types' approval probabilities by the same amount — better-tested code, smaller diffs, more familiar repos, all the unobserved quality effects — is in both gaps and cancels.

**Equivalent regression.** Fit a logit on the long-format dataset (two rows per PR, one per reviewer type, $2\times 119=238$ rows):

$$
\Pr(\text{approve}=1) \sim \text{logit}(\alpha + \beta_R\!\cdot\!\text{Reviewer}_\text{AI} + \beta_A\!\cdot\!\text{Author}_\text{AI} + \delta\!\cdot\!\text{Reviewer}_\text{AI}\!\times\!\text{Author}_\text{AI}).
$$

The interaction $\delta$ is the bias estimate; the test on $\delta$ is the headline.

**Why this addresses the confounders.**
- **Quality-selection (1):** if AI-authored PRs are intrinsically lower-(or higher-)quality, that shifts both reviewer types' approval probabilities *together*. The DiD differences it out under a homogeneity assumption (the quality effect on AI vs human approval is the same on log-odds or risk-difference scale) — much weaker than no-confounding.
- **Collider (2):** the design does not condition on AI-approval, so the collider is closed.
- **Reviewer concentration (4):** both author cells are evaluated by the same mix of AI bots; comparison is within reviewer-type.
- **Repo/time (6):** repo and month FEs can be added to the logit.

**Power problem to flag.** With $n_{\text{AI-auth}}=8$, marginal SEs on the AI-author cells are ${\sim}12$–$18$ pp. The MDE on $\delta$ is ${\sim}25$ pp at 80% power. The doubly-explicit cohort is *clean* but underpowered as a stand-alone test. **Sample (raw cells from the data):**

|  | AI-auth ($n=8$) | H-auth ($n=109$) |
|---|---:|---:|
| AI approves | 7/8 = 87.5% | 74/109 = 67.9% |
| Human approves | 8/8 = 100% | 98/109 = 89.9% |

DiD point estimate $\delta = (87.5\!-\!67.9) - (100\!-\!89.9) = +9.5$ pp (AI more selective for AI authors than the human) — directionally consistent with self-preference, but $n_{\text{AI-auth}}=8$ makes this an illustration, not a finding.

#### B-plus — Conservative regex-parsed AI approvals

To expand the AI-side approval pool without introducing an LLM classifier, parse `COMMENTED` review bodies for **explicit approval phrases only** (no rejection-side, no sentiment, no graded inference). Treat a regex hit as equivalent to an `APPROVED` state for the AI-side outcome; the AI-rejection side stays restricted to native `REQUEST_CHANGES`. The human side is unchanged.

**Conservative regex set (approve-only).** Match a `COMMENTED` review body if any of the following holds:

- `(?im)^\s*(\*\*)?\s*(LGTM|Approved|Ship\s*it|✅\s*Approved)\s*(\*\*)?\s*[!.]?\s*$` — a **standalone** verdict line (whole line, anchored).
- `(?is)##\s*(Verdict|Decision|Recommendation)\s*:?[^\n]*\n+\s*(\*\*)?\s*Approved\b` — Cubic-style structured verdict block whose verdict is exactly `Approved`.
- `(?im)^\s*:shipit:\s*$` or `(?im)^\s*:lgtm:\s*$` — emoji-alias verdicts on their own line.

Rules of thumb to stay conservative:
- Inline mentions ("LGTM modulo …", "approved with concerns", "looks good but") are **not** matched. The anchors `^…$` enforce standalone lines.
- Quoted text and code blocks (lines beginning with `>`, `\`\`\``, four-space indent) are stripped before matching.
- Only review bodies of bots on the existing AI-bot allowlist are scanned. Other bots (Dependabot, Renovate, etc.) are excluded by allowlist gate, regardless of body text.
- Sample 100 random matches and 100 random non-matches by hand to estimate precision/recall before reporting.

**Universe.** Same as B, with the AI-side `APPROVED` cell expanded to (native `APPROVED`) ∪ (regex-matched approval in a `COMMENTED` review by an allowlisted AI bot).

**Sample (estimate, to verify on the data).** Of 152,388 AI-bot `COMMENTED` events, a conservative parser of the kind above plausibly yields ~5–15% explicit-approval matches (Cubic and CodeRabbit emit structured verdicts on most reviews; Copilot and Greptile rarely approve). At 10% yield, the doubly-explicit-plus cohort would expand from $n=119$ to roughly $n=1{,}500$–$2{,}500$, with $n_{\text{AI-auth}}$ moving from 8 to perhaps 100–200. Marginal SEs on the AI-author cell drop from ${\sim}15$ pp to ${\sim}3$–$5$ pp — enough for a meaningful DiD point estimate without invoking matching machinery.

**Caveats.**
- Parser precision is the binding cost. Report it from the audit (target: ≥95% precision on a 200-review hand-coded sample).
- Asymmetric expansion (approve-side only) leaves the AI-rejection cell unchanged. The DiD interpretation still holds as long as the AI-side outcome is binary (`AI_approve $\in\{0,1\}$`) and rejection/silence both map to 0.
- Parser yield will be heavily Cubic/CodeRabbit-dominated; report the bot-mix breakdown alongside the headline.

**Verdict.** Promising bridge between strict B ($n=119$) and matched F (~$n=4{,}000$ pairs). Run if the parser audit clears the precision bar; otherwise stay with B.

### Option C — Direct quality-proxy adjustment (the one direct-control option)

**Universe.** Same PRs as B (or as A), full sample preserved.
**Model.**

$$
\Pr(\text{Hum}_\text{approve}=1)\sim\text{logit}(\alpha + \beta\!\cdot\!\text{AI-auth} + \boldsymbol{\gamma}^\top Q + \mu_\text{repo} + \tau_\text{month}),
$$

with $Q$ the observable quality proxies already in `pr_summary.parquet`: `additions`, `deletions`, `changed_files`, `commits_in_pr`, `body_len`, `title_len`, `review_comment_count`, `issue_comment_count`, `days_to_merge`. Repo and month fixed effects absorb between-repo and time selection.

**What it buys.** Adjustment for *observed* quality without dropping rows.
**What it costs.** Selection-on-unobservables remains. With $n_{\text{AI-auth}}=8$ in the doubly-explicit cohort, fit on a larger universe (e.g.\ doubly-engaged any-state, $n=13{,}651$) where regularised regression with FEs is identifiable.

**Verdict.** Run as a robustness panel under B and F, not standalone.

### Option D — Anchoring stratification

Stratify Option B (or F) by whether the AI bot's first review preceded or followed the first human review. *AI-first*: anchoring possible. *Human-first*: anchoring-free. If the bias is present only in AI-first, the result is anchoring, not preference.

In our data ${\sim}90\%$ of PRs are AI-first, so the human-first stratum is small. Useful as a sensitivity check, not a stand-alone test.

### Option E — Bot-onboarding event study

Use the date a code-review bot first appeared in each repo as a quasi-experimental shock. Event study with repo and calendar-month FEs; pre/post comparison of human approval rates on AI- vs human-authored PRs. ~30–80 onboarding events, 50–500 PRs in pre/post windows each.

This is a clean read on a *mediator* of disempowerment dynamics (do humans become more deferential after a bot arrives?) — closer to RQ2 than RQ3. Promising appendix figure.

### Option F — Twin / near-duplicate PR matching ⭐ recommended power-side complement

This is the cleanest within-repo design that *also* preserves a usable sample size. Concrete recipe:

1. **Candidate pool.** All PRs with `author_type` $\in\{\text{AI}, \text{human}\}$, with at least one explicit human reviewer event (so an outcome is defined).
2. **Per-AI-PR matching.** For each AI-authored PR, build a candidate set of human-authored PRs that satisfy:
   - same `repo`,
   - opened within $\pm 60$ days of the AI PR's `created_at`,
   - LOC ($|\text{additions}|+|\text{deletions}|$) within $\pm 25\%$ (or absolute difference $\leq 3$ for tiny PRs),
   - `changed_files` within $\pm 1$.
3. **Caliper on observables.** Estimate a propensity score $\Pr(\text{AI-auth}\mid Q)$ on $Q=$(LOC, files, commits, body length, day-of-week, month, repo) using a logit. Drop matches whose propensity differs by more than $0.1$ from the AI PR's score.
4. **Match.** 1-to-1 nearest-neighbour matching on the propensity score, with replacement, within the caliper. Tie-break by closest opening time.
5. **Outcomes per matched pair.** $\text{AI}_\text{approve}^{(AI)}, \text{AI}_\text{approve}^{(H)}, \text{Hum}_\text{approve}^{(AI)}, \text{Hum}_\text{approve}^{(H)}$. (Each may be missing if that reviewer did not engage; handle by listwise deletion or by treating "no engagement" as "did not approve" — report both.)
6. **Test.** Paired test on each outcome (McNemar for binary, sign test as backup); also fit the same DiD logit as Option B but on the matched dataset, with robust SEs clustered at the repo level.

**Sample feasibility (verified).** Out of 12,561 AI-author PRs, **5,849 have at least one strict human twin** (same repo, ±60d, ±25% LOC, ±1 file) and 7,481 have a loose twin. After 1-to-1 caliper matching with the propensity-score restriction, expect roughly 3,000–5,000 matched pairs.

**What this buys over B.**
- Same DiD framework, but matching on observables removes much of the AI-vs-human-author quality gap *before* differencing — so the homogeneity assumption is far easier to defend.
- $n$ is two orders of magnitude larger than doubly-explicit Option B; MDE on $\delta$ drops from ~25 pp to ~1–2 pp.
- Within-pair comparison absorbs the repo and time confounders without explicit FEs.

**Remaining threats.** Anchoring (3) is unaddressed, as is selection on unobservables that don't correlate with $Q$. Authorship-error (5) still attenuates. The matching algorithm has tuning knobs (caliper, ratio); pre-register or report sensitivity to caliper choice.

### Option G — Rejection-side estimand

Mirror of A, conditioning on AI `REQUEST_CHANGES`. ~200 unique PRs total, $n_{\text{AI-auth}}\approx 12$. Statistically dead in our window.

### Option H — Anchoring decomposition (no-AI-reviewer + AI-first strata) ⭐ recommended cleanest single-difference

This design avoids the *anchoring* confound (#3) entirely by partitioning the data into two strata defined by whether an AI bot reviewer could have nudged the human. It does *not* directly estimate $\rbt^A - \rbt^H$; it estimates the **human side alone** — $\rbt^H_{\text{AI-auth}} - \rbt^H_{\text{H-auth}}$ — and decomposes it into a pure component and an anchoring component.

**Stratum A — No-AI-reviewer cohort (anchoring-free).**
- *Universe.* PRs with **no** AI-bot review/comment event AND $\geq 1$ explicit human review (`APPROVED` or `REQUEST_CHANGES`).
- *Sample (verified).* $n=72{,}946$, $n_{\text{AI-auth}}=2{,}040$, $n_{\text{H-auth}}=70{,}906$.
- *Outcome.* $\text{Hum}_\text{approve}\in\{0,1\}$.
- *Single-difference estimand.* $\Pr(\text{Hum}_\text{approve}\mid\text{AI-auth}) - \Pr(\text{Hum}_\text{approve}\mid\text{H-auth})$.
- *Interpretation.* Direct human preference for AI-authored code, with **no AI-bot reviewer in the loop** to anchor on. This is the cleanest answer to "do humans, on their own, approve AI-authored PRs at a different rate than human-authored ones?"
- *MDE.* With $n_{\text{AI-auth}}=2{,}040$ at base rate 30%, SE on the gap is ~1.0 pp; MDE at 80% power $\approx 1.3$ pp.

**Stratum B — AI-first-with-clear-opinion cohort (anchoring-prone).**
- *Universe.* PRs where the AI bot expressed a **clear opinion** (explicit `APPROVED` / `REQUEST_CHANGES` review state, OR a regex-parsed verdict in a `COMMENTED` body — see §7) **before any human engagement** (comment or review), AND $\geq 1$ explicit human review.
- *Sample (verified).*
  - **Native explicit only:** $n=67$, $n_{\text{AI-auth}}=3$, $n_{\text{H-auth}}=63$ (power-dead).
  - **Native + regex-parsed (recommended):** $n=467$, $n_{\text{AI-auth}}=73$, $n_{\text{H-auth}}=380$. Approximate MDE on the gap at 80% power: ${\sim}7$ pp.
- *Same single-difference outcome.*
- *Interpretation.* Human preference *plus* anchoring on the AI's prior verdict.

**Cross-stratum DiD (the anchoring estimate).**

$$
\widehat{\text{anchoring effect}} = (\text{Stratum B gap}) - (\text{Stratum A gap}).
$$

If positive, humans are *additionally* more likely to approve AI-authored code when the AI's verdict came first — the anchoring confound is real.

In practice the cross-stratum DiD is dominated by the noise on Stratum B at $n_{\text{AI-auth}}=3$; treat it as a sensitivity row, not as the headline. Stratum A on its own remains the cleanest single-difference read on human preference for AI-authored code.

**Why this is causally cleaner than B/F.**
- *Anchoring (3):* Stratum A removes it by construction (no AI reviewer); Stratum B isolates it.
- *Collider (2):* avoided — no conditioning on any vote outcome.
- *Reviewer concentration (4):* irrelevant in Stratum A (no AI reviewer); within Stratum B, can be addressed by FE.
- *Authorship error (5):* still attenuates toward zero. (Note: lenient AI-authorship — `author_type='AI'` ∪ `ai_bot_commits>0` ∪ `coauthored_commits>0` — was checked but adds **0** PRs because the existing classifier already incorporates co-authored-by trailers.)

**What this does *not* give you.** A direct estimate of $\rbt^A - \rbt^H$. For that, combine Stratum A's anchoring-free human gap with the AI-side gap from B / B-plus / F. The cross-cohort DiD trades within-PR cancellation of code-quality confounders for within-cohort anchoring removal — a different identification, complementary to B/F.

**Remaining threats.**
- *Selection on the no-AI-reviewer stratum.* PRs with no AI-bot review may be systematically smaller, in less-AI-active repos, or in months before bot adoption. Mitigation: repo and month FEs in a logit; report by repo / by month.
- *Quality selection (1) is unaddressed.* AI-authored PRs may differ in unobserved quality from human-authored ones in either stratum. Pair Option H with Option C (covariate adjustment) on the same cohorts as a robustness check.

---

## 5. Recommendation

RQ3 splits naturally into two answerable questions, each with a clean design:

1. **"Do humans alone prefer AI-authored code?"** — answered by Option H-A (Stratum A, no-AI-reviewer cohort, $n=72{,}946$, $n_{\text{AI-auth}}=2{,}040$). Single-difference. No anchoring confound by construction. MDE ${\sim}1.3$ pp.
2. **"Does the AI's own preference for AI authorship exceed the human's?"** ($\rbt^A - \rbt^H$) — answered by Option F (twin-matched DiD, ${\sim}4{,}000$ pairs) or Option B-plus (audit-gated doubly-explicit-plus, $n\approx 2{,}000$).

**Suggested Figure 2 layout.**
- **Primary panel (left):** Stratum A single-difference (Option H-A). Two bars (human approval rate by author) on the no-AI-reviewer cohort, Wilson 95% CIs. This is the cleanest descriptive read on whether humans on their own favour AI-authored code.
- **Primary panel (right):** twin-matched DiD (Option F). Four bars (reviewer × author cells) on the matched cohort, interaction $\delta$ with repo-clustered 95% CI annotated. This is the headline $\rbt^A - \rbt^H$ estimate.
- **Inset / annotation:** cross-stratum DiD = (Stratum B gap) − (Stratum A gap), reported as the anchoring effect.
- **Appendix robustness:** Option B-plus (audit-gated parser) if it clears ≥95% precision, Option C (regression with $Q$ + FEs), Option D (anchoring stratification), Option E (bot-onboarding event study), and strict B at $n=119$.
- **Drop or demote:** Option A (current Fig 2). Retain as one row in a robustness table.

**Caveats to keep in the paper text.**
- Identification rests on a homogeneity assumption — quality shifts AI's and the human's approval probabilities on the same scale.
- Anchoring of humans on AI cannot be fully ruled out without timing variation.
- Authorship-classification error attenuates $\delta$ toward zero.

This shifts the third headline result from "no significant gap on $n=31$" to "twin-matched DiD estimate of $\widehat{\rbt^A - \rbt^H}$ on $n\approx 4{,}000$ pairs, identified up to a homogeneity assumption" — a much stronger causal claim on a sample that is finally adequate.

---

## 6. Sample-size summary

| Option | Cohort | $n$ | $n_{\text{AI-auth}}$ | Causal strength | Notes |
|---|---|---:|---:|---|---|
| A | AI-approved PRs | 282 | 31 | low | current Fig 2 |
| B | Doubly explicit votes | 119 | 8 | high (clean) but power-dead | strictest read |
| B-plus ⭐ | B + regex-parsed AI approvals (approve-only) | ~1,500–2,500 (est.) | ~100–200 (est.) | high (clean) | conservative parser, audit-gated |
| C | Doubly engaged + covariates | 13,651 | 2,163 | medium | direct adjustment, observed only |
| D | B or F, human-first stratum | small | small | sensitivity | tests anchoring |
| E | Bot-onboarding events | ~30–80 events | — | quasi-experimental | RQ2-flavoured |
| F ⭐ | Within-repo twin-matched pairs | ~3,000–5,000 pairs | same | high | DiD on matched observables |
| G | AI-rejected PRs | ~200 | ~12 | theoretical only | underpowered |
| H-A ⭐ | No-AI-reviewer + human explicit | 72,946 | 2,040 | high (no anchoring) | single-diff, human side only |
| H-B (native) | AI clear opinion before human + human explicit | 67 | 3 | sensitivity only | power-dead alone |
| H-B (native + parsed) ⭐ | Same, with regex-parsed COMMENTED verdicts (§7) | 467 | 73 | medium-high | MDE ~7pp; pairs with H-A |

---

## 7. Regex parser test for AI-bot verdicts in `COMMENTED` reviews

**Motivation.** AI bots issue explicit `APPROVED` / `REQUEST_CHANGES` review states on only 415 PRs (560 events). The remaining 70k+ AI-bot reviews carry the `COMMENTED` state — but most of them embed a structured verdict line written by the bot itself (e.g., CodeRabbit's `**Actionable comments posted: N**`). Parsing those headers with bot-specific regex recovers the verdict without an LLM classifier.

**Replicable script.** `paper/test_ai_verdict_regex.py` (standalone, not part of the pipeline). Run with:
```
source /home/ubuntu/mcp-monitoring/.venv/bin/activate
python paper/test_ai_verdict_regex.py
```
Scans `data/prs/*.jsonl` and prints per-bot match counts plus sample matches and unmatched bodies.

**Patterns (per-bot, first match wins).** Each pattern targets a fixed structured header that the bot itself writes — not free-text sentiment.

| Bot | Verdict | Regex (Python) | Yield |
|---|---|---|---:|
| Cubic | approve | `(?im)^\s*\*\*No issues found\b[^*]*\*\*` | 2,462 |
| Cubic | reject | `(?im)^\s*\*\*([1-9]\d*) issues? found\b[^*]*\*\*` | 1,313 |
| CodeRabbit | approve | `(?im)^\s*\*\*Actionable comments posted:\s*0\s*\*\*` | 4,491 |
| CodeRabbit | reject | `(?im)^\s*\*\*Actionable comments posted:\s*([1-9]\d*)\s*\*\*` | 12,888 |
| Copilot PR Reviewer | approve | `(?im)Copilot reviewed[^\n]*and generated no new comments` | 1,099 |
| Copilot PR Reviewer | reject | `(?im)Copilot reviewed[^\n]*and generated ([1-9]\d*) comments?` | 18,267 |
| Greptile | approve | `(?im)<sub>\s*\d+\s*files?\s*reviewed,\s*no comments?\s*</sub>` | 306 |
| Greptile | reject | `(?im)<sub>\s*\d+\s*files?\s*reviewed,\s*([1-9]\d*)\s*comments?\s*</sub>` | 720 |
| Cursor Bugbot | approve | `(?im)^\s*(###\s*)?✅\s*Bugbot reviewed your changes and found no (bugs\|new issues)!?` | 55 |
| Cursor Bugbot | reject | `(?im)Cursor (Bugbot\|Bug-Bot) has reviewed your changes and found ([1-9]\d*) potential issues?` | 1,504 |
| Sourcery | approve | `(?im)^Hey(\s+(there\|@\S+))?\s*-\s*I\'?ve reviewed your changes and they look great` | 986 |
| Sourcery | reject | `(?im)^Hey(\s+(there\|@\S+))?\s*-\s*I\'?ve (reviewed your changes\s*-\s*here\'?s some feedback\|reviewed your changes and found some issues\|left some\|found \d+)` | 1,635 |
| Claude | approve | `\ALGTM\b` (anchored to start of body) | 717 |

**Claude explained.** The pattern `\ALGTM\b` matches only when "LGTM" appears as the very first word of the review body (`\A` anchors to the absolute start of the body, not the start of any line). The trailing word-boundary `\b` accepts "LGTM —", "LGTM,", "LGTM\n", and "LGTM" alone, but rejects accidental substrings like "LGTMs" or "LGTMrebased".

- *Yield.* 717 of 1,042 Claude `COMMENTED` reviews (≈69%). All five randomly-sampled matches were genuine approvals followed by an `<details>Extended reasoning</details>` block.
- *Excluded by construction.*
  - 134 `## Claude Code Review` setup messages ("This repository is configured for manual code reviews. Comment `@claude review`…") — these are configuration notices, not opinions.
  - 22 `⚠️ **Code review skipped** — your organization's overage spend limit has been reached` billing notices — also no-opinion.
- *Knowingly missed.* About 20 reviews end with "— LGTM." rather than starting with it (e.g., "All previous review feedback has been addressed — LGTM."). The `\A` anchor rejects them. We accept this loss because relaxing the anchor (e.g., `\bLGTM\b` anywhere) would risk false positives in longer mixed-verdict bodies.
- *No reject pattern.* Claude's critique reviews are free-prose without a stable structured header (unlike Cubic's `**N issues found**` or CodeRabbit's `**Actionable comments posted: N**`). About 150 unclassified bodies are real reviews — some clear rejects ("This is a very large PR with breaking changes to core OAuth auth logic…"), some soft approvals with caveats ("No bugs found, but this is an RFC with open design questions…"). Recovering them would require an LLM sentiment classifier and is out of scope for the conservative parser.

**Cursor explained.** Cursor's reviewer (Bugbot) has two distinct verdict shapes that share no header:
- *No-bugs verdict.* `✅ Bugbot reviewed your changes and found no bugs!` (or `... no new issues!`), optionally prefixed with `### `. The leading green-check emoji is required — without it the line could be ambiguous prose. 55 hits in our corpus.
- *Issues-found verdict.* `Cursor Bugbot has reviewed your changes and found N potential issues.` Sometimes preceded on a previous line by `<!-- BUGBOT_REVIEW -->` (an HTML comment marker); the `(?im)` flag finds the verdict anywhere in the body, so this prefix is harmless. 1,504 hits.
- *Excluded by construction.* Status-only messages — `### This PR is being reviewed by Cursor Bugbot` (~200), `### This is the final PR Bugbot will review for you during this billing cycle` (~100), `**Bugbot free trial expires…**`, and `<!-- CURSOR_AUTOMATION_ID: …` blocked-task notices — never carry the verdict sentence and remain unmatched. They are also not opinions, so leaving them unclassified is correct.
- *Edge case.* About 100 reviews open with `<details open>` followed by a `<h3>Bug: …</h3>` block but lack the standard "found N potential issues" line. These are reject signals in a non-standard format; we leave them unclassified to keep the parser simple. Adding a fallback `(?im)^<details open>\s*<summary><h3>Bug:` would recover them but is optional.

**Sourcery explained.** Sourcery has three greeting variants — `Hey -`, `Hey there -`, and `Hey @username -` — each followed by either an approval or a critique. The optional group `(\s+(there|@\S+))?` covers all three:
- *Approve.* `... I've reviewed your changes and they look great!` (986 hits — covers all three greetings).
- *Reject.* Any of: `... here's some feedback`, `... found some issues that need to be addressed`, `... left some high level feedback`, or `... found N issues`. (1,635 hits.)
- *Excluded by construction.* `We've reviewed this pull request using the Sourcery rules engine` (156, lint-engine output, not the AI's opinion) and `Sorry @user, you have reached your weekly rate limit…` (rate-limit notices) remain unmatched. Both are correctly treated as no-opinion.

**Aggregate result over all 74,385 AI-bot `COMMENTED` reviews (corpus-wide):**

| Verdict | Events | Share |
|---|---:|---:|
| approve (parsed) | 10,116 | 13.6% |
| reject (parsed) | 36,327 | 48.8% |
| unmatched | 27,942 | 37.6% |

**Combined with native explicit votes**: ~10.3k approve and ~36.4k reject parsed verdict events. ~32k unique PRs gain at least one parsed AI opinion — a ~78× expansion of the AI-side opinion pool relative to the 415 PRs with native explicit `APPROVED`/`REQUEST_CHANGES` states.

**Coverage by bot (% of that bot's `COMMENTED` reviews classified):**

| Bot | Coverage | Unmatched | Why unmatched |
|---|---:|---:|---|
| Cubic | 100% | 0 | All reviews use the `**N issues found**` header |
| CodeRabbit | 86% | 2,771 | Follow-up reviews after the initial verdict; no new header |
| Copilot PR Reviewer | 57% | 14,889 | Substantive prose reviews without the "Copilot reviewed N out of N" template |
| Cursor Bugbot | 79% | 420 | Status-only messages and non-standard `<details open>` Bug-detail blocks |
| Greptile | 54% | 858 | Other format variants in early/late deployment |
| Sourcery | 79% | 706 | "Sourcery rules engine" auto-lint output and rate-limit notices (correctly excluded as no-opinion) |
| Claude | 69% | 325 | Setup messages ("## Claude Code Review", correctly excluded — these are no-opinion bodies) |
| Gemini Code Assist | 0% | 7,935 | Free-prose reviews; no stable structured verdict line. Would require an LLM classifier. |

**Precision (eye-test on samples).** All sample matches inspected for each pattern were correct verdicts (no false positives observed across ~50 hand-checked matches). A formal hand-audit of 100 random matches per pattern would tighten the precision estimate; we recommend it before promoting the parser into the pipeline.

**Decision.** With this parser, **Stratum B expanded** rises from $n=67$ ($n_{\text{AI-auth}}=3$) to $n=467$ ($n_{\text{AI-auth}}=73$) — power moves from "dead" to ~7 pp MDE on the gap. The cross-stratum DiD against Stratum A ($n=72{,}946$) becomes a usable estimate of the anchoring effect.

**Conservatism.** All patterns target the bot's *own* structured output. None infer verdicts from free-text sentiment. The unmatched 37.6% — dominated by Gemini and Copilot prose reviews — is left unclassified rather than guessed at.
