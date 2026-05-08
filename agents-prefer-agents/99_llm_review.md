# LLM Review of `paper.filled.tex`

Read-only review against `agents-prefer-agents/SI_review_prompt.md`. Findings only — **no source files have been edited**. Implementation is gated on human review.

## Top findings (most important)

5. **Two orphaned figure PDFs and several orphaned appendix files.** `figure1_merge_rates.pdf` and `figure3_any_ai_share.pdf` are generated but never `\includegraphics`'d. `old_repo_selection.filled.tex`, `audit.filled.tex`, `examples.filled.tex`, `regression.filled.tex`, `threats.filled.tex` are not `\input` from `paper.filled.tex` — `old_repo_selection` in particular describes a superseded methodology and risks confusion. (Cat 3 / Cat 9)
--> delete these orphaned files
6. **Terminology drift on the central measurement.** "AI reviewed", "AI opinion", "AI review", "explicit AI review", "AI bot general-review opinion" are used interchangeably in the main text without a single canonical definition; "general review" is used in Results without ever being defined in the main text. (Cat 1)
--> replace 'opinion' with 'review'. Check all mentions of 'review', rewrite it to either say '


## Severity counts by category

| Category | Blocker | Major | Minor | Nit | Total |
|---|---|---|---|---|---|
| 1 — Structure & Terminology | 0 | 5 | 5 | 2 | 12 |
| 2 — Writing, Grammar & Small Edits | 0 | 3 | 6 | 0 | 9 |
| 3 — Figures | 0 | 3 | 4 | 3 | 10 |
| 4 — Title | 0 | 0 | 0 | 0 | 0 |
| 5 — Abstract | 0 | 1 | 2 | 0 | 3 |
| 6 — Introduction | 0 | 3 | 2 | 0 | 5 |
| 7 — Results | 0 | 1 | 3 | 0 | 4 |
| 8 — Methods | 0 | 1 | 0 | 0 | 1 |
| 9 — Annex / Appendix | 0 | 2 | 2 | 0 | 4 |
| **Total** | **0** | **19** | **24** | **5** | **48** |

---

## Category 1 — Structure & Terminology

### Issue: RQ 3 framing mismatch with the actual finding
- **Location**: `paper.filled.tex:85` (RQ 3 statement) and `:156` (RQ 3 result subsection title)
- **Problem**: RQ 3 is phrased as "Do AI reviewers prefer AI-authored PRs more than human reviewers do?" — implying the hypothesis is "more". The Results subsection at line 156 is titled "AI systems dislike AI-authored code more than humans do" — a finding in the opposite direction. The RQ text and the result text point at each other inconsistently.
- **Recommended change**: Reframe RQ 3 as a neutral question ("Do AI and human reviewers differ in their approval of AI-authored PRs?") and explicitly note in the intro that motivation comes from lab predictions of self-preference, but the empirical question is open.
- **Severity**: major

### Issue: No explicit Conclusion section
- **Location**: `paper.filled.tex` (whole-document structure)
- **Problem**: Sections are Introduction → Framework → Methods → Results → Discussion; no Conclusion. The Discussion addresses limitations but does not (a) map findings back to the five accelerators, (b) restate the gradual-disempowerment claim in light of results, or (c) synthesise what the null/anti-self-preference RQ 3 result means.
- **Recommended change**: Add a short Conclusion section that maps RQ 1 / RQ 2 / RQ 3 findings to the accelerator framework, restates the core claim, and synthesises the implications of the reversal for "AI taste lock-in".
- **Severity**: major

### Issue: Ambiguous mapping between accelerators and RQs
- **Location**: `paper.filled.tex:85` (RQs), `:87–109` (Table 1 / framework), `:139` ("To illustrate how to measure the last two of the five accelerators")
- **Problem**: Table 1 lists five accelerators; only "the last two" are measured, and that scoping appears for the first time in Methods. Readers cannot tell which accelerators each RQ tests, supports, or refutes.
- **Recommended change**: In the Introduction or at the start of Methods, state explicitly: "RQ 1 provides indirect evidence on Accelerators 1–3; RQ 2 measures Accelerator 4 ($d\alpha_t/dt$); RQ 3 measures Accelerator 5 (AI-AI bias). Accelerators 1–3 are not directly tested." Re-bind this language in the Discussion.
- **Severity**: major

### Issue: Terminology drift on the core measurement (AI opinion / AI reviewed / AI review)
- **Location**: `paper.filled.tex:139` ("AI reviewed"), `:140` ("AI opinion"), `:154` ("explicit AI review"), `:156` ("AI bot general-review opinion"); `appendix/methods.filled.tex:3` (defines "explicit review" vs. "general review")
- **Problem**: The same construct is referred to with at least four different names across main text and appendix; "general review" is defined in Appendix E but used in the main Results section with no definition.
- **Recommended change**: Adopt one canonical scheme. Suggested: "explicit AI review" = native APPROVED/CHANGES_REQUESTED; "AI opinion" = explicit + regex-parsed verdicts; drop "AI reviewed" as a label (use only as the English participle). Define both terms once in Methods main text and use them consistently.
- **Severity**: major

### Issue: "General review" used in main Results without prior definition
- **Location**: `paper.filled.tex:156` ("AI bot general-review opinion")
- **Problem**: A reader of the main text encounters "general-review" without ever having seen it defined. The definition lives only in `appendix/methods.filled.tex:3`.
- **Recommended change**: Either define "general review" inline at first use in Methods (line 139–140) or replace it with the canonical term chosen above.
- **Severity**: major

### Issue: Discussion does not reconnect findings to RQs or framework
- **Location**: `paper.filled.tex:172–184`
- **Problem**: The Discussion opens with "AI taste lock-in is an important empirical question" and notes "we do not find strong evidence for AI taste lock-in here" but never explicitly states which accelerators are supported, refuted, or untested. The RQ 3 reversal is not interpreted in framework terms.
- **Recommended change**: Restructure the Discussion to (a) state RQ 1 + RQ 2 indirectly support Accelerators 1–4, (b) explicitly note that the AI-AI-bias mechanism (Accelerator 5) is *not* observed in production, and (c) discuss the alternative explanations (lower AI-code quality; review non-homogeneity) and their implications.
- **Severity**: major

### Issue: "AI taste lock-in" appears in title without nearby definition
- **Location**: `paper.filled.tex:50` (title), `:75` (definition)
- **Problem**: Title introduces "AI taste"; the definition appears only in paragraph 3 of the Introduction.
- **Recommended change**: Move the "By taste, we mean…" definition into the abstract or the first paragraph of the Introduction.
- **Severity**: minor

### Issue: Audit terminology overloaded
- **Location**: `appendix/audit.filled.tex:3, 16` (section "merger-detection audit", table "100.0% agreement")
- **Problem**: "Audit" + "100.0% agreement" reads like a precision claim on classification, but the table is actually a data-quality reconciliation between `merged_by` and the timeline `MergedEvent`. Given the project's flagged history of overstated audit/precision claims, this conflation is risky even though the numbers themselves check out.
- **Recommended change**: Rename to "Merger attribution consistency check" / "Merger field reconciliation"; change caption to "Field consistency on a sample of 1,000 PRs" and explicitly state this is not a classification accuracy audit. Also: this appendix is not currently `\input`, see Cat 9.
- **Severity**: minor

### Issue: Sample-size asymmetry across RQs not led with
- **Location**: `paper.filled.tex:152, :154, :156`
- **Problem**: RQ 1 uses 3,109,979 PRs; RQ 2 uses a subset; RQ 3 uses 46,852. Each subsection should lead with its own denominator so readers can calibrate confidence.
- **Recommended change**: Open each RQ Results paragraph with the active sample: "RQ 1 (n=3,109,979 PRs): …", etc.
- **Severity**: minor

### Issue: Inconsistent vocabulary in Table 1 "Layer" column
- **Location**: Table 1, `paper.filled.tex:98–106`
- **Problem**: The "Layer" column mixes values such as "deployment choice", "interfaces, deployment choice", "model propensity" without an explicit two-value taxonomy.
- **Recommended change**: Standardise to two values ("Model propensity", "Deployment choice") or list both with `;` separator if applicable.
- **Severity**: minor

### Issue: Preference-gap symbol introduced without verbal anchor
- **Location**: `paper.filled.tex:26–27` (`\rbt` macro), `:113` (first mathematical use)
- **Problem**: The robot glyph stands in for Δ representing the preference gap, but readers meet $\rbt^H$ at line 113 with no inline verbal anchor.
- **Recommended change**: At first mathematical use, write: "We denote the preference gap by Δ (rendered ⚙), with Δ^H := U^H(…) − U^H(…)."
- **Severity**: minor

### Issue: "Participation" scope (RQ 1) not crisp
- **Location**: `paper.filled.tex:85` (RQ 1) vs `:139–140` (operationalisation)
- **Problem**: RQ 1 asks about "AI participation". Methods operationalise it as "PRs with at least one AI-authored event". The figure caption (line 148) and the abstract use slightly different wordings.
- **Recommended change**: In RQ 1, state precisely: "Has the share of PRs with any AI-authored (committed or co-authored) event increased?" Then use the same wording in figure caption + abstract.
- **Severity**: minor

### Issue: Stale appendix comment headers (note only — no `.filled.tex` edit)
- **Location**: appendix `.filled.tex` files
- **Problem**: Some files have header comments referencing old appendix letters that don't match the current order. Per CLAUDE.md, `.filled.tex` files are regenerated from `.tex`, so fixes belong in the source `.tex` files, not the `.filled.tex` ones.
- **Recommended change**: When implementing, fix the header comments in the unfilled `paper/appendix/*.tex` files (not in `.filled.tex`).
- **Severity**: nit

### Issue: Results' first sentence is the metric, not the takeaway
- **Location**: `paper.filled.tex:152` (RQ 1 paragraph), `:154` (RQ 2 paragraph)
- **Problem**: Each subsection opens with a number rather than the directional claim. Lead with the takeaway and follow with the number.
- **Recommended change**: e.g. "AI participation grew sharply over the year: from 6.4% to 22.5% of PRs (3.5×)…".
- **Severity**: nit

---

## Category 2 — Writing, Grammar & Small Edits

### Issue: Abstract is a single ~265-word sentence-block
- **Location**: `paper.filled.tex:67–68`
- **Problem**: The abstract reads as one continuous run rather than a structured 3–5 sentence summary.
- **Recommended change**: Break into 4–5 sentences: motivation → claim → method → key results → call. See Cat 5 for a fresh rewrite to draw from.
- **Severity**: major

### Issue: Asymmetric verdict-detection method not surfaced in main Results
- **Location**: `paper.filled.tex:156`
- **Problem**: Main text expands "AI-side opinion pool from 9,841 to 174,113 unique PRs" via regex parsing; humans use only native APPROVED/CHANGES_REQUESTED. This asymmetry materially affects interpretation of the −14.17 pp DiD but is acknowledged only in Appendix B / F.
- **Recommended change**: Add a clause in the RQ 3 Results paragraph: "AI verdicts are extracted from native review states *and* regex-parsed COMMENTED bodies; human verdicts use native states only. The native-only sensitivity check (Appendix F) yields wider CIs, sometimes crossing zero." Mirrors a Cat 7 issue.
- **Severity**: major

### Issue: Buried interpretation in RQ 3 Results
- **Location**: `paper.filled.tex:156`
- **Problem**: "We expand the AI-side opinion pool from 9,841 explicit reviews to 174,113 unique PRs" mixes two units (review *events* vs. unique PRs) in one phrase.
- **Recommended change**: Pick one unit: e.g. "From 9,841 native explicit-review events covering N unique PRs, we expand to 174,113 unique PRs by parsing structured verdict lines from each bot's `COMMENTED` review bodies."
- **Severity**: major

### Issue: "This effect" referent ambiguous
- **Location**: `paper.filled.tex:156` ("This effect is not influenced by human anchoring…")
- **Problem**: "This effect" could refer to the −14.17 pp DiD or to the −70.72 pp gap on AI-authored PRs.
- **Recommended change**: Replace with "The within-PR DiD" or "The headline anti-self-preference result".
- **Severity**: minor

### Issue: "AI participation" scope shifts between sections
- **Location**: `paper.filled.tex:79` (intro) vs. `:139–140` (methods rollups)
- **Problem**: Intro uses "AI participation" loosely; methods distinguishes "AI bot" (autonomous) from "AI powered" (human+AI).
- **Recommended change**: Either define both terms in the intro at first use, or note "RQ 1 reports PRs with any AI bot or AI-powered event; RQ 2 / RQ 3 use bot-only review events."
- **Severity**: minor

### Issue: Section transition Intro → Framework lacks a "why"
- **Location**: `paper.filled.tex:73–87`
- **Problem**: After contributions and RQs, the paper jumps to the formal model with no bridge.
- **Recommended change**: Add a one-sentence transition: "To operationalise these mechanisms, we develop a framework that decomposes adoption-gap dynamics into measurable accelerators."
- **Severity**: minor

### Issue: Methods paragraph dense in passive voice
- **Location**: `paper.filled.tex:139`
- **Problem**: The 184-word data-classification paragraph is mostly passive ("is classified", "is rolled up", "are excluded", "are analysed").
- **Recommended change**: Break into 4–5 short active-voice sentences.
- **Severity**: minor

### Issue: "No commits between" constraint hidden as parenthetical
- **Location**: `paper.filled.tex:140`
- **Problem**: A constraint critical for causal interpretation (same commit graph evaluated by both reviewers) appears as a parenthetical aside.
- **Recommended change**: Move to a dedicated clause: "To ensure both reviewers evaluate the same code, we restrict to PRs where the human's first explicit review occurred without intervening commits after the AI's first opinion."
- **Severity**: minor

### Issue: Verdict-extraction prose mixes abstraction levels
- **Location**: `appendix/ai_verdict_regex.filled.tex:8`
- **Problem**: Cursor's emoji rule, Claude's trailing-LGTM rule, and the `<details>` stripping policy are listed in one run-on sentence.
- **Recommended change**: Split into three sentences, ordering "what" before "why".
- **Severity**: minor

### Checked: jargon — generally accessible for an ICML audience; no major issues found.
### Checked: British English consistency — no spelling/punctuation drift detected.
### Checked: naked pronouns — only the "This effect" instance above is materially ambiguous.
### Checked: missing citations — claims that depend on prior work are cited.
### Checked: speculation marking — "may", "might", "could" used where appropriate.

---

## Category 3 — Figures

### Issue: Orphan PDF — `figure1_merge_rates.pdf` not referenced
- **Location**: `paper/figures/figure1_merge_rates.pdf`
- **Problem**: The file exists in the figures directory but is never `\includegraphics`'d in any `.tex` file (main or appendix).
- **Recommended change**: Either include it (likely as a supporting Results figure on merge rates by author × reviewer type) with a caption, or delete the PDF if it has been superseded.
- **Severity**: major

### Issue: Orphan PDF — `figure3_any_ai_share.pdf` not referenced
- **Location**: `paper/figures/figure3_any_ai_share.pdf`
- **Problem**: A weekly time-series of any-AI share — generated but not used.
- **Recommended change**: Decide: include (e.g. in `extra_figures.filled.tex`) with a caption, or delete the PDF.
- **Severity**: major

### Issue: Appendix Figure 4 caption may not match the rendered visual
- **Location**: `appendix/extra_figures.filled.tex:6`
- **Problem**: Caption describes a boxplot ("Boxes show the IQR with whiskers at 1.5×IQR. Fliers are omitted.") while the figure appears to be a stacked-bar of chain-length buckets per quarter. If the actual figure is a stacked bar, the caption is wrong.
- **Recommended change**: Verify the rendered figure type. If stacked bar, rewrite caption to: "Share of PRs by chain-length bucket (1, 2, 3–4, 5–7, 8+) per quarter." If genuinely a boxplot, regenerate so its visual elements (median, IQR box, whiskers) are clearly labelled.
- **Severity**: major

### Issue: Three distinct visual styles across main figures
- **Location**: Figures 1, 2, 2b at `paper.filled.tex:147, :160, :167`
- **Problem**: Figure 1 = overlaid line series with markers and end-of-line annotations; Figures 2 / 2b = small-multiples bars with error bars and a DiD bracket panel. The visual languages do not share encoding.
- **Recommended change**: Either align colour and marker conventions, or accept the difference and call it out in captions ("Different visual conventions: trend in Fig. 1, contrast in Fig. 2.").
- **Severity**: minor

### Issue: Figure 2b at full width despite small sample
- **Location**: `paper.filled.tex:167` (`\linewidth`)
- **Problem**: Within-family Claude check uses 49 Claude-authored PRs of 689; rendered identically in size and prominence to the headline Figure 2 (n=46,852, 2,401 AI-authored). Visual prominence implies parity of evidence.
- **Recommended change**: Either reduce its width / move to appendix, or add a clearly visible "small sample" badge in-figure.
- **Severity**: minor

### Issue: CI styling not consistent across figures
- **Location**: Figures 2, 2b, and quarterly_did appendix figure
- **Problem**: Error bars, brackets, and box-and-whisker conventions all appear; readers must re-learn the encoding.
- **Recommended change**: Pick one style for rates (error bars) and one for differences (bracket CI) and apply consistently.
- **Severity**: minor

### Issue: Figure 1 in-plot legend may be small at print size
- **Location**: `paper.filled.tex:147`
- **Problem**: Three series labels in an interior legend at ~8pt after column scaling; end-of-line annotations sit outside the axis area, adding clutter.
- **Recommended change**: Direct-label series at the right edge of the line and drop the legend, or move the legend below the plot.
- **Severity**: minor

### Issue: Figure 2b DiD CI crosses zero without visual cue
- **Location**: `paper.filled.tex:167`
- **Problem**: The DiD CI of [−8.0, +14.0] pp includes zero, but the bracket is rendered identically to Figure 2's significant CI.
- **Recommended change**: Use a dashed bracket or lighter colour for non-significant CIs.
- **Severity**: nit

### Issue: pp / pp. / "percentage points" inconsistent
- **Location**: Figures and inline prose throughout
- **Problem**: Mixed punctuation/spelling.
- **Recommended change**: Standardise on `pp` (no period) globally.
- **Severity**: nit

### Issue: Figure 1 caption does not state aggregation granularity
- **Location**: `paper.filled.tex:148`
- **Problem**: X-axis is labelled "PR opened (month)" but the caption doesn't confirm monthly aggregation.
- **Recommended change**: Add "(monthly aggregation)" to the caption.
- **Severity**: nit

### Checked: every figure has a caption — pass (incl. appendix).
### Checked: first figure tight and professional — pass.
### Checked: Figures Test (title + abstract + figures + captions tells the story) — pass for the included set, but note that the two orphan PDFs above would *strengthen* the story if included.

---

## Category 4 — Title

**Current title:** "AI taste and gradual disempowerment"

Checked: title is distinctive vs. recent empirical work in this area (e.g. "How AI Coding Agents Modify Code", "Where Do AI Coding Agents Fail", "Code Review Agents in Pull Requests", "Self-Attribution Bias", "AI AI Bias"). The combination "AI taste" + "gradual disempowerment" is unique. The two-word hook "AI taste" is memorable. **No issues found.**

---

## Category 5 — Abstract

**Current abstract (paper.filled.tex:67–68):** the published paragraph beginning "When human workers… we call on researchers to assess model propensities and interfaces for AI taste lock-in."

**Fresh rewrite (drafted from scratch):**

> AI systems are increasingly deployed both to author and to review code in critical open-source software, raising the concern that AI-favouring "taste" could entrench AI-authored output even absent a real competence advantage. We introduce a framework of five accelerators that couple AI reviewer share with AI taste authority, and assess two of them empirically across 3,109,979 pull requests in 10,000 critical open-source repositories (April 2025 – March 2026). AI participation grew 3.5× (6.4% → 22.5%) and AI's share of explicit reviews rose from 0.23% to 0.42%. **Contrary to single-turn lab predictions of AI-AI bias, on the 46,852 PRs reviewed by both AI and humans, AI bots approve AI-authored code 14.17 pp less than humans do (p < 0.0001), while also disapproving of human-authored code more than humans do.** We discuss whether the reversal reflects AI-code quality, review non-homogeneity, or methodological asymmetry, and call for systematic assessment of model propensities and deployment interfaces to pre-empt taste lock-in.

**Side-by-side analysis:**
- **Retain from current**: the 5-accelerator framework; the precise numerical anchors (3,109,979 PRs / 10,000 repos / window); the headline RQ 1 + RQ 2 growth multiples; the closing call to action.
- **Replace in current**: opening "When human workers… gradual disempowerment" → too distant from the empirical contribution; the hyperbolic "Upstream of most software… GPT-5.5 and Mythos" → reads as marketing; the categorical "**no evidence for AI-AI bias**" → factually softer than the data, which actually shows the *opposite* sign; the in-line interpretation "may reflect poor human review, non-homogeneous reviews or other methodological limitations" → belongs in Discussion, not Abstract.

### Issue: "No evidence for AI-AI bias" overstates and mis-frames
- **Location**: `paper.filled.tex:67`
- **Problem**: The data show a *reversal* (AI is less, not more, favourable to AI-authored code); "no evidence for AI-AI bias" both undersells the result and is technically inaccurate (there is evidence — it points the other way).
- **Recommended change**: State the reversal directly: "Contrary to single-turn lab predictions, AI systems approve AI-authored code 14.17 pp less than humans do (p < 0.0001)…"
- **Severity**: major

### Issue: Promotional / hyperbolic domain context
- **Location**: `paper.filled.tex:67` ("Upstream of most software… GPT-5.5 and Mythos for critical tasks")
- **Problem**: Reads as advocacy; the "upstream" framing belongs in the Introduction.
- **Recommended change**: Replace with a neutral domain anchor ("10,000 critical open-source repositories") and move the upstream framing to the Intro paragraph that already discusses criticality.
- **Severity**: minor

### Issue: Excessive in-abstract hedging
- **Location**: `paper.filled.tex:67` ("As a limited empirical illustration, rather than a test of gradual disempowerment as a whole")
- **Problem**: Useful caveat, but consumes scarce abstract real estate when the body and Discussion both already make this clear.
- **Recommended change**: Drop from the abstract; keep the equivalent caveat in the Discussion.
- **Severity**: minor

---

## Category 6 — Introduction

### Issue: Categorical RQ 3 claim in abstract contradicts intro framing
- **Location**: `paper.filled.tex:67` (Abstract: "no evidence for AI-AI bias") vs. `:85` (RQ 3) and `:156` (Result text describing reversal)
- **Problem**: Same overstatement as Cat 5 Issue 1, but the Introduction's RQ statement should also be tightened so abstract / RQ / result form a consistent chain.
- **Recommended change**: Reframe RQ 3 as open ("Do AI and human reviewers differ in approval of AI-authored PRs?") and make the abstract/intro/result use the same wording.
- **Severity**: major

### Issue: "So what" buried — funnel is inverted
- **Location**: `paper.filled.tex:70–79`
- **Problem**: The intro opens with abstract gradual-disempowerment theory + definitions; the empirical gap appears only at line 79.
- **Recommended change**: Reorder to: (1) lead with the empirical gap ("Whether AI exhibits self-preference in real software-development deployments is unknown"), (2) broaden to the gradual-disempowerment frame, (3) define terms, (4) state contributions and RQs.
- **Severity**: major

### Issue: Related work cited but not contextualised
- **Location**: `paper.filled.tex:77–79` (Laurito 2025; Watanabe 2025; Ghaleb 2026)
- **Problem**: 3–4 closely related empirical papers are cited but the explicit "they showed X in setting Y; we test setting Z" mapping is absent.
- **Recommended change**: After each citation, add one explicit-extension sentence, e.g.: "Laurito et al. (2025) found AI systems prefer AI outputs by 20–30 pp in controlled choice experiments; we test whether this bias manifests in production code review where stakes and human oversight differ."
- **Severity**: major

### Issue: Five-accelerator scope vs. RQ scope not stated upfront
- **Location**: `paper.filled.tex:82–85`
- **Problem**: Mirror of Cat 1 Issue 3 from the Intro side — the reader doesn't learn that only "the last two" accelerators are tested until line 139.
- **Recommended change**: At end of Introduction (right after RQs): "Of the five accelerators, RQ 2 measures Accelerator 4 (reviewer-share growth) and RQ 3 measures Accelerator 5 (AI-AI bias); RQ 1 provides indirect evidence on Accelerators 1–3."
- **Severity**: minor

### Issue: "Propensities" and "interfaces" used as load-bearing terms but never defined
- **Location**: `paper.filled.tex:82` ("model propensities and interfaces") and recurring usage at `:175`
- **Problem**: These are central concepts in the contribution and conclusion but never defined.
- **Recommended change**: After the existing definitions in paragraph 3 of the Intro, add: "By **model propensities** we mean trained-in behavioural tendencies; by **interfaces** we mean the deployment / UX choices that shape model–human interaction."
- **Severity**: minor

### Checked: RQs are stated and addressed in Results — pass.
### Checked: no literature-review duplication later in the paper — pass.

---

## Category 7 — Results

### Issue: AI-vs-human verdict-detection asymmetry not surfaced in main Results
- **Location**: `paper.filled.tex:156`
- **Problem**: The headline DiD uses regex-parsed `COMMENTED` bodies for AI but only native review states for humans. The native-only sensitivity check (Appendix F / `quarterly_did.filled.tex`) shows non-monotone DiDs whose CIs cross zero in some quarters. This caveat is critical for interpreting the −14.17 pp result and currently lives only in the appendix.
- **Recommended change**: Add a single sentence at the end of the RQ 3 Results paragraph: "AI verdicts are extracted from native review states *and* regex-parsed `COMMENTED` bodies; human verdicts use native states only. A symmetric native-only check (Appendix F) yields wider CIs that cross zero in some quarters."
- **Severity**: major

### Issue: Per-cell Wilson 95% CIs not reported in main RQ 3 paragraph
- **Location**: `paper.filled.tex:156` and Figure 2 caption at `:161`
- **Problem**: Methods §E specifies Wilson 95% CIs on each cell, but the main paragraph reports the four cell rates (22.32%, 93.04%, 93.88%, 96.72%) without their CIs. The 22.32% AI-authored cell is the precision-critical one (n=2,401 of 46,852).
- **Recommended change**: Either inline the four CIs in the paragraph or add a small two-row table; at minimum, add the AI-authored CI parenthetically.
- **Severity**: minor

### Issue: Figure 2 caption conflates dual-review cohort size with AI-opinion-pool size
- **Location**: `paper.filled.tex:161`
- **Problem**: Caption: "Within-PR AI-AI bias DiD on the 46,852 PRs in the dual-review cohort". The AI approval rates that feed the DiD's left panel come from the expanded 174,113-PR verdict pool, not the 46,852 cohort directly.
- **Recommended change**: Clarify: "DiD computed on the 46,852 dual-review PRs; AI verdicts within those PRs come from native review states + regex-parsed `COMMENTED` bodies (174,113-PR pool universe), human verdicts from native states only."
- **Severity**: minor

### Issue: Within-family Claude cell underpowered, not flagged in main text
- **Location**: `paper.filled.tex:156` (the +2.99 pp / p=0.7533 line)
- **Problem**: Estimate rests on 49 Claude-authored vs. 640 human-authored PRs; quarterly numbers (e.g. 2025-Q4 +20 pp on 5 PRs) make the underpowered nature obvious.
- **Recommended change**: Append: "(based on 49 Claude-authored PRs; estimate is underpowered and should be interpreted only as a directional indication)."
- **Severity**: minor

### Checked: each subsection leads with key result (with the lead-with-takeaway nit logged in Cat 1) — pass.
### Checked: every Results metric has a corresponding Methods description — pass with one carry-over (174,113 universe explanation, see Cat 8).
### Checked: claims tightly bounded by data — pass; appropriate hedging in place.
### Checked: no fabricated precision/audit/100% claims in Results — verified, none found.

---

## Category 8 — Methods

### Issue: 174,113-PR "unique PRs" universe is introduced without bridging to other denominators
- **Location**: `paper.filled.tex:156` and `appendix/methods.filled.tex`
- **Problem**: Reader must hold three figures in mind — 9,841 native explicit reviews, ~70,000 regex-parsed `COMMENTED` reviews, 46,852 dual-review cohort — and is then handed a fourth, 174,113, with no explicit derivation.
- **Recommended change**: Add to Methods (main or appendix): "Combining 9,841 native explicit reviews and ~70,000 `COMMENTED` reviews with parseable verdicts yields 174,113 unique PRs that received at least one AI verdict. The dual-review cohort (RQ 3) is the subset of those PRs that also received a human explicit review with no intervening commits (n = 46,852)."
- **Severity**: major

### Checked: data-cleaning sequence (`repo_selection.filled.tex` Steps 1–5) — clear and ordered.
### Checked: regex (`ai_verdict_regex.filled.tex`) and allowlist (`allowlist.filled.tex`) — sufficient detail to regenerate.
### Checked: repo selection consistent between main text and appendix — pass.
### Checked: replicability parameters (Wilson CIs, HC0 SEs, logistic spec, "no commits between" filter) — documented.
### Checked: no fabricated audit/precision claims in Methods — verified, none found.
### Checked: page-limit discipline — Methods main text is lean (~302 words); detail correctly deferred to appendices A–E.

---

## Category 9 — Annex / Appendix

### Issue: Five appendix files exist but are not `\input` from the main paper
- **Location**: `paper/appendix/`: `audit.filled.tex`, `examples.filled.tex`, `old_repo_selection.filled.tex`, `regression.filled.tex`, `threats.filled.tex`
- **Problem**: These files are present in the appendix folder but never `\input{...}` from `paper.filled.tex`. `old_repo_selection` is especially risky — it documents a *superseded* methodology (500 repos via star search) while the paper actually uses the 10,000-repo criticality-score method in `repo_selection.filled.tex`.
- **Recommended change**: Decide for each file: keep (and `\input` it) or delete. At minimum, delete `old_repo_selection.filled.tex` (and its source `.tex`). For the others, either remove from the repo or add a top-of-file `% NOT INCLUDED IN PUBLISHED APPENDIX — ARCHIVED` banner. (The CLAUDE.md note about `.filled.tex` being regenerated implies edits should land in the source `.tex` files.)
- **Severity**: major

### Issue: Stale comment header in `old_repo_selection.filled.tex`
- **Location**: `appendix/old_repo_selection.filled.tex:1`
- **Problem**: Header says "Appendix E — Repository-selection pipeline (replication)", but this file is not rendered. If a reader stumbles on it, they will think it documents the live methodology.
- **Recommended change**: If keeping, mark "DEPRECATED — see `repo_selection.filled.tex` for current methodology"; if deleting (recommended), no further action.
- **Severity**: major

### Issue: Appendix Figure D ("chain-length distribution") not explicitly cited from main text
- **Location**: `appendix/extra_figures.filled.tex:3` (`fig:chainbox`)
- **Problem**: The appendix section is rendered but not referenced via `\Cref{fig:chainbox}` from the RQ 1 paragraph in the main text.
- **Recommended change**: Add a parenthetical to the RQ 1 paragraph (around `paper.filled.tex:152`): "(see Appendix D, Fig. \ref{fig:chainbox}, for the quarterly distribution)."
- **Severity**: minor

### Issue: `audit.filled.tex` orphaned and easy to misread as a precision audit
- **Location**: `appendix/audit.filled.tex` and `results/merger_detection_audit.md`
- **Problem**: The file is not `\input`, but if it were, the section title "merger-detection audit" + the "100.0% agreement" cell would read as a hand-audit precision claim. The numbers themselves are reproducible from `merger_detection_audit.md` and refer to a `merged_by` vs. timeline-actor consistency check, not a classification audit.
- **Recommended change**: If keeping the file: rename section to "Merger-attribution consistency check", rephrase caption ("Field consistency on a sample of 1,000 PRs"), drop the standalone "100.0%" without context. If not keeping: delete the file.
- **Severity**: minor

### Checked: every rendered appendix table has a caption and fits within the page (allowlist, ai_verdict_regex, quarterly_did) — pass.
### Checked: every rendered appendix figure has a caption (extra_figures fig:chainbox, quarterly_did fig and table) — pass.
### Checked: appendix font sizes (`footnotesize` / `small`) consistent with venue style — pass.
### Checked: appendix sections referenced from main text where relevant — pass except the chain-length figure (issue above).
### Checked: no fabricated audit/precision/100% claims survive in rendered appendices — verified. The `audit.filled.tex` "100.0%" cell is sourced from `results/merger_detection_audit.md` and is real, but framing risks remain (see issue above).

---

## Notes on out-of-scope items (not edited)

- The auto-memory rule "iterative paper edits land in `.filled.tex` first" applies when implementation begins; this review only points at issues. When a human authorises edits, both `paper.filled.tex` and `paper.tex` (and `*.tex` siblings) must be kept in sync.
- The auto-memory rule "proactively grep for hand-audit/precision/100% claims and flag" was applied across all categories; the only surviving issue is the framing risk in `audit.filled.tex` (orphaned) and the legacy "audit" terminology in the main-text appendix references. No new fabricated claims were detected in the current `.filled.tex` set.
