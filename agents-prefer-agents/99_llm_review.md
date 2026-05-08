# LLM Review of `paper.filled.tex`

Read-only review against `agents-prefer-agents/SI_review_prompt.md`. Findings only — **no source files have been edited**. Implementation is gated on human review.

## Top findings (most important)

5. **Two orphaned figure PDFs and several orphaned appendix files.** `figure1_merge_rates.pdf` and `figure3_any_ai_share.pdf` are generated but never `\includegraphics`'d. `old_repo_selection.filled.tex`, `audit.filled.tex`, `examples.filled.tex`, `regression.filled.tex`, `threats.filled.tex` are not `\input` from `paper.filled.tex` — `old_repo_selection` in particular describes a superseded methodology and risks confusion. (Cat 3 / Cat 9)
--> delete these orphaned files
**Implementation:** Deleted both `.tex` and `.filled.tex` twins for `old_repo_selection`, `audit`, `examples`, `regression`, `threats`; deleted `figure1_merge_rates.pdf` and `figure3_any_ai_share.pdf`. No doubt.

6. **Terminology drift on the central measurement.** "AI reviewed", "AI opinion", "AI review", "explicit AI review", "AI bot general-review opinion" are used interchangeably in the main text without a single canonical definition; "general review" is used in Results without ever being defined in the main text. (Cat 1)
--> replace 'opinion' with 'review'. Check all mentions of 'review': It should already be introduced in methods as 'explicit review' for an approve/request-changes event vs. 'comment review' (for comments with concrete reject/approve statements). Make sure this 'comment review' is properly introduced. Then drop any mentions of general review, and instead use 'explicit or comment review', and similar. 
**Implementation:** Swept 'opinion' → 'review' across all `.tex` / `.filled.tex` (verified by grep — zero residuals). Introduced bold-faced definitions of `explicit review` *and* `comment review` in main Methods (paper line 139), and aligned the corresponding paragraph in `appendix/methods*.tex` (replacing the old "general review" definition). 'general review' fully removed; 'AI opinion pool' → 'AI review pool' in `appendix/quarterly_did*.tex`; 'no-opinion' → 'no-verdict' in `appendix/ai_verdict_regex*.tex`.
**Doubt (critical reviewer):** I introduced both definitions in *main Methods* in addition to the appendix. Your instruction "It should already be introduced in methods" was ambiguous between main vs. appendix Methods; I went heavier (both). Cost is ~+25 words in the page-tight main Methods. If you want main Methods leaner, the bold-faced defs can move to appendix only.


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
-> reframe as '("Do AI and human reviewers differ in their approval of AI-authored and human-authored PRs?")
**Implementation:** RQ 3 in the intro now reads exactly that.
**Doubt (critical reviewer):** ⚠️ The *Results subsection title* at line 156 still reads "AI systems dislike AI-authored code more than humans do" — directional, and now mismatched against the neutral RQ. You did not ask to change it; flagging because abstract / RQ / result-title chain is now neutral / neutral / directional.

### Issue: Ambiguous mapping between accelerators and RQs
- **Location**: `paper.filled.tex:85` (RQs), `:87–109` (Table 1 / framework), `:139` ("To illustrate how to measure the last two of the five accelerators")
- **Problem**: Table 1 lists five accelerators; only "the last two" are measured, and that scoping appears for the first time in Methods. Readers cannot tell which accelerators each RQ tests, supports, or refutes.
- **Recommended change**: In the Introduction or at the start of Methods, state explicitly: "RQ 1 provides indirect evidence on Accelerators 1–3, but we do not test them directly. Instead we focus on measuring Acclerator 4 (RQ2) and 5 (RQ3)". 
- **Severity**: major
-> insert as a second sentence of the methods: "We do not measure accelerators 1-3 directly", and in the first sentence say "mea-
sure accelerator four (RQ2) and accelerator five (RQ3)"
**Implementation:** Methods line 139 now opens "To measure accelerator four (RQ 2) and accelerator five (RQ 3), we analyse all 3,109,979 PRs..." with the second sentence "We do not measure accelerators 1–3 directly."
**Doubt:** None.

### Issue: Terminology drift on the core measurement (AI opinion / AI reviewed / AI review)
- **Location**: `paper.filled.tex:139` ("AI reviewed"), `:140` ("AI opinion"), `:154` ("explicit AI review"), `:156` ("AI bot general-review opinion"); `appendix/methods.filled.tex:3` (defines "explicit review" vs. "general review")
- **Problem**: The same construct is referred to with at least four different names across main text and appendix; "general review" is defined in Appendix E but used in the main Results section with no definition.
- **Recommended change**: Adopt one canonical scheme. Suggested: "explicit AI review" = native APPROVED/CHANGES_REQUESTED; "AI opinion" = explicit + regex-parsed verdicts; drop "AI reviewed" as a label (use only as the English participle). Define both terms once in Methods main text and use them consistently.
- **Severity**: major
-> see above
**Implementation:** Covered by the top-finding-6 sweep. Canonical scheme: `explicit review` (native APPROVED/CHANGES_REQUESTED) vs. `comment review` (regex-parseable verdict in COMMENTED body or PR-thread issue_comment); `explicit or comment review` is the union phrase.
**Doubt:** None beyond the ambiguity flagged in top-finding-6.

### Issue: "General review" used in main Results without prior definition
- **Location**: `paper.filled.tex:156` ("AI bot general-review opinion")
- **Problem**: A reader of the main text encounters "general-review" without ever having seen it defined. The definition lives only in `appendix/methods.filled.tex:3`.
- **Recommended change**: Either define "general review" inline at first use in Methods (line 139–140) or replace it with the canonical term chosen above.
- **Severity**: major
-> see above, replace it
**Implementation:** Replaced everywhere. Main Results no longer references "general review"; appendix Methods replaced its "general review" definition with the new "comment review" definition.
**Doubt:** None.

### Issue: Audit terminology overloaded
- **Location**: `appendix/audit.filled.tex:3, 16` (section "merger-detection audit", table "100.0% agreement")
- **Problem**: "Audit" + "100.0% agreement" reads like a precision claim on classification, but the table is actually a data-quality reconciliation between `merged_by` and the timeline `MergedEvent`. Given the project's flagged history of overstated audit/precision claims, this conflation is risky even though the numbers themselves check out.
- **Recommended change**: Rename to "Merger attribution consistency check" / "Merger field reconciliation"; change caption to "Field consistency on a sample of 1,000 PRs" and explicitly state this is not a classification accuracy audit. Also: this appendix is not currently `\input`, see Cat 9.
- **Severity**: minor
-> I cannot find this? delete any 100% agreement audit style paragraphs, i think they should be deleted
**Implementation:** The "100% agreement" content lived only in `audit.filled.tex` / `audit.tex`, both already deleted as part of the orphan sweep (top finding 5). Grep confirms no other "100% agreement" or "audit" residuals.
**Doubt:** None — your "I cannot find this?" was correct: the file was already gone after the orphan deletion.

### Issue: Sample-size asymmetry across RQs not led with
- **Location**: `paper.filled.tex:152, :154, :156`
- **Problem**: RQ 1 uses 3,109,979 PRs; RQ 2 uses a subset; RQ 3 uses 46,852. Each subsection should lead with its own denominator so readers can calibrate confidence.
- **Recommended change**: Open each RQ Results paragraph with the active sample: "RQ 1 (n=3,109,979 PRs): …", etc.
- **Severity**: minor
-> implement this as brackets, that's a great idea!
**Implementation:** Each RQ paragraph title now carries the bracket: `RQ 1 (n=3,109,979 PRs)`, `RQ 2 (n=3,109,979 PRs)`, `RQ 3 (n=46,852 PRs)`.
**Doubt (critical reviewer):** I used n=3,109,979 for RQ 2 because the *rate* is computed over the full universe (share of all PRs that have an explicit AI review). A stricter reviewer might prefer the AI-participation sub-universe (~700k) since RQ 2 is conceptually about reviewer-share within AI-touched workflows. Honest-but-coarse choice; flag if you'd rather report a tighter denominator.

### Issue: Inconsistent vocabulary in Table 1 "Layer" column
- **Location**: Table 1, `paper.filled.tex:98–106`
- **Problem**: The "Layer" column mixes values such as "deployment choice", "interfaces, deployment choice", "model propensity" without an explicit two-value taxonomy.
- **Recommended change**: Standardise to two values ("Model propensity", "Deployment choice") or list both with `;` separator if applicable.
- **Severity**: minor
-> keep the current content of the table, but in the table description briefly say: "The layer describes the source and mitigation of an accelerator. A Model's propensity can be trained to reduce or increase the tendency to value its own outputs. Interfaces like websites can provide only LLM-optimised content or both human and LLM-compatible sites. Deployment choices, like the interoperability of coding agents with common office software like excel affects switching costs."
**Implementation:** Caption (line 92) now says: "The 'Layer' column describes the source and mitigation of an accelerator: a model's propensity can be trained to reduce or increase the tendency to value its own outputs; interfaces (e.g. websites) can provide only LLM-optimised content or both human- and LLM-compatible content; and deployment choices (e.g. the interoperability of coding agents with common office software like Excel) affect switching costs."
**Doubt (critical reviewer):** You said "briefly say". My implementation is a single semicolon-joined sentence (~50 words). Tighter than three separate sentences but not "brief" by paper-caption standards. Light edits to your text: "Model" lower-cased, "excel" → "Excel", joined with semicolons. If you want it tighter, this can compress to two clauses.

### Issue: Preference-gap symbol introduced without verbal anchor
- **Location**: `paper.filled.tex:26–27` (`\rbt` macro), `:113` (first mathematical use)
- **Problem**: The robot glyph stands in for Δ representing the preference gap, but readers meet $\rbt^H$ at line 113 with no inline verbal anchor.
- **Recommended change**: At first mathematical use, write: "We denote the preference gap by Δ (rendered ⚙), with Δ^H := U^H(…) − U^H(…)."
- **Severity**: minor
-> implement the rec
**Implementation:** At first mathematical use (line 113): "the human's *preference gap* for AI (denoted $\rbt$, a robot glyph in place of $\Delta$) is $\rbt^H := U^H(q_A^H, c_A) - U^H(q_H^H, c_H)$."
**Doubt:** I used the macro convention ($\rbt$ + $\Delta$) rather than the literal Unicode glyph; functionally identical at render time.


### Issue: "Participation" scope (RQ 1) not crisp
- **Location**: `paper.filled.tex:85` (RQ 1) vs `:139–140` (operationalisation)
- **Problem**: RQ 1 asks about "AI participation". Methods operationalise it as "PRs with at least one AI-authored event". The figure caption (line 148) and the abstract use slightly different wordings.
- **Recommended change**: In RQ 1, state precisely: "Has the share of PRs with any AI-authored (committed or co-authored) event increased?" Then use the same wording in figure caption + abstract.
- **Severity**: minor
-> use in figure caption 'participation (PRs with at least one AI-authored event'), keep current RQ1
**Implementation:** Figure 1 caption (line 148) now opens "Monthly AI participation (PRs with at least one AI-authored event) in GitHub PRs across 10,000 critical OS software repositories (monthly aggregation, Apr 2025–Mar 2026)." RQ 1 in intro and RQ 1 paragraph title left as-is per your instruction.
**Doubt:** None.


### Issue: Stale appendix comment headers (note only — no `.filled.tex` edit)
- **Location**: appendix `.filled.tex` files
- **Problem**: Some files have header comments referencing old appendix letters that don't match the current order. Per CLAUDE.md, `.filled.tex` files are regenerated from `.tex`, so fixes belong in the source `.tex` files, not the `.filled.tex` ones.
- **Recommended change**: When implementing, fix the header comments in the unfilled `paper/appendix/*.tex` files (not in `.filled.tex`).
- **Severity**: nit
-> implement the rec
**Implementation:** Updated `% Appendix X — …` header comments in all 7 active appendix files (and their `.filled.tex` twins, since both were wrong) to match the actual letter mapping after orphan deletion: A=allowlist, B=ai_verdict_regex, C=repo_selection, D=extra_figures, E=methods, F=anchoring_robustness, G=quarterly_did.
**Doubt (critical reviewer):** CLAUDE.md says "Treat the comment as flavour" — these header comments are non-rendered, so the change has zero output impact. Did it because you asked. Low-value cosmetic churn.



### Issue: Results' first sentence is the metric, not the takeaway
- **Location**: `paper.filled.tex:152` (RQ 1 paragraph), `:154` (RQ 2 paragraph)
- **Problem**: Each subsection opens with a number rather than the directional claim. Lead with the takeaway and follow with the number.
- **Recommended change**: e.g. "AI participation grew sharply over the year: from 6.4% to 22.5% of PRs (3.5×)…".
- **Severity**: nit
-> implement the rec
**Implementation:** RQ 1 now opens "AI participation grew sharply over the year: from 6.4% of PRs in April 2025 to 22.5% in March 2026 (3.5×; Figure 1)." RQ 2 now opens "AI's share as reviewer rose over the year: PRs with an explicit AI review grew from 0.23% to 0.42%."
**Doubt:** None.

---

## Category 2 — Writing, Grammar & Small Edits


### Issue: Asymmetric verdict-detection method not surfaced in main Results
- **Location**: `paper.filled.tex:156`
- **Problem**: Main text expands "AI-side opinion pool from 9,841 to 174,113 unique PRs" via regex parsing; humans use only native APPROVED/CHANGES_REQUESTED. This asymmetry materially affects interpretation of the −14.17 pp DiD but is acknowledged only in Appendix B / F.
- **Recommended change**: Add a clause in the RQ 3 Results paragraph: "AI verdicts are extracted from native review states *and* regex-parsed COMMENTED bodies; human verdicts use native states only. The native-only sensitivity check (Appendix F) yields wider CIs, sometimes crossing zero." Mirrors a Cat 7 issue.
- **Severity**: major
-> implement it but with "For sample size reasons, AI reviews are extracted from explicit reviews and regex-parsed comment reviews while human reviews are from explicit reviews only. Appendix F shows that explicit review-only comparisons are underpowered."
(Ensure that this is the only reference to appendix F in the results section to not duplicate and remove existing if needed)
**Implementation:** Inserted your exact sentence in the RQ 3 paragraph using `\Cref{app:quarterly-did}` so the appendix letter renders correctly via cleveref. Removed the existing duplicate ref ("The explicit-review-only sample is too small to replicate this finding…"); the new sentence subsumes that point and also handles the within-family pointer (Cat 3 Fig 2b move).
**Doubt (critical reviewer) — flagging:** ⚠️ You wrote "Appendix F" literally, but in the current appendix order quarterly_did is **Appendix G** (mapping is now A=allowlist, B=ai_verdict_regex, C=repo_selection, D=extra_figures, E=methods, F=anchoring_robustness, G=quarterly_did). At render time the sentence will read "Appendix G shows…" not "Appendix F shows…". Two ways to align: (a) leave it (renders "Appendix G"); (b) reorder so quarterly_did comes before anchoring_robustness, making it Appendix F. Need your call.


### Issue: Buried interpretation in RQ 3 Results
- **Location**: `paper.filled.tex:156`
- **Problem**: "We expand the AI-side opinion pool from 9,841 explicit reviews to 174,113 unique PRs" mixes two units (review *events* vs. unique PRs) in one phrase.
- **Recommended change**: Pick one unit: e.g. "From 9,841 native explicit-review events covering N unique PRs, we expand to 174,113 unique PRs by parsing structured verdict lines from each bot's `COMMENTED` review bodies."
- **Severity**: major
-> use 'From xxx PRs with explicit reviews to xxx PRs with comment reviews or explicit reviews'
**Implementation:** RQ 3 paragraph now reads "We expand the AI-side review pool from 9,841 PRs with explicit reviews to 174,113 PRs with comment reviews or explicit reviews by parsing structured verdict lines from each bot's COMMENTED review bodies." Both numbers verified against pipeline: 9,841 = `n_native_unique_prs` (PRs with at least one native explicit AI review); 174,113 = `n_unique_prs` (PRs with any AI verdict).
**Doubt:** None.

### Issue: "This effect" referent ambiguous
- **Location**: `paper.filled.tex:156` ("This effect is not influenced by human anchoring…")
- **Problem**: "This effect" could refer to the −14.17 pp DiD or to the −70.72 pp gap on AI-authored PRs.
- **Recommended change**: Replace with "The within-PR DiD" or "The headline anti-self-preference result".
- **Severity**: minor
-> use 'the within-PR DiD result... '
**Implementation:** Replaced — sentence now reads "The within-PR DiD result is not influenced by human anchoring on AI reviews…"
**Doubt:** None.

### Issue: Section transition Intro → Framework lacks a "why"
- **Location**: `paper.filled.tex:73–87`
- **Problem**: After contributions and RQs, the paper jumps to the formal model with no bridge.
- **Recommended change**: Add a one-sentence transition: "To operationalise these mechanisms, we develop a framework that decomposes adoption-gap dynamics into measurable accelerators."
- **Severity**: minor
-> end of intro should be this sentence added: 'To operationalise these mechanisms, we first derive a theoretical framework.'
**Implementation:** Appended verbatim to end of line 85 (right after the RQ 3 statement, immediately before the `\section{Theoretical framework}` heading).
**Doubt:** None.

### Issue: Methods paragraph dense in passive voice
- **Location**: `paper.filled.tex:139`
- **Problem**: The 184-word data-classification paragraph is mostly passive ("is classified", "is rolled up", "are excluded", "are analysed").
- **Recommended change**: Break into 4–5 short active-voice sentences.
- **Severity**: minor
-> turn all into active voice sentences without loosing detail. Keep the sentences short indeed.
**Implementation:** Rewrote the data-classification paragraph in active voice — 8 short sentences: "To measure accelerator four (RQ 2) and accelerator five (RQ 3), we analyse… We do not measure accelerators 1–3 directly. … We classify each actor–event pair… We pool AI bot + AI powered events as… For AI reviews we further distinguish… Humans rarely produce comment reviews, so… RQ 1 chain-length and RQ 2 explicit-review metrics use… RQ 1 participation and RQ 3 authorship-review metrics combine…"
**Doubt (critical reviewer):** the paragraph grew slightly (~+20 words) because the new explicit/comment review definitions (top finding 6) live here. If page count is tight, those defs can move to appendix Methods only — see top-finding-6 doubt.

---

## Category 3 — Figures

### Issue: Appendix Figure 4 caption may not match the rendered visual
- **Location**: `appendix/extra_figures.filled.tex:6`
- **Problem**: Caption describes a boxplot ("Boxes show the IQR with whiskers at 1.5×IQR. Fliers are omitted.") while the figure appears to be a stacked-bar of chain-length buckets per quarter. If the actual figure is a stacked bar, the caption is wrong.
- **Recommended change**: Verify the rendered figure type. If stacked bar, rewrite caption to: "Share of PRs by chain-length bucket (1, 2, 3–4, 5–7, 8+) per quarter." If genuinely a boxplot, regenerate so its visual elements (median, IQR box, whiskers) are clearly labelled.
- **Severity**: major
-> indeed, this is just a stacked bar. Keep a super concise caption of it
**Implementation:** Caption now reads: "Share of PRs by longest AI→AI chain-length bucket per quarter. The tail of long chains grows over time."
**Doubt (critical reviewer):** I dropped both the bucket labels (1, 2, 3–4, 5–7, 8+) AND the quantitative observation about p95 tripling, in the name of "super concise". 16 words vs. originally ~80. The bucket labels in particular would help a reader interpret the figure without inspecting it carefully — consider adding back "(buckets: 1, 2, 3–4, 5–7, 8+)" if you don't think it bloats the caption.

### Issue: Figure 2b at full width despite small sample
- **Location**: `paper.filled.tex:167` (`\linewidth`)
- **Problem**: Within-family Claude check uses 49 Claude-authored PRs of 689; rendered identically in size and prominence to the headline Figure 2 (n=46,852, 2,401 AI-authored). Visual prominence implies parity of evidence.
- **Recommended change**: Either reduce its width / move to appendix, or add a clearly visible "small sample" badge in-figure.
- **Severity**: minor
-> Move this figure to the RQ3 Extension annex as it is now. Only keep a small description of the claude results in the results section, pointing to the annex.
**Implementation:** Removed the full Fig 2b `\begin{figure*}…\label{fig:withinfamily}\end{figure*}` block from main paper. Inserted the same block (with placeholders / literals) in `appendix/quarterly_did.tex` and `.filled.tex` between the existing native-pool figure and the quarterly DiD table. Main paper retains a single-sentence Claude pointer in RQ 3 ("A within-family check on Claude-based reviewers vs Claude-authored PRs (689 PRs, only 49 Claude-authored) is direction-consistent with a small in-family preference but statistically indistinguishable from zero (DiD = +2.99 pp, p=0.7533). The estimate is underpowered and should be interpreted only as a directional indication.").
**Doubt (critical reviewer):** ⚠️ The within-family figure (`fig:withinfamily`) is no longer cross-referenced from main paper. A reader gets to the figure only by following the Cat 2 asymmetry pointer ("Appendix F shows that explicit review-only comparisons are underpowered and reports the within-family Claude check"). That's a section-level pointer, not a figure-level one. If you want a clean cross-reference, I can add `(Figure~\ref{fig:withinfamily})` to the within-family description sentence — but that would arguably violate your "only one reference to appendix F" rule depending on interpretation.


### Issue: pp / pp. / "percentage points" inconsistent
- **Location**: Figures and inline prose throughout
- **Problem**: Mixed punctuation/spelling.
- **Recommended change**: Standardise on `pp` (no period) globally.
- **Severity**: nit
-> implement rec
**Implementation:** Changed "less than one percentage point" → "less than 1 pp" in `appendix/anchoring_robustness*.tex` (only "percentage point" prose occurrence). Verified by grep: no `pp.` (abbreviation form) anywhere; all "pp" usages are bare `~pp`. Sentence-ending periods after "pp" remain because they're normal English punctuation, not abbreviation periods.
**Doubt:** None.

### Issue: Figure 1 caption does not state aggregation granularity
- **Location**: `paper.filled.tex:148`
- **Problem**: X-axis is labelled "PR opened (month)" but the caption doesn't confirm monthly aggregation.
- **Recommended change**: Add "(monthly aggregation)" to the caption.
- **Severity**: nit
-> implement rec
**Implementation:** Caption now contains "(monthly aggregation, Apr 2025–Mar 2026)".
**Doubt:** None.

---


---

## Category 6 — Introduction

### Issue: Categorical RQ 3 claim in abstract contradicts intro framing
- **Location**: `paper.filled.tex:67` (Abstract: "no evidence for AI-AI bias") vs. `:85` (RQ 3) and `:156` (Result text describing reversal)
- **Problem**: Same overstatement as Cat 5 Issue 1, but the Introduction's RQ statement should also be tightened so abstract / RQ / result form a consistent chain.
- **Recommended change**: Reframe RQ 3 as open ("Do AI and human reviewers differ in approval of AI-authored PRs?") and make the abstract/intro/result use the same wording.
- **Severity**: major
-> yes do it as above
**Implementation:** Removed the contradictory sentence "we find no AI-AI bias." from the abstract. Abstract now states the finding directly: "Contrary to single-turn lab findings, on the 46,852 PRs reviewed by both AI and humans, AI systems approve AI-authored code -14.17 pp (p<0.0001) less."
**Doubt (critical reviewer) — flagging:** ⚠️ The Discussion/Conclusion section (line 176) still says "We find that AI agents increasingly participate in and review critical open-source software development, but do not exhibit AI-AI bias." That sentence has the same overstatement problem as the original abstract — it claims absence of AI-AI bias when the data show *anti*-self-preference. You did not annotate it, so I left it untouched, but flagging because it's analogous to the abstract claim and would normally be aligned together.

### Issue: Related work cited but not contextualised
- **Location**: `paper.filled.tex:77–79` (Laurito 2025; Watanabe 2025; Ghaleb 2026)
- **Problem**: 3–4 closely related empirical papers are cited but the explicit "they showed X in setting Y; we test setting Z" mapping is absent.
- **Recommended change**: After each citation, add one explicit-extension sentence, e.g.: "Laurito et al. (2025) found AI systems prefer AI outputs by 20–30 pp in controlled choice experiments; we test whether this bias manifests in production code review where stakes and human oversight differ."
- **Severity**: major
-> change this sentence Whether
such biases manifest in real deployments is an open empiri-
cal question to 'Whether such biases manifest in real-world deployments is an open empirical question, which we explore here'.
**Implementation:** Replaced verbatim: "Whether such biases manifest in real-world deployments is an open empirical question, which we explore here."
**Doubt (critical reviewer) — flagging:** ⚠️ The original review issue was broader than this one bridging-sentence rewrite. The major-severity finding was that Laurito / Watanabe / Ghaleb each need an "extension" sentence ("they did X in setting Y; we test Z"). Your `->` only directs the bridge sentence change, so I did the lighter thing. The bigger contextualization gap (3–4 explicit empirical-paper extension sentences) is *not* addressed — let me know if you want me to add those.

---

### Issue: Figure 2 caption conflates dual-review cohort size with AI-opinion-pool size
- **Location**: `paper.filled.tex:161`
- **Problem**: Caption: "Within-PR AI-AI bias DiD on the 46,852 PRs in the dual-review cohort". The AI approval rates that feed the DiD's left panel come from the expanded 174,113-PR verdict pool, not the 46,852 cohort directly.
- **Recommended change**: Clarify: "DiD computed on the 46,852 dual-review PRs; AI verdicts within those PRs come from native review states + regex-parsed `COMMENTED` bodies (174,113-PR pool universe), human verdicts from native states only."
- **Severity**: minor
-> pls double check the 46k vs. 174k total number with the pipeline and ask me what you should do!
**Implementation:** Verified from `results/within_pr_stats.json`: `n_native_unique_prs=9841`, `n_unique_prs=174113`, dual-review cohort=46852. Asked you in chat with three caption-style options; you picked option 2 (show both numbers in caption with derivation). Figure 2 caption now reads: "Within-PR AI-AI bias DiD on 46,852 dual-review PRs (Any AI & human) — the subset of 174,113 PRs with any AI explicit or comment review that also received a human explicit review with no commits between the two."
**Doubt:** None.

### Issue: Within-family Claude cell underpowered, not flagged in main text
- **Location**: `paper.filled.tex:156` (the +2.99 pp / p=0.7533 line)
- **Problem**: Estimate rests on 49 Claude-authored vs. 640 human-authored PRs; quarterly numbers (e.g. 2025-Q4 +20 pp on 5 PRs) make the underpowered nature obvious.
- **Recommended change**: Append: "(based on 49 Claude-authored PRs; estimate is underpowered and should be interpreted only as a directional indication)."
- **Severity**: minor
-> implement 'The estimate is underpowered and should be interpreted only as a directional indication.'
**Implementation:** Appended verbatim to the within-family check sentence in RQ 3 ("…statistically indistinguishable from zero (DiD = +2.99 pp, p=0.7533). The estimate is underpowered and should be interpreted only as a directional indication.").
**Doubt:** None.


## Category 8 — Methods

### Issue: 174,113-PR "unique PRs" universe is introduced without bridging to other denominators
- **Location**: `paper.filled.tex:156` and `appendix/methods.filled.tex`
- **Problem**: Reader must hold three figures in mind — 9,841 native explicit reviews, ~70,000 regex-parsed `COMMENTED` reviews, 46,852 dual-review cohort — and is then handed a fourth, 174,113, with no explicit derivation.
- **Recommended change**: Add to Methods (main or appendix): "Combining 9,841 native explicit reviews and ~70,000 `COMMENTED` reviews with parseable verdicts yields 174,113 unique PRs that received at least one AI verdict. The dual-review cohort (RQ 3) is the subset of those PRs that also received a human explicit review with no intervening commits (n = 46,852)."
- **Severity**: major
-> same as above, check the pipeline for the actual dual-review cohort N and then give me options what to report. 
**Implementation:** Verified pipeline counts (n=46,852 dual-review cohort; 174,113 PRs with any AI verdict). Asked you with three options; you picked option A ("one terse sentence in main Methods"). **Initial pass had a bug** — I wove the structural definition into the dual-review cohort sentence but dropped the literal numbers. Fixed: Methods sentence (line 140) now reads "the subset of 174,113 PRs with any AI explicit or comment review (App B) that also received a human explicit review with no commits pushed between the AI's first review and the human's first explicit review (n=46,852 PRs)." Both numbers are now in Methods as your option A specified.
**Doubt (critical reviewer):** With this fix, n=46,852 now appears three times in main paper (Methods sentence, RQ 3 paragraph title, Figure 2 caption); 174,113 appears twice (Methods sentence, Figure 2 caption). That redundancy is the natural consequence of asking for derivations both upstream (Methods) and at report points (Results / Figure caption). If it bothers you, "n=46,852 PRs" can be dropped from the Methods sentence (the RQ 3 paragraph title already carries it) — or removed from the Figure 2 caption.


## Category 9 — Annex / Appendix

### Issue: Five appendix files exist but are not `\input` from the main paper
- **Location**: `paper/appendix/`: `audit.filled.tex`, `examples.filled.tex`, `old_repo_selection.filled.tex`, `regression.filled.tex`, `threats.filled.tex`
- **Problem**: These files are present in the appendix folder but never `\input{...}` from `paper.filled.tex`. `old_repo_selection` is especially risky — it documents a *superseded* methodology (500 repos via star search) while the paper actually uses the 10,000-repo criticality-score method in `repo_selection.filled.tex`.
- **Recommended change**: Decide for each file: keep (and `\input` it) or delete. At minimum, delete `old_repo_selection.filled.tex` (and its source `.tex`). For the others, either remove from the repo or add a top-of-file `% NOT INCLUDED IN PUBLISHED APPENDIX — ARCHIVED` banner. (The CLAUDE.md note about `.filled.tex` being regenerated implies edits should land in the source `.tex` files.)
- **Severity**: majo
-> delete the ones not used
**Implementation:** Deleted both `.tex` and `.filled.tex` for all five orphans (audit, examples, old_repo_selection, regression, threats).
**Doubt:** None.

### Issue: Stale comment header in `old_repo_selection.filled.tex`
- **Location**: `appendix/old_repo_selection.filled.tex:1`
- **Problem**: Header says "Appendix E — Repository-selection pipeline (replication)", but this file is not rendered. If a reader stumbles on it, they will think it documents the live methodology.
- **Recommended change**: If keeping, mark "DEPRECATED — see `repo_selection.filled.tex` for current methodology"; if deleting (recommended), no further action.
- **Severity**: major
-> delete the ones not used
**Implementation:** File deleted (covered by Cat 9 Issue 1). Header is moot.
**Doubt:** None.


### Issue: Appendix Figure D ("chain-length distribution") not explicitly cited from main text
- **Location**: `appendix/extra_figures.filled.tex:3` (`fig:chainbox`)
- **Problem**: The appendix section is rendered but not referenced via `\Cref{fig:chainbox}` from the RQ 1 paragraph in the main text.
- **Recommended change**: Add a parenthetical to the RQ 1 paragraph (around `paper.filled.tex:152`): "(see Appendix D, Fig. \ref{fig:chainbox}, for the quarterly distribution)."
- **Severity**: minor
-> yes do it like this
**Implementation:** Added "(quarterly distribution in `\Cref{fig:chainbox}`)" to the AI-AI chain-length sentence in the RQ 1 paragraph (line 152), so cleveref auto-renders "Figure 4" / "Appendix D" at compile time.
**Doubt:** None.

### Issue: `audit.filled.tex` orphaned and easy to misread as a precision audit
- **Location**: `appendix/audit.filled.tex` and `results/merger_detection_audit.md`
- **Problem**: The file is not `\input`, but if it were, the section title "merger-detection audit" + the "100.0% agreement" cell would read as a hand-audit precision claim. The numbers themselves are reproducible from `merger_detection_audit.md` and refer to a `merged_by` vs. timeline-actor consistency check, not a classification audit.
- **Recommended change**: If keeping the file: rename section to "Merger-attribution consistency check", rephrase caption ("Field consistency on a sample of 1,000 PRs"), drop the standalone "100.0%" without context. If not keeping: delete the file.
- **Severity**: minor
-> delete file
**Implementation:** Deleted (covered by Cat 9 Issue 1).
**Doubt:** None.
---
