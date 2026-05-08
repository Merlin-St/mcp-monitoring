# Paper Review Prompt (Read-Only)

## What this task is

Review the paper end-to-end against the checklist below and produce a single review document at `99_llm_review.md` (in the paper's working directory) listing every issue you find together with a recommended change.

**Do not edit any other file.** No edits to the main `.tex` file, appendix `.tex` files, figures, scripts, references, or anything else. The only file you may create or modify is `99_llm_review.md`. A human will review your findings; implementation happens later, in a separate pass, only after that review.

## Scope

The review must cover the entire write-up, including:
- The main paper source file.
- All appendix source files.
- All figure source files (open the actual PDFs/PNGs — do not rely on captions alone).
- The rendered PDF for layout, figure-size, and font-size checks.

If multiple parallel versions of the source exist (e.g. a working copy and a build-time variant), confirm with the human which is canonical before starting and review only that one.

## How to run the review (use subagents)

For each numbered category below, spawn a separate subagent (use a read-only exploration agent for prose-only categories, or a general-purpose agent for categories that need to inspect figure files and cross-reference text). Run independent categories in parallel where possible.

Each subagent should:
1. Read the relevant part(s) of the paper and appendices.
2. Check every checklist item in its category against the actual text/figures.
3. Return a list of concrete issues, each with: location (file + section/line or figure number), what's wrong, recommended change.

Then merge all subagent outputs into `99_llm_review.md` using the format in the *Output* section at the bottom.

If a checklist item has no issue, say so explicitly ("Checked: no issues found") rather than omitting it — the human needs to see the review was actually performed.

---

## Category 1 — Structure & Terminology

Single overall pass on how the paper hangs together and how concepts are named.

**Structure**
- The paper has ONE core idea. Any contributions (typically 2–4) all serve that one idea. If two distinct ideas are present, flag for splitting.
- Findings are not buried — each Results subsection leads with the key result, then explains.
- Methods and Results are tightly aligned: every analysis appearing in Results is described in Methods. Conversely, no Methods description is left dangling without a corresponding Result. Keep the alignment tight given the venue's main-text page limit (detail can live in the appendix).
- Research Questions stated in the introduction are the same Research Questions answered in Results and revisited in the Discussion/Conclusion — no mismatch between RQs and what is reported.
- The conclusion connects findings to broader implications (e.g. "what do these findings tell us about risks / policy / future work").
- Logical flow within and across sections: a reader should not stumble on a small section because it is out of order.

**Terminology**
- All key terms are defined when first introduced (typically in the introduction).
- Each defined term is used consistently throughout — concepts are not renamed or silently redefined mid-paper.
- All metrics and measurement names are audited: eliminate multiple names for the same thing (e.g. "used / published / downloaded / counted" referring to the same underlying count) and ensure each is explained once in Methods.
- Any sub-categories or distinctions introduced are actually used later in the paper. Remove (flag) the ones that are not.
- Definitions taken from cited literature are used consistently with how those terms are then used in the paper's own analysis.

---

## Category 2 — Writing, Grammar & Small Edits

Single overall pass on prose quality. This category is for small, line-level issues; structural issues belong in Category 1.

- Eliminate jargon — each paragraph should read as an explanation for a well-informed but non-specialist audience.
- Remove "notes-to-self" passages — every sentence is written for a reader, not as scaffolding.
- Confirm with the human which English variant the paper targets (British vs. American), then check grammar and spelling consistently against that variant throughout.
- No unnecessary repetition between sections — the same content is not restated.
- Each paragraph has one clear point — no rambling or jumbled paragraphs.
- Clear but minimal signposting between sections so a reader knows where they are and what's coming next.
- Bridging between the literature review and the paper's own contributions is explicit — the reader should see how this work relates to and extends prior work.
- The work is contextualised within at least 3–4 closely related empirical papers.
- The "so what" is foregrounded from the very start (abstract, intro) — why this matters for the relevant audience (policy, regulators, practitioners, safety, etc.).
- No "naked" pronouns: every `this`, `that`, `these`, `those`, `it` has an unambiguous referent. Flag every occurrence where the referent is not crystal clear.
- Sentences are short. Voice is active where possible. Simple words preferred over complex ones.
- Same word for the same concept (no stylistic variation that introduces ambiguity).
- Parallelism across sentence and section structures where it aids reading.
- Citations are generous — claims that depend on prior work are cited.
- Speculation is clearly marked as speculation.
- Affirmative sentences preferred to negative ones.
- For each sentence: is what is being asserted *precisely* correct? Flag claims that are imprecise, overstated, or that drift from what the data actually supports.

---

## Category 3 — Figures

Single overall pass on every figure in main text and appendix. Open the actual figure files (PDFs/PNGs) — do not rely on captions alone.

- Every figure has a self-contained legend/caption that fully explains what is shown. A reader should understand the figure without consulting the main text.
- All text within figures is large enough to read without zooming.
- No two unrelated metrics share the same figure without very clear justification.
- Colour usage is consistent across figures — the same colour does not denote different things in different figures.
- Multi-layered figures that try to capture too many levels at once are flagged for splitting into separate charts.
- Geographic figures: missing data and 0% are visually distinguished (different colour or hatch pattern).
- Geographic figures: labels do not overlap coloured regions; reposition labels for legibility where needed.
- Trend-line figures: the trend line is visually prominent (lighter background, darker/thicker line).
- Trend-line figures: uncertainty bands/ribbons appear throughout the line, not as a single error bar at one time point.
- Data points are not cut off at axis boundaries.
- Notation is standardised across figures — pick `n=` or `#` for counts and use it everywhere.
- Categorical data uses perceptually distinct palettes — not a near-continuous colormap across distinct categories.
- No in-figure title that simply repeats the caption — follow venue convention (caption only, unless the venue requires otherwise).
- The first figure is especially tight and professional — it sets the tone.
- Every figure has a caption, **including those in the appendix**.
- "Figures Test": strip the paper to title + abstract + figures + captions. Is the story still understandable? Flag what is missing.

---

## Category 4 — Title

- The title is distinctive relative to other recent empirical work in the same area.
- The title carries a catchy reference (e.g. a two-word hook or memorable phrase).

---

## Category 5 — Abstract

- The abstract clearly states: the problem, the approach, the results, and the implications.
- The "so what" is foregrounded — why this matters for the relevant audience.
- Recommend a side-by-side comparison: produce one fresh rewrite of the abstract from scratch, then identify which elements of the current abstract should be retained and which should be replaced. Put the comparison in the review.
- The abstract contains all three of: Context, Content, Conclusion.
- "Five-word test": if a reader retained only ~5 words from the abstract, would those be the right ones?

---

## Category 6 — Introduction

- Field domain → what the field knows → remaining gap → narrower gap that this paper addresses → summary of approach and results. Flag any of these that are missing or out of order.
- All key terms used later are defined here at first use.
- 3–4 closely related empirical papers are cited and explicitly contextualised — the reader should see how this paper extends them.
- The "so what" is led with, not buried.
- Research Questions are clearly stated and match what the rest of the paper actually addresses.
- No literature-review content is repeated later in the paper without purpose.

---

## Category 7 — Results

- Each Results subsection leads with the key result, then explains.
- Each paragraph follows: question → general method → answer sought → answer found, with figures supporting the relevant logic step.
- Every metric reported in Results has a corresponding description in Methods (cross-check).
- Logic chain is visible: raw data → processed data → final statistics.
- Claims are tightly bounded by what the data shows — flag any overreach.
- No restatement of Methods detail beyond what is necessary to interpret the result.

---

## Category 8 — Methods

- Each data-cleaning step is described clearly and in sequence.
- Every metric used anywhere in Results has a corresponding description here.
- The Methods section is replicable by another researcher — sufficient detail to reproduce, with overflow detail living in the appendix.
- Methods is kept tight enough to respect the venue's main-text page limit; check that nothing replicable-but-non-essential lives in main text when it could live in the appendix.

---

## Category 9 — Annex / Appendix

- Every appendix table has a caption and fits within the page.
- Every appendix figure has a caption.
- Tables use appropriate font sizes — flag any tiny text.
- Sufficient detail is present in the appendix for end-to-end replication of every analysis in the main text.
- Appendix sections are referenced from the main text where relevant.
- Anything in the appendix that is not actually referenced from the main text is flagged for review (keep or cut).

---

## Output: `99_llm_review.md`

Create `99_llm_review.md` (in the paper's working directory) with one top-level section per category above (1–9), in that order. Within each category, list issues as:

```
### Issue: <short title>
- **Location**: <file>:<section/line> (or figure N, table N)
- **Problem**: <what is wrong, in 1–3 sentences>
- **Recommended change**: <concrete, actionable change — not "consider revising">
- **Severity**: blocker / major / minor / nit
```

If a category has no issues, write `No issues found.` under that category heading — do not omit the heading.

At the very top of `99_llm_review.md`, include:
- A one-paragraph summary of the most important findings (top 3–5 blockers/majors).
- A count of issues by severity per category (small table).

## Hard rules

1. Do not edit any source file: main text, appendix, figures, scripts, or references.
2. Do not run the paper build, do not regenerate figures, do not rerun any analysis.
3. The only file you create or write is `99_llm_review.md`.
4. If something is ambiguous, record it as an issue with severity `minor` and let the human decide — do not act on the ambiguity.
5. Do not implement any of your own recommendations. Implementation is a separate, later pass that requires explicit human go-ahead.
