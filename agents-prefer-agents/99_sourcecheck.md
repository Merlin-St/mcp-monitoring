# Source Check — paper.filled.tex

Verifying that each cited source actually supports the claim made in the paper. Findings synthesised from six parallel verification passes (each subagent independently fetched abstracts / paper bodies / blog posts).

**Status:** ✅ remediation complete — see "Changes applied" below.

**Rating scale:**
- **OK** — source clearly supports the claim
- **WEAK** — source is related but does not directly support the specific claim, or is overstated
- **WRONG** — source does not support the claim, contradicts it, or is misattributed

---

## Changes applied (2026-05-04)

### Citations dropped (and bib entries removed)

| Cite key | Reason | Action taken |
|---|---|---|
| `peng2023copilot` | Cited for "deskilling"; paper actually shows 55.8% productivity gain, no deskilling. | Citation + claim deleted from Table 1 row 1 (no other source supports deskilling). |
| `motwani2024collusion` | Cited for "collusion in the wild"; paper is a lab/theoretical demo, "current models' steganographic capabilities remain limited." | Citation + claim "Collusion in the wild against humans" deleted from Table 1 row 5. |
| `ibrahim2025anthropomorphic` | Cited for "AI shaping human judgments without overt preference"; paper argues the *reverse direction* (humans anthropomorphise AI). | Citation + "anthropomorphic overreliance" phrase deleted from call-to-action. |
| `pike2020criticality` | Real authors are Arya, Lewandowski, Lorenc, Ferraioli — not "Rob Pike & Abhishek Lewandowski." Bib entry attribution fabricated. | Dropped from all 4 use sites (main methods, main limitations, appendix methods, appendix repo_selection). `ossf-criticality` covers the same claim. |
| `li2025teammates` | Cited for "increasing rates"; paper finds agent PRs are accepted *less* frequently than human PRs. | Citation removed from intro list. |
| `zhong2026synergy` | Cited for "increasing rates"; paper finds AI suggestions adopted at *significantly lower* rate than human reviewers'. | Citation removed from intro list. |
| `chowdhury2026claims` | Cited for "increasing rates"; paper explicitly debunks industry adoption claims (CRA 45% vs human 68% merge rate). | Citation removed from intro list. |
| `logicstar2025wild` | Cited for "increasing rates"; tracker explicitly *warns against* using merge rates as adoption signal. | Citation removed from intro list. |
| `eguiluz2025innovation` | Cited for Collingridge dilemma; supports it only by inheritance from Collingridge. | Dropped (redundant); `collingridge1980social` retained. |
| `staufer2026aiagentindex` | Cited for OSS code authoring + "increasing rates"; paper documents 30 deployed agents' *safety/transparency features*, neither claim well-supported. | Dropped from both use sites. |
| `shao2025workbank` | Cited for "deployed at scale"; paper audits *potential* automation via worker surveys, not deployment evidence. | Dropped; `appel2025economic` + `stein2026mcp` carry the deployment claim. |

### Claim rewordings (turning WEAK → OK)

| Original claim | Source | Revised claim |
|---|---|---|
| "Humans adopt AI-native phrasing and norms" | `juzek2025delve` | "Scientific writing shifts toward LLM-overrepresented vocabulary" — matches what the paper actually documents (lexical overrepresentation in scientific English). |
| "Unnecessarily verbose text or edits" | `huang2026morecode` | "Redundant code that ignores existing reuse opportunities" — matches paper's actual finding (silent technical debt from low code reuse). |
| "Infrastructure presumes AI agents" | `stein2026mcp` | "Infrastructure aimed at AI agents" — softer phrasing that the abstract supports directly. |
| "System prompts/constitutions encode commercial goals, then RLAIF on that objective" | `bai2022constitutional` | "Published model constitutions explicitly tie model behaviour to the developer's commercial success [`anthropic2026constitution`], with training feedback (e.g. RLAIF) optimising toward such objectives [`bai2022constitutional`]" — splits attribution: commercial-goal claim now backed by Anthropic's Jan 2026 Claude's Constitution, which states verbatim: *"Claude is also central to Anthropic's commercial success, which, in turn, is central to our mission."* `bai2022` retained for the RLAIF mechanism. |
| "Capability illegibility" | `bowman2022oversight` | Phrase deleted; `bowman2022oversight` moved to the preceding "Human evaluation too slow or expensive to keep up" citation, where it directly supports scalable-oversight challenges. |
| "Evaluation becomes a delegated task" | `ibrahim2025overreliance` | "Cognitive deskilling and over-reliance on LLM judgments" — matches the paper's actual focus. |
| "shape human judgments without overt preference" (call-to-action) | `thakkar2025iclr`, `cheng2025elephant` | "shape human judgments and behaviour" — drops the "without overt preference" interpretive layer; both papers do support general influence on human behaviour. |
| "AI systems built on LLMs are being deployed at scale across the economy" | `appel2025economic`, `stein2026mcp` | "AI systems built on LLMs are being adopted at growing rates across the economy" — matches Appel's directive-task-delegation rise (27%→39%) and Stein's MCP tool growth. |
| "AI-authored code suggestions are adopted at increasing rates" | `watanabe2025agentic`, `li2026aidev`, `ghaleb2026fingerprinting` | "AI-authored code is being submitted to open-source repositories in growing volumes and at high acceptance rates" — separates *volume* (which the papers do show) from *adoption rate trend* (which they don't). |

### Citations added

| Cite key | Document | Use |
|---|---|---|
| `anthropic2026constitution` | Anthropic, "Claude's Constitution" (January 2026), https://www.anthropic.com/constitution | Table 1 row 3, paired with `bai2022constitutional`. Includes verbatim quote in `note` field: *"Claude is also central to Anthropic's commercial success, which, in turn, is central to our mission."* |

### Laurito numbers verified ✅

User supplied the full PNAS paper text. Table 4 (GPT-4 as text generator):

| Dataset | Human | LLM | Match? |
|---|---|---|---|
| Product | 36% | 89% | ✅ matches paper claim "89% / 36%" |
| Paper | 61% | 78% | ✅ matches paper claim "78% / 61%" |

Both pairs match exactly. `laurito2025aibias` citation in the introduction (89%/36% product ads, 78%/61% abstracts) is fully verified.

---

## Files modified

- `paper/paper.filled.tex` (10 edits)
- `paper/paper.tex` (10 mirror edits)
- `paper/appendix/methods.filled.tex` (drop pike2020)
- `paper/appendix/methods.tex` (drop pike2020)
- `paper/appendix/repo_selection.filled.tex` (drop pike2020)
- `paper/appendix/repo_selection.tex` (drop pike2020)
- `paper/references.bib` (removed 11 unused/dropped entries; added `anthropic2026constitution`)

## Build status

LaTeX compile fails on a *pre-existing* Unicode emoji (`✅`, U+2705) at line 26 of `appendix/ai_verdict_regex.filled.tex`. This issue was present before my edits and is unrelated to the source-check work. To unblock compilation, either replace the emoji with `\checkmark` or add `\usepackage[utf8]{inputenc}` plus a `\DeclareUnicodeCharacter{2705}{\checkmark}` directive.

All citation keys in the edited paper resolve against `references.bib` (verified by `comm -23` of cited vs defined keys).

---

## Citations remaining in paper (final list)

agarwal2026ides, anthropic2026constitution, appel2025economic, bai2022constitutional, borges2018github, bowman2022oversight, chen2025selfprefer, cheng2025elephant, collingridge1980social, doshi2024homogeneity, ehsani2026fail, ghaleb2026fingerprinting, glasswing2026, goddard2012automation, huang2026morecode, ibrahim2025overreliance, juzek2025delve, kalliamvakou2014promises, kulveit2025gradual, laurito2025aibias, li2026aidev, munaiah2017curating, ossf-criticality, panickssery2024evaluators, pombal2026rubric, sharma2026whos, shumailov2024collapse, stein2026mcp, thakkar2025iclr, vanderweij2024sandbagging, watanabe2025agentic.

Total: 31 unique citations (down from 41 pre-remediation).
