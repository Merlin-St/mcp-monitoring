new changes, pls implement


drop this from the paper and the pipeline, not interested in it right now:  he Claude-vs-other-AI-reviewer
comparison is not estimable in our window (only 20 PRs
carry both a Claude and another-AI-bot opinion alongside a
human review).

Only claude vs. human is fine.


Make the section Methods called 'Empirical illustration methods'
Also, structure this section according to RQ1-3. (as measured by the share of PRs only reviewed or rejected by AI and by the AI-AI comment-chain length)

And the results section 'Empirical illustration results'
Keep the structure of the section along RQ1-3, and merge RQ3a and 3b into one, just share the results of 3b as part of a bigger 3. 

Instead of AI use 'manager' 

----
look at all my changes here, and also all the comments with %[..] and address them. 

\icmltitle{Accelerators of gradual disempowerment}

\icmlsetsymbol{equal}{*}
\begin{icmlauthorlist}
  \icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}
\icmlcorrespondingauthor{Anonymous}{anon@example.com}
\icmlkeywords{AI agents, gradual disempowerment, GitHub, code review, technical AI governance}

\vskip 0.3in
]

\printAffiliationsAndNotice{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{abstract}
When human workers, creators and managers are displaced by more competitive artificial intelligence (AI), societal systems might stop serving human interests~\citep{kulveit2025gradual}. We argue that such gradual disempowerment scenarios are accelerated --- \textit{independent} of the competitiveness of AI agents --- by the deployment of AI with propensities and interfaces that facilitate (1)~human preference drift toward AI, (2)~rising AI to human switching costs, (3)~AI preference drift toward AI, (4)~evaluation authority shifts from humans to AI, and (5)~AI-AI bias. 
In software development, Anthropic’s Mythos and other AI models are already deployed to fulfil critical security tasks [cite Project Glasswing, Anthropic \& Linux Foundation, 2026]. 
As a limited empirical illustration, rather than a test of gradual disempowerment as a whole, we review all 2,227,397 proposed modifications (i.e. pull requests (PR), April~2025--March~2026) to 10,000 critical open-source software repositories: a narrow slice of human economic activity, but an early site of AI substitution for human participation upstream of most modern software. 
First, in this small sample explicit AI participation grew 2.9{}$\times$ in one year (6.0\%{}$\to$17.3\%{} of all PRs), long ~$\geq$5 AI-AI chains grew 10{}$\times$. Second, AI evaluation authority grew from 0.07\%{} to 0.17\%{} of PRs only reviewed or rejected by AI. Third, we find no evidence for AI-AI bias: On the 34,743{} PRs reviewed by both AI and humans, AI agents are 69.21{}~pp less approving of AI-authored code than humans are, and only 58.48{}~pp less approving of human-authored code than humans are (difference significant at $p < 0.001$). This reversal from single-turn lab findings ~\citep{laurito2025aibias} may reflect measurement limits. We call on researchers and post-training teams to assess model propensities and interfaces for disempowering effects. 
\end{abstract}
%[make all in the text thereafter as "model propensities" or "interfaces and deployment choices". Keep it simple in abstract and in some other parts below as just "model propensities and interfaces", but at least table 1 should have the interfaces and other deployment choices in the layer column]
\section{Introduction}
\label{sec:intro}

``Gradual disempowerment'' scenarios~\citep{kulveit2025gradual} claim that the economy, society and culture have historically aligned with human goals because of human participation --- and such alignment will gradually decay when AI actions displace human actions. Once such drift manifests in economic, cultural, or political outcomes, it may be difficult to reverse~\citep{eguiluz2025innovation}. Some scholars argue that human participation in management tasks or oversight by trusted AI models may be able to ensure alignment to human goals [insert cites of these: https://arxiv.org/abs/2602.11865, arXiv:2211.03540]. There is first empirical work that chatbots based on large language models (LLMs) may disempower some users ~\citep{sharma2026whos}.

Previous researchers have showed in choice experiments, that AI systems fulfilling diverse tasks may exhibit \emph{AI-AI bias}. AI-AI bias means AI prefers AI (outputs) more than humans do, compared to humans (and their outputs). GPT-4 favoured LLM-written product ads 89\% of the time vs.\ a 36\% human baseline, and LLM abstracts 78\% vs.\ 61\%~\citep{laurito2025aibias}. AI models across model families and generations recognise and favour their own generations~\citep{panickssery2024evaluators, chen2025selfprefer, pombal2026rubric}. Whether such biases manifest in real deployments is an open empirical question. 

AI systems built on large-language models are being deployed at scale across the economy, especially in AI and software development~\citep{appel2025economic, stein2026mcp, shao2025workbank}. In software development, AI systems already complete both worker and management tasks: AI coding systems like Claude Code author and review code in public open-source repositories~\citep{ghaleb2026fingerprinting,staufer2026aiagentindex}. AI-authored code suggestions are adopted at increasing rates~\citep{watanabe2025agentic,li2026aidev,li2025teammates,logicstar2025wild,ghaleb2026fingerprinting,staufer2026aiagentindex,zhong2026synergy,chowdhury2026claims}, with concerns about  quality debt~\citep{agarwal2026ides,ehsani2026fail,huang2026morecode}.
Independent of quality, there is little empirical understanding whether the code suggestions and reviews of AI systems systematically favor AI systems. 

%[the entire paper should just be AI systems, not AI agents, replace whenever mentioned]

We make two contributions: 
%[turn this into bulleted list of 1), 2)]
1)~a taxonomy of \emph{accelerators} of gradual disempowerment: mechanisms through which model propensities, system interfaces and deployment choices increasingly disadvantage human participation, even when AI is not clearly more competitive than humans; and 2)~an illustrative empirical assessment of such acceleration in the wild in open-source software development. 
%[note: confusing whether the aceelerators are model propensities or deployment choices etc. because currently only AI-AI bias is an accelerator .... this needs review throughout the piece]
%[note: replace the word taxonomy everwhere with 'framework', also drop the word theoretical]
We ask three questions about AI involvement in GitHub PRs over Apr~2025--Mar~2026. 
\textbf{RQ 1}: Has the share of AI \emph{participation} in PRs increased? \textbf{RQ 2} (\textit{accelerator~4}): Has the share of AI systems involved in \emph{evaluation authority} increased? 
\textbf{RQ 3} (\textit{accelerator~5}): Do AI reviewers prefer AI-authored PRs more than human reviewers do, relative to human-authored PRs? 
\section{A theoretical framework of accelerators}
\label{sec:taxonomy}

Even slight AI self-preference, combined with growing AI evaluation authority, may over time reduce human participation and human-preferred outcomes---without AI being more competitive than humans.


----

## Claude's assessment on the propensity-vs-interface-vs-deployment-choice framing

The flagged tension is real, and worth resolving in the paper rather than papering over. Here's my read after a full pass:

**The framing currently mixes three different layers under one label.** "Accelerator" is being used to describe three structurally different things:
1. *Model propensities* — properties of the model itself, downstream of training (e.g.\ AI-AI bias, sandbagging, self-preference). These are the only things a post-training team can directly fix by changing training data, RLHF/RLAIF reward, or constitution.
2. *Interfaces* — the surfaces through which the model meets humans and other systems (e.g.\ AI-AI MCP routing, verbose-by-default chat UIs, "approve" buttons that gate humans out of the loop). A vendor product team owns these, and they are independent of the model.
3. *Deployment choices that aren't interface-level* — workflow, who-reviews-what policy, latency/cost SLOs, exposure rates, hiring decisions that put AI in the loop. An organisation deploying the model owns these, mostly independent of the model and the interface.

Putting (1) and (2)+(3) under one "accelerator" label is what creates the muddle the comment flags. Each accelerator in Table 1 actually decomposes into a propensity component and an interface/deployment-choice component:

| Accelerator | Propensity face | Interface / deployment-choice face |
|---|---|---|
| 1. Human pref drift | (none directly) | Exposure design, AI-native UX, deskilling-by-cost decisions |
| 2. AI-human switching costs | Verbose / homogeneous outputs | MCP-style infrastructure, AI-only channels |
| 3. AI pref drift | Self-reinforcing training loops, sycophancy, sandbagging | RLAIF reward design, constitution/system-prompt content |
| 4. Eval authority shift | Automation-bias-friendly outputs, capability illegibility | Workflow choice to delegate eval to AI |
| 5. AI-AI bias | The ranker actually preferring AI tokens / outputs | Stated gatekeeping rules, routing policy |

Only AI-AI bias has a *clean* propensity reading where you could measure it on the model in isolation (single-turn lab DiD). The other four are fundamentally socio-technical: they exist only when a model meets a deployment context, so calling them "model propensities" is a category error and calling them "deployment choices" leaves out the propensity ingredient.

**Recommended revision (low-cost, fits in one pass).** I have applied the user's "model propensities and interfaces" wording in the abstract/intro and put "interfaces, deployment choice, model propensity" tags in the Layer column of Table 1. To resolve the deeper ambiguity I'd suggest, in a future revision:
- Re-cast the contribution sentence so the *framework* doesn't claim the accelerators are categorisable as either "propensity" or "deployment choice"; state instead that each accelerator has a propensity face (what training can address) and an interface/deployment face (what product/governance can address), and that interventions need to span both.
- In §framework, add a one-sentence note that AI-AI bias is the only accelerator with a clean lab-isolatable propensity reading, which is why it is also the only one we put a number on; the others are framework, not measurement, claims.
- Keep the dynamics math as-is — it is honest about $d\rbt^A/dt$ and $d\alpha_t/dt$ being separately driven, which already encodes the propensity/deployment split.

**Why this matters for the call to action.** The current draft asks post-training teams to evaluate self-preference. If the framework stays muddled, a reasonable reader reads accelerators 1, 2, 4 as "post-training teams' job" too, and post-training teams will (correctly) push back that they cannot fix workflow choices in CodeRabbit's review-bot product. Keeping the propensity/interface/deployment split clean lets the call to action also cleanly split: model providers own propensities; platforms (GitHub, MCP hosts) own interfaces; deployers (companies running CodeRabbit, Cursor, etc.) own non-interface deployment choices; regulators set incentives across all three.

**On AI-AI bias being "the only one":** even AI-AI bias has an interface ingredient — gatekeeping rules and routing are interface choices. So "only one is an accelerator" overstates it; better is "only AI-AI bias has a clean post-training measurement". The framework still survives if the others are recognised as joint propensity-and-deployment phenomena.

