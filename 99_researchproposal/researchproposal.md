
DPhil Abstract 
Technical methods for governments’ monitoring of Advanced AI risks
Or: How can governments monitor advanced AI risks?

Merlin Stein
November 2025


197 words

Advanced artificial intelligence technology (advanced AI) - like Claude Code - may soon become a self-improving general-purpose technology, but its potentially large societal risks remain highly uncertain. Effective risk regulation of potential self-improving general-purpose technologies requires up-to-date information about how such systems interact with the economy. This article-based DPhil investigates how governments can monitor the risks of advanced AI by studying under which conditions governments should collect information, and by developing suitable monitoring methods. My mixed method approach empirically validates Menard’s (2004) hybrid governance theory, and fills empirical gaps on advanced AI monitoring. 
Article 1 (peer-reviewed, AIES '24) uses comparative case analysis of nine high-risk industries to determine who should evaluate advanced AI, finding that public bodies should conduct evaluations under high risk uncertainty and information sensitivity. Article 2 (draft complete) creates and demonstrates a monitoring approach capturing 100,538 public AI agent tools to identify what AI agents access and modify in the real economy, using bottom-up and top-down topic modelling. Article 3 (planned) will develop and evaluate, in collaboration with UK regulators, a self-improving monitoring system demonstrating how regulatory capacity can scale alongside advancing AI capabilities, implementing a novel reinforcement monitoring based on regulatory feedback (RMRF) algorithm.



DPhil Research Proposal 
Technical methods for governments’ monitoring of Advanced AI risks
Or: How can governments monitor advanced AI risks?

Merlin Stein
November 2025

Current word count: 3800 words

Introduction
A general-purpose technology is ‘a single generic technology, recognizable as such over its whole lifetime, that initially has much scope for improvement and eventually comes to be widely used, to have many uses, and to have many spillover effects’ (Lipsey et al. 2006, p.98). A transformative technology is a general-purpose technology that raises the productivity of research and development, becoming an ‘invention of a new method of invention’ like the industrial revolution (Crafts 2021). General-purpose AI could become a transformative, general-purpose technology - or not (Eloundou et al. 2024, Crafts 2021, Maas 2023, Gruetzemacher and Whittlestone 2022). Advanced AI could become transformative by becoming self-improving: Advanced AI contributes to its own development with less and less humans in the loop: advanced AI models used as coding agents already code the next generation of advanced AI models (Robeyns et al. 2025, Baratchi et al. 2024, Zoph & Le 2016, Anthropic 2025).
Risks, risk indicators and adequate regulation of general-purpose AI is highly uncertain (Carey 2025). For risk-based regulation, regulators require information about the risks and risk-factors of the subject of regulation (Black 2005). 

This DPhil aims to provide three empirical works to help answer subparts of the overarching question: How can regulators monitor the risks of advanced AI as it unfolds into the economy?
In these three articles/chapters, this DPhil contributes to the technical AI governance literature (Reuel et al. 2024) a governance design mechanism and two technical monitoring methods.

Chapter 1 (peer-reviewed, AIES ‘24) tests Menard’s (2004) hybrid governance theory empirically, answering ‘Who should run evaluations of general-purpose AI?’ with detailed scoring of nine case-studies of high-risk industries.
Chapter 1 shows how the uncertainty on risks and risk-factors in advanced AI implies a range of regulatory options - in this case, for the evaluation of advanced AI and associated regulatory capacity. 
Our findings suggest that public bodies should assess advanced AI if regulators expect high levels of risks, risk uncertainty, verification costs and information sensitivity of the technology. 
The findings of the chapter have already benefitted advanced AI regulation: The European AI Office hired me to implement the proposed external evaluation mechanism, working with the external Chairs on the Code of Practice for general-purpose AI (‘Independent external model evaluations’, EU CoP 2025, A.5). The International Scientific Report on the Safety of Advanced AI cites the paper (Bengio et al. 2025).

Chapter 2 (draft finished, available upon request [Note: Will be on arxiv in December, submissions to peer-reviewed venues thereafter]) fills the empirical gap on ‘What AI agents (the most advanced AI systems) access and modify in the real economy?’, based on a newly scraped dataset of 100,000 public AI agent tools.
Chapter 2 focuses on the start of the self-improvement cycle of advanced AI technology - on trends in public open-source developer repositories. It implements a method of monitoring critical uses of advanced AI systems.
A draft of this paper has already been used by the Bank of England to inform their monitoring approach on AI agents for financial stability (BoE 2025), by the UK Department for Science, Innovation and Technology to monitor public trends of AI development (2 publications, in Nov). Further, there is interest from the UK Competitions and Market Authority, as well as the UK Financial Conduct Authority in the gathered data.

Chapter 3 (scoping stage) fills the empirical gap on ‘How can regulators build self-improving  monitoring systems to track self-improving advanced AI?’, by jointly developing an example of such a system together with the Bank of England, and potentially other UK regulators. This research will be based on the dataset sourced in Chapter 2, Anthropic usage data (2025) and a combination of other data sources, access to which is currently negotiated (likely data from OpenRouter, glass.ai scraping of all agent products listed online, potentially usage data of large AI developers). 


Related literature
Information collection & risk governance literature
This study on information for risk regulation builds on three fields of research: 1) New institutional economics, particularly hybrid organization theory and 2) Risk governance theory and 3) mechanism-specific literature, e.g. on auditing regimes or post-deployment monitoring design.


Given the criticality and uncertainties described above, anticipatory regulation of advanced AI is unlikely to fit the technology’s continuous development. Advanced AI regulation needs to be adaptive. Adaptability and risk minimisation is often seen as a trade-off in regulation, though hybrid public-private regimes have proven able to address risks adaptively - when the oversight on auditors is sufficiently high and regulatory capture can be prevented (Bo 2006, Carpenter & Moss 2013, Stigler 1971, Pagallo et al. 2019, Radu 2021). At the forefront of regulating AI, the EU AI Act suggests such a hybrid approach for general-purpose AI, while the specific responsibilities of public and private bodies in collecting evaluation and monitoring information are yet to be specified. Regulatory directions in the US and UK are similar (NTIA 2023, UK Government 2023).

New institutional economics describes transaction costs, asset specificity and uncertainty as the main factors influencing organizational structures and contracts, and how these internalize externalities (Williamson 1979, Coase 1960, Williamson 2008). From a hybridity of governance lens (Quélin, Kivleniece and Lazzarini 2017), oversight regimes can be characterized as a continuous spectrum of contractual relations with a differing degree of hybridity between direct hierarchical and indirect market-driven oversight. Effective regulation is partly determined by the characteristics of the oversight body, mainly by the alignment of these characteristics with the underlying conditions of transaction costs and asset specificity (Menard 2004, Quélin et al. 2019). As hybridity shapes AI regulation (Radu 2019) and information collection mechanisms like auditing too (Rajala and Kokko 2022), I adapt Menard’s (2004) hybrid governance framework for the information collection framework and regulation as detailed below. 
Information collection for regulation is a structurally underinvested endogenous public good (Stephenson 2011). A reduction in uncertainty by government officers is not rewarded by the public, but only the quality of the decision with additional information. The fitting degree of hybridity of oversight with additional resources can lead to better incentives for government officers or private firms.

The question of hybridity shapes risk governance too. According to 2009’s Economics Nobel Prize Winners Ostrom and Williamson, risk governance can be characterized as a common good problem with questions of hierarchy. Hybridity has been linked by risk governance scholars to the degree of formalization of risk governance strategies. Hybrid strategies integrate risk-informed, precautionary, and discursive approaches, emphasizing systemic understanding and resilience. Successful risk governance adopts hybrid strategies to manage complexity, uncertainty, and ambiguity (Aven & Renn 2018).
Generally, risk governance scholars are concerned with the processes to identify, assess, manage and communicate risks. What are risks? “Risk is a situation or an event where something of human value (including humans themselves) is at stake and where outcome is uncertain” (Rosa 1998, 2003, 2010), according to (Aven & Renn 2009) the most widely adopted definition of risk in the social sciences.
Rosa, Renn & McCright 2015 define three risk problem classes: Simple, uncertain and complex.  Generally, AI risks are predominantly uncertain: The influencing factors of a lack of understanding of underlying phenomena and complexity, match the characterisation of advanced AI. Uncertain risks bear a potential for extreme consequences and large uncertainties concerning the nature and extent of consequences.. Main risk management strategies for this risk class are supposedly concerned with “knowledge aspects”, in line with the argument above. 
Systemic risks refer “to the risk or probability of breakdowns in an entire system, as opposed to breakdowns in individual parts or components, and is evidenced by co-movements (correlation) among most or all parts” (Kaufmann & Scott 2003). Systemic AI risks are uncertain and complex (Aven & Renn 2018), with normative ambiguity (long causation chains, many actors affected) and cascading effects beyond the source. 
Auditing, post-deployment monitoring and other information collection mechanisms can reduce risks effectively when implemented as hybrid risk regulation strategies. While each chapter will review the respective mechanism-specific literature, I provide a short general overview on information collection mechanisms for AI. There are institutional, software and hardware mechanisms to verify AI companies’ claims about their AI systems and increase specificity and diversity of external demands for claims (Brundage et al. 2021). This DPhil focuses on institutional mechanisms, while illustrating possible software mechanisms. 
High-quality & timely information might not be available and not in a form that governments can process. Without good info collection mechanisms, information asymmetries might widen and negative externalities spread as private commercial measurements are not aligned with public interest or there is not sufficient regulatory capacity to absorb and act upon new privately generated information (Whittlestone and Clark 2021).

Context: Information collecting mechanisms
For a comprehensive overview of potential mechanisms, I consider which information each player along the advanced AI value chain could provide to regulators. 

Figure 1: Information collection for advanced AI regulation. [Map asymmetries to each piece of information] 

[Potentially add description].

Technical AI governance literature
[Review TBD - mostly from https://openreview.net/forum?id=1nO4qFMiS0 and https://arxiv.org/pdf/2311.10538 and few related works.]
Theoretical framework
Terminology / Definitions
Advanced AI. Highly-capable general-purpose artificial intelligence models (GPAI models) and systems (GPAI systems)
‘GPAI model means an AI model, including where such an AI model is trained with a large amount of data using self-supervision at scale, that displays significant generality and is capable of competently performing a wide range of distinct tasks regardless of the way the model is placed on the market and that can be integrated into a variety of downstream systems or applications, except AI models that are used for research, development or prototyping activities before they are placed on the market’ 
‘GPAI system means an AI system which is based on a general-purpose AI model and which has the capability to serve a variety of purposes, both for direct use as well as for integration in other AI systems’, such as claude code
‘Highly-capable’ is approximated by GPAI of capability equivalent to GPT-5 or Claude Code, which could be proxied by the floating-point operations used in training the models being greater than 10^25. This definition thus follows the EU AI Act but will likely evolve.
Advanced AI providers or ‘AI Labs’. Legal person, public authority, agency or other body that develops advanced AI (‘developer’) or has advanced AI developed; and places advanced AI on the market or puts it into service (‘host’). (EU AI Act)
Vital physical and virtual environments cover “functions that society could not cope without for seven days or less without this threatening the safety and/or security of the population” (Norwegian Gov (2017))
Post-deployment monitoring. Conducting continuous or repeated measurements or observations of the AI system after it has been placed on the market or put into service.
Scale of risk externality: the degree of third-party exposure to harm in the event risks materialize 
Verification costs: the invasiveness and cost of establishing an auditee’s conformity with rules
Information sensitivity: the potential harm arising from unauthorized information access.
Risk uncertainty: degree of clarity regarding risks and risk measures.
Market concentration and size: distribution of the total industry revenue across firm
Chapter 1 research plan
See peer-reviewed research paper here
Contribution 
The first chapter focuses on the responsibility and resource perspective. I answer “(Q1.1) Which advanced AI audits should public bodies run, and which should private firms run? (Q1.2) Consequently, what regulatory capacity must public bodies develop?” by developing a two-step decision logic for public bodies. This logic is based on the criticality and market concentration perspective above: Fundamental factors (defined above) determine which auditor characteristics are important for which kind of audit, and in turn determine who is best suited to conduct a particular audit, and as a result, which resources public bodies may need to build.

New Institutional Economics: Applies and tests hybrid organization theory and transaction cost economics to explain the structure and effectiveness of auditing regimes.
Auditing Regimes Scholarship: Extends literature beyond auditor characteristics to focus on alignment of characteristics with fundamental factors. Previously, most scholars focused just on auditor characteristics without the underlying conditions
AI-Specific Auditing literature: Contributes to the emerging field of AI auditing by providing a structured 2-step approach to decision-making on auditor responsibilities and conditions influencing these decisions. 

Theoretical hypothesis
If advanced AI is critical and highly concentrated,  public bodies need to collect information for its regulation instead of private parties. (RQ 1.1) If public bodies collect information, then an adequate regulatory capacity for AI is more likely to resemble other high-risk industries like nuclear energy or life science. (RQ 1.2)

Following Menard’s (2004) and Williamson's (1999) work, hybridity is determined by transaction costs, asset specificity and uncertainty. I adapt this framework to fit the advanced AI risk regulation context, and quantify each indicator. 
Scale of risk externality: the degree of third-party exposure to harm in the event risks materialize 
Verification costs: the invasiveness and cost of establishing an auditee’s conformity with rules
Information sensitivity: the potential harm arising from unauthorized information access.
Risk uncertainty: degree of clarity regarding risks and risk measures.
Market concentration: distribution of the total industry revenue across firms
Factors 1 and 2 capture the transaction costs involved in auditing, relating to how markets facilitate the efficient exchange and dissemination of information about risks. Here I separate transaction costs into the extensive margin describing the external costs as the reasons for auditing transactions (“risk externalities”) and the intensive margin as the difficulty of the auditing transaction (“verification costs”).	
Factor 3 indicates asset specificity, implying how generally applicable the information and its risk impacts are. As auditing is the application of specific skills to specific information, I focus asset specificity into “skill specificity” and “information sensitivity”. The analysis reveals that information sensitivity is directly linked to hybridity, whereas skill specificity is more complex. Thus I drop it here in this overall description. 
Factor 4 relates to uncertainty and describes limitations in capturing valid and reliable information, which, in an auditing context, particularly relates to risks (here “risk uncertainty”). 
Economic hybrid governance theory is limited in focusing on economic hybridity and agents. This allows for this thesis’ actor-focused approach, but falls short of analyzing auditing from a within-organizational perspective (Bol et al. 2019) and power-distribution perspective (Levi-Faur 2011). To bridge the latter limitation, I build on regulation theorists (Behr 1985 and Stigler 1971) to establish factor 5, which pertains to the existing power distribution, from a societal and an economic perspective: “market concentration”. This factor is thus carving out existing within-market hierarchies to leave the hierarchy question solely between private companies and public regulators. 

Methodology
This comparative case study approach has proven effective for similar prescriptive questions on regulatory regimes (Levi-Faur 2004, Hill and Varone 2021). Given the small number of high-risk regimes and difficulty in capturing nuances in their variations quantitatively, I deploy a mixed-method approach. (More method details in the article itself)

Chapter 2 research plan
See draft research paper here [insert arxiv link when published]

[Quick summary of the research?]

Contribution
This paper contributes to the technical AI governance literature:
1)  	Agent tool distribution across tasks, through a newly curated agent tool dataset of 100,538 public MCP tools, sourced from MCP servers readmes on Github, Smithery, and the official MCP repository.
2)  	Agent tool usage trends on perception and action affordances based on npm and pypi download data of 4.2k MCP servers. Verification through a subset of 9k tools on official MCP servers built by legally registered entities. These entities turnover >3B £ UK AI-specific revenue (2024, 10% of entities with revenue data), which represents 20% of the UK AI sector revenue (DSIT 2025).
3)  	A method for early monitoring of high-stakes agent ecosystems. Monitoring public agent tools helps to anticipate large-scale agent deployment in particular sectors and occupations, to prepare for potential opportunities and systemic risks (Stein et al. 2024, Bernardi et al. 2024).



Theoretical hypothesis
The characteristics of advanced AI that lead to high criticality are the combination of action tools and fast and wide adoption in vital physical and virtual environments.
What do advanced AI agents already access and modify in high-stakes settings? (RQ2) (In short: Is Advanced AI already critical?)
We hypothesise that advanced AI does not yet have high criticality, thus does not yet require public evaluation or stringent regulation (Chapter 1), but a trend towards more criticality is visible.

[to extend, in line with this figure]
Note: I’m not focusing on the implication of the general-purpose characteristic on market concentration, as others have done this as reviewed above.

Methodology

Figure 1.  Methodology overview. Tool & server type proportions roughly indicate the distribution of servers and tools, along O-NET (2019) and US CAISI (2025) taxonomies. 
For the conducted human validation with n=13 computer science graduates, CUREC approval has been received (CUREC1A/BSG_C1A-24-10). 



Figure 2. Over time, early data shows that critical high-stakes actions tools increase (TO UPDATE FIGURE, and choose one of the two figures) [TO DETAIL: What is the exact regression done for my RQ]


Chapter 3 research plan
[TBD - some rough ideas only]

Reinforced monitoring based on regulatory feedback
Objective: Demonstrate a self-improving feedback loop: Regulator specifies monitoring target, automated monitoring, regulatory action on some monitoring results, and improved automated monitoring of agent tools and logs. Prototype automatic monitoring for consequentiality and law-alignment of agent actions
Theoretical hypothesis
The characteristic of advanced AI that might make it transformative might be its self-improving nature. Similarly, regulation might need
Can regulators benefit from adaptive technical monitoring methods to ensure regulatory capacity to govern advanced AI when it will advance (due to self-improvement) rapidly? (RQ3) 
We hypothesise that the benefit might depend on highly on the specified method [expand]

Methodology
This automated monitoring system approach addresses the scaling challenges of AI agent oversight. Given the volume of autonomous agent interactions (millions per day) and the inability of human oversight to scale proportionally, I deploy a hybrid algorithmic-human review method. The approach applies relevance-flagging algorithms to usage logs from multiple sources (OpenRouter, MCP agent tools, Glass.ai, and lab monitoring data) to surface the most critical incidents for human review, with automated labeling for clear cases and human judgment reserved for ambiguous scenarios. This method is particularly suited to bridging the gap between lab evaluations and real-world agent behavior, where distribution shifts and novel contexts prevent purely reactive incident discovery. [requires more details?]
Timeline of analysis
Note: Based on my original research proposal pre-PhD for Stages 1-3
As of December 2025, this PhD is in month 15. (Started Oct 2023, paused Oct 2024-Oct 2025.)
Stage 1: Research Design, Upskilling & Literature review 
Originally allocated time: PhD Months 1-12
Status: Mostly done
Research design above
Upskilling in qualitative policy methods & machine learning done
Overall literature reviewed, with chapter-specific literature partly outstanding

Stage 2: RQ 1.1 (Chapter 1)
Originally allocated time: PhD Months 3-15
Status: Mostly done

Stage 3: RQ 1.2 (Chapter 1)
Originally allocated time: PhD Months 9-24
Status: Mostly done, using secondary literature instead of interviews

Stage 4: RQ 2 (Chapter 2)
Allocated time: PhD Months 10-20
Status: First Draft finished, 

Stage 5: RQ 3 (Chapter 3)
Likely timeline: PhD Months 15-30
Status: Planning stage.

[PhD Months 30-36 as backup]

Challenges & limitations
[tbd]
References
[tbd]

Appendix: Fundamental factors quantification
To enable comparability across case studies of high-risk industries, the qualitative definitions of the fundamental factors were quantified in two steps. In a first step, a quantitative proxy variable was defined for each fundamental factor (cf. column 2).  In a second step, the value range of the proxy variable was divided into three intervals (“standardized value”): high values, medium values, low values (cf. columns 3 - 5) to simplify interpretation and comparability across factors.


Fundamental
Factor
Proxy
Standardized Rating




High
Medium
Low
Scale of risk externality
Impact and likelihood of risk event
Significant impact, >5% likelihood OR
Catastrophic impact, any likelihood
Moderate impact, 
>5% likelihood  OR
Significant impact, >1% likelihood
Minor/limited/ moderate impact,
<5% likelihood 
Verification costs
Invasiveness of test procedure

Source: Auditing rules
Experiments in use case environment
Onsites inspection & experiments in proxy environment OR
Onsites inspection & simulation of use case environment & outside logic verification
Simulation of use case environment AND/OR outside logic verification
Information sensitivity
Governmental classification requirements for "product" information
Access restricted to persons with legitimate
Access restricted to persons with legitimate interest within firm
No classification requirements
Risk uncertainty
ISO Standards (length & share currently under development vis-a-vis existing standards)



>2,000 pages of ISO documentation AND >50% share of ISO standards currently under development
>2,000 pages of ISO documentation OR >50% share of ISO standards currently under development
<2,000 pages of ISO documentation AND <50% ISO standards currently under development
Market concentration
Herfindahl Index (Points)
>1,000
500-1,000
0-500


Figure B.1: Quantification of fundamental factors



The decision logic for determining a suitable proxy variable in step 1 was the following:
1.	Does a suitable index already exist within the field of economics (c.f. market concentration)?
2. If not, did a government measure similar factors and publish their results (c.f. scale of risk externality)?
3. If not, did a third-party measure similar factors and publish their results (c.f. skill specificity)?
4. If not, are there government documents that can be used to extract data about the factor and quantify it by defining our own proxy (cf. verification resources, information sensitivity)?
5. If not, are there third-party documents that can be used to extract data about the factor and quantify it by defining our own proxy (cf. risk uncertainty, risk public salience)?
Scale of risk externality is assessed by contrasting the likelihood of the risk event with the societal-level impact. 
Since I are specifically interested in the risk externality, I infer the risk impact from the UK National Risk Register. As per its mandate, it focuses on the effect of risk incidents on the entire society, making it superior to other proxies, such as liability insurances, which tend to measure the risk internality. Verification costs are derived qualitatively from the primary building blocks of the testing procedure and their relative invasiveness. As such, I know that experiments in a true use case environment require substantially more resources to conduct than simulations. Ideally, I would have employed quantitative measures, such as the average cost of an audit in that industry, but unfortunately I could not gain access to such data. Skill specificity is inferred from the average private sector salary in a job that requires skills comparable to the typical profile of an auditor in that particular industry. I assume that more specific skills are linked to less labor supply. In turn, economic theory, backed by empirical evidence, predicts that more specific skills, and thus limited labor supply, are associated with higher salaries, at similar levels of labor demand (Broecke, 2016). Information sensitivity is assessed by analyzing the qualitative criteria for accessing product information in a given industry. Intuitively, the government prescribes greater access barriers to guard more sensitive information. Risk uncertainty is evaluated via the volume  of ISO standards, as well as the relative share of standards under development. Generally speaking, higher risk levels should necessitate more standards, intended to manage these risks. At the same time, it seems likely that a certain share of risks remains undetected, thus high risk levels should typically also correspond to somewhat higher risk uncertainty. This is particularly true in very nascent industries, where a substantial share of standards is still under development. Public salience is derived from the volume of Google News search results. While the use of internet search data as a proxy for issue salience has its pitfalls, prior research mostly corroborates its robustness (Mellon 2013). As I are primarily interested in the public’s attentiveness towards an industry’s risks, I limit our search to Google News, thereby excluding Google Search results which for some industries, like aviation, are heavily-driven by consumer offerings, e.g., regarding flights. Market concentration is measured via the Herfindahl Index which is among the most commonly applied measures in the economics literature when assessing and comparing industry-level market concentration (Knot & Pasipanodya 2023). Additionally, it is reported at a highly granular-level, down to 5-digit NAICS codes, which allows us to better approximate market concentration for particular applications within the wider industry, which is our analysis’ focus.
	The decision logic for determining the proxy variable categories  in step 2 was based on a distribution of the existing case studies, when possible along logical steps or along quartiles. Nevertheless, these categories can be seen as somewhat arbitrary and dependent on the selected case studies.




