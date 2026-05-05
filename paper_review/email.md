Dear Editor,

Thank you for the careful read and the encouraging signal. Have prepared a revised manuscript, that addresses each of the three points; a brief summary follows.

1. Data and code availability.

We have added a Code availability and Data Availability section at the end of the main text. The full dataset of 177,436 MCP tools is available on reasonable request. Uploaded the dataset already (one on tool level, one on server level, as described in the paper), and can share a replication code repo once needed (details for replication below already, which should work for most coding agents). Public release of the entire dataset is constrained by upstream licensing, unfortunately. PyPI metadata is distributed under CC-BY 4.0, but a portion of the GitHub repositories from which we analyse README content carry no explicit licence, and we therefore can only run online analysis on the data but cannot redistribute it. We are happy to share parts of the  dataset, including aggregate statistics, per-server classifications, and the list of public repository URLs covered, with referees during review under any confidentiality undertaking the journal would like to set in place.

2. Length and structure.

We have combined the Data and Methodology sections into a single Methods section at the end of the paper, after the Conclusions. 
We have also shortened Background sections 2.3 and 2.4 (compressed and tables moved to annex.)

3. Geographic representation of China (5%).

We agree that the 5% figure understates Chinese usage. A caveat in Results section 4.2 flags this now. PyPI is Western-centric and underrepresents activity in regions using alternative distribution channels. The mechanism behind that caveat is that our geography numbers are derived from pypi.org IP logs and do not capture installs proxied through Chinese domestic PyPI mirrors (Tsinghua TUNA, Aliyun, Tencent Cloud, USTC) nor code distributed via Gitee, the dominant Chinese code-hosting platform. Both Chinese-developed agent tooling and Chinese-side downloads of Western MCP servers are therefore systematically undercounted, and the rapid 2025-2026 growth of Chinese agent platforms is not visible in PyPI logs. Unfortunately, download statistics for Chinese platforms are not publicly available. 

Please let us know if you would like the dataset shared with referees through a different channel, or any further changes to address.

Best regards,
Merlin Stein


Annex: Note for referees on the uploaded datasets

We share two gzipped CSVs that together back every figure and table in the paper. Both files share the keys `server_id` so they can be joined on `server_id`.

`clservers_classified.csv.gz` (one row per MCP server, ~19k rows). Per server, it carries:
- Identification and provenance: `server_id`, `server_name`, `server_data_sources` (GitHub / Smithery / official / awesome list), `canonical_official` flag, `created_at`.
- Classifications used in the paper: server-level direct-impact (`tool_functionality_main`, plus `highest_automation_func` aggregating across the server's tools), generality of the environment the server accesses (`generality_environment`: 1 = unconstrained / general-purpose, 0 = constrained / narrow-purpose), payment-autonomy level (`payments_autonomy`, 0-4), and AI-coauthorship (`ai_authored` yes/no, plus `likely_ai_agent` naming the detected coding agent).
- O*NET task domain at server level: `level1_name` (12 high-level domains), `occupation_name`, `occupation_title`, and `impact_of_decisions` (the O*NET 0-100 stakes score).
- Usage: aggregate totals as `usage_npm_downloads`, `usage_pypi_downloads`, `usage_total_downloads`, `use_count`. Monthly and per-country detail is nested in the string column `usage_monthly_breakdown`, which on parsing (`ast.literal_eval` of the cell) yields a per-month dict with keys including monthly NPM/PyPI download counts and `pypi_by_country` (a `{country_code: downloads}` map for that month, available where pypi.org publishes IP-geolocation).

`cltools_classified.csv.gz` (one row per tool, ~177k rows). Per tool, it carries:
- Identification and join: `tool_id`, `server_id`, plus the inherited server fields above.
- Tool-level O*NET domain hierarchy: `level1_name`, `level2_name`, `level3_name` (12 / 400 / 18,796 categories).
- Tool-level direct-impact and functionality: `tool_functionality_main` (perception / reasoning / action) and `functionality_sub` (the more granular sub-functionality, e.g., computer use, software extension, code execution).

Together these two files are the immediate input to every figure and table in the paper. The trend figures on direct impact, generality, geography and AI co-authorship are produced by grouping the tool-level file by `year_month` and the relevant classification column, weighted by the download columns. The task-domain table and the cross-study comparison table are aggregations of the same fields at server or tool level. The high-stakes example for finance / payments uses the `is_finance_llm`, `payments_autonomy`, `impact_of_decisions` and time columns directly.

Per-figure replication guide (all from the two CSVs above):

- Figure 1, panels A-E. Panel A is a static schematic of the pipeline. Panel B: cumulative count of tools by `creation_date` month from `cltools_classified.csv.gz`, plus a second series restricted to rows with `ai_authored == "yes"` and indexed by the first month of detected AI evidence. Panel C: same cumulative count but split by `level1_name` (top 5 domains; remainder grouped as "Other"). Panel D: monthly sum of NPM + PyPI downloads from `clservers_classified.csv.gz`'s parsed `usage_monthly_breakdown` (monthly `npm_downloads + pypi_downloads`), total and AI-coauthored series. Panel E: for each month, share of total downloads that are action tools (`tool_functionality_main == "action"` from the tool-level file), with three further series restricted to (i) action AND `generality_environment == 1` (unconstrained), (ii) action AND `impact_of_decisions` in 50-75, (iii) action AND `impact_of_decisions` >= 75. WLS fits weighted by monthly downloads.
- Figure 2 (consequentiality scatter). From `cltools_classified.csv.gz` filtered to action tools: count distinct tools per `occupation_title`, plot against `impact_of_decisions` (one dot per occupation). Quadratic polynomial fit; pink-shaded region = `impact_of_decisions` > 75. Drop occupations with zero matched tools.
- Figure 3 (geographic world map). From the `pypi_by_country` entries inside `usage_monthly_breakdown` in `clservers_classified.csv.gz`, restricted to servers with any action tools (`highest_automation_func == "action"` or any tool with `tool_functionality_main == "action"`). Sum the per-month `{country_code: downloads}` maps over Nov 2024-Oct 2025, divide by global total to get the country share; compute H1-H2 2025 delta for the bracketed numbers. (Note: HK is folded into CN in the published map.)
- Figure 4 (perception/reasoning/action over time). Stacked area chart of monthly download share by `tool_functionality_main` and `functionality_sub`, computed on `cltools_classified.csv.gz` joined to monthly downloads from each server's `usage_monthly_breakdown`, grouped by `year_month` (server downloads divided across that server's tools). Asymptotic-convergence WLS fit on the action total.
- Figure 5 (general-purpose share over time). From `clservers_classified.csv.gz`: top panel = monthly download-weighted share where `generality_environment == 1` (unconstrained), using monthly downloads from `usage_monthly_breakdown`; bottom panel = cumulative count-share of `generality_environment == 1` servers by `created_at` month. Polynomial-convergence WLS fit.
- Figure 6 (AI-coauthored servers by coding agent). From `clservers_classified.csv.gz`: for each month group new servers (by `created_at`) into AI-coauthored vs not, and within AI-coauthored split by `likely_ai_agent` (Claude / Cursor / Copilot / Codex / other). Plot the stacked share series; quadratic WLS fit weighted by monthly server count.
- Figure 7 (agentic transaction tools). From `clservers_classified.csv.gz` filtered to `payments_autonomy >= 1`, plot the cumulative count of servers by month, stacked by `payments_autonomy` level (1-4).
- Appendix figure (cumulative usage distribution). Rank servers in `clservers_classified.csv.gz` by total NPM downloads (and separately by total PyPI downloads, and by `use_count`). For each ranking plot the cumulative share of downloads as a function of rank percentile.
- Appendix figure (bottom-up subclusters). The subcluster scatter is generated by the topic-modelling pipeline described in Methods (Stella-400M embeddings of server READMEs, UMAP, HDBSCAN), not by aggregation of the two CSVs. The cluster IDs and labels are saved alongside the analysis outputs; happy to share these on request.

We are happy to share the exact aggregation / plotting scripts with referees on request.
