Dear Editor,

Thank you for the careful read and the encouraging signal. We have prepared a revised manuscript (paper_nmi_05_2026.tex) that addresses each of the three points; a brief summary follows.

1. Data and code availability.

We have added a Data Availability section at the end of the main text. The full dataset of 177,436 MCP tools is available from the corresponding author (merlin.stein@bsg.ox.ac.uk) on reasonable request. Public release of the entire dataset is constrained by upstream licensing: PyPI metadata is distributed under CC-BY 4.0, but a substantial fraction of the GitHub repositories from which we collect README content carry no explicit licence, and we therefore cannot redistribute their text. We are happy to share the dataset, including aggregate statistics, per-server classifications, and the list of public repository URLs covered, with referees during review under any confidentiality undertaking the journal would like to set in place.

We have also added a Code Availability section. Code that replicates the analyses in the paper is available from the corresponding author on reasonable request. We are not yet linking the public repository while we consolidate it, but the classification prompts (Appendix) and methodology (Methods, at the end of the paper) are documented in sufficient detail to permit independent re-implementation.

2. Length and structure.

We have moved the Methodology to a Methods section at the end of the paper, after the Conclusions. We have shortened Background sections 2.3 and 2.4 by moving Tables 1, 2, and 3, the cross-study illustration paragraph, and the descriptive paragraphs on domains, practitioner studies, and worldwide usage to a new first appendix subsection ("Action space examples and cross-study comparison"). The remaining narrative in 2.3 and 2.4 has been compressed.

3. Geographic representation of China (5%).

We agree that the 5% figure understates Chinese usage. Our existing caveat in Results section 4.2 already flags this: PyPI is Western-centric and underrepresents activity in regions using alternative distribution channels. The mechanism behind that caveat is that our geography numbers are derived from pypi.org IP logs and do not capture installs proxied through Chinese domestic PyPI mirrors (Tsinghua TUNA, Aliyun, Tencent Cloud, USTC) nor code distributed via Gitee, the dominant Chinese code-hosting platform. Both Chinese-developed agent tooling and Chinese-side downloads of Western MCP servers are therefore systematically undercounted, and the rapid 2025-2026 growth of Chinese agent platforms is not visible in PyPI logs. We have not added new analysis in the manuscript because we believe the existing caveat is already sufficient to flag this, and a fuller treatment would require domestic-mirror and Gitee data we do not currently have access to. If you would prefer the manuscript to spell out the mirror/Gitee mechanism explicitly, we are happy to do so.

Please let us know if you would like the dataset shared with referees through a different channel, or any further changes to address.

Best regards,
Merlin Stein (on behalf of the authors)
