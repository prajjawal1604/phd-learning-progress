# Evening references index — 2026-11-09
**Lesson:** [../lesson.md](../lesson.md). **Morning dependencies:** [morning lesson](../../../../phd-morning/lessons/2026-11-09/lesson.md) and [morning references](../../../../phd-morning/lessons/2026-11-09/references/index.md). **Syllabus:** October 19 Gmail PDFs *Advanced Generative AI and Foundation Models Learning Pathway*, Unit 2 Session 6, and *AI for Cybersecurity Learning Pathway*, Unit 2 Session 4. Morning sources were read before evening writing.

## Papers and source annotations
1. Vaswani et al., **Attention Is All You Need** (2017), arXiv:1706.03762, https://arxiv.org/abs/1706.03762 . Attention tensors and architecture; not a latency benchmark.
2. Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022), arXiv:2205.14135, https://arxiv.org/abs/2205.14135 . IO-aware exact attention; device-dependent speedups, not asymptotically linear exact attention.
3. Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (2023), arXiv:2305.13245, https://arxiv.org/abs/2305.13245 . KV-head sharing and conversion; compare accuracy, not memory alone.
4. Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention** (2023), arXiv:2309.06180, https://arxiv.org/abs/2309.06180 . Paged KV serving; throughput depends on workload and scheduler.
5. Sommer and Paxson, **Outside the Closed World: On Using Machine Learning for Network Intrusion Detection** (2010), DOI:10.1109/SP.2010.25, https://doi.org/10.1109/SP.2010.25 . Deployment and open-world ML critique; no universal threshold.
6. Saxe and Berlin, **eXpose: A Character-Level Convolutional Neural Network with Embeddings For Detecting Malicious URLs, File Paths and Registry Keys** (2017), arXiv:1702.08568, https://arxiv.org/abs/1702.08568 . String-based classification; historic data and drift limitations.
7. NIST, **Incident Response Recommendations and Considerations for Cybersecurity Risk Management**, SP 800-61 Rev. 3 (2025), DOI:10.6028/NIST.SP.800-61r3, https://doi.org/10.6028/NIST.SP.800-61r3 . Governance and response safeguards.
8. MITRE, **ATT&CK Enterprise** https://attack.mitre.org/ . TTP classification, not probabilistic evidence.
9. CISA, **#StopRansomware Guide** (2023), https://www.cisa.gov/stopransomware/ransomware-guide . Preparedness and response, check updates.

## Local papers and permissions
**Local PDF paths: none.** No open-access PDFs were committed; per-paper redistribution rights and license terms were not independently established. arXiv availability does not automatically grant redistribution permission. Stable canonical links above are provided instead. Source summaries are limited to established paper scope, not unverified claims of reproduced results.

## Source-to-claim map
- KV tensor/memory derivation: evening A1 and Lab A; sources 1,3.
- IO-bound exact attention and cache paging: A1–A2; sources 2,4.
- Queueing example: A3 (explicit M/M/1 assumptions; elementary analytical model, not a paper's empirical result).
- Bayesian LR fusion and decision costs: B1–B2 and Lab B (derivations self-contained); source 5 for real-world evaluation cautions.
- Phishing domain generalization: B3; sources 5,6.
- Defensive response, governance and safety: B3; sources 7–9.
- All numerical scenarios and expected lab results are synthetic; no production incident or hardware benchmark is asserted.

**Validation status:** code is standard-library Python with assertions; source URLs are canonical but not all were live-opened during this run. No downloaded papers or narration/audio artifacts are claimed.
