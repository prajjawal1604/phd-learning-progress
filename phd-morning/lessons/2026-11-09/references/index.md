# References and paper access index — 2026-11-09 morning
**Lesson:** [../lesson.md](../lesson.md). **Syllabus:** October 19, 2026 Gmail attachments *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 2 Session 6, p.6) and *AI for Cybersecurity Learning Pathway* (Unit 2 Session 4, p.6). Both PDFs were read through Gmail attachment extraction for the relevant syllabus sections. The attachments themselves are not copied into the repository.

## Research sources and annotations
1. Vaswani, A. et al. (2017), **Attention Is All You Need**, arXiv:1706.03762. https://arxiv.org/abs/1706.03762 . Transformer architecture and attention arithmetic; not a production-serving performance guarantee.
2. Dao, T. et al. (2022), **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**, arXiv:2205.14135. https://arxiv.org/abs/2205.14135 . IO-aware tiled exact attention; performance depends on hardware and kernels.
3. Ainslie, J. et al. (2023), **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**, arXiv:2305.13245. https://arxiv.org/abs/2305.13245 . Reducing KV heads can reduce cache memory, with model-quality trade-offs.
4. Kwon, W. et al. (2023), **Efficient Memory Management for Large Language Model Serving with PagedAttention**, arXiv:2309.06180. https://arxiv.org/abs/2309.06180 . Paging KV memory and serving throughput; workload/scheduler affect replication.
5. Saxe, J. and Berlin, K. (2017), **eXpose: A Character-Level Convolutional Neural Network with Embeddings For Detecting Malicious URLs, File Paths and Registry Keys**, arXiv:1702.08568. https://arxiv.org/abs/1702.08568 . Character-level security strings; historical dataset and distribution shift constrain generalization.
6. Sommer, R. and Paxson, V. (2010), **Outside the Closed World: On Using Machine Learning for Network Intrusion Detection**, DOI:10.1109/SP.2010.25. https://doi.org/10.1109/SP.2010.25 . Critical account of operational deployment and open-world evaluation.
7. Anderson, H. S. and Roth, P. (2018), **EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models**, arXiv:1804.04637. https://arxiv.org/abs/1804.04637 . Prior-session static malware baseline, not a ransomware behavior benchmark.

## Official guidance
- MITRE ATT&CK, **Enterprise techniques and knowledge base**: https://attack.mitre.org/ . Observed TTP taxonomy; not a calibrated attack probability.
- NIST, **SP 800-61 Rev. 3: Incident Response Recommendations and Considerations for Cybersecurity Risk Management** (2025), https://doi.org/10.6028/NIST.SP.800-61r3 . Incident response governance and organizational risk integration.
- CISA, **#StopRansomware Guide** (2023), https://www.cisa.gov/stopransomware/ransomware-guide . Defensive preparedness, detection and response. Check current revision before operational adoption.

## Local PDFs and access/redistribution status
**Local paper paths:** none. No research paper PDFs were downloaded or committed in this run; individual licenses and redistribution permissions were not independently established. arXiv abstract pages offer access to manuscripts, but availability alone is not a blanket redistribution license. Use the stable source links above. No claim is made that the source pages were freshly fetched during this run.

## Claim-to-source map
- Attention and cache tensors: sources 1–4; mathematical example and deterministic simulator in lesson §§1–3.
- Inference speed versus throughput and memory: sources 2–4; numeric latency is an explicit *illustrative model* rather than a measured claim.
- Ransomware/C2/phishing evidence and safety boundaries: MITRE ATT&CK, CISA, NIST; §5 synthetic scenario.
- Base-rate and false-positive economics: Bayes identity in lesson §6 and Lab C; source 6 contextualizes deployment challenges.
- Character-level URL detection and dataset shift: source 5; source 7 is historical malware comparison.
- Research critiques and proposed experiments are analyses, not reproductions or results.

**Reproducibility:** Both labs use Python 3.10+ standard library only and include assertions; execute locally with `python3` after extracting code blocks. Execution has not been independently confirmed by this repository write.
