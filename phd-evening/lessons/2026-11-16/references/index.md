# Companion references and source index — evening 16 November 2026

## Dependency and syllabus provenance
Read committed [morning lesson](../../../../phd-morning/lessons/2026-11-16/lesson.md) and [morning references](../../../../phd-morning/lessons/2026-11-16/references/index.md) on GitHub main before drafting. Both original PDF attachments were retrieved from Gmail's 19 October 2026 message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*: *Advanced Generative AI and Foundation Models Learning Pathway* (p.10 Unit 4 Session 1, retrieval) and *AI for Cybersecurity Learning Pathway* (p.8 Unit 3 Session 6, robustness). These are private syllabus files and not redistributed.

## Primary and official references
| Reference / canonical link | Main contribution, limitations and critical reading | Local paper path / access note |
|---|---|---|
| Patrick Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS, arXiv:2005.11401, https://arxiv.org/abs/2005.11401 | RAG-Sequence and RAG-Token marginalize retrieved passages; gradient in evening G2 follows from their model. Enterprise ACL, temporal validity and claim entailment are outside the paper's proof of concept. | None. Canonical arXiv record verified in morning; no PDF committed, version-specific redistribution terms not independently verified. |
| Stephen Robertson & Hugo Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in Information Retrieval, DOI:10.1561/1500000019, https://doi.org/10.1561/1500000019 | BM25 and probabilistic retrieval foundations. Scores are not calibrated truth or authorization; lexical retrieval can miss paraphrase. | None. Publisher permission not established; use DOI link. |
| Gordon V. Cormack, Charles L. A. Clarke & Stefan Buettcher (2009), *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR, DOI:10.1145/1571941.1572114, https://doi.org/10.1145/1571941.1572114 | Rank fusion without raw-score comparability; benchmark result does not establish tenant-safe retrieval or score calibration. | None. Canonical DOI supplied; live page verification not independently completed this run; no publisher PDF committed. |
| Aleksander Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR, arXiv:1706.06083, https://arxiv.org/abs/1706.06083 | Min-max adversarial training and first-order attacks; vision norm-ball threat models do not automatically map to causal telemetry. | None. Canonical arXiv record verified in morning; no PDF committed because version-specific reuse terms unverified. |
| Apostol Vassilev et al. (2025), *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025 | Official taxonomy for attacker goals, capabilities, lifecycle and mitigations; not a guarantee that any tested defense is secure. | None. Official download exists; precise reuse conditions for committing were not independently checked. |

## Paper index and download status
No research PDFs were added to \`references/papers/\` in this run. **No local paper paths exist or are implied.** Open-access readability alone does not establish a redistribution license. Use canonical links above; license/access status should be rechecked before any future commit. Both private syllabus PDFs were read but not copied.

## Claim-to-source map for offline teaching
- G1 authorization, date validity and data/control boundaries: morning lesson G1/G4, syllabus GenAI Unit4 Session1, evening G1/G4; engineering design rather than a direct paper measurement.
- G2 latent-passage likelihood and posterior responsibility: Lewis 2020; gradient p-r and numeric example derived in evening G2 and lab G5.
- G3 BM25: Robertson & Zaragoza 2009; RRF: Cormack et al. 2009; exact ranks, recall/MRR definitions and caveats in evening G3/G5.
- C1 robust min-max and first-order attacks: Madry 2018; dual norm and nullspace projection are explicit mathematical derivations in evening C1.
- C2 thresholds, abstention and base-rate precision: local derivation and synthetic calculations; NIST 2025 provides broader risk context.
- C3/C5 threat-model taxonomy, attack budgets, adaptive testing and limitations: NIST 2025, Madry 2018, evening C3/C5.
- Research hypotheses in G6/C5 are proposed experiments, not established findings.

## Reproduction checklist
Run \`rag_deep_lab.py\` and \`robust_deep_lab.py\` by copying the corresponding fenced blocks from the evening lesson, using Python 3.10+ standard library. Both include deterministic assertions and expected outputs. Record Python version and actual console output if rerun. These toy labs do not certify any real deployed model or system. No narration, audio or email artifacts are produced.