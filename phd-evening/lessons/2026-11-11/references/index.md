# Companion research and paper index — 11 November 2026 evening
[Full evening lesson](../lesson.md) | [Morning lesson](../../../../phd-morning/lessons/2026-11-11/lesson.md) | [Morning index](../../../../phd-morning/lessons/2026-11-11/references/index.md)

**Syllabus:** October 19 Gmail attachments: *Advanced Generative AI and Foundation Models Learning Pathway*, p.8 Unit 3 Session 2; *AI for Cybersecurity Learning Pathway*, pp.6–7 Unit 2 Session 6. The original attachments were accessed for this run, but are private and must not be copied into a public repository.

## Canonical papers, claims and critique
1. Long Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155, https://arxiv.org/abs/2203.02155 . Demonstration SFT plus subsequent preference-based alignment; cannot isolate SFT's contribution from full pipeline outcomes. Lesson G1/G4.
2. Jason Wei et al. (2022), *Finetuned Language Models Are Zero-Shot Learners*, arXiv:2109.01652, https://arxiv.org/abs/2109.01652 . Instruction transfer; scrutinize task overlap, model scale and unseen-task validity. G4.
3. Edward J. Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, https://arxiv.org/abs/2106.09685 . Low-rank weight update; parameter reduction does not establish whole-system memory savings or privacy. G2/G4.
4. Colin Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21, arXiv:1910.10683, https://arxiv.org/abs/1910.10683 . Unified text-to-text formulation; not evidence of guaranteed factual grounding. G4.
5. Iman Sharafaldin, Arash Habibi Lashkari and Ali A. Ghorbani (2018), *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, DOI:10.5220/0006639801080116, https://doi.org/10.5220/0006639801080116 . CICIDS2017 construction; question capture-specific shortcuts and labeling. C1/C4.
6. Hyrum S. Anderson and Phil Roth (2018), *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637, https://arxiv.org/abs/1804.04637 . Static prepared PE feature benchmark; not dynamic malware execution. C1/C4.
7. Min Du et al. (2017), *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*, DOI:10.1145/3133956.3134015, https://doi.org/10.1145/3133956.3134015 . Sequential log anomaly model; depends on log-template consistency and operational drift. C4.
8. Takaya Saito and Marc Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, DOI:10.1371/journal.pone.0118432, https://doi.org/10.1371/journal.pone.0118432 . Precision/recall under skew; does not itself optimize analyst cost. C2/C4.

## Local paper inventory and access
**Local PDFs: none; no files added to `references/papers/`.** The canonical abstract and DOI landing links above are provided for reading. Permission to redistribute the publisher PDFs in a public GitHub repository was not independently verified. Open-to-read does not always mean licensed for redistribution. Before adding a PDF, verify its specific license, record direct source URL, retrieval date, license terms, SHA-256 and relative local path. The syllabus PDFs are private and must not be republished.

## Claim-to-source map
- Response-only loss, gradients, masking, sequence weighting: evening G1/G3 derivations, morning G2/G3; theoretical calculations, not reported experiment.
- LoRA rank and parameter savings: G2, paper 3; memory caveats in G2.
- SFT versus RAG and evidence checks: G2/G4, morning G1/G4; proposed experiment rather than established universal outcome.
- Dataset provenance and unit/split validity: C1/C4, papers 5–7, morning C1/C4.
- Rare-event precision and cost threshold: C2/C3, paper 8 for imbalance context; numbers are constructed.
- Prior-shift adjustment, group bootstrap and p95 caveats: C2/C3 explicit derivations; methodological recommendations, not empirical claims from papers 5–8.
- Ethical guardrails and integrated SOC use: G2/C4/cross-track; reasoned operational recommendations.

**Verification boundary:** The source identifiers are canonical stable links; no claim is made that every destination was live-resolved during this run. No paper PDF download, audio, email or narration was performed.
