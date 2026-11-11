# References and source index — 12 November 2026 morning

[Return to full lesson](../lesson.md). **Syllabus:** private Gmail message of 19 October 2026, *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*, attachments *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 3: SFT/adapters/LoRA) and *AI for Cybersecurity Learning Pathway* (Unit 3: threat models/evasion). Both were accessed and read for this run; do not publish them in this repository. Previous learning continuity: [11 November morning lesson](../../../2026-11-11/lesson.md) and its [references](../../../2026-11-11/references/index.md).

## Canonical papers and critical summaries
- **[G1] Edward J. Hu et al. (2022)**, *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, https://arxiv.org/abs/2106.09685 . Freeze pretrained weights and train low-rank factors; study adaptation capacity and the actual choice of projections, rank, datasets and baselines.
- **[G2] Tim Dettmers et al. (2023)**, *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314, https://arxiv.org/abs/2305.14314 . Quantized frozen model weights with LoRA adaptation; distinguish memory-saving method from claims about universal accuracy or runtime.
- **[G3] Neil Houlsby et al. (2019)**, *Parameter-Efficient Transfer Learning for NLP*, arXiv:1902.00751, https://arxiv.org/abs/1902.00751 . Bottleneck adapters as an alternative PEFT family; module insertion differs from LoRA's additive weight update.
- **[C1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy (2015)**, *Explaining and Harnessing Adversarial Examples*, arXiv:1412.6572, https://arxiv.org/abs/1412.6572 . First-order adversarial perturbation mechanism; unconstrained image-style perturbations do not prove feasibility in security telemetry.
- **[C2] Battista Biggio et al. (2013)**, *Evasion Attacks against Machine Learning at Test Time*, DOI:10.1007/978-3-642-40994-3_25, https://doi.org/10.1007/978-3-642-40994-3_25 . Evasion and adversarial capability framing; inspect assumed knowledge and feasible transformations before comparison.
- **[C3] NIST (2025)**, *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025 . Standardized attack and defense taxonomy; not an empirical robustness guarantee.

## Local paper index, licensing and access
**Local paths: none.** No research PDF was downloaded or committed in this run. Canonical abstract and DOI pages above provide stable routes. Redistribution rights and PDF licenses were not individually verified, so no file is represented as licensed for repository redistribution. To add a paper later under `references/papers/`, record original URL, retrieval date, explicit license, redistribution authorization, SHA-256 and citation. The original syllabus PDFs remain private.

## Claim-to-source map
- LoRA architecture, factorization and parameter efficiency: [G1], lesson G1–G3; numerical derivatives are derived in the lesson.
- Quantized adapter tuning: [G2], lesson G2/G4.
- Bottleneck adapter comparison: [G3], lesson G4.
- Gradient-based adversarial sensitivity: [C1], lesson C2; constrained box bound is a mathematical derivation.
- Test-time evasion threat modeling: [C2], lesson C1/C4.
- Attack taxonomy and lifecycle: [C3], lesson C1/C4.
- Practical feasibility constraints and defensive governance: lesson C1–C4 and cross-track synthesis, clearly labeled design analysis.
- All lab data are synthetic; no reported numerical results are attributed to the papers.

**Verification status:** canonical arXiv IDs/DOIs recorded; external links not individually live-resolved in this run. The labs have expected outputs and assertions, but remote CI execution is not claimed.
