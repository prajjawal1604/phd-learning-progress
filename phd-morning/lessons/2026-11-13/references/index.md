# References and claim-to-source index — 13 November 2026 morning

## Syllabus and continuity
- Private source: 19 October 2026 Gmail message “PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027”; attachment `Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf`, Unit 3 Session 4 (quantization, efficient adaptation), and `AI_for_Cybersecurity_Learning_Pathway.pdf`, Unit 3 Session 3 (poisoning, backdoors, supply chain). These are private Gmail attachments, **not** committed to GitHub.
- [12 Nov morning](../../../../phd-morning/lessons/2026-11-12/lesson.md) and [12 Nov evening](../../../../phd-evening/lessons/2026-11-12/lesson.md) provide the prior LoRA and constrained evasion groundwork.

## Primary research / official references
1. **Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). “QLoRA: Efficient Finetuning of Quantized LLMs.” arXiv:2305.14314.** Canonical https://arxiv.org/abs/2305.14314 . Explains frozen quantized base, NF4, double quantization, paged optimizers and empirical adaptation evaluations. Critique: chatbot benchmarks, hardware-specific kernels and peak-memory methodology limit generalization. Supports G2/G4. **Local PDF:** not downloaded; canonical open-access paper link only; arXiv availability is not blanket permission to redistribute.
2. **Edward J. Hu et al. (2022). “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv:2106.09685.** https://arxiv.org/abs/2106.09685 . Establishes frozen-base low-rank adaptation and trainable factor accounting. Does not establish 4-bit quantization. Supports G2. **Local PDF:** not downloaded; link only.
3. **Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg (2017 preprint; revised 2019). “BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.” arXiv:1708.06733.** https://arxiv.org/abs/1708.06733 . Shows conditional trigger failures with relatively preserved clean behavior in studied image-model settings; cannot be assumed to transfer unchanged to SOC telemetry. Supports C1/C3/C5. **Local PDF:** not downloaded; link only.
4. **Apostol Vassilev et al. (2025). “Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations.” NIST AI 100-2e2025. DOI:10.6028/NIST.AI.100-2e2025.** https://doi.org/10.6028/NIST.AI.100-2e2025 . Official taxonomy and terminology for attacker objectives, capabilities, poisoning and defenses; not an effectiveness evaluation. Supports C1/C5. **Local PDF:** not downloaded; official link only.

## Paper inventory / access status
No research PDF has been committed under `references/papers/` in this run. The links above identify canonical, publicly accessible sources. Distribution and repository license permissions were not independently established, so PDFs were deliberately not copied into the repo. This index records the exact gap rather than suggesting files exist.

## Claim map and reproduction
- Uniform affine quantizer, quantization error and norm bound: lesson G1 and `quantization_lab.py` embedded in G3; derived mathematically, not attributed to QLoRA's NF4.
- Storage and adapter gradient flow: lesson G2; compare Dettmers and Hu.
- Contamination mixture and implicit-gradient sensitivity: lesson C2 (local derivation, requiring invertible Hessian and local optimum); NIST for terminology.
- Backdoor conditional audit: lesson C3/C4; Gu et al. for the motivating case, **not** for toy numerical rates.
- Integrity controls: lesson C3/C5; NIST taxonomy plus local security engineering recommendations.
- Expected code output is synthetic. Use Python 3.10+; G3 needs NumPy, C4 is standard library. No live target or malware is required.

**Validation boundary:** canonical arXiv and NIST identifiers checked during preparation; no claim that every linked PDF was downloaded, licensed for redistribution or independently replicated.
