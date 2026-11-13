# Companion research and source index — evening 13 November 2026
**Scope:** Deepens same-day [morning lesson](../../../../phd-morning/lessons/2026-11-13/lesson.md) and [morning reference index](../../../../phd-morning/lessons/2026-11-13/references/index.md). Private syllabus sources: 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*, PDFs *Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf* (Unit 3 Session 4) and *AI_for_Cybersecurity_Learning_Pathway.pdf* (Unit 3 Session 3). The PDFs were read but are not redistributed.

## Primary papers and official material
1. **Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314.** https://arxiv.org/abs/2305.14314 . Frozen 4-bit backbone, NF4, double quantization, paged optimizers. Critique: performance on chatbot/instruction benchmarks does not establish rare SOC-event recall or universal latency benefit. Local path: **none** (PDF not downloaded; repository redistribution license not established). Supports G1/G2/G4.
2. **Edward J. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.** https://arxiv.org/abs/2106.09685 . Low-rank trainable factors with frozen pretrained weights. Critique: not evidence of 4-bit accuracy. Local path: **none**, link only. Supports G2/G4.
3. **Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg (2017 preprint, revised 2019). BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. arXiv:1708.06733.** https://arxiv.org/abs/1708.06733 . Outsourced/pretrained model trust and conditional trigger behavior. Critique: studied visual triggers, not measured SOC telemetry attack prevalence. Local path: **none**, link only. Supports C1/C3/C5.
4. **Apostol Vassilev et al. (2025). Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations. NIST AI 100-2e2025. DOI:10.6028/NIST.AI.100-2e2025.** https://doi.org/10.6028/NIST.AI.100-2e2025 . Official poisoning/evasion terminology and defenses taxonomy; not an efficacy trial. Local path: **none**, link only. Supports C1/C5.

## Paper inventory, licensing and access
No local research PDFs under `references/papers/` were created or committed. Public reading access at canonical paper pages does not by itself establish permission to redistribute copies in this repository. Before future PDF commits, inspect each paper's explicitly stated version-specific license and repository rights, then record the license, source URL, retrieved date and SHA256. No download is claimed.

## Claim-to-source map
| Claim | Local explanation and reproduction | Reference |
|---|---|---|
| Quantization output error bound, null-space and margin counterexamples | Evening G1 and G3 code | Local derivation; Dettmers for context |
| NF4, double quantization, paged optimizers | Evening G1/G2 | Dettmers 2023 |
| Adapter gradients, zero-initialization and merge risk | Evening G2/G3 | Hu 2022; local derivation |
| Poisoning sensitivity via implicit differentiation | Evening C2/C4 | Local derivation; NIST for terminology |
| Conditional trigger failure despite clean accuracy | Evening C3/C4 | Local synthetic audit; Gu 2017/2019 |
| Signed artifact identity is not behavioral safety | Evening C1/C5 | Local threat-model analysis; NIST |
| Base-rate precision | Evening C3/C4 | Bayes theorem and runnable arithmetic |
| Cross-track study design | Evening synthesis | Research proposal, not established empirical claim |

## Validation boundary
The morning lesson and references were fetched from GitHub `main` before preparation. Both Gmail syllabus PDFs were retrieved and read. The code is designed for Python 3.10+ (NumPy for G3; C4 standard library). Assertions and numerical calculations should be checked independently; no production model or real incident is evaluated. External canonical links were checked selectively; no claim of exhaustive paper review, PDF redistribution rights, or current-day breakthroughs. No email, audio or narration.
