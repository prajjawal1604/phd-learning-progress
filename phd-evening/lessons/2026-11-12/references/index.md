# Companion references index — evening 12 November 2026

[Full self-contained evening lesson](../lesson.md) | [Same-day morning lesson](../../../../phd-morning/lessons/2026-11-12/lesson.md) | [Morning index](../../../../phd-morning/lessons/2026-11-12/references/index.md)

## Syllabus and provenance
Both private attachments from the 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027* were accessed and read: **Advanced Generative AI and Foundation Models Learning Pathway**, Unit 3 (SFT, adapters, LoRA, quantization); **AI for Cybersecurity Learning Pathway**, Unit 3 (adversarial threat models, evasion and mitigations). They are not republished. The committed morning lesson and index were fetched from `main` before this evening lesson was written. No unrelated syllabus unit is introduced.

## Canonical research sources and critical summaries
- **[G1] Hu, Edward J., et al. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. https://arxiv.org/abs/2106.09685 . Frozen pretrained matrices plus trainable low-rank deltas. Key limitation: adaptation effectiveness depends on tasks, rank, target modules and baselines; parameter count does not equal end-to-end training-memory savings.
- **[G2] Dettmers, Tim, et al. (2023).** *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314. https://arxiv.org/abs/2305.14314 . Quantized frozen weights and trainable adapters; distinguish memory savings, numerical effects and task performance. Not a guarantee for all hardware or tasks.
- **[G3] Houlsby, Neil, et al. (2019).** *Parameter-Efficient Transfer Learning for NLP.* arXiv:1902.00751. https://arxiv.org/abs/1902.00751 . Bottleneck adapter modules; useful PEFT comparator but not the same as a low-rank additive projection delta.
- **[G4] Eckart, Carl & Gale Young (1936).** *The Approximation of One Matrix by Another of Lower Rank.* DOI:10.1007/BF02288367. https://doi.org/10.1007/BF02288367 . Foundational optimal low-rank approximation result under specified norms; does not prove neural task quality.
- **[C1] Goodfellow, Ian J., Jonathon Shlens & Christian Szegedy (2015).** *Explaining and Harnessing Adversarial Examples.* arXiv:1412.6572. https://arxiv.org/abs/1412.6572 . First-order adversarial sensitivity; image perturbation assumptions may not yield feasible changes to structured telemetry.
- **[C2] Biggio, Battista, et al. (2013).** *Evasion Attacks against Machine Learning at Test Time.* DOI:10.1007/978-3-642-40994-3_25. https://doi.org/10.1007/978-3-642-40994-3_25 . Attacker-aware test-time evasion; operational relevance depends on knowledge, editable features and valid constraints.
- **[C3] Carlini, Nicholas & David Wagner (2017).** *Towards Evaluating the Robustness of Neural Networks.* arXiv:1608.04644. https://arxiv.org/abs/1608.04644 . Stronger attack evaluations expose weaknesses of some defenses; the work does not certify feasible manipulation of every cyber feature type.
- **[C4] NIST (2025).** *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations.* NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025. https://doi.org/10.6028/NIST.AI.100-2e2025 . Standardized language for attack capabilities and defenses; not an empirical robustness result.

## Local paper index and access notes
**Local paper paths: none.** No open-access research PDFs were downloaded or committed. Canonical arXiv/DOI landing pages above are stable reading routes, but individual PDF licenses, redistribution authorization and live HTTP availability were not verified here. Do not imply that “open access” necessarily authorizes redistribution. For a future permitted import into `references/papers/`, record exact local path, original PDF URL, retrieval date, license evidence, SHA-256, and any attribution conditions. The two private syllabus PDFs must remain private.

## Source map
- Low-rank factorization, trainable-parameter savings: G1; lesson G1–G3.
- Quantized frozen base and adapters: G2; lesson G3/G5.
- Bottleneck adapters: G3; lesson G5.
- Rank-one impossibility and SVD error: G4; lesson G1 and Lab G.
- Gradient, zero initialization, gauge freedom, composition: lesson G2–G4 (mathematical derivations, synthetic assertions; not empirical literature claims).
- Gradient sensitivity and adversarial examples: C1; lesson C2.
- Evasion threat models: C2; lesson C1.
- Weak attack versus security certification: C3; lesson C3/C5.
- Lifecycle taxonomy and governance: C4; lesson C1/C5 and cross-track synthesis.
- Conservation-coupled feasible sets, precision and threshold examples: lesson C2–C4, original mathematical exercises.
- No real malware, production telemetry or unauthorized probing was used; no current-day release claim is made.

**Verification limits:** Morning source files and private syllabus PDFs were retrieved; canonical IDs recorded, but individual external links and redistribution licenses were not independently verified. Labs include assertions and expected outputs; remote CI execution is not claimed. No audio, narration or email was produced.
