# Evening references — 22 November 2026
Companion to [evening lesson](../lesson.md). The [same-day morning lesson](../../../../phd-morning/lessons/2026-11-22/lesson.md) and [morning reference index](../../../../phd-morning/lessons/2026-11-22/references/index.md) were read before writing. Both original 19 October 2026 syllabus PDFs (*Advanced Generative AI and Foundation Models Learning Pathway* and *AI for Cybersecurity Learning Pathway*, p.12 Unit 5 Session 3) were retrieved from Gmail.

|ID|Citation / canonical source|Detailed use, limits and critique|Local paper / license status|
|---|---|---|---|
|R1|Ho, Jonathan; Jain, Ajay; Abbeel, Pieter. **Denoising Diffusion Probabilistic Models** (2020), NeurIPS; arXiv:2006.11239. https://arxiv.org/abs/2006.11239|Gaussian forward process, posterior and denoising objective. The evening lesson derives the posterior and score; published image-synthesis benchmarks do not authenticate historical images.|No local PDF committed; arXiv open access for reading, version-specific redistribution rights not independently established.|
|R2|Rombach, Robin et al. **High-Resolution Image Synthesis with Latent Diffusion Models** (2022), CVPR; arXiv:2112.10752; DOI:10.1109/CVPR52688.2022.01042. https://arxiv.org/abs/2112.10752|Learned compression, latent denoising, cross-attention conditioning. Tiny-character failure is an explicitly proposed test, not a reported result from this paper.|No local PDF; repository redistribution license not independently established.|
|R3|Ho, Jonathan; Salimans, Tim. **Classifier-Free Diffusion Guidance** (2022), arXiv:2207.12598. https://arxiv.org/abs/2207.12598|Joint conditional/unconditional estimates and sample quality/diversity tradeoff; not a guarantee of prompt truth, calibrated probabilities or image provenance.|No local PDF; exact arXiv-version redistribution license not checked.|
|R4|Yao, Shunyu et al. **ReAct: Synergizing Reasoning and Acting in Language Models** (2023), ICLR; arXiv:2210.03629. https://arxiv.org/abs/2210.03629|Interleaved actions/observations; task performance is not a security proof and does not replace reference-monitor authorization.|No local PDF; version license not checked.|
|R5|OWASP GenAI Security Project. **LLM06:2025 Excessive Agency** (2025). https://genai.owasp.org/llmrisk/llm062025-excessive-agency/ ; also https://genai.owasp.org/llm-top-10/|Excessive functionality, permissions, autonomy; complete mediation and least privilege. Guidance is a threat model and checklist, not a measured success rate.|Official web guidance, not a local paper PDF.|
|R6|NIST. **Artificial Intelligence Risk Management Framework 1.0** (2023), NIST AI 100-1, DOI:10.6028/NIST.AI.100-1. https://doi.org/10.6028/NIST.AI.100-1|Govern, Map, Measure, Manage; governance reference, not an authorization implementation or certification.|Official stable publication link; no local PDF.|

## Claim-to-source map for offline tutoring
- Evening A1–A3: R1; algebra and synthetic worked values fully contained in lesson.
- Evening A4: R3; CFG formula, guidance extrapolation and limitations.
- Evening A5: R2; latent-space compression and cross-attention; numeric OCR hypothesis is original research design.
- Evening B1–B4: R4 for action-observation pattern; R5 for excessive agency and complete mediation; the invariant, TOCTOU demonstration and policy gateway are original defensive constructions.
- Evening B6 and research ethics: R5/R6. Labs are original, synthetic, no external service calls.
- Syllabus continuity: both October 19 PDFs and same-day morning lesson/index.

## Paper index and access log
No new paper files were written under references/papers/ in this run. Canonical arXiv records R1–R3 and official OWASP R5 were opened or surfaced through live research during preparation; R4 and R6 bibliographic identifiers were carried forward from the morning index and were not independently re-opened in this run. Open-access reading alone does not establish a license to redistribute a particular PDF version. Future paper commits require a recorded source URL, version, explicit license, retrieval date, SHA-256 and verified local path; never infer a local file from an external link.

## Reproduction and uncertainty
Both labs require only Python 3 and include deterministic assertions. The lesson's numerical examples are analytic/synthetic, not reproductions of published model-training results. No live SOC environment, real incident, generated audio, narration, or full diffusion training was used. Research questions are hypotheses, not established findings.
