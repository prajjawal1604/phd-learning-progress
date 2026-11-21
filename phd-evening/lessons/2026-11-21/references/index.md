# Companion references — evening 21 November 2026
Parent: [evening lesson](../lesson.md). Dependency: [same-day morning lesson](../../../../phd-morning/lessons/2026-11-21/lesson.md) and [morning references](../../../../phd-morning/lessons/2026-11-21/references/index.md). Syllabus: 19 October 2026 Gmail attachments *Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf* and *AI_for_Cybersecurity_Learning_Pathway.pdf*, Unit 5 Session 2. Both attachments retrieved during this run.

|ID|Primary or official source|What to learn; what not to infer|Local PDF status|
|---|---|---|---|
|R1|Junnan Li et al., **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** (2022), arXiv:2201.12086, https://arxiv.org/abs/2201.12086|Caption bootstrapping and unified vision-language tasks; no guarantee of numerical or cell-level correctness.|Not downloaded|
|R2|Yupan Huang et al., **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking** (2022), arXiv:2204.08387, https://arxiv.org/abs/2204.08387|Masked text/image and word-patch alignment for document AI; evaluate OCR/domain shift and region provenance separately.|Not downloaded|
|R3|Patrick Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2020), arXiv:2005.11401, https://arxiv.org/abs/2005.11401|Retrieval-conditioned sequence generation; original study does not test tenant ACLs or SOC action controls.|Not downloaded|
|R4|Akari Asai et al., **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (2023), arXiv:2310.11511, https://arxiv.org/abs/2310.11511|Adaptive retrieval and reflection; self-critique is not an independent verifier or authorization boundary.|Not downloaded|
|R5|NIST, **AI Risk Management Framework 1.0** (2023), NIST AI 100-1, DOI:10.6028/NIST.AI.100-1, https://doi.org/10.6028/NIST.AI.100-1|Govern, Map, Measure, Manage framework; not experimental proof of model safety.|Not downloaded|
|R6|OWASP, **Top 10 for LLM Applications** (2025 edition), https://genai.owasp.org/llm-top-10/|Injection and disclosure threat categories; mitigation checklist is not certification.|Not downloaded|

## Paper index and licensing/access
No PDF files were downloaded or committed to references/papers/ during this run. Public abstract/landing access is not equivalent to an explicit redistribution license for the exact version. Local paper paths therefore do not exist and are not invented. Before any later PDF commit, confirm the paper/version license, record canonical source URL, download timestamp, checksum, permission notes and local path. Stable links above provide lawful fallback. The arXiv landing pages for R2/R3/R4 were live-checked in this run; R1/R5/R6 were inherited from the morning index or official identifiers and not all reopened individually.

## Claim-to-source map
- Evening A1–A3: BLIP R1, LayoutLMv3 R2; Jacobian and finite-difference exercise are original mathematical teaching derivations, not quoted paper findings.
- Evening A3–A4: exact cell-binding failures and synthetic test code; no external performance claims.
- Evening B1–B4: RAG R3 and Self-RAG R4 for retrieval and reflection context; eligibility prefilter, time-validity and authorization rechecks are proposed application controls, not established by those experiments.
- Evening B5 and ethics: NIST R5, OWASP R6; study designs are proposed, not completed.
- Local continuity: same-day morning files above, plus 20 November prior lessons linked from morning source map.

## Reproducibility checklist
Python 3 standard library, synthetic data, deterministic assertions; preserve script text in evening lesson. Run labs separately; record interpreter version, stdout, exit code, and changes. This run did not perform neural-model training or a production SOC experiment. No audio or narration artifacts were created.
