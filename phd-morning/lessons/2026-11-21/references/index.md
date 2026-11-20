# References index — morning 21 November 2026
Parent: [lesson](../lesson.md). Source syllabus: 19 October 2026 Gmail message `PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027`, attachments `Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf` p.12 Session 2 and `AI_for_Cybersecurity_Learning_Pathway.pdf` p.12 Session 2. Continuity: [20 Nov morning](../../../../phd-morning/lessons/2026-11-20/lesson.md), [20 Nov evening](../../../../phd-evening/lessons/2026-11-20/lesson.md).

## Primary research and official guidance
|ID|Bibliographic record and stable link|What it establishes; critique|Local PDF|
|---|---|---|---|
|R1|Li, Junnan; Li, Dongxu; Xiong, Caiming; Hoi, Steven. *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation* (2022), arXiv:2201.12086, https://arxiv.org/abs/2201.12086|Flexible understanding/generation with caption generation/filtering for noisy pairs. Does not guarantee numeric extraction, region grounding or SOC reliability.|Not downloaded|
|R2|Huang, Yupan; Lv, Tengchao; Cui, Lei; Lu, Yutong; Wei, Furu. *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking* (2022), arXiv:2204.08387, https://arxiv.org/abs/2204.08387|Joint text/image masking and word-patch alignment for document understanding; OCR/layout shifts and exact table-cell provenance remain application-specific tests.|Not downloaded|
|R3|Lewis, Patrick et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), NeurIPS, arXiv:2005.11401, https://arxiv.org/abs/2005.11401, DOI:10.48550/arXiv.2005.11401|Parametric generator plus nonparametric retrieval improves knowledge-intensive QA in studied settings. Does not test tenant ACLs, effective-time policy or security tool use.|Not downloaded|
|R4|NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (2023), NIST AI 100-1, https://doi.org/10.6028/NIST.AI.100-1|Govern/Map/Measure/Manage risk-management guidance; not empirical proof of any SOC assistant's safety.|Not downloaded|
|R5|OWASP, *Top 10 for LLM Applications* (2025 edition), https://genai.owasp.org/llm-top-10/|Threat taxonomy for injection and information leakage; checklist does not certify mitigation efficacy.|Not downloaded|

## Access and licensing
Canonical abstract/landing pages are publicly readable. **No research PDF was downloaded or committed** in this run: version-specific license/redistribution permissions were not independently verified for every item. The table deliberately has no invented `references/papers/*.pdf` paths. Access to a PDF does not itself establish repository redistribution permission. Consult each version's license and repository policy before committing bytes; otherwise use the canonical links above. This index is the paper index and records download status explicitly.

## Source-to-claim map
- A1–A4: BLIP R1 and LayoutLMv3 R2 for architecture/motivation; cross-attention math is independently derived in lesson A2; exact-cell examples are synthetic.
- B1–B3: RAG R3 for retrieval+generation formulation; BM25 and RRF formulas are explained and independently instantiated in the lesson; ACL/time controls are security design requirements, not claims of R3.
- B1/B3 and ethics: R4/R5 for governance and application threat taxonomy.
- A5/B4: standard-library executable toy exercises; expected results are mathematical checks only.
- Syllabus topic selection: the two original Gmail PDF attachments and the linked November 20 local lessons.

## Verification status
Canonical arXiv records R1/R2/R3 were checked by web lookup during preparation. R4/R5 are official stable identifiers/landing links but were not individually re-opened in this run. No claim of recent 2026 experimental results is made without a fresh primary source.