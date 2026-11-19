# References and source index — 20 November 2026 morning
**Syllabus:** October 19, 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*: *Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf* p.12 Unit 5 Session 1; *AI_for_Cybersecurity_Learning_Pathway.pdf* p.12 Unit 5 Session 1. Both original attachments were accessed and read. Continuity: [Nov 19 morning](https://github.com/prajjawal1604/phd-learning-progress/blob/main/phd-morning/lessons/2026-11-19/lesson.md) and [Nov 19 evening](https://github.com/prajjawal1604/phd-learning-progress/blob/main/phd-evening/lessons/2026-11-19/lesson.md).

**Local paper PDF paths:** none. No paper PDFs have been downloaded or committed in this run. Canonical reading links do not by themselves establish a specific version's redistribution license. If later committing a PDF under `references/papers/`, verify its version-specific license/permission, record source URL, retrieval date, SHA-256 and attribution first.

## Readings
**[R1]** Alec Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (2021), ICML, **arXiv:2103.00020**, https://arxiv.org/abs/2103.00020 ; PDF https://arxiv.org/pdf/2103.00020 . Canonical identifier is listed in the syllabus and stable arXiv link supplied; no claim of a live HTTP check in this run. **Summary:** paired image/text encoders, normalized similarities, contrastive pretraining and prompt-based zero-shot classification. **Critique:** web captions, dataset bias, domain shift, prompt sensitivity and lack of guaranteed region-level grounding. **Local PDF:** none.

**[R2]** Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), NeurIPS, **arXiv:2005.11401**, https://arxiv.org/abs/2005.11401 . **Summary:** combines retrieval with conditioned generation; differentiates retrieval relevance from generated response. **Critique:** benchmark gains do not establish tenant ACLs, citation entailment or valid-time SOC evidence. **Local PDF:** none.

**[R3]** Shunyu Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022 preprint, ICLR 2023), **arXiv:2210.03629**, https://arxiv.org/abs/2210.03629 . **Summary:** interleaves language-model reasoning/action/observation on QA and interactive benchmarks. **Critique:** does not prove tool authorization, reliable self-explanations or injection resistance. **Local PDF:** none.

**[R4]** National Institute of Standards and Technology, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (2023), **NIST AI 100-1**, DOI **10.6028/NIST.AI.100-1**, https://doi.org/10.6028/NIST.AI.100-1 ; official https://www.nist.gov/itl/ai-risk-management-framework . **Summary:** Govern, Map, Measure and Manage risk functions; guidance, not a validation experiment. **Local PDF:** none.

**[R5]** OWASP, *Top 10 for Large Language Model Applications* (2025 edition), https://genai.owasp.org/llm-top-10/ . **Summary:** application-security risk taxonomy including prompt injection and sensitive information disclosure. **Critique:** risk checklist, not proof that any prompt-based mitigation works. **Local PDF:** none.

## Claim-to-source map
- GenAI contrastive training, image/text encoders and zero-shot similarity: [R1], lesson A1–A4.
- Contrastive gradient and 2×2 examples: algebra from [R1]-style loss; lesson A2 and Lab A; not an empirical CLIP result.
- SOC retrieval: [R2], lesson B1–B4; permission gates and temporal versioning are additional design requirements, not claims of [R2].
- Tool-use architecture: [R3], November 19 lessons and B1–B2; no production authorization guarantee.
- Governance and attack surface: [R4] and [R5], B4–B6.
- Alert precision: Bayes theorem, B3 and Lab B; numbers entirely synthetic.
- Original curricular scope: two October 19 PDFs cited above.

## Validation and access notes
Both runnable labs are standard-library-only and include exact assertions; execute in Python 3.10+. No model weights, proprietary SOC logs or real incident records used. No full CLIP training, model evaluation, or operational security test is claimed. Reference identifiers and canonical URLs are provided; not all destinations were live-verified during this run. Research paper downloads were not attempted because reuse terms were not confirmed. Preserve version, license and checksum before future distribution.
