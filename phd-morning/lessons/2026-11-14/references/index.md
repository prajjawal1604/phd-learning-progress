# Companion references and paper index — 14 November 2026
**Syllabus sources (private, not copied into public repository):** 19 October 2026 Gmail message "PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027"; *Advanced Generative AI and Foundation Models Learning Pathway*, p.8, Unit 3 Session 5; *AI for Cybersecurity Learning Pathway*, p.8, Unit 3 Session 4. Previous local context: [13 Nov morning](../../../../2026-11-13/lesson.md) (see canonical repo path if relative resolution differs) and [13 Nov evening](../../../../../phd-evening/lessons/2026-11-13/lesson.md). The PDFs were accessed through Gmail but are not redistributed.

## Paper inventory and critical summaries
| Paper and canonical identifier | What it supports | Limits and access / local path |
|---|---|---|
| Ouyang et al., **Training language models to follow instructions with human feedback** (2022), arXiv:2203.02155, https://arxiv.org/abs/2203.02155 | Demonstration SFT, preference ranking, reward modeling, RLHF, annotator evaluations | Not a universal safety guarantee; arXiv abstract and PDF access checked 2026-11-14; local PDF **not committed** |
| Rafailov et al., **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (2023; v3 2024), arXiv:2305.18290, DOI:10.48550/arXiv.2305.18290, https://arxiv.org/abs/2305.18290 | DPO's KL-derived preference-classification objective; selected empirical tasks | Does not prove all tasks, groups or factuality improve; canonical abstract/PDF access checked 2026-11-14; local PDF **not committed** |
| Shokri et al., **Membership Inference Attacks Against Machine Learning Models** (2017), DOI:10.1109/SP.2017.41, https://doi.org/10.1109/SP.2017.41 | Membership inference, shadow-model concept and evaluation | Interface/distribution assumptions matter; canonical identifier supplied from established bibliography, not individually live checked; local PDF **not committed** |
| Tramèr et al., **Stealing Machine Learning Models via Prediction APIs** (2016), USENIX Security, https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer | Model extraction via prediction interfaces, fidelity and query access | Results not universal to modern LLM endpoints; link not individually live checked; local PDF **not committed** |
| Carlini et al., **Extracting Training Data from Large Language Models** (2021), USENIX Security, https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting | Memorization and possible training-data exposure in studied language models | Does not imply every record is extractable; link not individually live checked; local PDF **not committed** |

**Access/license status:** arXiv provides accessible PDF links for the two verified papers, but a download link alone does not establish redistribution permission for a public GitHub repository. No paper PDFs were downloaded or committed during this run; hence **no local paths under references/papers/** exist for these five papers. Check each version's license and redistribution terms before committing binaries. Stable links and critical summaries above substitute for unverified local copies. All teaching calculations are locally derived, not quoted from papers.

## Source-to-claim map
- RLHF stages and feedback limitations → Ouyang 2022; lesson G1.
- Bradley–Terry, KL-regularized optimum and DPO log-ratio loss → Rafailov 2023; lesson G2, with complete local derivation.
- Gradient signs and toy DPO loss → lesson G2 and G4 runnable code.
- Extraction threat, fidelity vs task accuracy → Tramèr 2016; lesson C1–C2.
- Membership inference and attack evaluation → Shokri 2017; lesson C3–C4.
- Generative memorization risk → Carlini 2021; lesson C5.
- Bayes prevalence examples and AUC toy → lesson C3–C4 local derivation/code.
- Cross-track claims about DPO increasing privacy leakage are **research hypotheses only**, not established paper findings.

**Reproduction note:** Code uses Python 3.10+ standard library, deterministic constants and explicit assertions. No actual victim API, training records, audio or email are involved.
