# Companion references index — morning 18 November 2026
Syllabus: private PDFs from 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*: *Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf*, Unit 4 Session 3 p.10; *AI_for_Cybersecurity_Learning_Pathway.pdf*, Unit 4 Session 2 p.10. Both consulted, not redistributed.
Lesson: [../lesson.md](../lesson.md). Continuity: [Nov 17 morning](../../../../phd-morning/lessons/2026-11-17/lesson.md), [Nov 17 evening](../../../../phd-evening/lessons/2026-11-17/lesson.md).

## Paper index and access
**Local PDFs: none; local paths under references/papers/: none.** Canonical arXiv abstracts are openly accessible; exact PDF license/redistribution permissions were not established, so no PDF was downloaded or committed. Linked PDF availability is not a license assertion.

| ID | Bibliography and canonical source | Summary, critique, access |
|---|---|---|
| R1 | Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), arXiv:2005.11401, https://arxiv.org/abs/2005.11401 | Couples parametric generation and retrieved nonparametric memory; motivates separating retrieval from answer behavior. No automatic claim-level citation guarantee. Prior index checked; not independently re-opened this run. PDF https://arxiv.org/pdf/2005.11401 ; license unverified. |
| R2 | Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert, *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023), arXiv:2309.15217, https://arxiv.org/abs/2309.15217 | Reference-free RAG evaluation across context, faithfulness and answer quality dimensions. Automated judges can be biased and do not replace expert adjudication. Canonical arXiv record located 2026-11-18. PDF https://arxiv.org/pdf/2309.15217 ; license unverified. |
| R3 | Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi, *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (2023), arXiv:2310.11511, DOI:10.48550/arXiv.2310.11511, https://arxiv.org/abs/2310.11511 | Trains reflection tokens for adaptive retrieval and response critique; self-assessment does not ensure authorized or correct evidence. Canonical arXiv record checked 2026-11-18. PDF https://arxiv.org/pdf/2310.11511 ; license unverified. |
| R4 | William L. Hamilton, Rex Ying, Jure Leskovec, *Inductive Representation Learning on Large Graphs* (2017), arXiv:1706.02216, DOI:10.48550/arXiv.1706.02216, https://arxiv.org/abs/1706.02216 | GraphSAGE samples and aggregates neighborhood features for inductive embeddings. Original benchmarks are not SOC deployment evidence; time leakage and edge integrity need separate controls. Canonical arXiv record checked 2026-11-18. PDF https://arxiv.org/pdf/1706.02216 ; license unverified. |
| R5 | Petar Veličković et al., *Graph Attention Networks* (2017), arXiv:1710.10903, https://arxiv.org/abs/1710.10903 | Learns masked neighbor-attention weights in graph representation learning. Weights do not prove causal explanations or security effectiveness. Canonical arXiv record located 2026-11-18. PDF https://arxiv.org/pdf/1710.10903 ; license unverified. |

## Claim-to-source map
- Claim support, recall, completeness, citation coverage and abstention: lesson A1–A5 original definitions, arithmetic and decision theory; R1–R3 motivate retrieval and evaluation, not toy results.
- RAG judge failure modes and reflection limitations: lesson A4, R2 and R3; date/ACL examples are authored counterexamples.
- GraphSAGE inductive neighborhood update: lesson B2, R4; scalar worked example is original.
- Graph attention weights: lesson B2, R5; causal interpretation caution is a methodological limitation.
- Contrastive logistic derivative and time-restricted graph: lesson B3–B5, original math and threat-model reasoning; do not attribute this exact toy implementation to R4.
- Operational SOC architecture, privacy and ethics: lesson B4–B6 and pathway PDFs, explicitly proposed controls rather than validated deployments.

## Reproduction and permissions
Lab A: standard-library evidence audit; Lab B: standard-library time-safe aggregation and finite-difference contrastive gradient. Both use synthetic numbers and no network access. Cite exact arXiv IDs rather than claiming paper PDFs are in the repo. Research PDF rights must be checked per version before any future binary upload. No email, audio or narration.
