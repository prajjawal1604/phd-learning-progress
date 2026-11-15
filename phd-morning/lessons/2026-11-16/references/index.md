# References and paper index — 16 November 2026
## Provenance and access
Syllabus sources were **actually retrieved** from Gmail message dated 19 October 2026, subject *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*: *Advanced Generative AI and Foundation Models Learning Pathway* p.10 Unit4 Session1, and *AI for Cybersecurity Learning Pathway* p.8 Unit3 Session6. Private PDFs are not redistributed. Prior session: [15 Nov morning](../../../2026-11-15/lesson.md), [15 Nov evening](../../../../phd-evening/lessons/2026-11-15/lesson.md).

## Primary research and official guidance
| Reference | Contribution, assumptions and critique | Local paper file / access |
|---|---|---|
| Patrick Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS, arXiv:2005.11401, https://arxiv.org/abs/2005.11401 | Joint parametric/nonparametric memory; RAG-Sequence and RAG-Token on knowledge-intensive benchmarks. Not proof of grounded answers or ACL safety in enterprise deployments. Canonical arXiv record checked 16 Nov 2026. | No local PDF. arXiv source accessible; version-specific reuse license not established for repository redistribution. |
| Stephen Robertson and Hugo Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in IR, DOI:10.1561/1500000019, https://doi.org/10.1561/1500000019 | Probabilistic foundations and BM25 ranking; lexical mismatch and score-calibration limitations. Bibliographic record checked 16 Nov 2026. | No local PDF; publisher reuse permission not established. |
| Aleksander Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR, arXiv:1706.06083, https://arxiv.org/abs/1706.06083 | Min–max robust optimization, first-order attacks and adversarial training; image/norm-ball setting not automatically feasible for security logs. arXiv/institutional bibliographic records checked 16 Nov 2026. | No local PDF; institutional accepted version carries rights notice, so no redistribution asserted. |
| Apostol Vassilev, Alina Oprea, Alie Fordyce, Hyrum Anderson, Xander Davies and Maia Hamin (2025), *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025 | Formal taxonomy of attacker goals, capabilities, stages and mitigations; a framework, not a security certification. NIST official publication record checked 16 Nov 2026. | No local PDF; official download available but precise reuse/commit conditions not independently checked. |

**Paper index:** No files committed under \`references/papers/\` this run. A publicly readable PDF does not automatically grant repository redistribution rights. Links above are stable canonical access points; do not claim local binaries exist.

## Claim-to-source map
- G1 ingestion/retrieval/generation pipeline: GenAI pathway p.10 and lesson G1; original parametric/nonparametric formulation: Lewis 2020.
- G2 BM25 expression and intuition: Robertson & Zaragoza 2009; toy calculation and RRF counterexample: lesson G2/G5 (local derivations, not measured paper results).
- G3 passage marginalization: Lewis 2020; enterprise ACL/validity design is an engineering requirement discussed in lesson G1/G4, not a measured result of Lewis.
- C1/C2 threat model, robust risk and PGD: Madry 2018; NIST 2025; protocol-constrained score example is local hypothetical calculation.
- C3 abstention, checks and red-team evaluation: NIST 2025 taxonomy; lesson C3/C5 discusses limitations, not certified performance.
- Cross-track research hypotheses: lesson G6/C6; proposals only.

## Reproduction
Both labs use Python 3.10+ standard library, fixed synthetic data and assertions. No sensitive data or live attack target. Record execution environment and actual results separately; do not confuse deterministic expected output with empirical deployment validation.
