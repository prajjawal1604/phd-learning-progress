# Evening references and claim index — 2026-11-18
Companion to [evening lesson](../lesson.md). Dependency: [morning lesson](../../../../phd-morning/lessons/2026-11-18/lesson.md) and [morning references](../../../../phd-morning/lessons/2026-11-18/references/index.md). Both fetched from main before authoring. Original syllabus attachments consulted from 19 October 2026 Gmail: *Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf* (Unit 4 Session 3, p.10) and *AI_for_Cybersecurity_Learning_Pathway.pdf* (Unit 4 Session 2, p.10). Private syllabus PDFs not redistributed.

## Paper index and access
**Local research paper PDFs: none.** No files were committed under references/papers/; arXiv PDF availability alone does not establish a redistribution license for committing binaries. Use stable canonical links below. Bibliographic identifiers follow the already-reviewed morning companion index; independent live re-verification for this evening is not claimed.

| ID | Primary reference / canonical link | Deep summary, critical limitation and access |
|---|---|---|
| R1 | Patrick Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), arXiv:2005.11401, https://arxiv.org/abs/2005.11401 | RAG combines retrieved documents with sequence generation. Retrieval relevance is not atomic evidence entailment, temporal authority or authorization. PDF https://arxiv.org/pdf/2005.11401 ; repository redistribution license not checked. |
| R2 | Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert, *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023), arXiv:2309.15217, https://arxiv.org/abs/2309.15217 | Automated reference-free evaluation components for retrieval and response quality; judges can fail on negation, conflicting sources and domain context, so human adjudication remains necessary. PDF https://arxiv.org/pdf/2309.15217 ; license unverified. |
| R3 | Akari Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (2023), arXiv:2310.11511, DOI:10.48550/arXiv.2310.11511, https://arxiv.org/abs/2310.11511 | Learns retrieval and reflection behavior; internal critique cannot certify external truth or permission. PDF https://arxiv.org/pdf/2310.11511 ; license unverified. |
| R4 | William L. Hamilton, Rex Ying, Jure Leskovec, *Inductive Representation Learning on Large Graphs* (2017), arXiv:1706.02216, DOI:10.48550/arXiv.1706.02216, https://arxiv.org/abs/1706.02216 | GraphSAGE samples and aggregates neighbor representations for inductive embedding. Temporal telemetry, adversarial edges and SOC generalization require separate experiments. PDF https://arxiv.org/pdf/1706.02216 ; license unverified. |
| R5 | Petar Veličković et al., *Graph Attention Networks* (2017), arXiv:1710.10903, https://arxiv.org/abs/1710.10903 | Neighbor attention learns differential weighting; attention is not proof of causal influence or forensic validity. PDF https://arxiv.org/pdf/1710.10903 ; license unverified. |

## Claim map for offline tutoring
- Evidence completeness, joint recall and end-to-end probability: evening I.1–I.2 (original derivations); R1 retrieval motivation.
- Claim graph and weighted support: evening I.2 (proposed audit method); R2 automated-evaluation context.
- Selective answering, Brier/calibration and risk-coverage: evening I.3/I.5 (decision-theoretic derivation and synthetic lab), not claims of R2 experimental results.
- Reflection limitations and authorization: evening I.4; R3, with additional original threat-model reasoning.
- GraphSAGE and typed-neighbor variants: evening II.1; R4 for GraphSAGE, typed variant explicitly illustrative.
- Contrastive logistic derivative and false-negative sampling: evening II.2/II.6, original mathematical exercise, not a claim of a specific R4 loss.
- Graph attention and shortcut risk: evening II.3, R5 for attention mechanism; causal caution is methodological.
- SOC case, privacy, leakage and combined pipeline: evening II.3–II.5, original hypothetical controls and evaluation proposals.
- Source scope: October 19 private syllabus attachments and the committed November 18 morning lesson/index.

## Reproduction and rights notes
Lab A and Lab B are Python 3.10+ standard-library scripts, deterministic synthetic fixtures with explicit assertions. No operational detector or third-party system is touched. The paper links provide stable access; **no research PDFs were downloaded or committed**. For future downloads, inspect each arXiv version's license and repository redistribution rights before adding a binary, and record version, URL, date, hash and license. No email, narration or audio.
