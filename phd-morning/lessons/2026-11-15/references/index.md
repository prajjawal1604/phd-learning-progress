# References and paper index — 15 November 2026
## Syllabus provenance
Private Gmail message **PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027**, 19 October 2026. Both attachments actually read today: *Advanced Generative AI and Foundation Models Learning Pathway* (p.8, Unit 3 Session 6) and *AI for Cybersecurity Learning Pathway* (p.8, Unit 3 Session 5). Not redistributed to the public repository. Prior context: [14 November morning](../../../2026-11-14/lesson.md) and [14 November evening](../../../../phd-evening/lessons/2026-11-14/lesson.md).

## Primary reading and critical summaries
| Reference and canonical link | Contribution / critique | Local PDF / rights |
|---|---|---|
| Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks*, PNAS, DOI [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114) | EWC and importance-weighted retention; diagonal approximation and benchmark-specific results limit generalization | Not downloaded; publisher access/license not independently checked |
| Ilharco et al. (2023), *Editing Models with Task Arithmetic*, ICLR, [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) | Task vectors for selected model edits; arithmetic can interfere or fail for incompatible checkpoints | Not downloaded; version license not independently checked |
| McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS/PMLR 54, [PMLR paper](https://proceedings.mlr.press/v54/mcmahan17a.html), [arXiv:1602.05629](https://arxiv.org/abs/1602.05629) | FedAvg and communication trade-offs; no universal adversarial robustness or privacy guarantee | Not downloaded; reuse terms not independently checked |
| Blanchard et al. (2017), *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*, NeurIPS, [arXiv:1703.02757](https://arxiv.org/abs/1703.02757) | Byzantine aggregation under specified assumptions; non-IID and collusion remain difficult | Not downloaded; version license not independently checked |
| Greshake et al. (2023), *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*, [arXiv:2302.12173](https://arxiv.org/abs/2302.12173) | Documents indirect prompt-injection risk via lower-trust inputs; not a proof every payload succeeds | Not downloaded; version license not independently checked |
| NIST (2025), *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, [NIST publication](https://doi.org/10.6028/NIST.AI.100-2e2025) | Threat taxonomy and mitigation terminology; guidance, not empirical detector benchmark | Not downloaded; public-sector rights/access need checking for exact version |

**Access honesty:** Canonical identifiers are drawn from established primary bibliographies; individual external links and exact version-specific licenses were not all live-checked during this run. No paper binaries were downloaded or committed. Therefore **no local paths under references/papers/** are claimed. Do not infer redistribution rights from a public PDF URL.

## Source map
- G1 gradient-interference first-order approximation: local Taylor derivation; G1.
- G2 curvature-aware merge optimum and exact concatenated LoRA factorization: local algebra and lab; G2/G4. Task-vector context: Ilharco 2023.
- G3 EWC Bayesian quadratic approximation: Kirkpatrick 2017; local derivation G3.
- C1 EMA contamination example: local recurrence and lab C1/C5, hypothetical attack model.
- C2 discounted reward perturbation: local discounted-return algebra C2.
- C3 FedAvg and non-robust mean: McMahan 2017; C3/C5. Byzantine limitations: Blanchard 2017.
- C4 indirect prompt injection: Greshake 2023; NIST 2025; C4. Lab string assertion is **not** a defense test.
- Cross-track tenant-adapter security proposal: hypothesis only; not a paper-established result.

## Reproduction and access
Python 3.10+ standard library; deterministic constants and explicit assertions. All data synthetic. Verify independent runtime execution separately; no real user data, victim APIs, email, audio or narration.
