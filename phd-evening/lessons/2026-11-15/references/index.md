# Companion references and paper index — 15 November 2026 evening

## Dependency and syllabus
Read same-day committed [morning lesson](../../../../phd-morning/lessons/2026-11-15/lesson.md) and [morning references](../../../../phd-morning/lessons/2026-11-15/references/index.md). Syllabus sources actually retrieved from the 19 October 2026 Gmail message: *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 3 Session 6, p.8) and *AI for Cybersecurity Learning Pathway* (Unit 3 Session 5, p.8). Private attachments are **not** copied to this public repository.

## Primary references: critical reading, identifiers and access
| Paper / official source | What it supports and what it does not | Canonical link and verification | Local PDF |
|---|---|---|---|
| Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (2017), PNAS, DOI 10.1073/pnas.1611835114 | EWC as importance-weighted sequential-learning regularization; diagonal importance and benchmark assumptions limit extrapolation to LLM safety | https://doi.org/10.1073/pnas.1611835114 ; DOI redirect attempted 2026-11-15 but publisher page inaccessible to live checker | None; license unverified |
| Ilharco et al., *Editing Models with Task Arithmetic* (2023), ICLR, arXiv:2212.04089 | Task vectors steer selected models; not a guarantee that arbitrary adapters compose safely | https://arxiv.org/abs/2212.04089 ; arXiv abstract/metadata verified 2026-11-15 | None; version-specific redistribution not checked |
| McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (2017), AISTATS, PMLR 54 | FedAvg, communication savings and non-IID experiments; not a Byzantine or DP guarantee | https://proceedings.mlr.press/v54/mcmahan17a.html ; publisher metadata verified 2026-11-15 | None; redistribution terms not checked |
| Blanchard, El Mhamdi, Guerraoui & Stainer, *Byzantine-Tolerant Machine Learning* (2017), arXiv:1703.02757 | Krum and a defined Byzantine robustness setting; not a universal guarantee under heterogeneous or colluding clients | https://arxiv.org/abs/1703.02757 ; arXiv title/abstract verified 2026-11-15. Note: morning index used a different title; this index follows the live arXiv record | None; version-specific rights not checked |
| Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (2023), arXiv:2302.12173 | Lower-trust retrieved content as an injection channel; no guarantee every attack succeeds | https://arxiv.org/abs/2302.12173 ; arXiv abstract/metadata verified 2026-11-15 | None; version-specific rights not checked |
| NIST, *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations* (2025), NIST AI 100-2e2025, DOI 10.6028/NIST.AI.100-2e2025 | Vocabulary and threat-model taxonomy, not empirical evidence for any detector score | https://doi.org/10.6028/NIST.AI.100-2e2025 ; canonical identifier from morning index, not live-checked this run | None; exact-version access/rights not checked |

**Paper inventory:** no files under `references/papers/` were created or committed in this run. Accessible abstracts/PDF links do not by themselves establish redistribution rights. Stable canonical links and substantive summaries are provided instead; no invented local paper paths.

## Source map: claim -> evidence
- G1 second-order old-loss change and step-size reversal: local Taylor derivation and Lab G; no experimental paper result claimed.
- G2 Laplace/EWC posterior penalty: Kirkpatrick 2017; full-Hessian counterexample and diagonal comparison are local algebra.
- G3 adapter factor concatenation and SVD compression: local linear algebra; Ilharco 2023 motivates task arithmetic, not its safety.
- C1 accepted staircase EMA drift: local recurrence and Lab C, constructed threat model.
- C2 reward-distortion bound: local geometric series, not an empirical attack rate.
- C3 FedAvg, Byzantine threat: McMahan 2017; Blanchard 2017. Clipping influence and minority-median counterexample are local derivations.
- C4 retrieved-data instruction risk: Greshake 2023; NIST taxonomy. Independent authorization is a design recommendation, not a measured guarantee.
- Integrated hypothesis: proposed future research, **not** established fact.

## Reproducibility and safety
Python 3.10+; Lab G requires NumPy, Lab C standard library only. Both are deterministic, synthetic and contain assertions. No real tenant data, production tools, credentials or external targets. Full LLM training and operational red-team experiments were not performed. Audio, narration and email intentionally not used.
