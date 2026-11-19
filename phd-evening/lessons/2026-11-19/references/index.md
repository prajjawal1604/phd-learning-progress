# Companion references — evening 2026-11-19
**Morning dependencies read:** [lesson](../../../phd-morning/lessons/2026-11-19/lesson.md), [index](../../../phd-morning/lessons/2026-11-19/references/index.md). **Syllabus PDFs read from October 19 Gmail:** `Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf` (p.10 Unit 4 Session 4); `AI_for_Cybersecurity_Learning_Pathway.pdf` (p.10 Unit 4 Session 3). These private syllabus attachments are not republished.

## Papers, summaries, critical access notes
- **R1** Shunyu Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022 preprint, ICLR 2023), arXiv:2210.03629, https://arxiv.org/abs/2210.03629 ; PDF https://arxiv.org/pdf/2210.03629 . Interleaves reasoning/action/observation on QA and interactive tasks. Benchmark performance does not certify authorization or reasoning faithfulness. **Local PDF:** none.
- **R2** Volodymyr Mnih et al., *Human-level control through deep reinforcement learning* (2015), *Nature* 518, 529–533, DOI:10.1038/nature14236, https://doi.org/10.1038/nature14236 . Deep Q-learning with replay/target network on Atari. No transfer guarantee to SOC safety. **Local PDF:** none; publisher reuse rights not established.
- **R3** Javier García and Fernando Fernández, *A Comprehensive Survey on Safe Reinforcement Learning* (2015), *JMLR* 16(42):1437–1480, https://jmlr.org/papers/v16/garcia15a.html ; publisher PDF https://jmlr.org/papers/volume16/garcia15a/garcia15a.pdf . Taxonomy of safe exploration and risk objectives, not an operational certification. **Local PDF:** none.
- **R4** Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra, *Planning and Acting in Partially Observable Stochastic Domains* (1998), *Artificial Intelligence* 101(1–2):99–134, DOI:10.1016/S0004-3702(98)00023-X, https://doi.org/10.1016/S0004-3702(98)00023-X . Formal belief-state planning and POMDP treatment; known-model assumptions and computational complexity are limitations. **Local PDF:** none.
- **R5** Miroslav Dudík, John Langford, Lihong Li, *Doubly Robust Policy Evaluation and Learning* (2011), ICML, arXiv:1103.4601, https://arxiv.org/abs/1103.4601 ; PDF https://arxiv.org/pdf/1103.4601 . Propensity/outcome model combination for contextual-bandit evaluation; absence of support remains fatal to identification. **Local PDF:** none.

**Access/license note:** Stable publisher and arXiv reading links are supplied; no specific version's redistribution license was independently confirmed, so no PDFs were downloaded or committed to `references/papers/`. These are bibliographic summaries, not claims of complete PDF review. R1–R3 bibliographic records were verified in the morning lesson; R4–R5 identifiers are canonical references but not independently live-checked during this evening run.

## Claim-to-source map
| Major claim | Local explanation | Source |
|---|---|---|
| Interleaved reasoning and action | lesson A1 | R1 |
| Bayes-risk and VOI arithmetic | lesson A2, Lab A | Explicit derivation, synthetic assumptions |
| Tenant policy, memory provenance | lesson A1/A3 | Proposed design; not empirically verified |
| Belief update | lesson B1, Lab B | R4 |
| Discounted Bellman and DQN distinction | lesson B2 | R2 |
| Risk constraints versus hard approval | lesson B2/B5 | R3 + proposed shield |
| Importance sampling, support and variance | lesson B3/B4 | R5 + local demonstration |

## Reproduction and future reading
Python 3.10+; extract fenced scripts into `agent_evening.py` and `soc_evening.py`, run with `python3`. The expected outputs are written next to each lab. No model training, operational containment, or adversarial live system testing. Before committing any paper PDF, check source URL, version, license, permitted redistribution, retrieval date and SHA-256; add an entry under `references/papers/` only when lawful.
