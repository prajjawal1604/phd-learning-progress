# Evening references, paper index and claim map — 2026-11-14
**Syllabus:** private 19 October 2026 Gmail attachments, *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 3 preference/alignment) and *AI for Cybersecurity Learning Pathway* (Unit 3 model extraction and inference). Do not redistribute private course attachments. **Morning dependency:** [lesson](../../../../phd-morning/lessons/2026-11-14/lesson.md) and [index](../../../../phd-morning/lessons/2026-11-14/references/index.md) were fetched from GitHub main before authoring.

## Canonical reading list and critical summaries
1. **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. (2023).** *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290; DOI:10.48550/arXiv.2305.18290. https://arxiv.org/abs/2305.18290. **Contribution:** derives a direct pairwise classification objective from KL-regularized reward optimization and evaluates selected generation tasks. **Critique:** finite-model optimization, annotation biases, truthfulness and distribution shift require separate evaluation. **Access:** arXiv abstract and paper link publicly available; redistribution license for the particular version not independently checked. **Local PDF:** not committed.
2. **Ouyang, L., et al. (2022).** *Training language models to follow instructions with human feedback*. arXiv:2203.02155. https://arxiv.org/abs/2203.02155. **Contribution:** demonstrations, ranking, reward modeling and RLHF evaluation. **Critique:** human ratings and annotator/rubric selection do not guarantee truth or universal values. **Access:** arXiv link; specific redistribution license not independently checked. **Local PDF:** not committed.
3. **Shokri, R., Stronati, M., Song, C., Shmatikov, V. (2017).** *Membership Inference Attacks Against Machine Learning Models*. IEEE S&P. DOI:10.1109/SP.2017.41. https://doi.org/10.1109/SP.2017.41. **Contribution:** membership attacks using shadow models and target outputs. **Critique:** target access, sampling and output richness vary by deployment. **Access:** publisher DOI; open redistribution not established. **Local PDF:** not committed.
4. **Tramèr, F., et al. (2016).** *Stealing Machine Learning Models via Prediction APIs*. USENIX Security. https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer. **Contribution:** demonstrates model extraction under studied API interfaces. **Critique:** surrogate fidelity differs from task accuracy and weight recovery; query interfaces vary. **Access:** official USENIX page; redistribution terms not independently checked. **Local PDF:** not committed.
5. **Carlini, N., et al. (2021).** *Extracting Training Data from Large Language Models*. USENIX Security. https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting. **Contribution:** evidence that memorized sequences can be elicited in studied LMs. **Critique:** not all records or models are equally vulnerable. **Access:** official USENIX page; redistribution terms not independently checked. **Local PDF:** not committed.
6. **Yeom, S., Giacomelli, I., Fredrikson, M., Jha, S. (2018).** *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*. arXiv:1709.01604. https://arxiv.org/abs/1709.01604. **Contribution:** formal/empirical connection between overfitting, membership and attribute inference, and limitations of generalization as privacy evidence. **Critique:** theoretical assumptions and distributions must be checked before applying bounds to modern LLMs. **Access:** canonical arXiv record found; license not independently checked. **Local PDF:** not committed.
7. **Abadi, M., et al. (2016).** *Deep Learning with Differential Privacy*. ACM CCS. DOI:10.1145/2976749.2978318; arXiv:1607.00133. https://arxiv.org/abs/1607.00133. **Contribution:** practical DP-SGD and privacy accounting for deep learning. **Critique:** reported privacy/utility depends on adjacency, accountant, noise and task. **Access:** canonical arXiv record and DOI confirmed; redistribution license not independently checked. **Local PDF:** not committed.

## Local paper inventory
**references/papers/**: no PDFs committed. Downloadability is not equivalent to an explicit right to redistribute a particular version publicly. For lawful access, use the canonical links above; a later pass may inspect per-version licensing and add permitted PDFs with checksums, original URLs, retrieval dates and license statements. No binary downloads are claimed.

## Claim-to-source map
| Claim | Local explanation / validation | Primary reference |
|---|---|---|
| KL-optimal policy and DPO loss | lesson §§1–3 derivation | Rafailov 2023 |
| Preference pipeline and reward-model limitations | lesson §§1–2,5 | Ouyang 2022 |
| Symmetric label-noise optimum | lesson §2 analytic derivation, §4 deterministic lab | Local mathematical derivation, not a claimed paper result |
| Extraction fidelity vs task accuracy | lesson §6 | Tramèr 2016 |
| MI and shadow-model threat model | lesson §§6–8 | Shokri 2017 |
| Memorization can expose training sequences | lesson §10 | Carlini 2021 |
| Generalization is not a privacy certificate | lesson §§8,10 | Yeom 2018 |
| DP-SGD and neighboring-dataset guarantee | lesson §8–9 | Abadi 2016 |
| Prevalence/decision-cost examples | lesson §§7,9 deterministic Bayes lab | Local mathematical derivation |
| Privacy-aware DPO improves utility/privacy | cross-track **hypothesis only**, not a finding | Proposed experiment |

**Verification boundary:** Morning GitHub source was read, both original syllabus PDFs were retrieved, canonical references were cross-checked selectively, and lab calculations can be reproduced with Python 3.10+ standard library. No live-model experiments, actual privacy attack, paper redistribution, email, audio or narration.
