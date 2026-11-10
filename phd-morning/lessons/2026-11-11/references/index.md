# References and access index — 11 November 2026 morning

[Return to lesson](../lesson.md). Syllabus: 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*, attachments *Advanced Generative AI and Foundation Models Learning Pathway* (21 pages), p.8 Unit 3 Session 2; *AI for Cybersecurity Learning Pathway* (20 pages), pp.6–7 Unit 2 Session 6 and lab. Both PDFs were read; private attachments are not reproduced in this public repository.

## Canonical research references and critical summaries
1. **Long Ouyang et al. (2022)**, *Training language models to follow instructions with human feedback*, arXiv:2203.02155, https://arxiv.org/abs/2203.02155 . SFT demonstrations followed by preference/reward alignment; separate the effects of stages when evaluating behavior, and scrutinize annotation distributions.
2. **Jason Wei et al. (2022)**, *Finetuned Language Models Are Zero-Shot Learners*, arXiv:2109.01652, https://arxiv.org/abs/2109.01652 . Instruction tuning and transfer to unseen tasks; check task overlap and scale dependence.
3. **Colin Raffel et al. (2020)**, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21, arXiv:1910.10683, https://arxiv.org/abs/1910.10683 . Text-to-text task formulation; not evidence that a tuned model knows updated domain facts.
4. **Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani (2018)**, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, DOI:10.5220/0006639801080116, https://doi.org/10.5220/0006639801080116 . CICIDS2017 dataset design and traffic generation; scrutinize capture artifacts and deployment realism.
5. **Hyrum S. Anderson and Phil Roth (2018)**, *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637, https://arxiv.org/abs/1804.04637 . Prepared static PE features and labels; not a dynamic malware sandbox benchmark.
6. **Min Du et al. (2017)**, *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*, DOI:10.1145/3133956.3134015, https://doi.org/10.1145/3133956.3134015 . Sequence anomaly detection from system logs; evaluate drift and log-template assumptions before reuse.
7. **Takaya Saito and Marc Rehmsmeier (2015)**, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, DOI:10.1371/journal.pone.0118432, https://doi.org/10.1371/journal.pone.0118432 . Precision–recall evaluation under class imbalance; prevalence and operational costs still matter.

## Paper files, licensing and access
**Local paper paths: none.** No research PDFs were downloaded or committed. Links above are canonical abstract/DOI landing pages; direct PDF licensing and redistribution permissions were not verified for this run. Public access does not automatically authorize repository redistribution. Before adding a file to `references/papers/`, record source URL, date retrieved, license text, permitted redistribution, and SHA-256. Do not copy private syllabus PDFs to the public repository.

## Claim-to-source map
- SFT, instruction tuning and text-to-text behavior: refs 1–3, lesson G1–G4.
- Response-only token loss and gradients: lesson G2–G3, derivation from autoregressive maximum likelihood; numeric examples synthetic.
- Network IDS capture assumptions: ref 4, lesson C1/C4.
- Static malware prepared-feature evaluation: ref 5, lesson C1/C4.
- Log sequence benchmark limitations: ref 6, lesson C4.
- Precision–recall under imbalance: ref 7, lesson C2/C3.
- Point-in-time intelligence, costs, and operational safety: lesson C1–C4, explicitly reasoned design recommendations; not attributed to the above empirical studies.

**Validation caveat:** Identifiers and canonical URLs are recorded; live HTTP resolution of each external link and full paper license verification were not performed. The paper summaries are limited to well-established high-level scope; no unsupported numerical findings are attributed to them.
