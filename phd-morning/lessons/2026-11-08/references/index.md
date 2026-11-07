# References index — Morning 2026-11-08
Parent: [complete lesson](../lesson.md). Syllabus: October 19, 2026 Gmail attachments *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 2 Session 5) and *AI for Cybersecurity Learning Pathway* (Unit 2 Session 3). The syllabus PDFs are private attachments, not reproduced here.

## Paper inventory and access
No research paper PDFs were downloaded or committed in this run. **Local paper paths: none.** The links below point to canonical arXiv abstract pages where lawful individual access is available. No redistribution license was independently established; do not infer one from public readability.

| ID | Paper, year, stable identifier | Summary and scope | Local PDF |
|---|---|---|---|
| G1 | Brown et al., *Language Models are Few-Shot Learners* (2020), arXiv:2005.14165, https://arxiv.org/abs/2005.14165 | Describes task performance using in-context examples in large autoregressive models. Does not imply model weights change during prompting. | Not downloaded |
| G2 | Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (2022), arXiv:2109.01652, https://arxiv.org/abs/2109.01652 | Instruction fine-tuning and cross-task generalization; conclusions depend on model and benchmark setup. | Not downloaded |
| G3 | Ouyang et al., *Training language models to follow instructions with human feedback* (2022), arXiv:2203.02155, https://arxiv.org/abs/2203.02155 | Instruction-following and preference optimization; not proof of truthfulness or security guarantees. | Not downloaded |
| C1 | Anderson and Roth, *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models* (2018), arXiv:1804.04637, https://arxiv.org/abs/1804.04637 | Static PE malware features, labels and reproducible baseline setting; dataset and temporal limitations matter. | Not downloaded |
| C2 | Saxe and Berlin, *Deep Neural Network Based Malware Detection Using Two Dimensional Binary Program Features* (2015), arXiv:1508.03096, https://arxiv.org/abs/1508.03096 | Neural malware classification from binary program features; check deployment shift and robustness. | Not downloaded |

## Official documentation and learning aids
- C3: MITRE ATT&CK software catalog, https://attack.mitre.org/software/ — living taxonomy of documented software, not a classifier.
- Python JSON module, https://docs.python.org/3/library/json.html — parsing; schema and semantic verification must be added separately.
- scikit-learn model evaluation, https://scikit-learn.org/stable/modules/model_evaluation.html — precision/recall.
- Dive into Deep Learning, https://d2l.ai/ — foundational softmax, loss and gradient tutorials.
- Listening directories (specific episode not verified): TWIML AI https://twimlai.com/podcast/ ; SANS ISC https://isc.sans.edu/podcast.html.

## Claim-to-source map
| Claim | Local lesson | Evidence |
|---|---|---|
| Few-shot changes context, not model parameters | G1 | G1; local conditional-probability definition |
| SFT minimizes teacher-forced token NLL | G1 worked derivation | G2, G3; algebra locally derived |
| JSON and ID checks do not establish factual support | G2 / Lab G | Executable counterexample; Python docs |
| Static malware feature baselines and limits | C1/C3 | C1, C2 |
| Byte entropy and base-rate precision | C1 / Lab C | Explicit local derivation and synthetic assertions |
| Malware technique taxonomy | C3 | MITRE ATT&CK C3 |
| Cross-track incident triage workflow | Cross-track case | Proposed design, **not** independently validated |

## Validation and rights notes
Research abstracts and identifiers are cited for further reading, not a claim of complete paper reproduction. No open-access PDFs were committed because licensing/redistribution was not checked. No recent AI/cyber news items are presented as verified in this package. Lab results are stated as expected unless independently executed.