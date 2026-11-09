# References and paper access index — 10 November 2026 morning

[Return to lesson](../lesson.md). **Syllabus sources actually retrieved:** October 19, 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*; attachment *Advanced Generative AI and Foundation Models Learning Pathway* (21 pages), p.8 Unit 3 Session 1, and attachment *AI for Cybersecurity Learning Pathway* (20 pages), p.6 Unit 2 Session 5. Private syllabus PDFs are not copied to this public repository.

## Papers, stable links and critical reading notes
1. **Katherine Lee et al. (2022)**, *Deduplicating Training Data Makes Language Models Better*, arXiv:2107.06499. https://arxiv.org/abs/2107.06499 . Deduplication, memorization and train/evaluation overlap; compare their duplicate criteria and corpus to a domain-specific security dataset.
2. **Jordan Hoffmann et al. (2022)**, *Training Compute-Optimal Large Language Models*, arXiv:2203.15556. https://arxiv.org/abs/2203.15556 . Empirical trade-off between data and parameters under fixed compute; not a universal guarantee for every architecture or dataset.
3. **Jesse Dodge et al. (2021)**, *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*, arXiv:2104.08758. https://arxiv.org/abs/2104.08758 . Corpus composition and documentation; discuss coverage, bias and reproducibility.
4. **Leo Breiman (2001)**, *Random Forests*, Machine Learning 45, DOI:10.1023/A:1010933404324. https://doi.org/10.1023/A:1010933404324 . Bagged randomized tree ensemble; not an operational intrusion-detection benchmark.
5. **Jerome H. Friedman (2001)**, *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of Statistics 29(5), DOI:10.1214/aos/1013203451. https://doi.org/10.1214/aos/1013203451 . Stagewise additive modeling via loss optimization; performance depends on labels, features and split.
6. **OASIS (2021)**, *STIX Version 2.1*, standard. https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html . Structured threat intelligence with object relationships, provenance and validity semantics; not a guarantee of feed quality.
7. **MITRE ATT&CK**, Enterprise techniques. https://attack.mitre.org/ . Taxonomy for behaviors and investigation, not probabilities or confirmed incident labels.

## Local PDFs, source and license/access notes
**Local paper paths: none.** No PDFs have been downloaded or committed. Stable abstract/DOI/specification links above are provided instead. A publicly accessible manuscript does not automatically imply redistribution rights; licenses were not independently checked for this run. The two private syllabus PDFs are user-provided course documents and remain outside the repository. Do not describe any paper as locally available until a licensed PDF is actually checked into `references/papers/` with its source and license metadata.

## Claim-to-source map
- Corpus quality, deduplication, memorization and provenance: sources 1 and 3; lesson §§1–3.
- Compute/data allocation and limits: source 2; lesson §2.
- Random forests, bagging and boosting: sources 4–5; lesson §§5–7.
- Threat indicator typing, evidence relationships, time-of-knowledge: sources 6–7; lesson §§5–8.
- Numeric leakage, majority-error and loss examples: mathematical assumptions and derivations explicitly given in lesson; they are **synthetic illustrations**, not results from these papers.
- Cross-track SOC safety and abstention: lesson integration section; proposed design, not a published empirical result.

**Validation note:** Canonical identifiers are provided, but live resolution of every link and independent execution of all labs must be reported separately; do not infer verification from inclusion here.
