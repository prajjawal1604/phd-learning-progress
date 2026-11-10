# Evening references index — 10 November 2026
[Lesson](../lesson.md) · [Same-day morning lesson](../../../../phd-morning/lessons/2026-11-10/lesson.md) · [Morning references](../../../../phd-morning/lessons/2026-11-10/references/index.md).

**Syllabus provenance:** October 19, 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*. Both PDFs read: *Advanced Generative AI and Foundation Models Learning Pathway* (21 pp.; Unit 3 Session 1, p.8) and *AI for Cybersecurity Learning Pathway* (20 pp.; Unit 2 Session 5, p.6). Private syllabus PDFs are intentionally not redistributed in this repository.

## Primary research and official references
1. Katherine Lee et al. (2022), **Deduplicating Training Data Makes Language Models Better**, arXiv:2107.06499, https://arxiv.org/abs/2107.06499 . Evidence on duplication, memorization and evaluation overlap. Critique: corpus and matching criteria may not transfer to rare security incident templates; avoid invented effect sizes.
2. Jordan Hoffmann et al. (2022), **Training Compute-Optimal Large Language Models**, arXiv:2203.15556, https://arxiv.org/abs/2203.15556 . Empirical compute/data trade-offs in tested model regimes; does not prove interchangeability of high- and low-quality tokens.
3. Jesse Dodge et al. (2021), **Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus**, arXiv:2104.08758, https://arxiv.org/abs/2104.08758 . Corpus composition, documentation and representational limitations; documentation alone is not permission to redistribute.
4. Andrei Z. Broder (1997), **On the Resemblance and Containment of Documents**, DOI:10.1109/SEQUEN.1997.666900, https://doi.org/10.1109/SEQUEN.1997.666900 . Shingle resemblance and document similarity foundations; tokenization and boilerplate affect measured overlap.
5. Leo Breiman (2001), **Random Forests**, DOI:10.1023/A:1010933404324, https://doi.org/10.1023/A:1010933404324 . Ensemble construction and error correlation; not a validated SOC benchmark.
6. Jerome H. Friedman (2001), **Greedy Function Approximation: A Gradient Boosting Machine**, DOI:10.1214/aos/1013203451, https://doi.org/10.1214/aos/1013203451 . Stagewise additive optimization; assess noise sensitivity and temporal drift.
7. OASIS (2021), **STIX Version 2.1**, official standard, https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html . Indicator and relationship data model; structured indicators remain claims needing provenance and temporal validity.
8. MITRE, **ATT&CK Enterprise**, official maintained knowledge base, https://attack.mitre.org/ . Behavioral taxonomy for analyst explanation; not a source of calibrated incident probabilities.

## Claim-to-source map
- Duplicate influence on gradients and ESS: lesson §§1–3, derived explicitly under stated weighting/independence assumptions; Lee, Hoffmann and Dodge supply research context.
- Shingle/MinHash candidate generation and false merges: lesson §2 and Lab G; Broder 1997. Numerical examples are synthetic.
- Training checkpoint and loss gradient: lesson §3; mathematical derivation included; neither a performance benchmark nor a claim from a specific paper.
- Bagging, boosting and correlated-voter counterexample: lesson §6; Breiman and Friedman; numerical .104 is a derived illustrative probability.
- Bayesian fusion and containment costs: lesson §6 and Lab C; explicit assumptions/derivations, not empirical SOC performance.
- Typed temporal graph, point-in-time indicators and safety: lesson §§7–9; OASIS STIX and MITRE ATT&CK; synthetic cases.
- Research protocols and ethics: lesson §§5,9 and synthesis; proposed falsifiable designs, not published results.

## Local paper index and rights
**Local PDF paths: none.** No paper PDFs were downloaded or committed. The stable abstract, DOI and official-standard links above remain the access paths. Open access does not automatically imply repository redistribution permission; license status and live access were not individually established for this run. The private syllabus attachments remain in Gmail only. Do not infer local availability or paper-license grants from these links.

## Validation scope
The labs use Python standard library and include assertions and expected output. Verify execution independently; the GitHub commit does not itself prove code was run. Canonical identifiers are provided; individual live link resolution is not asserted. No unverified November 10, 2026 current-development claim is included.
