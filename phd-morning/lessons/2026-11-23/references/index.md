# Companion references — morning 23 November 2026
Parent: [full lesson](../lesson.md). Syllabus: 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*; `Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf` p.12 Unit 5 Session 4 (speech, audio, video); `AI_for_Cybersecurity_Learning_Pathway.pdf` p.12 Unit 5 Session 4 (code review, cyber ranges, synthetic data). Continuity: [22 Nov morning](../../../2026-11-22/lesson.md) and [22 Nov evening](../../../../phd-evening/lessons/2026-11-22/lesson.md).

|ID|Reference / canonical link|Summary and limitation|Local PDF / access|
|---|---|---|---|
|R1|Graves, Alex; Fernández, Santiago; Gomez, Faustino; Schmidhuber, Jürgen. *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks* (2006), ICML, DOI: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)|CTC marginalizes monotonic alignment paths using blanks and dynamic programming; does not guarantee recognition under overlapping speakers or poor audio.|No local PDF; publisher DOI, redistribution permission not checked.|
|R2|Radford, Alec et al. *Robust Speech Recognition via Large-Scale Weak Supervision* (2023), ICML, arXiv:[2212.04356](https://arxiv.org/abs/2212.04356)|Whisper-style large-scale weakly supervised ASR, multilingual and noisy-domain evaluation; broad benchmarks do not certify high-stakes command recognition.|No local PDF; arXiv read access; version-specific license not confirmed.|
|R3|Ho, Jonathan; Jain, Ajay; Abbeel, Pieter. *Denoising Diffusion Probabilistic Models* (2020), NeurIPS, arXiv:[2006.11239](https://arxiv.org/abs/2006.11239)|Gaussian denoising foundation from 22 Nov; no guarantee of authenticity for generated media.|No local PDF; link to open manuscript, license not checked.|
|R4|Rombach, Robin et al. *High-Resolution Image Synthesis with Latent Diffusion Models* (2022), CVPR, DOI:[10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042), arXiv:[2112.10752](https://arxiv.org/abs/2112.10752)|Latent diffusion lowers spatial compute and supports conditioning; reconstruction fidelity and source authenticity remain separate questions.|No local PDF; license not checked.|
|R5|Pearce, Hammond et al. *Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions* (2022), IEEE S&P, arXiv:[2108.09293](https://arxiv.org/abs/2108.09293)|Controlled security assessment of generated code on vulnerability-relevant prompts; not an evaluation of all models, contexts or modern production tools.|No local PDF; arXiv read access; redistribution license not checked.|
|R6|NIST. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (2023), NIST AI 100-1, DOI:[10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1)|Govern, Map, Measure and Manage risk framework relevant to provenance, evaluation and human oversight; not certification of a particular system.|Official stable document; no local PDF.|

## Claim-to-source map
|Major claim|Local section|Reference|
|---|---|---|
|CTC path sum and blank alignment|A3|R1|
|Weakly supervised ASR and need for task-specific evaluation|A1, A3, A4|R2|
|Sampling/STFT and frame-hit formula|A2, A5|Lesson's explicit signal-processing derivation and assumptions|
|Generative outputs are not authenticated captures|A1, A5, B7|Lesson provenance reasoning, R6; R3/R4 are context only|
|AI code suggestions need security review|B1, B2, B3|R5; local SQLite demonstration B6|
|Base-rate and prior-shift math|B3, B5, B6|Lesson derivations, not empirical claims from R5|
|Synthetic-range isolation, leakage and evaluation|B4–B7|Lesson proposed protocol and R6 governance|

## Paper index and download/rights log
No new files were placed under `references/papers/` in this run. The canonical links above provide lawful reading routes; open access for reading does not automatically establish permission to redistribute a particular PDF in a repository. Before any future commit, record paper version, explicit license, source URL, checksum and retrieval date. **No local paper paths are claimed.**

## Verification status
Syllabus attachment pages were read directly. Bibliographic identifiers and links are provided as canonical records, but not every landing page was live-opened during this run; do not treat them as freshly link-tested. Both labs use Python standard library and include assertions and expected outputs. The content makes no claim of full-scale model training, real SOC testing or reproduction of published benchmarks.
