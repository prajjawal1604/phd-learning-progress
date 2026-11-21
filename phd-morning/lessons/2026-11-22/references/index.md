# Companion references — morning 22 November 2026
Parent: [lesson](../lesson.md). Syllabus: 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*, attachments `Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf` p.12 and `AI_for_Cybersecurity_Learning_Pathway.pdf` p.12, both Unit 5 Session 3. Continuity: [21 Nov morning](../../../../phd-morning/lessons/2026-11-21/lesson.md) and [21 Nov evening](../../../../phd-evening/lessons/2026-11-21/lesson.md).

|ID|Primary reference and canonical link|Substantive summary and critical boundary|Local paper path / access|
|---|---|---|---|
|R1|Ho, Jonathan; Jain, Ajay; Abbeel, Pieter. *Denoising Diffusion Probabilistic Models* (2020), NeurIPS, arXiv:2006.11239, https://arxiv.org/abs/2006.11239|Defines Gaussian forward corruption, learned reverse process and denoising objectives; sample quality does not validate exact provenance or forensic truth.|Not downloaded; arXiv landing and PDF publicly accessible, redistribution license not verified|
|R2|Rombach, Robin et al. *High-Resolution Image Synthesis with Latent Diffusion Models* (2022), CVPR, arXiv:2112.10752, https://arxiv.org/abs/2112.10752, DOI:10.1109/CVPR52688.2022.01042|Autoencoder latent-space diffusion and cross-attention conditioning reduce computational burden; compression may lose small text and fine evidence.|Not downloaded; license for repository redistribution not independently established|
|R3|Yao, Shunyu et al. *ReAct: Synergizing Reasoning and Acting in Language Models* (2023), ICLR, arXiv:2210.03629, https://arxiv.org/abs/2210.03629|Interleaves reasoning and actions to interact with external environments; no guarantee of authorization or injection resilience.|Not downloaded; license not independently verified|
|R4|OWASP. *Top 10 for LLM Applications* (2025), https://genai.owasp.org/llm-top-10/|Prompt injection, excessive agency and sensitive-information risks; guidance taxonomy is not experimental proof of controls.|Official online guidance; no local PDF|
|R5|NIST. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (2023), NIST AI 100-1, DOI:10.6028/NIST.AI.100-1, https://doi.org/10.6028/NIST.AI.100-1|Govern, Map, Measure, Manage framework; not a SOC agent certification.|Official publication link; no local PDF|

## Source-to-claim map
- Forward Gaussian marginal, noise prediction, reverse chain: R1; mathematical derivation in lesson A2–A3.
- Latent compression and conditioning: R2; exact-number caveats are lesson's proposed evaluation, not reported R2 results.
- Agent observation/action cycle: R3; authorization invariant and toy gateway are original defensive engineering examples in B2/B4.
- Threats and governance: R4/R5; no claim that either proves efficacy.
- All lab records synthetic. No paper content beyond abstracts and well-established published methods is claimed as independently reproduced.

## Download and license log
`references/papers/` contains **no new PDFs from this run**. Open access for reading is not necessarily permission to redistribute the exact version in this repository. Canonical links above remain the authoritative access route; no invented local paths or claimed downloads. Before future commit, record the version, explicit license, original URL, checksum and retrieval date in this index.

## Verification limits
Canonical identifiers are stable bibliographic records; this run did not independently live-open every external landing page. No claim of a novel November 2026 research result. Lab numerical examples are teaching simulations, not reproductions of research benchmarks.
