# References index — 2026-11-19 morning
Syllabus: original Gmail message dated 2026-10-19, *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*, attachments **Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf** (p.10 Unit 4 Session 4) and **AI_for_Cybersecurity_Learning_Pathway.pdf** (p.10 Unit 4 Session 3). These were read for this lesson. Prior continuity: [Nov 18 morning](../../../../2026-11-18/lesson.md) and [Nov 18 evening](../../../../../phd-evening/lessons/2026-11-18/lesson.md). The previous morning link may be used relative to this index as ../../../2026-11-18/lesson.md; see absolute repository paths if uncertain.

**Local research PDFs:** none. The repository's references/papers/ was not modified. Availability of an open reading URL does not automatically establish a right to redistribute a particular PDF version; links and summaries are supplied instead. The syllabus PDFs were accessed via the user's Gmail attachments, not republished.

## Primary reading and critique
**[R1]** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao, *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 preprint / 2023 ICLR, **arXiv:2210.03629**, https://arxiv.org/abs/2210.03629 ; DOI https://doi.org/10.48550/arXiv.2210.03629 . Verified canonical arXiv metadata in this run. **Summary:** interleaves reasoning-oriented text, actions and environmental observations; tests QA and interactive decision benchmarks. **Limitations:** benchmark success does not validate production authorization, explanation faithfulness or prompt-injection resistance. **Local path:** not downloaded; canonical PDF https://arxiv.org/pdf/2210.03629 (read-only link; PDF license/version-specific reuse not established).

**[R2]** Volodymyr Mnih et al., *Human-level control through deep reinforcement learning*, *Nature* 518, 529–533 (2015), **DOI:10.1038/nature14236**, https://doi.org/10.1038/nature14236 . Verified publisher metadata in this run. **Summary:** deep Q-network learns Atari policies from visual inputs; replay and target networks stabilize training. **Limitations:** Atari reward structure and action space do not establish safe SOC response; publisher article is not assumed open for redistribution. **Local path:** none.

**[R3]** Javier García and Fernando Fernández, *A Comprehensive Survey on Safe Reinforcement Learning*, *JMLR* 16(42):1437–1480 (2015), https://jmlr.org/papers/v16/garcia15a.html . Verified JMLR-hosted publication record in this run. **Summary:** categorizes safety-aware optimization criteria and exploration guidance, with a survey of methods. **Limitations:** a survey is not evidence of a safe deployed SOC policy. **Local path:** none; publisher provides a publicly readable PDF but redistribution permission was not checked.

## Claim map and reproduction
- ReAct loop, action/observation distinction: [R1], lesson A1–A2.
- Bellman/deep value-learning context: [R2], lesson B1–B2.
- Risk-aware learning and constraints: [R3], lesson B3.
- Tool-policy gates, tenant isolation, memory TTL: proposed architecture in lesson A4–A5, not externally validated security guarantees.
- Synthetic VOI: lesson A3; reproduce arithmetic with Bayes.
- Synthetic value iteration: lesson B2/B4; run \`python3 rl_defense.py\`.
- All numeric results are illustrative, not operational statistics.

## Access, integrity and future reading
Check canonical publisher/arXiv pages before any subsequent PDF commit; record the exact version, license/terms, source URL, retrieval date, checksum and redistribution justification. A public abstract alone does not grant reuse rights. No paper contents or PDF downloads are claimed beyond the source summaries above.
