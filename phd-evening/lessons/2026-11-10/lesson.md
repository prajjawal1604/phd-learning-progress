# Evening PhD deep dive — 10 November 2026 (Asia/Kolkata)
**Dependency checked:** Read the committed [morning lesson](../../../phd-morning/lessons/2026-11-10/lesson.md) and its [reference index](../../../phd-morning/lessons/2026-11-10/references/index.md) on main before preparing this lesson. Same subjects only: **Generative AI Unit 3 Session 1 — data curation and pretraining** and **AI for Cybersecurity Unit 2 Session 5 — ensembles, typed graphs and threat intelligence**. The October 19 Gmail attachments *Advanced Generative AI and Foundation Models Learning Pathway* (p.8) and *AI for Cybersecurity Learning Pathway* (p.6) were read as the governing syllabi. All numbers, incidents, and datasets below are synthetic unless explicitly attributed. No audio/email requested.

## Voice tutor route
**10 minutes:** Ask: Why is a document split not sufficient for near-duplicate corpora? Derive the probability of sibling leakage. Contrast a correlated ensemble with independent majority voting. End with time-of-knowledge and evidence provenance.
**30 minutes:** Work through importance-weighted gradient and effective sample size, MinHash collision probability, probability calibration, dependent likelihood ratios, and temporal graph joins. Ask Q1–Q10.
**Extended 90–150 minutes:** Run both executable labs, audit assumptions and negative controls, critique three papers, design a preregistered research experiment, then answer all 18 oral questions. Pause after each question and invite the learner's derivation before revealing the answer.

## Prerequisites, symbols and definitions
Assume the morning lesson's next-token log likelihood, softmax gradient, train/validation/test discipline, bagging/boosting, base rates and graph edge types. A **duplicate cluster** is a set of documents deemed equivalent under a specified algorithm, not necessarily semantically identical. A **shingle** is a contiguous sequence of k tokens. **MinHash** estimates set overlap from random permutations/hashes. **Effective sample size (ESS)** measures how concentrated weights are. **Calibration** means predicted probabilities match frequencies under a specified population. **Likelihood ratio (LR)** is P(evidence|attack)/P(evidence|benign). **Temporal graph** stores relationships and when each was observed and ingested. **Point-in-time join** allows only information known by the decision time. **Entity resolution** merges identifiers thought to refer to one object, with potentially serious false merges.

# Part I — Generative AI: pretraining data as an optimization system

## 1. Why duplicates are not simply extra examples
An autoregressive language model with parameters θ assigns pθ(x₁:T)=∏ₜpθ(xₜ|x₍<t₎). For a document dᵢ with nᵢ valid tokens, let ℓᵢ(θ)=−(1/nᵢ)Σₜlog pθ(xᵢₜ|xᵢ,<t). An explicit document-weighted objective is L(θ)=Σᵢwᵢℓᵢ(θ)/Σᵢwᵢ, with wᵢ≥0. Then ∇L=Σᵢwᵢ∇ℓᵢ/Σᵢwᵢ, so copies increase their cluster's gradient share. **Caution:** this document-normalized objective differs from uniform *token*-weighted pretraining: long documents otherwise receive greater influence. Always specify sampling and loss normalization.

Worked example: unique documents A and B each contribute gradient +2 and −1 in a one-dimensional toy parameter. Unique average gradient =(+2−1)/2=+0.5. Duplicate A four times and retain B once: gradient=(4×2−1)/5=+1.4. Same architecture, very different update. A duplicate can improve coverage of important patterns, but excessive duplication reduces diversity and increases memorization risk; the best filter threshold is an empirical question.

**ESS derivation:** for normalized nonnegative weights aᵢ=wᵢ/Σwⱼ, variance of weighted independent equal-variance observations is σ²Σaᵢ². Equating this to σ²/n_eff yields n_eff=1/Σaᵢ²=(Σwᵢ)²/Σwᵢ². For weights [4,1,1,1], ESS=49/19≈2.579 rather than four. Independence and equal variance are assumptions; correlated duplicate gradients make this only a warning statistic, not an exact training-data diversity measure.

## 2. Duplicate detection and leakage: exact, lexical, semantic
Pipeline: licensed acquisition → immutable source ID/hash → parse and normalize → PII/license review → duplicate candidates → human audit → group split → tokenizer/version → packed causal sequences → weighted loss → checkpoint/evaluation. For k-word shingle sets A,B, Jaccard J=|A∩B|/|A∪B|. For ideal MinHash, P[h_min(A)=h_min(B)]=J. With b bands of r independent hash rows, the approximate probability of becoming a candidate pair is 1−(1−Jʳ)ᵇ; this is *candidate retrieval*, not proof of semantic equivalence. A short boilerplate header can inflate overlap; aggressive clustering can merge distinct security incidents.

**Counterexample:** two incident reports share 95% of a template but differ in the final sentence: “block the IP” versus “do not block the IP.” A similarity threshold can collapse a safety-critical contrast. Preserve both labels/decisions, record cluster exceptions, and test recall on rare counterexamples.

**Leakage revisited:** with one test copy and two siblings independently assigned to train with probability .8, P(at least one train sibling)=1−(.2)²=.96. If copies are grouped before splitting, this specific overlap is eliminated. But semantic contamination, public benchmark answer memorization, and evaluation-time prompt tuning remain possible. Define a sealed test corpus and maintain a dated audit of suspected overlaps.

## 3. Optimization, compute and reproducible checkpoints
For a softmax over vocabulary classes j, pⱼ=exp(zⱼ)/Σₖexp(zₖ), loss −Σⱼyⱼlog pⱼ, derivative ∂L/∂zⱼ=pⱼ−yⱼ. If target class probability is .1, its logit derivative is −.9; at .9 it is −.1. Weighting a record multiplies its contribution to the *batch aggregate*, not necessarily the final parameter update linearly under Adam, clipping or nonlinear training dynamics.

With D data-parallel workers, microbatch m per worker and accumulation a, global batch G=Dma, assuming equal batches and no dropped/partial final microbatch. To resume reproducibly, save weights, optimizer moments, scheduler, scaler, random generator states, sampler position, data/tokenizer revision, distributed configuration and software versions. Determinism can still vary across kernels/hardware. An Adam mixed-precision training memory budget must include model weights, gradients, FP32 moments (~8 bytes per parameter), possibly FP32 master weights, activations, communication buffers and fragmentation. Do not conflate inference KV-cache memory with training activations.

**Research implication:** Hoffmann et al. (2022) estimated compute-optimal model/data allocation in tested regimes. It does not imply that repeating a tiny low-quality corpus is equivalent to obtaining new high-quality tokens. Dodge et al. (2021) documented provenance/representation limitations in large web corpora; documenting a dataset is necessary but does not by itself establish lawful reuse or safety.

## 4. Advanced Lab G — shingle false merges, split leakage, ESS
Python 3.10+, standard library; save as `lab_g_evening.py` and run `python3 lab_g_evening.py`. No external data.
```python
from collections import defaultdict
def shingles(s,k=3):
    t=s.lower().split()
    return set(tuple(t[i:i+k]) for i in range(len(t)-k+1))
def jac(a,b):
    x,y=shingles(a),shingles(b)
    return len(x&y)/len(x|y) if x|y else 1.0
template="incident report host contacted unusual domain analyst reviewed"
a=template+" block destination after verification"
b=template+" do not block destination without verification"
c="unrelated scientific corpus with novel terms"
assert jac(a,b)>jac(a,c)
print("template similarity",round(jac(a,b),3))
# Same template, opposing operational instructions: high overlap is NOT equivalence.
groups={"incident-A":["a1","a2","a3"],"incident-B":["b1"],"incident-C":["c1"]}
copy_split={"train":{"a1","b1"},"test":{"a2","a3","c1"}}
leaks=[g for g,ids in groups.items()
       if set(ids)&copy_split["train"] and set(ids)&copy_split["test"]]
assert leaks==["incident-A"]
group_split={"train":{"a1","a2","a3","b1"},"test":{"c1"}}
assert not any(set(ids)&group_split["train"] and set(ids)&group_split["test"]
               for ids in groups.values())
weights=[4,1,1,1]
ess=sum(weights)**2/sum(w*w for w in weights)
assert abs(ess-49/19)<1e-12
gradient=(4*2-1)/5
assert gradient==1.4
print("leaked groups",leaks,"group-split leakage",0)
print("ESS",round(ess,3),"weighted gradient",gradient)
```
Expected: first similarity is greater than comparison with unrelated text; `leaked groups ['incident-A'] group-split leakage 0`; `ESS 2.579 weighted gradient 1.4`. The first similarity value depends only on the deterministic strings; its operational interpretation is deliberately contested. Extend by changing k, adding boilerplate-only reports, comparing precision/recall of candidate clusters against manually defined semantic labels, and reporting false-merge costs. **Do not** interpret a toy gradient as a trained LLM result.

## 5. Paper critique and falsifiable pretraining study
Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022; arXiv:2107.06499): valuable evidence about duplicate-induced memorization/contamination in their tested corpora; critique the match criterion, domain shift and cost of deleting rare security examples. Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022; arXiv:2203.15556): empirical scaling fit, not a causal statement that arbitrary tokens are equally valuable. Dodge et al., *Documenting Large Webtext Corpora* (2021; arXiv:2104.08758): useful documentation framework; audit coverage of collection permissions and removal requests. Do not quote effect sizes without inspecting original experiments.

**Preregister:** H₀: cluster-aware splitting plus quality filtering does not improve held-out incident QA or reduce memorization relative to a token-budget-matched unfiltered baseline. Same model, tokenizer, training tokens, compute, and sealed future incident families. Randomize training seeds; compare exact/near-duplicate audit, factual support, PII leakage probes, rare-family recall and paired incident-level uncertainty. Ablate dedup threshold and template exceptions. **Reject simplistic success:** a higher benchmark score paired with worse rare-family coverage or leaked answers is not a clean win.

# Part II — Cybersecurity: dependent evidence, temporal graphs and threat intelligence

## 6. Ensemble diversity and why naive fusion fails
For three *conditionally independent* classifiers each wrong with probability e=.2, majority wrong probability = C(3,2)e²(1−e)+e³=3(.04)(.8)+.008=.104. If errors are perfectly correlated, majority error remains .2. In realistic SOCs, shared sensors, common training labels and cloud service changes create correlated errors. Diversity must be measured on time-held-out incidents, not asserted because algorithms have different names.

**Bayesian derivation:** Let A be attack, prior odds O(A)=π/(1−π). For evidence E, posterior odds O(A|E)=O(A)×LR(E), LR(E)=P(E|A)/P(E|¬A). For E₁,E₂, multiply individual LRs **only if** they are conditionally independent under *both* A and ¬A. Otherwise use joint LR=P(E₁,E₂|A)/P(E₁,E₂|¬A). Suppose prior π=.01, each independent sensor LR=10: posterior odds= (.01/.99)×100≈1.0101; posterior ≈.5025. But if E₂ is a duplicate of E₁, joint LR=10, odds=.10101 and posterior ≈.0917. A duplicate alert must not transform a ~9% posterior into ~50% confidence.

**Decision threshold:** Let C_FP be cost of containing a benign entity, C_FN cost of missing a true attack. Under calibrated p=P(A|evidence), expected cost of contain=C_FP(1−p), cost of do-not-contain=C_FN p. Contain only if p>C_FP/(C_FP+C_FN), under the simplifying assumption of no other action costs, fixed consequences and authorized policy. At C_FP=100, C_FN=900, threshold=.1. Real containment requires operational constraints and human authorization; these illustrative costs are not organizational policy.

## 7. Point-in-time graph construction
```text
[EDR process event] ----(executed, event_at, ingested_at, evidence_id)----> [file hash]
        |                                                      
      [host] ----(contacted, event_at, ingested_at)----> [domain]
        |                                                   |
     [user login]                              [STIX indicator, valid_from,
                                                valid_until, published_at]
                     -> time-of-knowledge filter -> candidate paths
                     -> correlated-sensor fusion -> evidence-linked analyst review
```
Store `event_at` (when observed), `ingested_at` (when available), `source`, `tenant`, `confidence semantics`, `evidence_id`, and any source corrections. At decision time τ, require ingested_at≤τ; for a feed indicator also require published_at≤τ and applicable validity interval. Event time alone is insufficient: an IOC published tomorrow cannot support today's historical prediction. Do not treat a host→domain edge as evidence that the domain caused malicious execution. NAT, CDNs, DHCP and recycled cloud IPs generate ambiguous joins. Graph propagation should not cross tenants without authorization.

**Counterexample:** A benign update domain is on a stale IOC list. Two host telemetry products derive alerts from the *same* update process. Treating the two alerts as independent and propagating suspicion through a high-degree domain may mass-quarantine legitimate hosts. Negative controls: stale IOC, missing EDR, duplicated feed item, delayed ingestion, benign shared-domain hub and entity-ID collision.

## 8. Advanced Lab C — point-in-time joins and dependent evidence
Python 3.10+, standard library; save as `lab_c_evening.py`.
```python
from math import isclose
def posterior(prior,lr):
    odds=prior/(1-prior)*lr
    return odds/(1+odds)
prior=.01
independent=posterior(prior,100)
duplicate=posterior(prior,10)
assert isclose(independent,100/199,rel_tol=1e-12)
assert isclose(duplicate,10/109,rel_tol=1e-12)
events=[
 {"host":"h1","domain":"updates.example","event_at":10,"ingested_at":11,"evidence":"flow-1"},
 {"host":"h1","domain":"bad.example","event_at":12,"ingested_at":20,"evidence":"flow-late"},
 {"host":"h2","domain":"bad.example","event_at":9,"ingested_at":10,"evidence":"flow-2"},
]
intel={
 "updates.example":{"published_at":5,"valid_until":8,"source":"synthetic-stale"},
 "bad.example":{"published_at":15,"valid_until":30,"source":"synthetic-feed"},
}
def known_matches(host,decision_at):
    out=[]
    for e in events:
        if e["host"]!=host or e["ingested_at"]>decision_at: continue
        i=intel.get(e["domain"])
        if i and i["published_at"]<=decision_at<=i["valid_until"]:
            out.append(e["evidence"])
    return out
assert known_matches("h1",13)==[]
assert known_matches("h1",21)==["flow-late"]
assert known_matches("h2",13)==[]
print("independent",round(independent,4),"duplicate",round(duplicate,4))
print("h1 at 13",known_matches("h1",13),"h1 at 21",known_matches("h1",21))
```
Expected: `independent 0.5025 duplicate 0.0917`; `h1 at 13 [] h1 at 21 ['flow-late']`. The synthetic time units are not wall-clock dates. **Important:** a historical event can be matched with an IOC that became known later *for retrospective investigation*, but not for prospective backtesting at the earlier decision timestamp. Extend with a source-retraction timestamp, host-ID collision and a feed-poisoning negative control. Evaluate precision at 30 alerts/day, incident recall, median detection lead time and confidence intervals by incident (not individual edges).

## 9. Research critique, real-world constraints, ethics
Breiman, *Random Forests* (2001; DOI:10.1023/A:1010933404324), formalizes randomized tree ensembles; it does not establish their performance in modern adversarial SOC environments. Friedman, *Greedy Function Approximation: A Gradient Boosting Machine* (2001; DOI:10.1214/aos/1013203451), explains stagewise loss optimization; noisy analyst labels and shifting baselines can mislead boosting. OASIS STIX 2.1 (2021) standardizes intelligence objects/relationships, but data interoperability is not proof of truth. MITRE ATT&CK supplies behavioral vocabulary, not calibrated posterior probabilities. Ask of each study: what was the split unit, which evidence was known at prediction time, how were duplicates and analyst feedback handled, and were confidence intervals computed at the incident level?

**Defensive case:** A SOC sees a suspicious host–domain contact, endpoint execution and a threat feed indicator. An evidence-backed assistant must state three observations with timestamps and provenance, list alternative benign explanations, distinguish correlated sensors and request human authorization before containment. Never execute untrusted feed text as a tool instruction. Respect tenant isolation, privacy, retention, licensing, analyst review, appeal and documented false-positive harms. No live malware, scanning, or operational blocking is performed in these labs.

**Preregister:** At a fixed analyst budget of 30 alerts/day, does a time-aware typed graph improve *incident-level* recall and detection lead time beyond a calibrated endpoint+flow ensemble? Compare graph/no-graph, fresh/stale IOC, deduplicated/duplicated sensor alerts and known/late feeds; split chronologically by incident family and organization; report precision, review minutes, uncertainty, subgroup false positives, and abstention rate. A result that improves recall but violates time-of-knowledge is invalid.

# Cross-track synthesis
Training data and SOC evidence share an epistemic contract: record source, legal permission, identity, transformation, timestamp, confidence and the point at which information became available. An LLM summarizer trained on contaminated incident reports can sound confident about a graph that contains stale indicators. Require cited evidence IDs and uncertainty, preserve the distinction between *observation*, *model inference*, and *authorized action*, and make unsupported conclusions fail closed to human review.

## Socratic oral examination — ask first, pause, reveal answer, then probe
1. **What changed from morning?** We analyze weight concentration and dependency, not merely define duplicates. **Probe:** Why isn't ESS a count of unique meanings?
2. **Derive weighted gradient.** ∇L=Σwᵢ∇ℓᵢ/Σwᵢ. **Probe:** What if examples have different token counts?
3. **What is ESS for [4,1,1,1]?** 49/19≈2.579. **Probe:** Which independence assumptions underlie its interpretation?
4. **Why can near-duplicate filtering be unsafe?** Opposite decisions may share templates. **Probe:** Construct a false merge.
5. **What does MinHash estimate?** Set Jaccard via hash-collision frequency. **Probe:** Is a candidate pair a duplicate verdict?
6. **Why is cluster splitting not a full contamination defense?** Paraphrases and benchmark selection still leak. **Probe:** Design a sealed holdout.
7. **What is the correct logit gradient?** pⱼ−yⱼ. **Probe:** Explain the sign for a correct class at p=.1.
8. **What must a training checkpoint save?** Model, optimizer, scheduler, RNG, sampler/data revisions and config. **Probe:** Why might GPU runs still diverge?
9. **What does Chinchilla not prove?** That all tokens have equal quality. **Probe:** Design a fixed-compute test.
10. **When does three-voter majority reach .104 error?** Independent .2-error voters. **Probe:** What if their errors coincide?
11. **When can likelihood ratios multiply?** Conditional independence under both classes. **Probe:** What if one alert copies another?
12. **Compute duplicated-alert posterior.** Prior .01, one LR 10 gives ≈.0917. **Probe:** Why isn't it .5025?
13. **Derive cost threshold.** C_FP/(C_FP+C_FN) under simplified two-action costs. **Probe:** What about irreversible containment?
14. **Why store ingested_at?** To exclude knowledge unavailable at decision time. **Probe:** Can retrospective investigation use later intel?
15. **Why are shared cloud domains dangerous graph hubs?** They create many benign edges. **Probe:** Name a negative control.
16. **What does STIX standardize?** Representation and relationships, not truth or calibration. **Probe:** What metadata supports retraction?
17. **How would you evaluate graph utility?** Time-held-out incident recall at fixed analyst budget, with ablations. **Probe:** Why bootstrap incidents rather than edges?
18. **Propose a falsification of cross-track claims.** Insert synthetic duplicated corpus answers and stale/late IOCs; check the assistant abstains and graph backtest excludes future knowledge. **Probe:** Which outcome would disprove improvement?

## Final recall and artifacts
Recall without notes: ESS formula; 96% sibling-leakage assumption; MinHash/Jaccard; why duplicate alerts cannot multiply LR; posterior .0917 vs .5025; event time vs ingestion time; one dangerous graph false merge. Save lab outputs, data/graph schema, paper critique matrix, one-page preregistration and negative controls. No paper PDFs are represented as locally available. See [companion references](references/index.md) for source map, stable identifiers and access notes.
