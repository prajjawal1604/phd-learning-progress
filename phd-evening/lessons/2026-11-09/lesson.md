# Evening PhD deep dive — 9 November 2026 (Asia/Kolkata)
**Dependency verified:** Read `phd-morning/lessons/2026-11-09/lesson.md` and its `references/index.md` from main before composing. This is an extension of **Generative AI Unit 2 Session 6: scaling and inference** and **AI for Cybersecurity Unit 2 Session 4: ransomware, botnets, phishing**; not a new syllabus session. Both October 19 Gmail pathway PDFs were accessed: *Advanced Generative AI and Foundation Models Learning Pathway* and *AI for Cybersecurity Learning Pathway* (both Unit 2, page 6). Examples and events below are synthetic, not field measurements.

## Spoken navigation
**10 minutes:** derive KV-cache memory, explain why decode is memory-sensitive, derive a posterior from likelihood ratios, and distinguish a benign backup from ransomware. Ask Q1–Q5.
**30 minutes:** add roofline reasoning, paged memory fragmentation, multi-sensor dependence, calibration and response costs; Q1–Q11.
**Extended 90–150 minutes:** execute both labs, critique papers, design ablations and complete Q1–Q18. Tutor: ask one question, wait, request a counterexample, then reveal the model answer. No coding while walking.

## Prerequisites and definitions
Recall morning's causal attention, prefill, decode, grouped-query attention (GQA), key/value (KV) cache, batching, TTFT and p95 latency. Also Bayes theorem, prevalence, sensitivity/specificity, endpoint and DNS telemetry, phishing and command-and-control (C2). A **roofline model** bounds achievable computation by arithmetic throughput and memory bandwidth; **arithmetic intensity** is operations per byte transferred; **calibration** means events assigned probability q occur about q of the time in comparable groups; a **likelihood ratio** is evidence probability under attack divided by its probability under benign conditions. **Incident-level** evaluation counts correlated alerts from one episode together.

# A. Advanced LLM inference

## A1. Why caching changes the bottleneck
For a single new query q, head keys K of shape [S,d] and values V of shape [S,d], attention is `softmax(q K^T / sqrt(d)) V`. Cache stores K,V for S historical tokens, avoiding repeated projection of history; nevertheless each new token reads roughly 2 S d scalars per KV head. For B sequences, L layers, H_kv KV heads, dimension d and scalar size b bytes, **M_KV=2 B L S H_kv d b**. Query heads H_q may exceed H_kv with GQA; K/V are shared among query-head groups. This is a *logical* tensor budget, not allocated GPU memory.

**Derivation of marginal cost:** `dM/dS=2 B L H_kv d b` bytes per additional retained token. With B=4,L=32,H_kv=8,d=128,b=2, marginal cost = 524,288 bytes/token = 0.5 MiB/token across the batch. An additional 2,048 tokens requires 1 GiB. At B=1, marginal cost is 0.125 MiB/token. This makes context length a scheduling and admission-control decision.

**Compute versus bandwidth:** Suppose one decoding step needs F floating-point operations and moves D bytes from high-bandwidth memory. On hardware with peak compute C FLOP/s and sustained bandwidth W byte/s, ideal time is at least `max(F/C,D/W)`. This is a lower bound, *not* a predicted latency; launch overhead, cache misses, synchronization and scheduler delays add cost. Arithmetic intensity I=F/D; the crossover is C/W FLOP/byte. A tiny batch can read large model weights to produce few tokens and become bandwidth-bound. Batching amortizes some weight movement, but increases KV residency and queueing. A request's output tokens/second can decrease even while aggregate tokens/second rises.

**Worked synthetic bound:** F=1.2e12 FLOP, D=80e9 bytes, C=120e12 FLOP/s, W=2e12 bytes/s. Compute bound=.010 s, bandwidth bound=.040 s, so minimum=.040 s. Doubling arithmetic throughput alone does not improve this lower bound; halving bytes makes bound max(.010,.020)=.020 s. These numbers are hypothetical, not GPU specifications.

## A2. Paged KV and fragmentation
Contiguous preallocation for maximum length S_max wastes memory when actual lengths vary. A paged cache divides tokens into blocks of P positions; each sequence needs `ceil(S/P)` blocks, with internal slack `P ceil(S/P)-S` positions. For S=33 and P=16, three blocks hold 48 positions, leaving 15 unused. A second sequence of 16 uses one block with zero slack. Across requests, block mapping avoids requiring one giant contiguous region; bookkeeping and noncontiguous reads introduce overhead. Shared-prefix reuse requires correct reference counts and strict tenant isolation; never infer security from identical textual prefixes alone. A shared block must be immutable or copy-on-write when a sequence diverges.

**Counterexample:** P=64 for a workload of many 5-token requests can waste most of the last block; P=4 lowers internal slack but increases metadata and mapping operations. There is no universally optimal block size. Fragmentation, preemption, swapping and recomputation are scheduler decisions to benchmark.

## A3. Tail latency and controlled comparisons
For arrival rate λ requests/s and service capacity μ requests/s, a simple M/M/1 queue (Poisson arrivals, exponential independent service, one server, λ<μ) has expected *system time* `1/(μ−λ)`. With λ=8 and μ=10, mean=.5 s; at λ=9.5, mean=2 s. Real LLM serving is not M/M/1: service times depend on prompt/output length, batching, preemption and cache. The queue example illustrates why utilization near saturation can explode latency; it is not a production estimator. Report per-request TTFT, inter-token latency, end-to-end p50/p95, throughput, peak allocated memory, accuracy and energy when measured.

**Architecture:**
```text
requests -> admission/tenant policy -> queue -> prefill workers
                                            |         |
                                     scheduler <-> paged KV blocks
                                            |         |
                                            +-> decode steps -> validated stream
metrics: TTFT / p95 / throughput / KV residency / evictions / correctness
```
**Failure modes:** longer context may improve retrieval yet worsen latency; quantization can degrade rare-token evidence; prefix cache can leak cross-tenant data if isolation is faulty; aggressive eviction causes recomputation; batching optimizes average throughput but may harm interactive fairness. Keep sensitive security logs out of unapproved inference services.

### Lab A: runnable inference-capacity experiment (Python 3.10+, standard library)
```python
from math import ceil
def kv_bytes(B,L,S,H,d,b=2):
    return 2*B*L*S*H*d*b
def block_slack(lengths, block_size):
    return sum(ceil(s/block_size)*block_size-s for s in lengths)
def roofline_lower_bound(flops, bytes_moved, flops_per_sec, bytes_per_sec):
    return max(flops/flops_per_sec,bytes_moved/bytes_per_sec)
assert kv_bytes(4,32,4096,8,128)==2*2**30
assert kv_bytes(4,32,6144,8,128)-kv_bytes(4,32,4096,8,128)==2**30
assert block_slack([33,16],16)==15
assert block_slack([5]*10,64)==590
assert abs(roofline_lower_bound(1.2e12,80e9,120e12,2e12)-.04)<1e-12
print("KV GiB",kv_bytes(4,32,6144,8,128)/2**30)
print("slack",block_slack([33,16],16))
print("lower bound s",roofline_lower_bound(1.2e12,80e9,120e12,2e12))
```
**Expected:** `KV GiB 3.0`, `slack 15`, `lower bound s 0.04`. **Research extension:** sweep block sizes [4,8,16,32,64] on realistic *anonymized* length distributions; measure internal slack and metadata cost separately. This model does not benchmark a GPU.

# B. Advanced ransomware, botnet and phishing detection

## B1. Evidence fusion without double counting
For attack A and evidence E, posterior odds equal prior odds times likelihood ratio: `P(A|E)/(1-P(A|E)) = [π/(1-π)] × [P(E|A)/P(E|not A)]`. If E1 and E2 are **conditionally independent given both A and not-A**, their likelihood ratios multiply. If both alerts arise from the *same backup job*, independence fails. Multiplying their LRs overstates certainty.

**Worked example:** π=.001, E1 rapid rewrites LR=20, E2 backup-tamper LR=30. Prior odds=.001/.999≈.001001; combined odds=.6006, posterior≈.3752. But if benign maintenance jointly triggers both, empirical joint LR might be only 35; posterior≈.03385. Do not substitute an assumed product for a measured joint likelihood. A third sensor is useful only if it adds conditional information and arrives before the decision deadline.

**Temporal graph:**
```text
mail link -> identity login -> process launch -> file rewrites
                                    |                  |
                                DNS/C2 flow       backup modification
                                    \                  /
                                    evidence timeline
                                           |
                              calibrated risk + provenance
                                           |
                             human approval / reversible action
```
A ransomware chain may include privilege misuse, backup interference and mass file modifications; legitimate administration can overlap. Botnet periodic DNS may resemble monitoring; phishing lookalike domains may be legitimate. Evidence must include host identity, time ordering, process ancestry, allowlists, telemetry gaps and sensor reliability. A missing event is not evidence of innocence if logging failed.

## B2. Decision theory and analyst budgets
Let q be calibrated attack posterior, C_FP cost of containment when benign, C_FN cost of noncontainment when malicious. Expected loss of containment = `(1−q)C_FP`; expected loss of no containment = `q C_FN`. Contain only if `q > C_FP/(C_FP+C_FN)` **within this two-action model**. For C_FP=40,C_FN=160, threshold=.2. But if evidence gathering costs 2 units and can improve decisions, a third *investigate* action may dominate both. Delayed response costs, reversible containment and human capacity should be modeled explicitly.

At π=.001, sensitivity=.9, specificity=.99, precision≈.08264: 1,000 alerts need not imply 1,000 incidents. For 100,000 independent decision units, expected TP=90, FP=999, precision=90/1089≈8.26%; confidence intervals and clustering matter. Never extrapolate per-event FPR to per-host-day alert counts without aggregation.

**Counterexample:** A domain blacklist achieves 99.9% specificity on random historical URLs but fails on new phishing domains and sends many alerts on a domain-heavy enterprise; a random URL split leaks domains across train/test. Evaluate chronologically, group by domain/campaign/host, freeze thresholds on validation, and report time-to-detection, incident-level recall, false alerts per host-day, analyst workload and missed high-impact events. Calibrate on a separate held-out set; monitor prevalence drift.

### Lab B: synthetic Bayesian sensor-fusion and cost tests (no malicious code/network access)
```python
def posterior(prior, likelihood_ratio):
    odds=prior/(1-prior)*likelihood_ratio
    return odds/(1+odds)
def threshold(fp_cost, fn_cost):
    return fp_cost/(fp_cost+fn_cost)
def precision(prevalence,sensitivity,specificity):
    tp=prevalence*sensitivity
    fp=(1-prevalence)*(1-specificity)
    return tp/(tp+fp)
independent=posterior(.001,20*30)
dependent=posterior(.001,35)
assert independent>dependent
assert abs(independent-.3752345216)<.001
assert abs(dependent-.0338504937)<.001
assert threshold(40,160)==.2
assert abs(precision(.001,.9,.99)-.0826446281)<1e-8
print(round(independent,4),round(dependent,4),threshold(40,160))
```
**Expected:** `0.3752 0.0338 0.2` (rounding depends on exact values). **Challenge:** Replace arbitrary LRs with counts from a labeled, temporally held-out synthetic table; use Laplace smoothing, confidence intervals and correlation checks. **Failure test:** increase the benign co-occurrence rate; does your joint evidence become less persuasive? This lab is pedagogical, not a validated detector.

## B3. Cases, safeguards and paper critique
**Synthetic case 1:** approved backup tool rewrites thousands of files and connects to cloud storage; alert requires process allowlist, signed binary, approved schedule and backup audit. **Synthetic case 2:** an unknown child process modifies files and tampers with recovery settings after suspicious login; prioritize incident response, preserve evidence, use preauthorized containment gates. **Synthetic case 3:** a lookalike phishing domain redirects through several hosts; do not visit live suspicious pages in class; analyze offline metadata and known-safe artifacts. **Synthetic case 4:** periodic DNS to an unfamiliar endpoint is an approved health check; avoid treating periodicity as proof of C2.

**Paper critique:** Kwon et al. (2023), arXiv:2309.06180, demonstrates serving-memory techniques under particular workloads; study workload sensitivity and overhead, not just peak throughput. Dao et al. (2022), arXiv:2205.14135, is about IO-aware exact attention, not a universal linear-time algorithm. Ainslie et al. (2023), arXiv:2305.13245, explores GQA checkpoint conversion; evaluate model quality under target workloads. Sommer & Paxson (2010), DOI:10.1109/SP.2010.25, explains open-world deployment challenges; it does not establish your local incident prevalence. Saxe & Berlin (2017), arXiv:1702.08568, concerns character-level malicious strings; historical datasets may not represent current phishing. Compare methods using pre-registered splits, avoid test-set tuning, document negative results.

**Ethics and security:** obtain authorization for endpoint/network telemetry; minimize personal data; never upload credentials or private incident data to an unapproved LLM; require human review for disruptive response; retain provenance and appeal/recovery mechanisms. An LLM may explain evidence IDs but must not treat retrieved log text as instructions or autonomously isolate hosts. False positives can interrupt critical operations.

# C. Research program and reproducibility
**Inference hypothesis:** At fixed security-answer evidence recall, paged-KV scheduling and GQA reduce p95 latency versus an MHA baseline under heterogeneous context lengths. Independent variables: KV-head count, block size, prompt-length distribution, concurrency. Dependent: p95 TTFT, end-to-end latency, throughput, peak allocated bytes, answer entailment. Hold model family/hardware/revision/quantization constant when possible; report when GQA comparison necessarily changes weights. Replicate with seeds and warm/cold separation. Negative result: metadata overhead dominates short requests.
**Security hypothesis:** Joint endpoint ancestry + flow evidence improves incident-level recall at a fixed 10 false alerts/1000 host-days over either sensor alone. Split by time and incident/campaign, calibrate on validation, test missing-sensor scenarios, report bootstrap CIs by incident and deployment cost. Negative result: correlated sensors provide no additional information.
**Cross-track question:** Does a grounded LLM explanation improve analyst triage speed without raising unsupported claims? Randomize case order and blind analysts to model variant; score factual entailment against evidence IDs, decision correctness, time and trust calibration. No live malicious samples required.

# D. Socratic oral examination (answer, then follow-up)
1. **What is stored in KV cache?** Per-layer historical keys and values, not all model activations. *Follow-up:* what else occupies GPU memory?
2. **Why does GQA save memory?** Fewer KV heads than query heads; cache scales with H_kv. *Follow-up:* why might quality change?
3. **What is the cache marginal byte cost per token in our batch?** 524,288 bytes. *Follow-up:* compute for B=1.
4. **Does FlashAttention remove O(S²) dense-attention arithmetic?** No; it reduces memory traffic/materialization. *Follow-up:* distinguish from sparse attention.
5. **Why is low-batch decode often bandwidth-bound?** Large weight/cache movement relative to per-step arithmetic. *Follow-up:* when can batching help?
6. **What is roofline lower bound?** max(compute time, transfer time), ignoring overhead. *Follow-up:* why is measured latency higher?
7. **Why use KV blocks?** Flexible allocation and reduced external fragmentation. *Follow-up:* define internal slack.
8. **What is block slack for S=33,P=16?** 15 tokens. *Follow-up:* what changes at P=8?
9. **Why does p95 worsen near saturation?** Queueing grows rapidly as arrival approaches service capacity. *Follow-up:* why M/M/1 is insufficient?
10. **What is likelihood ratio?** P(E|A)/P(E|not A). *Follow-up:* what if E is common in backups?
11. **When can LRs multiply?** Conditional independence under both attack and benign classes. *Follow-up:* show shared-cause counterexample.
12. **What is posterior with π=.001 and joint LR=600?** About .375. *Follow-up:* with measured joint LR=35?
13. **What is calibrated probability?** Comparable events predicted q are attacks about q fraction of the time. *Follow-up:* how does drift break calibration?
14. **Why does 99% specificity still create noise?** At low prevalence, benign volume dominates false positives. *Follow-up:* calculate expected FP in 100k units.
15. **What is the two-action containment threshold for costs 40 and 160?** .2. *Follow-up:* why add investigation as third action?
16. **How do you avoid phishing benchmark leakage?** Group by domain/campaign and split chronologically. *Follow-up:* what if campaigns overlap?
17. **What distinguishes ransomware from backup activity?** Corroborated context, ancestry, intent, timing and recovery tampering; no single signature proves it. *Follow-up:* how handle missing logs?
18. **What is a publishable negative result?** A preregistered, adequately powered experiment showing no incremental benefit under specified conditions. *Follow-up:* which assumptions limit transfer?

## Source map and further learning
- Cache formula, roofline derivation, paging arithmetic: sections A1–A3, Lab A; Vaswani et al. (2017) arXiv:1706.03762; Dao et al. (2022) arXiv:2205.14135; Ainslie et al. (2023) arXiv:2305.13245; Kwon et al. (2023) arXiv:2309.06180.
- Detection posterior and loss derivation: B1–B2, Lab B; Sommer & Paxson (2010), DOI:10.1109/SP.2010.25 for deployment critique.
- Phishing feature limitations: B3; Saxe & Berlin (2017), arXiv:1702.08568.
- Operational taxonomy and safety: MITRE ATT&CK (https://attack.mitre.org/), NIST SP 800-61 Rev.3 (2025), DOI:10.6028/NIST.SP.800-61r3; CISA #StopRansomware Guide (2023), https://www.cisa.gov/stopransomware/ransomware-guide .
- **Tutorials:** PyTorch scaled-dot-product attention docs https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html ; vLLM docs https://docs.vllm.ai/ ; NIST incident-response guidance at DOI above.
- **Audio/podcast discovery (not episode-verified):** Stanford CS25 lecture recordings https://web.stanford.edu/class/cs25/ ; SANS Internet Storm Center podcast https://isc.sans.edu/podcast.html . Choose an episode after checking its date and availability; none is claimed reviewed here.
- Detailed reference annotations and local-PDF status: [companion index](references/index.md). No fresh news claims without primary-source date verification.

**Final recall:** Explain the difference between logical KV bytes and allocated memory; derive a joint-evidence posterior without double counting; name one optimization that hurts latency; identify a benign ransomware lookalike; describe an ethically safe, falsifiable experiment.
