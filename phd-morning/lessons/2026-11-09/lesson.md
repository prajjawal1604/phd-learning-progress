# Morning PhD interactive teaching package — 9 November 2026 (Asia/Kolkata)

**Syllabus alignment:** The October 19, 2026 email contains *Advanced Generative AI and Foundation Models Learning Pathway* (21 pages), Unit 2 Session 6, **Scaling and inference**, and *AI for Cybersecurity Learning Pathway* (20 pages), Unit 2 Session 4, **Ransomware, botnets and phishing**. This advances November 8's prompting/instruction tuning and static/dynamic malware analysis by exactly one syllabus session in each track. These are course-session labels, not a claim that a university contact hour was delivered. Numerical scenarios below are synthetic unless explicitly attributed.

## Walking lesson routes and learning contract
**10-minute route:** Tell a partner the difference between prefill and decode; derive KV-cache bytes; explain why a ransomware indicator is not proof of ransomware; answer Q1–Q5. **30-minute route:** Include batching, memory bottlenecks, latency calculations, ransomware/phishing/botnet timelines, and Q1–Q10. **Extended 90–150 minutes:** perform both runnable labs, critique the cited research, rehearse all oral questions and formulate one preregistered experiment per track. In spoken mode, the tutor should ask one question at a time, wait for an answer, then give a counterexample and follow-up. Full material follows, so the route is a navigation aid, not a replacement.

**Prerequisites:** conditional next-token probability, Transformer attention, causal masks, token sampling, supervised evaluation, confusion matrices, calibrated posteriors, basic network/DNS/process logs, and static/dynamic malware inspection. **Definitions:** *prefill* processes the supplied prompt and produces initial KV tensors; *decode* generates new tokens autoregressively; *KV cache* stores per-layer past key and value tensors; *throughput* is completed tokens or requests per second; *time to first token (TTFT)* includes prompt processing and queueing; *inter-token latency* is elapsed time between successive output tokens; *ransomware* malicious activity that denies access to data, often by encryption and sometimes extortion; *botnet* coordinated compromised devices; *C2* command-and-control communication; *phishing* deceptive messages or sites that induce credential disclosure or other harmful action; *IOC* indicator of compromise, a clue requiring contextual corroboration; *TTP* tactic, technique or procedure, a behavioral pattern.

# Track G — Unit 2 Session 6: scaling and inference

## 1. Intuition and mechanics
Imagine reading a 2,000-token case file and then speaking a 100-token answer. The reading pass is **prefill**: a decoder-only Transformer computes activations for many prompt positions in parallel, under a causal mask, and caches keys and values. Speaking is **decode**: token t depends on tokens 1…t−1; each step must wait for the preceding sampled token. The cached past K,V are reused so earlier tokens need not be projected again at every generation step. This improves arithmetic efficiency but grows memory with sequence length, batch size and layers. **Cache reuse is not equivalent to zero-cost long-context generation**: each new query attends to stored keys, so per-step attention work still grows with retained context. Prefill may be compute-intensive; decode often becomes constrained by moving model weights and cache data through memory, especially at low batch sizes. Exact bottlenecks depend on hardware, kernels, attention architecture, quantization and scheduler.

```text
incoming prompt -> tokenize -> request queue -> prefill (parallel positions)
                                               | 
                                      layer-wise KV cache
                                               |
                         decode token 1 -> append KV -> sample
                                               |
                         decode token 2 -> append KV -> sample -> ...
                                               |
                                    validate / stream answer
metrics: queue delay | TTFT | tokens/sec | peak memory | p95 latency
```

At inference, model **weights** are fixed (unless an adaptation mechanism is explicitly active); activations and KV entries change with each request. A serving system may use continuous batching to admit new requests between decode steps. This improves utilization but introduces contention and variable per-user latency. Prefix caching can share identical trusted prefixes under compatible implementation constraints; it must not allow one tenant's private context to leak to another. Prompt injection and private retrieved documents remain untrusted inputs regardless of caching.

## 2. Derive the cache size and compute budget
Let B = simultaneous sequences, L = Transformer layers, S = retained sequence tokens per sequence, H_kv = key/value attention heads, d_h = dimension per head, and b = bytes per scalar. A layer stores K and V, each shaped [B,S,H_kv,d_h]. Therefore

**M_KV = 2 × B × L × S × H_kv × d_h × b bytes.**

The factor 2 is for K and V. For conventional multi-head attention H_kv=H_q; grouped-query attention (GQA) uses fewer KV heads than query heads, reducing cache memory without necessarily reducing all compute by the same factor. Cache allocation fragmentation, page tables, quantization scales, padding and other runtime overhead are omitted.

**Worked example:** B=4, L=32, S=4096, H_kv=8, d_h=128, b=2 (FP16). M_KV=2×4×32×4096×8×128×2 = 2,147,483,648 bytes = **2 GiB**. If H_kv=32 instead, cache = **8 GiB**. At S=8192 with the original GQA settings, cache = **4 GiB**. These are KV-only budgets, not GPU memory estimates. If a model has P=7 billion parameters and stores weights at 2 bytes each, nominal weight bytes = 14 billion bytes ≈13.04 GiB before framework and auxiliary buffers. A 16-GiB GPU is therefore **not** proven sufficient by adding nominal weight and cache sizes; temporary allocations and runtime kernels matter.

With a dense attention layer, full-sequence score construction has S×S entries per head, giving O(S²d_h) attention arithmetic per head in a naïve prefill formulation. FlashAttention-like tiling can avoid materializing the full S² matrix in high-bandwidth memory; it does **not** make exact dense attention asymptotically linear in S. At each decode step, the new query compares against S previous keys: O(Sd_h) per query head, while caching avoids recomputing historical K,V. For T generated tokens after a prompt of length S, cumulative attention comparisons scale roughly with TS + T(T+1)/2 per head, apart from kernels and architectural details.

**Latency budget example:** queue 0.20 s + prefill 0.35 s + first decode 0.05 s gives TTFT≈0.60 s (under this explicit definition). For 100 output tokens with mean 0.05 s/token, generation adds ≈5 s; if first token was included in TTFT, total≈0.60+99×0.05=5.55 s. Never conflate TTFT with full-response latency or per-request tokens/sec with aggregate server throughput. Measure p50 and p95, not just averages.

## 3. When optimization helps—and when it fails
Reducing H_kv with GQA reduces cache footprint but may alter quality; quantizing weights lowers storage and bandwidth demand but can hurt certain tasks and may require dequantization work; quantizing KV cache saves memory but can perturb attention; batching increases aggregate throughput but may degrade TTFT and tail latency; limiting context lowers cost but risks omitting evidence; speculative decoding can accelerate accepted tokens when a cheaper draft model agrees with a target model, but rejection overhead and low acceptance can erase gains. These are testable hypotheses, not universal performance guarantees.

**Counterexample:** Suppose a shorter context removes the only log line proving a host was patched. The model may respond faster but make a worse security recommendation. Another: doubling batch size improves tokens/second overall while doubling p95 user latency. A good benchmark fixes prompt lengths, output limits, decoding settings, model revision, concurrency, warmup, hardware, tokenizer, quantization, and workload mix; logs queue time, TTFT, output tokens, p95 latency, peak allocated memory and correctness. Randomize request order; separate cold and warm runs.

### Lab G — deterministic KV-memory and latency capacity simulator (Python 3.10+, standard library)
This is an analytical simulator, **not a measured GPU benchmark**. It teaches dimensional analysis and trade-offs.
```python
def kv_gib(batch, layers, tokens, kv_heads, head_dim, bytes_per_scalar=2):
    return 2*batch*layers*tokens*kv_heads*head_dim*bytes_per_scalar / 2**30

def latency_seconds(queue, prefill, seconds_per_token, output_tokens):
    assert output_tokens >= 1
    ttft = queue + prefill + seconds_per_token
    return ttft, ttft + (output_tokens-1)*seconds_per_token

base = kv_gib(4, 32, 4096, 8, 128)
full = kv_gib(4, 32, 4096, 32, 128)
double_context = kv_gib(4, 32, 8192, 8, 128)
ttft, total = latency_seconds(.20, .35, .05, 100)
assert base == 2.0 and full == 8.0 and double_context == 4.0
assert abs(ttft-.60)<1e-12 and abs(total-5.55)<1e-12
print(f"KV GiB: GQA={base:.1f}, MHA={full:.1f}, long={double_context:.1f}")
print(f"TTFT={ttft:.2f}s total={total:.2f}s")
```
**Expected:** `KV GiB: GQA=2.0, MHA=8.0, long=4.0` and `TTFT=0.60s total=5.55s`. **Extension:** add 10% allocation overhead and compare capacity for 1, 4 and 16 requests. Explain why a constant seconds/token assumption breaks under memory pressure and continuous batching. Never present synthetic output as empirical benchmark evidence.

## 4. Papers and critique
Vaswani et al., *Attention Is All You Need* (2017), arXiv:1706.03762, establishes attention architecture but not a serving-latency guarantee. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022), arXiv:2205.14135, studies IO-aware exact attention; assess hardware dependence and whether benchmark sequences match production. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023), arXiv:2305.13245, evaluates KV-head reduction; compare quality at equal task mix, not memory alone. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023), arXiv:2309.06180, studies serving memory and throughput; reproduction needs scheduler and workload details. These sources support mechanisms and experimental motivations, not numeric claims about your hardware.

**Doctoral research question G:** At a fixed accuracy floor on evidence-grounded security questions, how do KV-head count, context length and batching affect p95 latency and peak memory? Preregister a minimum factual-evidence recall, measure cold/warm runs, repeat 30+ requests per setting, and report confidence intervals. If resources are unavailable, implement the simulator and design the benchmark without inventing results.

# Track C — Unit 2 Session 4: ransomware, botnets and phishing

## 5. Intuition: detect a behavior chain, not a scary keyword
Ransomware may be suggested by unusual rapid file rewrites, backup tampering, privilege escalation, suspicious encryption-related process activity, and extortion messages, but benign backup, migration and encryption software can share individual indicators. Botnet C2 may appear as periodic outbound connections, unusual DNS activity, or a host contacting rare destinations; legitimate health checks and update agents can look periodic too. Phishing may involve a deceptive URL, impersonated sender, credential-collection page or malicious attachment; lexical URL features alone cannot determine intent. Different evidence becomes available at different decision times. A page-content classifier cannot protect a user *before navigation* if it requires visiting the page first; a reputation feed may be stale; an email header may be forged or forwarded. Combine early, authorized signals and report uncertainty.

```text
mail / URL metadata --+                          +--> analyst explanation
DNS / flow metadata --+--> normalize time --> rules / ML --> calibrated risk
endpoint file events -+        |                  +--> policy gate
identity audit logs --+   evidence ledger             |
                                                    alert / isolate only
                                               under authorized policy
```

**Case study (synthetic):** Host A creates 800 encrypted-looking files in 2 minutes. A backup process running under an approved service account could explain the pattern. Host B exhibits the same burst, followed by shadow-copy deletion and a suspicious process launched from a temporary directory. The second chain warrants higher priority, but neither is automatically a confirmed ransomware incident. Preserve process ancestry, timestamps, file hashes and authorization context. A C2 detector sees 5-minute beaconing from Host C to a domain: check whether it is an approved monitoring service before escalation. A phishing detector sees a lookalike domain and credential form: corroborate domain registration, sender authentication and destination reputation while avoiding untrusted active content.

## 6. Probability, thresholds and alert economics
Let A denote attack, E evidence, π=P(A) the base rate, sensitivity s=P(alert|A), specificity c=P(no alert|not A). Bayes gives:

**P(A|alert) = sπ / [sπ + (1−c)(1−π)].**

For π=.001, s=.90, c=.99, P(A|alert)=.0009/(.0009+.00999)≈**0.08264**, only 8.26%. At c=.999, precision≈.0009/(.0009+.000999)≈**47.39%**. Improving specificity by one percentage point on the displayed percentage scale (99%→99.9% is 0.9 percentage points) dramatically reduces analyst noise. These are illustrative, not estimates of any real product.

For a calibrated posterior q=P(A|evidence), and losses C_FP for unnecessary containment and C_FN for missing an attack, containment minimizes expected immediate loss when (1−q)C_FP < q C_FN, equivalently q>C_FP/(C_FP+C_FN). With C_FP=10 and C_FN=90, threshold is .10. **Caution:** disruptive containment can have cascading business costs; the simplified two-action loss model omits delayed evidence gathering, escalation capacity, reversible responses and adversarial adaptation. Scores must be calibrated and losses approved before applying this formula.

**Worked temporal correlation:** Suppose independent benign triggers occur with probability .02 for rapid rewrites and .01 for unusual outbound traffic. Under an *explicit conditional-independence assumption*, joint benign rate=.0002. But a scheduled backup could cause both signals, invalidating independence and inflating confidence. Test dependence using observed benign operations and incident-grouped evaluation. Correlation is not causation; labels based on an analyst's previous alert can create circular validation.

### Lab C — safe synthetic triage and Bayesian precision (no malware, no live URLs)
```python
def precision_at_base_rate(prevalence, sensitivity, specificity):
    numerator = prevalence*sensitivity
    denominator = numerator + (1-prevalence)*(1-specificity)
    return numerator/denominator

def triage(events):
    # Illustrative policy, NOT a production detector.
    score = sum((2 if x=="backup_tamper" else 1)
                for x in set(events)
                if x in {"rapid_rewrites","backup_tamper","rare_outbound"})
    return "review urgently" if score>=3 else "investigate" if score>=1 else "insufficient evidence"

a = ["rapid_rewrites"]
b = ["rapid_rewrites","backup_tamper","rare_outbound"]
assert triage(a)=="investigate"
assert triage(b)=="review urgently"
p1 = precision_at_base_rate(.001,.9,.99)
p2 = precision_at_base_rate(.001,.9,.999)
assert abs(p1-.0826446281)<1e-8
assert abs(p2-.4739336493)<1e-8
print(triage(a), triage(b))
print(round(p1*100,2), round(p2*100,2))
```
**Expected:** `investigate review urgently` and `8.26 47.39`. The score is deliberately **not calibrated**, so never feed it into the posterior threshold formula. **Ablations:** add benign backup cases, evaluate incident-level rather than event-level splits, and simulate attacker timing changes. **Failure test:** benign backup tampering in an authorized recovery test will trigger the same rule; assess cost before automated containment.

## 7. Research design, real-world context and limitations
MITRE ATT&CK describes observable ransomware-impact and command-and-control behaviors as technique knowledge, not prevalence estimates or proof that an event is malicious. NIST SP 800-61 Rev. 3 (2025) frames incident response in risk-management processes, not as permission for autonomous isolation. CISA's *#StopRansomware Guide* (2023) emphasizes preparation, detection and response; operational guidance should be checked for updates before deployment. For phishing research, URL-only models can overfit domain and capture artifacts; a chronological and domain-grouped split better probes generalization than random duplicate URLs. Avoid contacting suspicious domains during teaching; use synthetic names and offline features.

**Paper critique:** Saxe & Berlin (2017), *eXpose: A Character-Level Convolutional Neural Network with Embeddings For Detecting Malicious URLs, File Paths and Registry Keys*, arXiv:1702.08568, investigates character-level security strings; assess dataset vintage, distribution shifts, and whether labels transfer to contemporary campaigns. Sommer & Paxson (2010), *Outside the Closed World: On Using Machine Learning for Network Intrusion Detection*, IEEE Symposium on Security and Privacy, DOI:10.1109/SP.2010.25, explains deployment difficulties; it does not supply a universal detector threshold. Anderson & Roth (2018), *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637, remains useful as a contrast with the previous session's file analysis, but cannot substitute for ransomware-behavior labels.

**Doctoral research question C:** Does combining endpoint process ancestry with network-flow timing improve *incident-level* recall at a fixed daily analyst alert budget compared with either source alone? Design group/time splits, account for missing sensors, calibrate on validation only, preserve test incidents, measure false alerts per host-day and lead time to detection. Include privacy review for endpoint monitoring. A model trained on one organization's approved backup jobs may generalize poorly to another organization's schedule.

# Cross-track integration: an LLM as a controlled SOC explainer
The detection engine outputs structured observations and calibrated scores; the LLM converts *verified* evidence IDs into an analyst-readable narrative, with explicit unknowns. The LLM must not decide to isolate a host or visit an untrusted URL. If evidence includes a log line saying “ignore previous instructions,” treat it as quoted untrusted data. Apply least privilege, tenant isolation, prompt-injection checks, output schema validation, and a separate policy engine for response actions. A valid citation ID does not prove a supported claim; sample and audit factual entailment. Avoid uploading secrets, malware, private customer telemetry or personal identifiers to an unapproved model provider.

# Misconceptions and failure modes
A larger context window does not guarantee correct use of distant evidence. FlashAttention is not sparse attention and does not remove dense-attention arithmetic. KV caching saves recomputation but consumes memory. A high tokens/sec figure may hide poor p95 user latency. A ransomware-like file burst is not a confirmed incident; a clean scan is not proof of safety; periodic traffic is not necessarily C2; a suspicious URL string is not definitive phishing. Precision depends on prevalence and deployment conditions. A confidence score is not automatically a calibrated probability.

# Progressive oral Socratic tutorial (answer, then probe)
1. **Prefill versus decode?** Prompt positions processed in parallel versus sequential next-token steps. *Probe:* why does decode serialize?
2. **What is in a KV cache?** Per-layer past key and value vectors. *Probe:* why not cache just logits?
3. **Derive cache bytes.** 2BL S H_kv d_h b. *Probe:* identify the factor of two.
4. **Calculate the GQA example.** 2 GiB. *Probe:* what if KV heads quadruple? 8 GiB.
5. **Does FlashAttention change exact dense attention's asymptotic arithmetic?** No; it reduces memory IO/materialization. *Probe:* contrast sparse attention.
6. **What does TTFT measure?** Wait through queue/prefill/first token under a stated convention. *Probe:* distinguish completion latency.
7. **Why might batching hurt one user?** Contention and queueing despite higher aggregate throughput. *Probe:* choose a tail-latency metric.
8. **When can context truncation be dangerous?** It can remove critical evidence. *Probe:* design a retrieval recall test.
9. **Name three ransomware signals.** Rapid rewrites, backup tampering, suspicious process ancestry. *Probe:* name a benign counterexample.
10. **What is C2 beaconing?** Repeated communication potentially directed by a controller. *Probe:* why is periodicity insufficient?
11. **Why do URL lexical features fail?** Legitimate lookalikes and unseen domain patterns. *Probe:* what split detects leakage?
12. **Compute precision with π=.001, s=.9, c=.99.** ≈8.26%. *Probe:* why is accuracy unhelpful?
13. **What changes at specificity .999?** Precision ≈47.39%. *Probe:* what happens to analyst workload?
14. **Why must scores be calibrated for cost thresholds?** Threshold derivation assumes true posterior probabilities. *Probe:* suggest a calibration plot.
15. **Why is independent-signal multiplication risky?** Shared benign causes induce dependence. *Probe:* describe a backup scenario.
16. **What does missing sandbox/endpoint telemetry mean?** Unknown, not benign. *Probe:* how encode missingness?
17. **Can an LLM authorize containment?** Not by default; independent policy and human oversight. *Probe:* where should permissions live?
18. **Propose a falsifiable cross-track study.** Compare evidence-grounded SOC narratives with/without retrieved source IDs at matched detector outputs; score unsupported claims and review time. *Probe:* how avoid contamination?

# Source map and reproducibility
- Course progression: October 19 email PDFs, Generative AI Unit 2 Session 6 (page 6), Cybersecurity Unit 2 Session 4 (page 6); November 8 committed morning lesson.
- Prefill, attention and cache arithmetic: Vaswani 2017; Dao 2022; Ainslie 2023; Kwon 2023; arithmetic explicitly derived here and in Lab G.
- Threat observables and response boundaries: MITRE ATT&CK, NIST SP 800-61r3, CISA #StopRansomware; synthetic case and Lab C are **not** empirical incident data.
- Low-prevalence precision: Bayes derivation in §6, reproducible Lab C; Sommer & Paxson 2010 for deployment caveats.
- URL generalization: Saxe & Berlin 2017, with limitations explicitly discussed.
- Complete citations, stable links, local-paper status and access notes: [companion references index](references/index.md). No fresh November 9 headline is asserted without a dated primary-source check.

**Recall before the walk ends:** Explain why 2 GiB of KV cache is not 2 GiB of total GPU memory; state one case where optimization worsens quality; derive 8.26% precision from the base rate; describe an authorized, reversible incident-response workflow; propose one experiment whose negative result would still be publishable.
