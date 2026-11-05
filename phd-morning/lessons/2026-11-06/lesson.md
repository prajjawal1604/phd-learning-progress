# Morning PhD teaching package — Friday 6 November 2026 (Asia/Kolkata)
**Tracks:** Advanced Generative AI and Foundation Models; AI for Cybersecurity. **Accelerated sequence:** after Nov 5 attention masks/model families and temporal drift, advance to **Generative AI Unit 2 Session 3: context and efficient attention**, and **Cybersecurity Unit 2 Session 1: network versus host intrusion detection**. The two 19 October 2026 Marwadi University pathway PDFs specify these topics. This is independent accelerated study, not a modification of university contact-hour requirements.

## Spoken navigation
**10 minutes:** define context and IDS; narrate quadratic attention; contrast network and host evidence; answer Q1–Q3. **30 minutes:** add attention memory arithmetic, a flow-to-incident case, both worked examples, and Q4–Q8. **Extended (90–150 minutes):** run both labs, critique the papers, design a prospective evaluation, answer Q9–Q12. Pause after each question; learner answers aloud before revealing the answer. No external browsing is needed for core explanations.

## Prerequisites and vocabulary
Recall Q,K,V matrices, softmax, causal/padding masks, and training/test chronological separation from Nov 4–5. A **token** is a tokenizer-defined unit; **context length L** is the number of tokens considered at once; **attention head** is one set of query/key/value projections; **prefill** processes input tokens, **decode** generates subsequent tokens; **KV cache** stores previous keys and values for autoregressive inference. A **network IDS (NIDS)** observes packet/flow/network metadata; a **host IDS (HIDS)** observes endpoint processes, files, system calls, authentication and audit events. A **flow** aggregates packets for a conversation over an interval; an **incident** is a human/security conclusion, not automatically one flow. **Telemetry provenance** is where and when evidence was observed. **False positive** is a benign observation flagged as suspicious. These distinctions define the unit of prediction.

# Part I — Context and efficient attention
## 1. Intuition and mechanism
Imagine reading a long investigation transcript. Dense self-attention lets each token compare with every visible token. This is flexible but its pairwise score matrix grows with the square of sequence length. The model's *advertised* context window is not a guarantee it will faithfully retrieve evidence placed far from the question. Distractors, recency bias and task difficulty must be evaluated independently.

For one head, input X has shape L×d_model; Q=XW_Q, K=XW_K, V=XW_V each have shape L×d_k (assuming value width d_k). The attention output is
```
S = Q K^T / sqrt(d_k) + M                 [L,L]
P_ij = exp(S_ij) / sum_{k allowed} exp(S_ik)
O = P V                                  [L,d_k]
```
Here i is the query position, j is a key position, M is zero for allowed pairs and negative infinity for blocked pairs. Dividing by sqrt(d_k) moderates score variance when query/key components have roughly unit variance and weak dependence. For h heads, dense score storage is O(hL²) elements; QK multiplication costs O(L²d_model) under standard equal-head dimensions. This is a scaling model, not a benchmark of actual kernel memory use. FlashAttention [P2] tiles exact attention to reduce GPU memory traffic and avoids materializing the entire score matrix in high-bandwidth memory; it does **not** remove the underlying all-pairs arithmetic. Sparse/local attention changes the set of pairs and may lose global dependencies.

```text
tokens -> embedding + positions -> Q,K,V
                                      |
                          [scaled QK^T + visibility mask]
                                      |
                                  row softmax
                                      |
                                   weights @ V -> projection
Long context choices: dense exact | tiled exact | local/sparse | retrieve-then-read
```

### Worked calculation: scaling
Take 16 heads, L=2048, float16 attention scores (2 bytes each). A naive score tensor alone occupies 16×2048²×2 = 134,217,728 bytes = **128 MiB**, excluding gradients, activations and other buffers. Doubling L to 4096 yields **512 MiB** for that same tensor. Ratio is four, not two. This does not predict actual FlashAttention peak allocation. At L=4096, a local window of 256 previous positions would consider at most ~1.05 million pairs per head rather than ~16.8 million dense pairs, but could miss an important event 500 tokens away. Causal attention has approximately half the unmasked pairs yet common dense implementations can still allocate a square score array.

### When each approach works or fails
Dense exact: useful for moderate context and unconstrained dependencies, expensive at large L. IO-aware exact: useful on supported hardware/kernels for the same attention result within floating-point tolerance; gains vary by GPU, dtype, mask and shape. Local/sparse: suitable when locality is a justified prior, vulnerable to distant evidence. Retrieval: efficient for finding external documents, but retrieval recall, chunk boundaries, stale indexes and prompt injection become failure points. KV cache accelerates sequential decoding by reusing earlier projections, but grows with sequence length and batch size and is not a training substitute.

**Misconception:** more context always means more factual accuracy. Correction: capacity and reliable evidence use are separate measurable properties. **Misconception:** FlashAttention makes dense attention linear-time. Correction: its central innovation is IO-aware exact attention, not universal linear arithmetic.

## 2. Lab A — attention shape, mask and scaling (NumPy, offline)
Requires Python 3.10+ and `pip install numpy`. This is a small illustrative CPU calculation, not a GPU speed benchmark.
```python
import numpy as np
rng=np.random.default_rng(6)
def causal_attention(x):
    L,d=x.shape
    q,k,v=x,x,x
    s=q@k.T/np.sqrt(d)
    s[np.triu_indices(L,1)]=-np.inf
    z=s-np.max(s,axis=1,keepdims=True)
    w=np.exp(z); w/=w.sum(axis=1,keepdims=True)
    return w,w@v
for L in (4,8,16):
    x=rng.normal(size=(L,8))
    w,y=causal_attention(x)
    assert y.shape==(L,8)
    assert np.allclose(w.sum(axis=1),1)
    assert np.allclose(np.triu(w,1),0)
    print("L",L,"score_elements",L*L,"first_visible_keys",np.count_nonzero(w[0]))
print("PASS")
```
**Expected:** score_elements 16, 64, 256; first_visible_keys 1 each; PASS. The count is the size of a dense score array, not observed memory. **Experiment:** change the mask to bidirectional; compare first-row weights, then double L and predict fourfold score growth. **Failure mode:** all-masked rows produce undefined softmax; explicitly define a zero-output policy if implementing padded queries.

## 3. Real-world case and research test
A SOC assistant receives a 40-page incident timeline; the decisive log line is in the middle and distractor paragraphs repeat an incorrect host name. Compare (A) full-context answer, (B) retrieval of top-k timestamped snippets followed by answer, and (C) fixed local window. Build 30 synthetic or sanitized questions with exact evidence IDs and planted distractors. Report evidence recall@k, exact answer correctness, unsupported claims, token count, latency, and confidence intervals. Use *the same* gold questions; avoid placing answer keys in prompts. Never feed untrusted retrieved text into privileged tool instructions.

# Part II — Network versus host intrusion detection
## 4. Evidence and detection contracts
A NIDS can see flow counts, ports, durations, bytes and, where legally available, packet metadata; encryption often limits payload inspection. A HIDS can observe executable lineage, command-line, authentication and file changes, but requires endpoint access and can be disabled or evaded. They complement, rather than replace, signatures, firewall policy and analyst judgment.

```text
network sensors -> timestamped flow records ----                                                  -> normalize + provenance
host agents ----> process/auth/file events ------/          |
                                                       entity/time joins
                                                            |
                                                rules + statistical detector
                                                            |
                                             scored events -> dedupe/correlate
                                                            |
                                              analyst queue -> investigation
                                                            |
                                          delayed labels -> evaluation/drift
```

The choice of **unit** matters: a flow detector answers 'is this flow suspicious?', a host-window detector answers 'is this host suspicious during five minutes?', and incident correlation answers 'do multiple observations indicate a case?'. Treating correlated flows as independent test examples inflates confidence. A process-tree signal may add context but introduces collection overhead and sensitive data handling.

### Worked example: alarm volume and base rate
Suppose 100,000 flows/day, 0.2% attacks (200), true positive rate 0.9, false positive rate 0.01. Expected true alerts = 200×0.9 = **180**; false alerts = 99,800×0.01 = **998**; total **1178**, precision = 180/1178 ≈ **15.28%**. A classifier with 90% recall is not necessarily operationally useful when 998 benign flows must be reviewed. If each analyst reviews 100 alerts/day, roughly 12 analyst-days of review would be needed *under the unrealistic assumption every alert is separately reviewed*. Correlation and prioritization change workload; measure them explicitly.

**When NIDS fails:** asymmetric routing, missing sensors, TLS payload blindness, benign high-volume transfers, changing infrastructure. **When HIDS fails:** missing audit policies, tampered agents, short-lived processes, telemetry delays, OS-specific behavior. **Ethics/security:** collect minimal data, restrict access, preserve timestamps and evidence integrity, test on authorized datasets only; do not execute malware or scan networks without permission.

## 5. Lab B — synthetic NIDS/HIDS fusion with held-out time (stdlib only)
Illustrative generated events, not a measured IDS or real threat dataset. The labels are synthetic; this lab demonstrates evidence fusion and evaluation, not model superiority.
```python
from collections import Counter
# (day, network_flag, host_flag, true_incident)
rows=[
(1,0,0,0),(1,1,0,0),(1,1,1,1),(1,0,1,0),
(2,1,1,1),(2,1,0,0),(2,0,1,1),(2,0,0,0),
(3,1,1,1),(3,1,0,0),(3,0,1,0),(3,0,0,0)]
train=[r for r in rows if r[0]<3]
test=[r for r in rows if r[0]==3]
def metrics(data,fn):
    tp=fp=fn_count=0
    for _,n,h,y in data:
        p=int(fn(n,h))
        tp+=p==1 and y==1
        fp+=p==1 and y==0
        fn_count+=p==0 and y==1
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn_count) if tp+fn_count else 0
    return (tp,fp,fn_count,round(precision,3),round(recall,3))
for name,rule in [("network",lambda n,h:n),
                  ("host",lambda n,h:h),
                  ("both",lambda n,h:n and h)]:
    print(name,metrics(test,rule))
assert metrics(test,lambda n,h:n and h)==(1,0,0,1.0,1.0)
```
**Expected test output:** network (1,1,0,0.5,1.0); host (1,1,0,0.5,1.0); both (1,0,0,1.0,1.0). **Do not generalize** from four test examples: the AND rule misses attacks with only one sensor firing (see day 2). The tiny test is deliberately underpowered; no uncertainty or deployment performance claims follow. **Extension:** generate 30 days, group by host and time, preserve timestamps, use day 1–20 train, 21–25 validation and 26–30 untouched test; evaluate recall, precision, FPR, review volume, missing-agent rate and latency. A sensor outage must be represented as missing data, not automatically as 'no threat'.

## 6. Case study and experiment design
Imagine a compromised account makes many outbound connections while a legitimate backup service makes similar flows. NIDS alone cannot reliably distinguish them. Add endpoint process lineage and change windows, then compare three policies: network-only, host-only, and fusion with missing-data handling. Define ground truth through incident review, not merely rule agreement. Report per-host/day metrics and latency; enforce an analyst capacity limit. Split by time and, where possible, previously unseen hosts. Validate threshold on earlier data only. Investigate correlated false positives and privacy impact before deployment.

# Cross-track synthesis
Long-context models and IDS both face **evidence selection**: a model may miss a relevant sentence among distractors, while a SOC may miss a malicious host among millions of flows. More raw input does not guarantee better decisions. A retrieval step and an event-correlation step each reduce volume but introduce their own recall bottleneck. Formulate the end-to-end probability of success as P(evidence selected) × P(correct reasoning | evidence selected); this is a useful decomposition only if stages and outcomes are operationally defined.

## Progressive oral tutoring — pause before each answer
1. **What is context length?** Number of tokens in the considered sequence, not guaranteed retrieval reliability. *Follow-up:* what if decisive evidence is at token 10,000?
2. **Why divide by sqrt(d_k)?** To moderate dot-product scale under simple variance assumptions. *Follow-up:* what happens to softmax saturation?
3. **Why does doubling L quadruple naive scores?** L×L pairs. *Follow-up:* distinguish arithmetic from peak GPU allocation.
4. **Does FlashAttention approximate softmax?** Its central dense algorithm computes exact attention (up to numerical differences) with IO-aware tiling. *Follow-up:* what is still quadratic?
5. **When does local attention fail?** When relevant dependencies lie outside its receptive field. *Follow-up:* propose a long-range test.
6. **What is a flow?** Aggregated network conversation metadata, not an incident. *Follow-up:* why are flow-level splits risky?
7. **What does HIDS see that NIDS may not?** Process/file/authentication evidence. *Follow-up:* what if endpoint agent is offline?
8. **Compute precision from 180 true and 998 false alerts.** 180/1178≈15.28%. *Follow-up:* why is recall alone insufficient?
9. **Why avoid random splitting of adjacent flows?** Correlation and temporal leakage can inflate generalization estimates. *Follow-up:* propose a split.
10. **Should two sensors always be ANDed?** No; fewer false alarms can mean more missed attacks. *Follow-up:* design missing-data behavior.
11. **How could retrieval and IDS correlation share a failure mode?** Both can discard decisive evidence before downstream reasoning. *Follow-up:* define recall at each stage.
12. **Propose a publishable hypothesis.** Example: under a fixed review budget, time-aware network/host fusion improves incident recall on unseen hosts compared with network-only; preregister dataset, splits, latency and statistical uncertainty. *Follow-up:* what result would falsify it?

## Research tasks and reproducibility
Write a one-page experiment preregistration: unit, population, observation cutoff, missing telemetry, data rights, split boundaries, baseline, threshold, costs, primary metric, confidence interval, and negative-result criteria. For the context experiment, vary document length, evidence position and distractor count independently; for IDS, vary attack prevalence, sensor coverage and entity shift. Save seed, software versions, dataset manifest, sanitized examples, and errors. Do not present synthetic labs as real security findings.

## Primary references and source map
- **[P1]** Vaswani et al., *Attention Is All You Need* (2017), arXiv:1706.03762, https://arxiv.org/abs/1706.03762 — attention formula and Transformer mechanism. 
- **[P2]** Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022), arXiv:2205.14135, https://arxiv.org/abs/2205.14135 — IO-aware exact attention, tiling, memory traffic. 
- **[P3]** Ring et al., *A Survey of Network-based Intrusion Detection Data Sets* (2019), DOI:10.1016/j.cose.2019.06.005, https://doi.org/10.1016/j.cose.2019.06.005 — dataset limitations and evaluation caution; verify full text before attributing detailed findings.
- **[P4]** Moustafa & Slay, *UNSW-NB15: a comprehensive data set for network intrusion detection systems* (2015), DOI:10.1109/MilCIS.2015.7348942, https://doi.org/10.1109/MilCIS.2015.7348942 — benchmark dataset origin; do not treat as proof of field performance.
- **[S1]** October 19 syllabus PDF, *Advanced Generative AI and Foundation Models Learning Pathway*, Unit 2 Session 3 (context and efficient attention).
- **[S2]** October 19 syllabus PDF, *AI for Cybersecurity Learning Pathway*, Unit 2 Session 1 (network and host intrusion).
**Source map:** mechanism/equation → [P1]; exact IO-aware optimization → [P2]; benchmark selection caveats → [P3,P4]; lesson sequencing → [S1,S2]; 128/512 MiB and alert arithmetic → derivations in this lesson; lab claims → runnable synthetic scripts in this lesson. **Freshness note (6 Nov 2026):** canonical paper records for [P1,P2] were checked during preparation. No specific November 2026 AI/cybersecurity news item was included without a verifiable dated primary source. This lesson deliberately prioritizes validated durable research over unverified headlines.

**Reading route:** read [P1] sections on scaled dot-product attention; [P2] introduction and IO complexity; then compare with the syllabus's network/host IDS tasks. Companion `references/index.md` lists access notes. **Paper PDF policy:** no external PDF bytes are asserted to have been downloaded or committed; stable canonical links are provided instead.
