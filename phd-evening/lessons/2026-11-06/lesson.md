# Evening PhD deep dive — 6 November 2026 (Asia/Kolkata)
**Continuity contract.** Read and deepen [today's morning lesson](../../../phd-morning/lessons/2026-11-06/lesson.md) and its [references](../../../phd-morning/lessons/2026-11-06/references/index.md). Same topics only: **long-context/efficient attention** (Generative AI Unit 2 Session 3) and **network-versus-host intrusion detection** (AI for Cybersecurity Unit 2 Session 1). The two syllabus PDFs were located in the 19 October 2026 Gmail message, *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*. This session deepens, rather than replaces, morning teaching. Syllabus PDFs remain private and are not redistributed.

## Spoken tutoring routes
**10 minutes:** narrate the memory-versus-compute distinction, derive online softmax, explain why AND sensor fusion loses recall, answer oral Q1–Q4. **30 minutes:** add two numerical examples, the sensor-missingness counterexample, and Q5–Q9. **Extended 90–150 minutes:** execute both labs, compare ablations, critique primary research, answer Q10–Q16, write an experiment preregistration. Ask each question aloud; pause for an answer; then read the solution and probe.

## Prerequisites and precise definitions
Know Q, K, V, masked row softmax, causal attention, and the morning 128 MiB versus 512 MiB example. Let L be tokens, d the key dimension, B the tile width, h the number of heads, and b the bytes per stored element. **Exact attention** means evaluating the same mathematical dense attention (subject to floating-point rounding), not necessarily storing the L×L score matrix. **IO complexity** measures reads/writes between memory levels, distinct from FLOPs. **KV cache** stores projected past keys/values at decoding; it saves recomputation but its storage grows with retained tokens. For cybersecurity, define **NIDS** as network observation and **HIDS** as host observation; a **missing sensor** is not a negative observation. **Event time** is when activity occurred; **ingest time** is when the detector received it; **label time** is when a case was adjudicated. A **decision threshold** maps a score to an alert; **incident-level recall** measures found incidents, not merely correctly classified flows.

# Track A — Efficient attention under a microscope

## A1. Why attention is quadratic yet can use subquadratic *working* memory
For one head, Q,K in R^(L×d), V in R^(L×d_v), attention is O = softmax(QKᵀ/√d + M)V. The dot products require approximately L²d multiply-accumulates; forming the complete score and probability arrays naively requires O(L²) storage. IO-aware tiling can avoid writing those complete arrays to global memory. Thus the slogan “FlashAttention makes attention linear” confuses memory traffic with arithmetic. A GPU implementation's total allocation also includes Q/K/V, outputs, gradients, temporary buffers and allocator behavior.

**Derive stable online softmax.** For a query row with logits s_j, maintain running maximum m, normalizer l and weighted accumulator a. Initially m=-∞, l=0, a=0. For a new tile T, let m_T=max(s_j:j∈T), l_T=Σ_T exp(s_j-m_T), a_T=Σ_T exp(s_j-m_T)v_j. Merge with
```
m' = max(m,m_T)
alpha = exp(m-m')       (take alpha=0 for empty initial state)
beta  = exp(m_T-m')
l' = alpha*l + beta*l_T
a' = alpha*a + beta*a_T
output = a'/l'
```
**Proof:** multiplying both old and new exponential sums by exp(-m') expresses every term in one common normalization frame; addition preserves the exact numerator and denominator. This associativity permits tiled computation. Masked entries have zero exponential contribution. A fully masked row needs an explicit policy; division by zero is not meaningful. Online softmax also requires numerical care when both running and tile maxima are -∞.

**Worked example:** logits [0, log(2), log(4)] and scalar values [10,20,40]. First tile [0,log2] gives m=log2, l=1/2+1=1.5, a=5+20=25. Next tile [log4] has m_T=log4, l_T=1, a_T=40. Rescale old by exp(log2-log4)=1/2: l'=0.75+1=1.75, a'=12.5+40=52.5; output=52.5/1.75=30. Direct softmax gives (10+2×20+4×40)/7=30. **Counterexample:** averaging each tile's separately normalized output gives (25/1.5+40)/2≈28.33, wrong because tile masses differ.

```text
Q tile --> scores against K tile --> local max/sum/weighted V
   |                 |                       |
   +-----------------+------ running (m,l,a) +--> normalized output
                     no complete [L,L] matrix written
```

**Complexity comparison:** naive score tensor hL²b bytes. At h=16, L=4096, b=2: 512 MiB just for scores. A window w=256 has O(Lw) permitted pairs, but a dense mask over a dense kernel does not automatically yield sparse compute savings. Sliding windows can miss dependencies beyond w. Retrieval-augmented context changes the candidate evidence set and therefore adds retrieval recall as a bottleneck.

## A2. Prefill versus decoding: an often-missed cost
For autoregressive generation at step t, one new query attends to t cached keys, costing O(td) attention arithmetic per head and storing approximately 2×t×h_kv×d_head×b bytes per sequence for cached keys plus values (excluding overhead; h_kv is number of distinct KV heads). With h_kv=8, d_head=128, t=8192, b=2: 2×8192×8×128×2=33,554,432 bytes = 32 MiB **per sequence**. If 32 independent sequences have separate caches, this component alone is roughly 1 GiB. Grouped-query attention reduces h_kv relative to query-head count but does not guarantee correctness at longer context. Prefill is typically a many-query operation; decoding is often constrained by repeated KV memory reads. Hardware, batching, quantization and kernel selection determine actual throughput.

**Failure cases to articulate:** numerical drift at low precision; an all-masked row; context truncation; cache invalidation after modifying earlier tokens; stale retrieval index; prompt injection in a retrieved incident report; a middle-position fact overlooked despite fitting in the window. Never infer answer fidelity solely from model context-window specification.

## A3. Advanced runnable Lab A — tiled stable attention, exact reference and tests
Requires Python 3.10+ and NumPy (`python -m pip install numpy`). This is CPU reference code, **not** a FlashAttention GPU performance implementation.
```python
import numpy as np
rng = np.random.default_rng(20261106)
def reference(q,k,v,causal=False):
    s=q@k.T/np.sqrt(q.shape[1])
    if causal:
        i=np.arange(q.shape[0])[:,None]; j=np.arange(k.shape[0])[None,:]
        s=np.where(j>i,-np.inf,s)
    z=np.exp(s-np.max(s,axis=1,keepdims=True))
    return (z/z.sum(axis=1,keepdims=True))@v

def tiled(q,k,v,tile=3,causal=False):
    n,d=q.shape
    out=np.empty((n,v.shape[1]))
    for i in range(n):
        m=-np.inf; den=0.; num=np.zeros(v.shape[1])
        for start in range(0,len(k),tile):
            js=np.arange(start,min(start+tile,len(k)))
            scores=(k[js]@q[i])/np.sqrt(d)
            if causal: scores=np.where(js>i,-np.inf,scores)
            valid=np.isfinite(scores)
            if not valid.any(): continue
            mt=scores[valid].max()
            ex=np.exp(scores[valid]-mt)
            newm=max(m,mt)
            oldscale=0. if m==-np.inf else np.exp(m-newm)
            newscale=np.exp(mt-newm)
            den=den*oldscale+ex.sum()*newscale
            num=num*oldscale+(ex@v[js[valid]])*newscale
            m=newm
        if den==0: raise ValueError("all keys masked")
        out[i]=num/den
    return out

for causal in (False,True):
    q=rng.normal(size=(11,8)); k=rng.normal(size=(11,8))
    v=rng.normal(size=(11,5))
    expected=reference(q,k,v,causal)
    for tile in (1,2,3,5,11):
        actual=tiled(q,k,v,tile,causal)
        assert np.allclose(actual,expected,atol=1e-12)
print("PASS: 10 tiled/reference comparisons")
```
**Expected:** `PASS: 10 tiled/reference comparisons`. Test additionally with logits multiplied by 100 to challenge stability. Ablate the rescaling term and explain the resulting wrong answer. Test a row with no allowed keys and specify behavior before production use. Benchmark memory and latency separately; Python-loop speed is not representative of fused GPU kernels.

## A4. Research critique and experiment
Vaswani et al. [P1] establish Transformer scaled dot-product attention, but do not prove arbitrary long-context retrieval reliability. Dao et al. [P2] study exact IO-aware attention; reduced memory movement should not be misrepresented as a linear-time attention algorithm. **Research extension:** preregister 4 context lengths × 3 evidence positions × 3 distractor densities, with 20 independent examples per cell. Compare dense full-context, fixed-window, and retrieve-then-read using identical questions. Primary metric: evidence-ID recall; secondary: grounded answer accuracy, latency, cost, hallucination rate. Report paired bootstrap intervals and failure examples. Retrieval may win speed but lose decisive evidence before reasoning.

# Track B — NIDS/HIDS fusion as decision theory

## B1. From alerts to calibrated incident decisions
Suppose event x contains network evidence n, host evidence h, and telemetry-availability indicators a_n,a_h. Desired output is P(incident | n,h,a_n,a_h,context). The availability indicators matter: a missing host agent can be correlated with attack or maintenance and is not equivalent to host evidence h=0. Naive Bayes multiplication of network and host likelihoods assumes conditional independence; shared infrastructure and common rules violate that assumption, leading to overconfident fusion.

**Derive a decision threshold.** Let p=P(incident|evidence), C_FP be cost of investigating a benign case, C_FN be cost of missing an incident. Alert expected cost is (1-p)C_FP; no-alert expected cost is p C_FN. Alert when (1-p)C_FP < p C_FN, equivalently p > C_FP/(C_FP+C_FN). For C_FP=1 and C_FN=19, threshold is 0.05, **provided** p is calibrated and these costs capture the real decision. With analyst capacity 100 cases/day, a fixed cost threshold alone may exceed capacity; evaluate top-k ranking and delayed queue effects.

**Worked base-rate update:** morning's 100,000 flows, 200 attacks, TPR=.90, FPR=.01 yielded 180 true and 998 false alerts: precision 180/1178≈15.28%. If prevalence halves to 0.1% while TPR/FPR stay fixed: TP=90, FP=999, precision=90/1089≈8.26%. Same classifier, much worse operational precision. At incident level, multiple flagged flows may collapse to one case; never treat flow precision as incident precision.

## B2. Temporal joins, leakage, and sensor failure
```text
NIDS flow (event_time, src/dst, flow_id) ----+
                                            +--> clock correction --> entity/time join
HIDS process (event_time, host, pid) --------+                            |
             agent heartbeat / coverage ---------------------------------+
                                                                         |
                                                   score + missingness + provenance
                                                                         |
                                                 dedupe -> incident queue -> analyst
                                                                         |
                                        delayed confirmed labels -> time-split audit
```
A valid join needs an entity mapping (NAT, DHCP, VPN can change identities), time window, known clock error, and provenance. Joining on future events leaks information into an online detector. A label assigned three days later must not be used as an input feature at decision time. If an agent goes offline, an AND policy silently suppresses alerts; OR can explode false positives; learned fusion requires coverage-aware validation. A hostile endpoint may tamper with HIDS, and encrypted traffic may obscure NIDS payloads.

**Counterexample:** NIDS=1, HIDS=0 for a genuine attack whose endpoint agent was disabled. AND predicts benign; OR predicts suspicious. Neither is universally optimal. With NIDS=1, HIDS missing, a correct system records uncertainty rather than treating missing as 0. Evaluate coverage-stratified recall and missingness as a possible attack signal, without asserting every outage is malicious.

## B3. Advanced runnable Lab B — prevalence, fusion and missing telemetry
Standard library only; all events below are synthetic and illustrate trade-offs, not deployment performance.
```python
from collections import defaultdict
rows = [
 # day, host, net_flag, host_flag (None=missing), truth
 (1,"a",1,1,1),(1,"b",1,0,0),(1,"c",0,1,0),
 (2,"a",1,None,1),(2,"b",1,0,0),(2,"d",0,1,1),
 (3,"e",1,1,1),(3,"f",1,None,1),(3,"g",1,0,0),
 (3,"h",0,1,0),(3,"i",0,None,0)
]
def predict(n,h,policy):
    if policy=="network": return bool(n)
    if policy=="and": return bool(n and h is not None and h)
    if policy=="or": return bool(n or (h is not None and h))
    if policy=="coverage-aware":
        return bool(n if h is None else n and h)
    raise ValueError(policy)
def evaluate(records,policy):
    tp=fp=fn=0
    for _,_,n,h,y in records:
        p=predict(n,h,policy)
        tp+=bool(p and y); fp+=bool(p and not y); fn+=bool(not p and y)
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":round(tp/(tp+fp),3) if tp+fp else None,
            "recall":round(tp/(tp+fn),3) if tp+fn else None}
test=[r for r in rows if r[0]==3]
for policy in ("network","and","or","coverage-aware"):
    print(policy,evaluate(test,policy))
assert evaluate(test,"and")["FN"]==1
assert evaluate(test,"coverage-aware")["TP"]==2
print("PASS")
```
**Expected test-day results:** network TP2 FP1 FN0 precision .667 recall 1; AND TP1 FP0 FN1 precision 1 recall .5; OR TP2 FP2 FN0 precision .5 recall 1; coverage-aware TP2 FP0 FN0 precision 1 recall 1; PASS. The last rule's perfect result is an artifact of five held-out examples. **Ablations:** change row (3,"f",1,None,1) to a benign outage and watch coverage-aware precision drop; change (3,"e",1,1,1) to NIDS=0 and observe AND recall loss. Expand to thousands of events with host-group and time holdouts, explicit delayed labels, calibration curves and queue-capacity constraints. Never tune a rule using the test day.

## B4. A realistic authorized case
A legitimate backup server and a compromised workstation both show high outbound traffic. NIDS flow volume alone cannot identify intent; endpoint process lineage and maintenance schedule help. But if HIDS telemetry arrives late, the detector must choose between early low-confidence escalation and delayed correlation. Define incident onset, detection delay, and the maximum permissible waiting window before evaluating the policy. Protect command-line and user identifiers; preserve raw evidence access controls, retention limits and audit trails. Use authorized synthetic or appropriately licensed datasets, not live intrusive scanning.

## B5. Research critique and extensions
Ring et al. [P3] survey network IDS datasets; use the survey to frame representativeness questions, not to claim that any benchmark establishes field performance. Moustafa and Slay [P4] describe UNSW-NB15; benchmark origin is not evidence of real-world efficacy. **Falsifiable hypothesis:** under a fixed 100-cases/day analyst budget, coverage-aware NIDS/HIDS fusion improves incident-level recall on previously unseen hosts compared with network-only, without increasing median detection latency beyond five minutes. Pre-register label adjudication, event-time joins, host-disjoint chronological split, confidence intervals, sensor outage strata and a negative-result criterion. Report alert-volume distribution, privacy cost and false-negative case studies.

# Cross-track synthesis: evidence bottlenecks
In retrieval-augmented long-context reasoning, a relevant paragraph can be discarded before the model sees it. In incident correlation, a relevant host event can be lost before the analyst sees it. For a two-stage system, P(success)=P(evidence retained)×P(correct decision | evidence retained) when the events are explicitly defined. Improving downstream reasoning cannot recover evidence irrevocably dropped upstream. Both systems need provenance, coverage measurements, adversarial-input boundaries and evaluation at the final decision unit. A malicious retrieved document must never become an instruction; an attacker-controlled log line must never become a privileged tool command.

# Progressive Socratic oral examination (pause before reading answers)
1. **What does quadratic attention describe?** All-pairs score arithmetic/naive matrix size O(L²). *Probe:* which costs does tiling reduce?
2. **What is IO-aware attention?** It schedules exact attention to reduce expensive memory traffic. *Probe:* why can FLOPs remain quadratic?
3. **Why keep running maximum m?** Stable exponentials across tiles. *Probe:* derive the rescaling factors.
4. **Why not average tile outputs?** Their partition-function masses differ. *Probe:* recompute the [1,2,4] weights.
5. **How much KV cache for 8192 tokens, 8 KV heads, width 128, fp16?** 32 MiB per sequence for K and V alone. *Probe:* what happens at batch 32?
6. **Does fitting in context imply evidence recall?** No. *Probe:* propose a middle-position distractor test.
7. **What does retrieval change?** Candidate evidence and cost, with a recall bottleneck. *Probe:* define recall@k.
8. **What is the difference between missing HIDS and HIDS=0?** Unknown versus observed negative. *Probe:* how should the schema represent it?
9. **Why does AND reduce recall?** A true incident may fire only one sensor. *Probe:* give an outage example.
10. **Derive the alert threshold.** Compare (1-p)C_FP and pC_FN; threshold C_FP/(C_FP+C_FN). *Probe:* what if probabilities are uncalibrated?
11. **Why can precision fall with stable TPR/FPR?** Lower attack prevalence changes the base-rate denominator. *Probe:* calculate 8.26%.
12. **What is incident-level versus flow-level evaluation?** Case outcomes versus individual network records. *Probe:* how does deduplication change review workload?
13. **How can a time join leak labels?** Joining future endpoint events or adjudications into past decisions. *Probe:* state an observation cutoff.
14. **What does a perfect five-event lab prove?** Only that the implementation matches this toy dataset. *Probe:* design a prospective test.
15. **How do long-context and SOC fusion fail similarly?** Early evidence loss cannot be repaired downstream. *Probe:* formalize the two-stage probability.
16. **Propose a falsifiable experiment.** Fixed-budget fusion on unseen hosts with recall, latency, missingness and uncertainty pre-registered. *Probe:* what negative outcome rejects the hypothesis?

# Source map, paper critiques and verification boundaries
**Syllabus:** 19 October 2026 Gmail attachments *Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf* [S1] and *AI_for_Cybersecurity_Learning_Pathway.pdf* [S2]; morning lesson states Unit 2 Session 3 and Unit 2 Session 1 respectively. **Attention formula/Transformer:** Vaswani et al. [P1], *Attention Is All You Need* (2017), arXiv:1706.03762, https://arxiv.org/abs/1706.03762. **Tiled exact attention/IO:** Dao et al. [P2], *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022), arXiv:2205.14135, https://arxiv.org/abs/2205.14135. **IDS dataset caution:** Ring et al. [P3], *A Survey of Network-based Intrusion Detection Data Sets* (2019), DOI:10.1016/j.cose.2019.06.005, https://doi.org/10.1016/j.cose.2019.06.005; Moustafa and Slay [P4], *UNSW-NB15: a comprehensive data set for network intrusion detection systems* (2015), DOI:10.1109/MilCIS.2015.7348942, https://doi.org/10.1109/MilCIS.2015.7348942. **Local derivations:** online-softmax merge, KV cache arithmetic, Bayes decision threshold and prevalence arithmetic are shown in this lesson; **synthetic findings** belong only to the two runnable labs, not to the cited empirical literature. See [companion index](references/index.md) for access/licensing notes. Canonical links inherited from today's committed morning source map; publisher full-text findings and redistribution licenses have not been independently verified in this run. No paper PDFs are claimed downloaded. No current-news claim is made without dated source verification.

**Research deliverable:** write a preregistration with dataset license, threat model, ground truth, observation cutoff, baseline, split, confidence intervals, privacy review, false-negative analysis and falsification criteria. Save code output and Python/NumPy versions. No email, audio generation or narration is part of this preparation.
