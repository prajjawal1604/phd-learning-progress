# Morning PhD teaching package — 7 November 2026 (Asia/Kolkata)

**Syllabus continuity.** Both PDFs attached to the 19 October 2026 message “PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027” were read: *Advanced Generative AI and Foundation Models Learning Pathway* and *AI for Cybersecurity Learning Pathway* (Marwadi University PhD coursework, 2026). Yesterday completed efficient context/attention (Generative AI Unit 2 Session 3) and network-versus-host IDS (Cybersecurity Unit 2 Session 1), including an evening deep dive. Today progresses to **Generative AI Unit 2 Session 4: autoregressive decoding, sampling and prompting** and **Cybersecurity Unit 2 Session 2: time series and security logs**. This is accelerated independent study; it does not alter university contact hours.

## Spoken lesson routes
**10 minutes:** define conditional generation, temperature, top-k and top-p; explain event-time windows and delayed log arrival; solve examples G1 and C1; answer Q1–Q4 aloud.
**30 minutes:** narrate decoding loop and log pipeline diagrams, compare beam/sampling, derive lateness and anomaly scores, work through labs conceptually, answer Q1–Q9.
**Extended 90–150 minutes:** execute both labs, change parameters, critique papers, design a controlled experiment, answer all questions and research extensions. Pause after each question for the learner's attempt; only then read the answer and follow-up. Core teaching is self-contained.

## Prerequisites and definitions
From previous days: token IDs and embeddings, causal attention and KV caching, likelihood and softmax, precision/recall and sensor provenance. **Logit** is an unnormalized model score; **softmax** converts logits into probabilities. **Decoding** converts model conditional distributions into a token sequence. **Temperature** T>0 rescales logits before softmax. **Top-k** retains k highest-scoring tokens; **top-p/nucleus** retains the smallest sorted prefix with cumulative probability at least p. **Prompt** is the input context, not a guarantee of correct behavior. **Event time** is when an action happened; **ingest time** is when the pipeline saw it; **processing time** is when computation ran. **Window** aggregates events over time; **watermark** is a policy estimate that events before a timestamp are sufficiently complete. **Late event** arrives after the processing policy's allowed delay. **Sequence anomaly** is an unexpected order, timing or frequency, not necessarily malicious intent.

# Track G — Decoding and prompting from first principles

## G1. Why generation needs a policy
A causal language model learns P(x_1,...,x_n)=product from t=1 to n of P(x_t | x_<t). At inference, for vocabulary V, logits z_i yield p_i=exp(z_i/T)/sum_j exp(z_j/T). Stable implementation subtracts max(z/T). Temperature is **not** a confidence calibration method: T<1 sharpens and T>1 flattens the distribution; T approaching zero approximates argmax if a unique maximum exists. Greedy decoding chooses argmax at every step. It maximizes each *local* conditional probability, not necessarily the probability of the *whole* completed sequence. Beam search keeps B high-scoring partial sequences but can prefer generic or short completions unless length handling is specified. Sampling explores alternatives and needs a fixed seed for reproducibility.

**Worked example G1.** Logits for tokens [A,B,C] are [2,1,0]. At T=1, exponentials relative to max are [1,0.3679,0.1353], sum=1.5032; probabilities are approximately [0.6652,0.2447,0.0900]. Top-k=2 removes C and renormalizes to [0.7311,0.2689,0]. Top-p=0.8 retains A and B (A alone 0.6652<0.8, A+B 0.9099≥0.8). At T=0.5 the relative exponentials become [1,0.1353,0.0183], so A rises to ~0.8668. Distinguish truncation from rescaling: they have different effects.

**Counterexample to greedy optimality.** First step chooses A with 0.6 versus B with 0.4. The best completion after A has probability 0.2, whereas after B it has 0.9. Completed path probabilities are 0.6×0.2=0.12 and 0.4×0.9=0.36. Greedy chooses A and loses the higher-probability sequence. This is an illustrative two-step tree, not a statement that beam search always improves factuality.

### Mechanism and architecture
```text
user question + trusted system policy + retrieved evidence
                       |
                 tokenize + prefill
                       |
       causal model -> logits for next token
                       |
       temperature -> top-k / nucleus / greedy
                       |
       sample or select -> append token -> KV cache
                       |                      |
              EOS / token budget? --no-------+
                       |
           generated answer -> citation/claim checks
```
Prompt instructions can specify a schema and require evidence, but cannot force the underlying model to know missing facts. Treat retrieved webpages, emails and log text as untrusted *data*, not privileged instructions. In a security assistant, tool permissions must be enforced outside the model. A low temperature does not eliminate hallucinations, and a long context does not guarantee retrieval of the relevant line.

### Lab G — reproducible decoder (Python 3.10+, standard library)
```python
import math, random
def distribution(logits, temperature=1.0, k=None, p=None):
    assert temperature > 0
    vals=[v/temperature for v in logits]
    weights=[math.exp(v-max(vals)) for v in vals]
    probs=[w/sum(weights) for w in weights]
    order=sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)
    keep=set(order[:k]) if k is not None else set(order)
    if p is not None:
        assert 0<p<=1
        running=0; nucleus=set()
        for i in order:
            nucleus.add(i); running+=probs[i]
            if running>=p: break
        keep &= nucleus
    assert keep
    mass=sum(probs[i] for i in keep)
    return [probs[i]/mass if i in keep else 0.0 for i in range(len(probs))]
base=distribution([2,1,0])
assert abs(sum(base)-1)<1e-12
assert abs(base[0]-0.66524096)<1e-6
assert distribution([2,1,0],k=2)[2]==0
assert distribution([2,1,0],p=.8)[2]==0
rng=random.Random(7)
weights=distribution([2,1,0],temperature=.7,p=.9)
samples=rng.choices(['A','B','C'],weights=weights,k=20)
print('base',*[round(x,4) for x in base])
print('sample_count',len(samples),'allowed',set(samples))
print('PASS')
```
**Expected:** `base 0.6652 0.2447 0.09`, sample_count 20, allowed subset of A/B, PASS. This lab uses fixed toy logits; it does not train a language model. **Exercise:** vary temperature 0.5,1,2 and compute entropy H=-sum p_i ln p_i; observe higher entropy at higher T before truncation. **Failure:** a naive softmax can overflow on large logits; a top-p implementation must include the threshold-crossing token. For k and p together, this lab takes their intersection and renormalizes; libraries can apply truncation in a different order, so record exact decoding semantics.

### Real-world design case
A SOC analyst asks an LLM to summarize an incident from time-stamped logs. Compare deterministic greedy generation against temperature=0.7 top-p=0.9 across 40 sanitized incidents with gold evidence IDs. Measure claim-level support, missing critical evidence, schema validity, token usage and latency. Use paired prompts and a blinded reviewer; sampling diversity is not itself correctness. Never put secrets or private production logs into an unapproved external model.

# Track C — Security time series, log sequences and delayed evidence

## C1. Why chronology matters
Network flows and host events form a time-indexed, partially observed stream. Aggregation can reveal burst patterns, while sequence order can reveal authentication followed by unusual process creation. A raw event may include timestamp, host, user, event type and sensor. Clocks can drift; timestamps may be forged or missing; the pipeline must preserve event time, ingestion time, source and parsing quality. Aggregation is lossy: a five-minute count cannot reconstruct exact order.

**Worked example C1: window boundary.** Fixed event-time windows are [10:00,10:05), [10:05,10:10). Login failures at 10:04:58 and 10:05:02 fall in different windows even though four seconds apart. A rolling five-minute window at 10:05:02 includes both. An event stamped 10:04:58 but ingested at 10:07:30 is late relative to a policy closing the first window at 10:06:00. If the system drops it, its count becomes biased. A watermark is a completeness *assumption*, not proof that all events arrived.

**Worked example C2: baseline anomaly.** A host typically produces mean μ=4 failed logins per five minutes with standard deviation σ=2. A window with x=12 has z=(12-4)/2=4. But counts are often overdispersed and nonstationary; z=4 is not automatically a calibrated compromise probability. A Poisson model with λ=4 assumes equal mean and variance, which may be inappropriate. Day-of-week, maintenance and shared identities may change baselines. Evaluate false alerts across hosts and hours.

### Event-time architecture
```text
network flow / endpoint / identity logs
               |
      parser + schema + provenance
               |
      timestamp sanity + dedupe
               |
      event-time buffer + watermark
               |
      tumbling / sliding windows + sequence joins
               |
      rule + statistical anomaly score
               |
      correlation -> analyst queue -> delayed adjudication
               |
      label store -> temporal evaluation + drift audit
```
A process creation at 10:02, a network connection at 10:03, and a privilege change at 10:04 may be more informative together than any one event. But merging by a reused hostname without stable identity can create false correlations. Missing telemetry is unknown, not benign. Treat data from different collection agents as having different delays and trust levels.

### Lab C — event time versus arrival time (stdlib only)
```python
from datetime import datetime,timedelta
events=[
 ('10:04:58','10:05:01','fail'),
 ('10:05:02','10:05:03','fail'),
 ('10:04:59','10:07:30','fail'),
 ('10:06:00','10:06:01','success')]
def sec(t):
    h,m,s=map(int,t.split(':')); return 3600*h+60*m+s
def window(t): return (sec(t)//300)*300
all_counts={}; accepted={}
for event,arrival,kind in events:
    if kind!='fail': continue
    w=window(event)
    all_counts[w]=all_counts.get(w,0)+1
    # This intentionally simplistic policy closes each window 60 seconds after its end.
    close=w+300+60
    if sec(arrival)<=close:
        accepted[w]=accepted.get(w,0)+1
print('true',sorted(all_counts.values()))
print('accepted',sorted(accepted.values()))
assert sum(all_counts.values())==3
assert sum(accepted.values())==2
assert all_counts[window('10:04:58')]==2
print('PASS')
```
**Expected:** true [1,2], accepted [1,1], PASS. **Interpretation:** arrival-policy censoring loses one event, changing the observed first-window count. This is a synthetic demonstration, not a production watermark algorithm. **Extend:** add per-sensor lateness distributions, retractions/upserts for late events, rolling windows and a deduplication key. Measure both incident recall and processing latency. **Failure:** dropping late evidence can hide an attack; accepting indefinitely late evidence can make alerts unstable. Choose a documented correction policy.

### Research case and ethics
For a hypothetical enterprise authentication system, build chronological train/validation/test weeks, group by identity and device, and retain timestamps as audit fields. Compare (A) per-window count threshold, (B) a sequence rule with event ordering, (C) a learned anomaly model. Report incident recall at a fixed analyst alert budget, detection delay, false alerts per host-day, missing sensor rate and confidence intervals clustered by incident. Public IDS datasets may not capture enterprise identity patterns; do not claim deployment performance from synthetic examples. Collect minimal user data, apply retention and access controls, and avoid labeling unusual employee behavior as malicious without investigation.

# Joint synthesis: LLM summarization over imperfect logs
A model's decoding policy and the SOC's event-time policy can both change the narrative. Missing late logs cause evidence omissions upstream; sampling can add unsupported assertions downstream. Build a two-stage test: (1) replay the same events under different lateness policies; (2) generate evidence-constrained summaries under different decoding policies. Ground-truth evidence IDs must be fixed. Compare evidence coverage, unsupported claims, alert latency and reviewer burden. This is a defensible doctoral hypothesis about the *interaction* of data quality and language generation, not proof of a specific model's reliability.

## Misconceptions and failure modes
- “Greedy always finds the most likely completed answer”: false; local maxima need not maximize the path.
- “Temperature 0 guarantees truth”: false; it changes token selection, not grounding.
- “Top-p=0.9 means each token has 90% probability”: false; it describes cumulative mass of a retained candidate set.
- “No event in a window means no attack”: false if collection or ingestion failed.
- “An anomaly score is a probability of maliciousness”: false without calibration and a defined target.
- “A random train/test split validates a streaming detector”: often false due to temporal leakage and correlated entities.

## Progressive oral Socratic Q&A (answer after pause)
**Q1. What is a logit?** An unnormalized score; softmax converts a vector of logits to probabilities. **Follow-up:** why subtract the maximum?
**Q2. Why is decoding distinct from training?** Training estimates conditional distributions; decoding chooses a realized sequence under a policy. **Follow-up:** can a policy recover knowledge absent from the model?
**Q3. Compute the greedy first token for [2,1,0].** A. **Follow-up:** does that establish best whole sequence?
**Q4. Explain temperature without metaphors.** Divide logits by T then softmax; lower T concentrates mass. **Follow-up:** what happens as T→∞?
**Q5. What does nucleus sampling retain?** Minimal sorted token prefix whose original cumulative probability reaches p, then renormalizes. **Follow-up:** why must the crossing token remain?
**Q6. Why can beam search disappoint?** Search objective and length bias differ from truthfulness and human utility. **Follow-up:** propose a grounded evaluation.
**Q7. Event time versus ingest time?** Occurrence timestamp versus arrival timestamp. **Follow-up:** how does clock skew affect ordering?
**Q8. Where do 10:04:58 and 10:05:02 fall?** Separate five-minute tumbling windows. **Follow-up:** how does a sliding window differ?
**Q9. What is a watermark?** A processing completeness policy/estimate for event time, not certainty. **Follow-up:** what is the late-event correction strategy?
**Q10. Why is z=4 not P(attack)=0.999?** Standardization is not a calibrated posterior and baseline assumptions may fail. **Follow-up:** what validation data would calibrate it?
**Q11. Why preserve provenance?** To distinguish trustworthy observations, missing sensors, and duplicate or delayed evidence. **Follow-up:** which fields enable reproducible replay?
**Q12. What is the joint research confound?** Upstream missing events and downstream unsupported generation can both create incorrect incident summaries. **Follow-up:** propose a factorial experiment with fixed gold labels.

## Research extensions and reproducibility contract
1. Pre-register whether low-temperature decoding improves evidence citation precision without harming recall; hold model, corpus, prompt and test cases fixed.
2. Estimate a late-arrival survival curve per sensor and test how correction policies affect incident-level recall versus alert stability.
3. Test whether retrieval of time-aligned evidence reduces unsupported claims when data is incomplete; report uncertainty and failure cases.
Record seed, Python version, data hashes, exact windows, model revision, tokenizer, prompt, decoding parameters, raw counts, reviewer rubric and negative results. Do not expose private identities or execute unauthorized attack traffic.

## Learning resources and source map
**Primary references:** [G1] Holtzman et al., *The Curious Case of Neural Text Degeneration* (2020), arXiv:1904.09751, https://arxiv.org/abs/1904.09751 — nucleus sampling and degeneration; [G2] Vijayakumar et al., *Diverse Beam Search* (2018), arXiv:1610.02424, https://arxiv.org/abs/1610.02424 — decoding diversity and beam trade-offs; [G3] Vaswani et al., *Attention Is All You Need* (2017), arXiv:1706.03762, https://arxiv.org/abs/1706.03762 — causal Transformer foundation. [C1] He et al., *Drain: An Online Log Parsing Approach with Fixed Depth Tree* (2017), DOI:10.1109/ICWS.2017.13, https://doi.org/10.1109/ICWS.2017.13 — log templates and online parsing; [C2] Du et al., *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning* (2017), DOI:10.1145/3133956.3134015, https://doi.org/10.1145/3133956.3134015 — sequence-based log anomaly detection; [C3] Apache Flink, *Event Time and Watermarks* documentation, https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/ — stream-time semantics (check documentation version when implementing).

**Claim-to-source map:** autoregressive factorization and causal masking → G3 and derivation G1; sampling/truncation and degeneration → G1 and Lab G; beam alternatives → G2 and greedy counterexample; event-time windows and late data → C3 and Lab C; log parsing → C1; sequence anomalies → C2 and C1 architecture; synthetic numerical claims → worked examples and runnable code here. These citations identify foundational material, not independent validation of today's toy outputs. Links are canonical identifiers known from the papers/docs; live link resolution and any 2026 news claims were not independently verified during preparation. No speculative news is included.

**Tutorial/lecture:** Dive into Deep Learning, https://d2l.ai/ — attention and sequence modeling; Apache Flink event-time docs above for stream processing. **Podcast listening:** *Practical AI* (search its official episode catalog for decoding/LLM evaluation) and *Risky Business* (search its official episode catalog for SOC logging); no specific episode or current claim is asserted without verification. For a commute, explain the two worked examples aloud before listening to external material.

**Paper handling:** see [companion references index](references/index.md). No research PDFs are claimed downloaded; redistribution license was not independently checked. The two syllabus PDFs remain private Gmail attachments, not committed. No email, narration or audio generation is requested or performed.
