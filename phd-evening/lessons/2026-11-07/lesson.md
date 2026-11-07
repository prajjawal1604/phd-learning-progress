# Evening PhD deep dive — 7 November 2026 (Asia/Kolkata)
**Continuity verified:** read the committed [morning lesson](../../../phd-morning/lessons/2026-11-07/lesson.md) and its [reference index](../../../phd-morning/lessons/2026-11-07/references/index.md) before writing. Both October 19 PhD pathway PDFs were read. This evening deepens **Generative AI Unit 2 Session 4 (autoregressive decoding, sampling, prompting)** and **AI for Cybersecurity Unit 2 Session 2 (event-time log sequences, watermarks and anomaly detection)**; no new syllabus unit. Morning lab's top-k/top-p *intersection* is a pedagogical convention, not the only library implementation.

## Spoken routes and prerequisite check
**10-minute walk:** Explain why greedy search is not sequence-optimal; derive the odds effect of temperature; distinguish watermark from guaranteed completeness; solve examples G1 and C1; answer oral Q1–Q4. **30-minute walk:** Add beam length bias, nucleus order, late-event retractions, precision at alert budget, and Q1–Q10. **Extended 90–150 minutes:** execute both labs, inspect counterexamples, critique papers, answer all 16 questions and preregister a factorial experiment. Pause after each oral question before reading its answer.
Prerequisites: conditional probability, log probabilities, softmax, entropy, causally masked attention, token-level KV cache; UTC/event timestamps, sliding/tumbling windows, delayed ingestion, precision/recall and provenance. A *logit* is an unnormalized score. A *decoder* selects tokens from model conditionals; it does not retrain model weights. A *watermark* is a heuristic event-time progress boundary; a *retraction* corrects an earlier aggregate or alert.

# G — Deep mathematics of decoding, search and grounded prompting
## G1. Temperature as a change in odds
For logits z_i and T>0, p_T(i)=exp(z_i/T)/Σ_j exp(z_j/T). For two tokens i,j, **log[p_T(i)/p_T(j)]=(z_i-z_j)/T**. Thus halving T doubles log-odds, but does not add evidence. If z_A=2,z_B=1, at T=1 odds A:B=e≈2.718; at T=.5 odds=e²≈7.389. For [2,1,0], T=1 probabilities [0.66524,0.24473,0.09003]. Entropy H(T)=-Σp_i ln p_i has derivative **dH/dT=Var_{p_T}(z)/T³ ≥0** (finite logits, natural logs): proof: H=ln Z - E[z]/T; dE[z]/dT=-Var(z)/T²; substitute. This monotonicity holds *before truncation*; after changing a top-p support, entropy can jump discontinuously. Temperature is not epistemic confidence or a factuality guarantee.

**Worked G1:** For logits [2,1,0], top-p=.8 retains A+B at T=1 (mass .90997). At T=.5 A alone has mass ≈.86681 and nucleus becomes {A}. Thus a small T change can change not only probabilities but *which words are legal* under top-p. At p=1 all tokens remain; at p=.8 candidate count is dynamic.

## G2. Search objective versus response quality
Autoregressive joint likelihood P(y_1:L|x)=Π_t P(y_t|x,y_<t); maximize sum of log probabilities, not independent token scores. Greedy selects local argmax. Counterexample: first A=.6, B=.4; best second-token conditional after A=.2, after B=.9: greedy path .12 versus B path .36. Beam search approximates the global search but is not exact for finite beam and can discard prefixes that later win. Raw summed log probabilities systematically reward shorter sequences when comparing completed candidates; common length normalization uses score=Σ log p_t/L^α, α≥0, but changes the objective and may reward verbose outputs. End-of-sequence handling matters.

**Worked G2:** Sequence X has two tokens each probability .8: log score 2ln(.8)≈-.4463. Sequence Y has three tokens each probability .9: 3ln(.9)≈-.3161, so Y wins raw likelihood despite being longer because its token probabilities are higher. Now compare X=.8²=.64 with Y=.8³=.512: X wins. For length-normalized average log-likelihood both tie at ln(.8); this demonstrates why blanket claims about length bias require specified distributions and scoring. A high-likelihood summary can still be wrong about an incident.

## G3. Truncation semantics and reproducibility
Top-k retains k most probable tokens. Nucleus top-p retains minimal descending prefix whose **original** probability sum crosses p; the crossing token stays. If both are configured, implementations may apply top-k first then top-p on renormalized mass, or intersect sets computed on the original distribution. They are **not equivalent**. With base p=[.6,.25,.15], k=2, nucleus p=.7: original nucleus {A,B}, intersection {A,B}; after top-k renormalization [.70588,.29412], nucleus .7 keeps {A} alone. Document processor order, tie-breaking, EOS, seed, tokenizer revision and stopping criteria. A prompt specifying "cite only evidence IDs" must be backed by an external validator; malicious instructions inside a log line are untrusted data.

### Architecture: bounded-evidence incident answer
```text
trusted policy + question
          |                   untrusted log payloads
          +---> evidence retrieval <--- event-time corrected store
                        |
              evidence IDs + timestamps
                        |
               model prefill + KV cache
                        |
              logits -> T -> k/p -> token
                        |             |
                        +-- append <---+
                        |
             schema + evidence-ID validator
                        |
               human-reviewed summary
```
At every stage record evidence version; if a late event changes the evidence snapshot, mark the prior narrative stale rather than silently rewriting history.

### Advanced Python Lab G (Python 3.10+, stdlib)
```python
import math
def softmax(z,T=1.):
    assert T>0 and len(z)>0
    a=[v/T for v in z]; m=max(a)
    w=[math.exp(v-m) for v in a]; return [v/sum(w) for v in w]
def nucleus(p,threshold):
    assert 0<threshold<=1
    order=sorted(range(len(p)),key=lambda i:(-p[i],i))
    keep=[]; s=0.
    for i in order:
        keep.append(i); s+=p[i]
        if s>=threshold-1e-12: break
    return set(keep)
def trunc(z,k,p,order="original"):
    q=softmax(z); ids=sorted(range(len(q)),key=lambda i:(-q[i],i))
    top=set(ids[:k])
    if order=="original": keep=top & nucleus(q,p)
    else:
        r=[q[i]/sum(q[j] for j in top) if i in top else 0 for i in range(len(q))]
        keep=top & nucleus(r,p)
    return keep
assert trunc([math.log(x) for x in [.6,.25,.15]],2,.7,"original")=={0,1}
assert trunc([math.log(x) for x in [.6,.25,.15]],2,.7,"sequential")=={0}
z=[2.,1.,0.]
for T in [.5,1.,2.]:
    q=softmax(z,T); H=-sum(v*math.log(v) for v in q)
    print("T",T,"entropy",round(H,4),"nucleus",sorted(nucleus(q,.8)))
assert nucleus(softmax(z,.5),.8)=={0}
assert nucleus(softmax(z,1),.8)=={0,1}
print("PASS")
```
Expected: increasing entropies (approximately .4411, .8324, 1.0202), nucleus [0] at .5 and [0,1] at 1; PASS. The output at T=2 can include more tokens. **Ablation:** compare k=2,p=.7 with the two processor orders and log the *effective* retained mass. **Failure tests:** equal logits, extreme logits 1e6, p near 0 or 1, EOS-only support, deterministic tie-breaking. This is a toy probability lab, not an LLM quality benchmark.

## G4. Critical reading
Holtzman et al. (2020, arXiv:1904.09751) motivate nucleus sampling by analyzing degeneration of high-probability decoding; their experiments do not establish that any top-p value is universally optimal for evidence-grounded security summaries. Vijayakumar et al. (2018, arXiv:1610.02424) introduce diversity-aware beam search; diversity is not factual correctness. Vaswani et al. (2017, arXiv:1706.03762) give the Transformer attention architecture; a causal model's logits still require a separate inference-time decoding policy. **Research question:** At fixed model, prompt and evidence, does nucleus sampling improve coverage while worsening unsupported claims relative to greedy? Measure paired claim-level precision and omission rate with blinded adjudication, not subjective fluency alone.

# C — Deep event-time reasoning, corrections and sequence anomaly detection
## C1. Watermarks are probabilistic engineering policies
Let event timestamp e, arrival timestamp a, lateness D=a-e≥0 (ignoring clock errors). A tumbling window W=[s,s+w) closes for *on-time* updates when watermark ≥s+w. A bounded-out-of-orderness policy may use W_m(t)=max_{arrived by t}(event_time)-δ, δ a tolerated lag; this can stall when no events arrive and can be misleading across skewed partitions. **Key counterexample:** a watermark of 10:06 says "we choose to finalize up to 10:06", not "no older event will arrive." Event 10:04:59 arriving 10:07:30 may require a late-data side output, correction or explicit loss accounting.

**Worked C1:** Morning window [10:00,10:05) contains 10:04:58 and late 10:04:59; [10:05,10:10) contains 10:05:02. A policy closing first window at 10:06 counts 1 instead of 2 until correction. Suppose alert threshold is ≥2 failures: provisional state has no alert; corrected state has an alert **90 seconds after the close** (10:07:30). Detection delay and eventual recall are distinct. If an alert already fired on a provisional count, a correction may instead require retracting a false alert. Distinguish *arrival-time deadline*, *watermark advancement* and *allowed lateness*: they are separate knobs in real stream processors.

## C2. Sequence evidence and its failure modes
For event alphabet {login_fail,login_ok,process_start,net_connect}, a simple Markov baseline estimates P(E_t|E_{t-1}) with Laplace smoothing: (count(pair)+α)/(count(previous)+α|V|). A sequence surprise score S=-Σ_t ln P(E_t|E_{t-1}) is high for rare transitions, but a rare transition may reflect benign software updates or missing sensor events. If a collector drops login_ok, the apparent sequence login_fail→process_start can be spuriously anomalous. Fit transitions only on training-period records; test chronologically and stratify by device and sensor health.

**Worked C2:** With 2 event types and α=1, counts after A: A→A=8, A→B=0. Smoothed P(B|A)=(0+1)/(8+0+2)=.1, surprise=-ln(.1)=2.3026 nats. A rare transition does not imply P(attack)=.9; it is only a likelihood under a baseline.

## C3. Decision economics
Let q=P(incident|evidence) be a *calibrated* posterior, C_FP cost of investigating a benign alert, C_FN cost of missing an incident. Alert if (1-q)C_FP < q C_FN, equivalently **q>C_FP/(C_FP+C_FN)** under the simplified two-action model. With C_FP=1 and C_FN=19, threshold=.05. But raw anomaly surprise is not q; calibrating against representative incident labels and prevalence is necessary. Under a fixed 30-alert/day analyst capacity, choose operating threshold on validation data, report recall and delays on held-out time periods. Include uncertainty clustered by incident and avoid treating correlated events as independent.

### Architecture: versioned event-time detection
```text
endpoint + identity + network sensors
       |        |          |
 schema, clock-skew checks, deduplication, provenance
                   |
       partitioned event-time buffers
                   |
       watermark + late side output
                   |
   window counts + ordered sequence features
                   |
   versioned aggregate / correction ledger
                   |
    score + calibrated decision threshold
                   |
     analyst queue / alert update or retract
                   |
      adjudication and chronological evaluation
```
Missing telemetry must be marked **unknown**, never zero risk. Keep a durable raw log for lawful replay, use retention limits and access controls, and avoid uploading employee activity to an unapproved LLM.

### Advanced Python Lab C (Python 3.10+, stdlib)
```python
from collections import defaultdict
def seconds(s):
    h,m,x=map(int,s.split(":")); return 3600*h+60*m+x
rows=[("10:04:58","10:05:01","fail"),
      ("10:05:02","10:05:03","fail"),
      ("10:04:59","10:07:30","fail")]
counts=defaultdict(int); updates=[]; seen=set()
for event,arrival,kind in sorted(rows,key=lambda r:seconds(r[1])):
    key=(event,arrival,kind)
    if key in seen: continue
    seen.add(key)
    if kind!="fail": continue
    start=seconds(event)//300*300
    old=counts[start]; counts[start]+=1
    close=start+300+60
    updates.append((start,old,counts[start],seconds(arrival)>close))
first=seconds("10:04:58")//300*300
assert counts[first]==2
assert len([u for u in updates if u[3]])==1
assert any(u[0]==first and u[1]==1 and u[2]==2 and u[3] for u in updates)
print("first-window final",counts[first],"late corrections",sum(u[3] for u in updates))
print("PASS")
```
Expected: `first-window final 2 late corrections 1`, `PASS`. This shows **upsert/correction accounting**, not a production Flink watermark implementation. **Extend:** add a duplicate record (must not increase counts), clock skew of +90 seconds, two independent partitions, and alert state changes at count threshold 2. Preserve both provisional and final metrics. **Counterexample:** if a host's sensor goes offline during a burst, the observed count may fall while attack risk rises.

## C4. Critical reading and reproducibility
He et al., *Drain* (2017, DOI:10.1109/ICWS.2017.13) describe online log template parsing, not proof that parsed templates are semantically sufficient for detecting attacks. Du et al., *DeepLog* (2017, DOI:10.1145/3133956.3134015) model log sequences; benchmark performance is sensitive to parsing, labels and environment shift. Apache Flink's event-time/watermark documentation defines processing semantics, but a toy deadline in the morning lab is **not** equivalent to its runtime semantics. Research extension: compare (a) drop late data, (b) correct aggregates, (c) hold windows longer; pre-register incident recall, alert correction frequency, p95 detection delay, CPU/state overhead and analyst burden across realistic delay distributions.

# Joint doctoral experiment — uncertainty compounds across stages
A 2×2×2 factorial study: lateness policy {drop,correct} × decoding {greedy,nucleus} × evidence validator {off,on}. Hold source events, gold incident labels, model revision, prompt, and evidence IDs fixed. Primary outcomes: incident evidence recall, unsupported-claim rate, detection latency and human review minutes. Randomize narrative evaluation order and blind reviewers to conditions. Interaction hypothesis: correction reduces missing evidence; validator reduces unsupported claims; sampling may affect both coverage and unsupported content. Do not infer causation from a single synthetic trace. Attack surface includes forged timestamps, prompt injection embedded in log messages, poisoned templates and leakage of private identifiers; enforce tool authorization outside the model.

# Progressive oral Socratic questions (pause, then answer and follow-up)
1. **What does T change?** Relative logit odds, not model knowledge. *Follow-up:* compute e^(1/.5).
2. **Derive the two-token log odds.** Subtract log-softmax expressions; partition functions cancel. *Follow-up:* what if logits tie?
3. **Why can entropy rise with T?** dH/dT=Var(z)/T³≥0 before truncation. *Follow-up:* why may top-p break smoothness?
4. **What happens to top-p support at T=.5 in G1?** A alone. *Follow-up:* why not at T=1?
5. **Why can greedy lose?** It optimizes local, not complete-path probability. *Follow-up:* state the .12/.36 example.
6. **Does beam search prove factuality?** No; it searches model likelihood. *Follow-up:* what external metric tests evidence support?
7. **Why record processor order?** Sequential top-k then top-p differs from original-mass intersection. *Follow-up:* demonstrate with [.6,.25,.15].
8. **What must an incident prompt never trust?** Instructions inside log payloads. *Follow-up:* where enforce authorization?
9. **What is a watermark?** Estimated event-time progress, not completeness guarantee. *Follow-up:* what if one partition stalls?
10. **Why can a late event change alert state?** Corrected window count crosses threshold. *Follow-up:* should prior output be versioned?
11. **What is a retraction?** A correction invalidating earlier output. *Follow-up:* how notify an analyst?
12. **Compute surprise when P(B|A)=.1.** -ln(.1)=2.3026 nats. *Follow-up:* why not a probability of attack?
13. **Why can missing telemetry inflate surprise?** Dropped intermediate events create artificial transitions. *Follow-up:* what sensor-health feature helps?
14. **Derive Bayes decision threshold.** Compare (1-q)C_FP and q C_FN; q>C_FP/(C_FP+C_FN). *Follow-up:* what if analyst capacity binds?
15. **Why temporal splits?** Avoid future-to-past leakage and test realistic deployment drift. *Follow-up:* how group repeated incidents?
16. **Design a falsifiable joint experiment.** Vary lateness, decoding and validator while fixing gold evidence; measure claim support and latency. *Follow-up:* identify one interaction and one confound.

## Source map, reading and listening
**Local morning prerequisites:** morning [Track G](../../../phd-morning/lessons/2026-11-07/lesson.md) for basic decoding and Lab G; morning [Track C](../../../phd-morning/lessons/2026-11-07/lesson.md) for event-time baseline and Lab C. **Syllabus:** the two October 19 private pathway PDFs, Generative AI Unit 2 and Cybersecurity Unit 2. **G1–G3:** mathematical derivations and Lab G above; foundational Transformer [R3], decoding critique [R1], beam diversity [R2]. **C1–C3:** derivations and Lab C above, official stream-time docs [R6], log parsing [R4], log sequence anomaly [R5]. **Joint experiment:** original proposed protocol, not a published result. See [companion index](references/index.md) for citations, links, summaries and rights status.
**Tutorials:** Dive into Deep Learning sequence modeling (https://d2l.ai/); Apache Flink event-time documentation (https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/time/). **Listening:** Practical AI (https://changelog.com/practicalai) for model evaluation themes and Risky Business (https://risky.biz/) for operational security discussions; no specific episode was independently reviewed. Papers are cited by stable identifiers; this lesson does not assert new November 2026 news. 
**Validation and scope:** Labs include deterministic assertions and expected results; code execution should be separately checked before treating results as reproduced. No research PDFs are represented as downloaded. No email, audio generation or narration trigger is requested.
