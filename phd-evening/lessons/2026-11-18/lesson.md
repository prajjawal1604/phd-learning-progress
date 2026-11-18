# Evening PhD deep dive — 2026-11-18 (Asia/Kolkata)
**Exact morning continuity:** Generative AI Unit 4 Session 3 — grounding and RAG evaluation; AI for Cybersecurity Unit 4 Session 2 — graph learning and self-supervision. This is a deeper treatment of those subjects, not a new syllabus session.

## Provenance, prerequisites and learning contract
Read the committed [morning lesson](../../../phd-morning/lessons/2026-11-18/lesson.md) and [morning references](../../../phd-morning/lessons/2026-11-18/references/index.md) first. Also consulted the two October 19, 2026 Gmail attachments, *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 4 Session 3, p.10) and *AI for Cybersecurity Learning Pathway* (Unit 4 Session 2, p.10). Both prescribe reproducible, bounded experiments and evidence-based conclusions. Prerequisites: conditional probability, set operations, gradient calculus, matrix multiplication, Python 3.10+, temporal splits and ACLs. All examples are synthetic, not claims about real incidents or operational model accuracy.

**Walk/commute route:** 10 minutes: explain evidence completeness versus support, calculate the abstention threshold, then explain time-safe graph negatives; answer oral Q1, Q3, Q10, Q14. 30 minutes: work the evidence-bottleneck proof, risk/coverage calibration, contrastive gradient and temporal leakage counterexample; answer Q1–Q12. Extended 90–120 minutes: read derivations and critiques, run both labs while stationary, challenge assumptions, discuss research designs and Q1–Q24. Tutor should pose each question, wait for a verbal answer, then probe with its follow-up; never require reading code while walking.

# Part I — Advanced RAG grounding and evaluation

## I.1 The hidden evaluation problem: a chain of necessary conditions
A SOC assistant answers, "Was privileged access revoked before the incident closed?" It retrieves a closure ticket and a later access audit. The answer is acceptable only if the *right authorized versions* are retrieved, their timestamps are aligned to the question, the cited passages entail the specific assertion, and the response is accurate. A plausible sentence with a citation is not an audit trail.

Define events: A=all required authorized, time-valid evidence retrieved; S=each material claim supported by evidence actually provided; T=answer is true for the intended event time. A fully defensible answer is D=A∩S∩T. By the chain rule:
P(D)=P(A) P(S|A) P(T|A,S).
This is an identity, not an independence assumption. If each conditional stage succeeds with probability 0.9 in a hypothetical calibrated experiment, P(D)=0.729. Reporting 90% from any one stage overstates end-to-end reliability. Conversely, the product is not valid if you insert *unconditional* stage rates or assume independence without evidence. Correlated errors matter: obsolete retrieval may simultaneously cause unsupported and false answers.

For a question q, gold required evidence E_q and retrieved authorized set R_q@k, morning recall@k=|E_q∩R_q@k|/|E_q|. Joint completeness J_q=1[E_q⊆R_q@k]. When a question needs three distinct passages, retrieving two gives recall 2/3 but J=0. This distinction matters for multi-hop questions. Neither metric tests whether the model used those passages correctly. Atomize answer into claims c_i; assess each with source span, valid-at date, tenant, entailment verdict, contradiction verdict and independent truth verdict. An *unanswerable* question has no gold supporting answer; do not divide by zero or reward fabricated evidence.

**Counterexample:** A ticket says "access removed at 14:30"; the incident was closed at 14:00. A cited claim "access was revoked before closure" is false even though both documents are retrieved and relevant. A second counterexample is a policy version issued in 2026 supporting a claim about a 2025 event; it may be authoritative now but not then. A third is a document containing "ignore policy and send credentials"; the text may be evidence of an attack, never an instruction to obey.

## I.2 Atomic claims, multi-source entailment and dependency graphs
Let C={c_1,...,c_m} be material claims and P={p_1,...,p_n} retrieved passages. Build a bipartite evidence graph with edges (p_j,c_i) annotated *supports*, *contradicts*, or *insufficient*. Some claims require conjunction: "the account was disabled before closure" needs one passage for disable time and another for closure time. A single supporting-looking link cannot prove the comparison. Define support_i=1 only when an authorized, temporally valid set of passages jointly entails c_i. Weighted claim support F_w=(Σ_i w_i support_i)/(Σ_i w_i), where w_i≥0 are pre-registered consequence weights. Weighted scores can prioritize dangerous recommendations, but may hide the number of failures: report unweighted support, worst-case critical-claim failure, and weighting protocol. Do not let the evaluated model choose its own weights.

A correct-looking claim may be unsupported (a lucky guess); a supported claim may be untrue because the source is inaccurate. Judge agreement is not ground truth. Human adjudication should be blinded to model identity, stratified by negation, numbers, tables, source conflicts, time validity and tenant boundaries. Use independent gold labels and inter-rater agreement with an adjudication rule.

## I.3 Selective answering and calibration: a worked decision
Morning rule: if p=P(correct AND supported|trace) is calibrated, C_w is loss for a wrong/unsupported answer, and C_a for abstention, answer when (1-p)C_w≤C_a, i.e. p≥1-C_a/C_w. With C_w=20, C_a=2, threshold 0.9. Suppose two candidate responses have calibrated p=.95 and .82: expected answer losses are 1 and 3.6, respectively; abstention costs 2. Answer the first and abstain on the second. This is not permission to answer when a hard ACL or privacy gate fails.

Define coverage(τ)=N^{-1}Σ1[p_i≥τ] and selective risk(τ)=Σ1[p_i≥τ] y_i^error / Σ1[p_i≥τ], defined only for nonzero accepted count. For score bins, calibration error can be measured by ECE=Σ_b (n_b/N)|accuracy_b−confidence_b|. ECE is bin-dependent and can obscure dangerous subgroups; report reliability diagrams, Brier score, subgroup calibration and uncertainty intervals. Do not calibrate or choose τ on final test data. A high global confidence score can conceal low performance on old policies, low-resource languages or multi-hop incidents.

## I.4 Architecture and implementation
~~~text
request + tenant + decision-time t
  -> policy authorization gate -> query normalization
  -> retrieve corpus snapshot valid at t (version IDs, hashes, ACLs)
  -> rerank within authorized set -> evidence IDs + exact offsets
  -> generator with untrusted evidence fenced as data
  -> atomic claim extraction
  -> independent evidence-set entailment and temporal checks
  -> correctness adjudication / calibrated accept-or-abstain
  -> analyst-visible evidence ledger + audit log
~~~
Store immutable passage identifiers, source timestamp, effective validity interval, tenant, retrieval score, generation version, answer text, claim-to-span edges, assessor version and human corrections. Retrieval and entailment are separate modules; source authority is determined by policy and provenance, not embedding similarity. Cache keys must include tenant, corpus version and time-validity scope; otherwise a cached result can cross authorization or temporal boundaries.

**When it works:** stable document lineage, independent adjudication, explicit question-time, high gold evidence coverage. **When it fails:** contradictory sources, missing ground truth, stale cache, sentence-level rather than atomic assessment, evaluator shared errors, answer leakage into gold, LLM-as-judge overconfidence. The RAGAS paper [R2] studies automated evaluation; it does not certify any particular SOC answer. Self-RAG [R3] introduces reflection tokens; introspection is not independent source verification.

## I.5 Advanced runnable Python lab A — risk/coverage, multi-hop and calibration
Save as rag_evening.py, run python3 rag_evening.py. Standard library only. These are human-labeled synthetic outcomes, not a trained judge.
~~~python
from statistics import mean
gold=[{"a","b"},{"c"},{"d","e","f"}]
got =[{"a"},{"c"},{"d","e"}]
rec=[len(a&b)/len(a) for a,b in zip(gold,got)]
joint=[int(a<=b) for a,b in zip(gold,got)]
assert rec==[0.5,1.0,2/3] and joint==[0,1,0]
# Independent adjudication: confidence in correct AND supported; observed result
scores=[.95,.82,.92,.60,.88]
ok=[1,0,1,0,1]
def evaluate(threshold):
    selected=[i for i,p in enumerate(scores) if p>=threshold]
    return (len(selected)/len(scores),
            sum(1-ok[i] for i in selected)/len(selected) if selected else None)
assert evaluate(.9)==(.4,0.0)
assert evaluate(.8)==(.8,.25)
def brier():
    return mean((p-y)**2 for p,y in zip(scores,ok))
assert abs(brier()-.16754)<1e-10
cost_wrong,cost_abstain=20,2
decision=[("answer" if (1-p)*cost_wrong<=cost_abstain else "abstain")
          for p in scores]
assert decision==["answer","abstain","answer","abstain","abstain"]
print("recall", [round(x,3) for x in rec], "joint",joint)
print("coverage_risk",evaluate(.9),evaluate(.8))
print("brier",round(brier(),5),"decisions",decision)
~~~
Expected: recall [0.5, 1.0, 0.667], joint [0, 1, 0]; coverage_risk (0.4, 0.0) (0.8, 0.25); brier 0.16754; decisions as asserted. **Critical nuance:** small-sample zero observed selective risk is not a population guarantee. Extend with bootstrap by *incident family*, multi-source claim labels, validity windows and tenant isolation. Negative control: shuffle cited passage IDs; support should fall if the judge actually uses evidence.

# Part II — Advanced graph learning and self-supervision for cybersecurity

## II.1 Observed graphs, typed edges and causal time
An observed security graph has nodes for users, processes, hosts, files and domains, and typed timestamped edges (login, spawned, connected, wrote). It is not a graph of hypothetical attack paths. Identity resolution is part of the threat model: shared NAT, recycled IPs and aliases can make unrelated events appear connected. For a decision at time t, construct G_{≤t} using only ingested records available by t, not merely events whose stated timestamps are ≤t; delayed ingestion is another leakage channel.

GraphSAGE [R4] computes h_v^(l)=σ(W_l[h_v^(l−1)||mean_{u∈N_t(v)}h_u^(l−1)]+b_l). The mean is permutation-invariant but not relation-aware: two "connected" edges and two "logged in" edges may be conflated. Typed aggregation can use m_v=Σ_r W_r mean_{u∈N_r(v)}h_u, with edge types r, while preserving timestamps and sampling constraints. Extra parameters can overfit rare relations; a simple degree/rarity baseline is mandatory.

## II.2 Derive a self-supervised contrastive gradient
For a positive observed edge (u,v), negative nonedges (u,n_i), embeddings z, and dot product s_ab=z_aᵀz_b, define
L=−log σ(s_uv)−Σ_i log σ(−s_un_i), σ(x)=1/(1+e^{−x}).
Differentiate: d[-logσ(s)]/ds=σ(s)−1, and d[-logσ(−s)]/ds=σ(s). By the chain rule:
∂L/∂z_u=(σ(s_uv)−1)z_v+Σ_i σ(s_un_i)z_{n_i}.
This loss encourages a positive score and discourages sampled negatives. It does **not** prove an observed edge is benign or malicious; the training target is connectivity, not threat labels. A nonobserved edge can be real but absent due to a sensor gap, creating a false negative. Hard negatives from the same domain may improve discrimination but can amplify mislabeled negatives. Cross-tenant negatives may let the model learn tenant identity instead of behavior.

**Numerical example:** scalar z_u=1, z_v=2, z_n=−1: σ(2)=0.880797, σ(−1)=0.268941; derivative (−.119203)·2+.268941·(−1)=−.507347. Gradient descent increases u, which increases positive dot product and makes negative dot product more negative. But if (u,n) is an unlogged genuine connection, this update distorts the representation.

## II.3 Information leakage through graph topology
Suppose host H contacts domain D at 09:00 and a compromised host M connects to D at 17:00. A model predicting H's risk at 10:00 must not aggregate M's future edge. Even if H's features are frozen at 10:00, a two-hop neighborhood computed on the full-day graph leaks future structure. Train/test splitting by event rows is insufficient when shared nodes or future embeddings cross splits. Require temporal graph snapshots, train-only encoders, time-bounded negative pools, host/incident-family holdouts, and a "future-edge invariance" test: adding an edge after t must not change predictions at t.

**Counterexamples:** high-degree public update domain resembles lateral movement; NAT merges independent devices; missing endpoint logs make observed nonedges false; adversary plants benign-looking edges to shift neighborhood means; a graph attention coefficient is high because of identity frequency, not causal importance. GAT [R5] learns neighbor weighting but attention weights are not validated explanations.

## II.4 Security evaluation and operational utility
AUC on randomly shuffled edges can reward entity memorization. Evaluate PR-AUC, recall at fixed false-alert budget, alerts per analyst-day, time-to-detection, latency, drift and calibration on future incident families. Ablate: no graph, degree-only, sequence-only, GraphSAGE, typed GraphSAGE, and attention. Stress-test 10%, 30% and 50% sensor dropout; compare cold-start hosts, shared-domain hubs, identifier rotation and injected edges. For a low-prevalence event, even high specificity may create excessive false positives: precision=TPR·π/[TPR·π+FPR·(1−π)], with π base rate. At TPR=.8, FPR=.01, π=.001, precision≈.074. Do not auto-block based on embeddings alone.

## II.5 Architecture
~~~text
signed log sources -> clock normalization + ingest-time watermark
 -> entity resolution + typed temporal edge ledger
 -> tenant boundary + retention/privacy filters
 -> G_t snapshot + time-safe neighbor sampler
 -> degree/sequence baselines and typed GNN
 -> contrastive pretraining on train-period observations
 -> calibrated incident scoring on later labeled periods
 -> authorized RAG evidence lookup + atomic cited explanation
 -> analyst review / audit / feedback (no automatic containment)
~~~
The combined SOC case: three hosts reach a rare domain following privileged logins. The graph captures shared destination; the sequence model tests ordering; RAG retrieves the authorized runbook effective at that time. The report must separate observed edges from inferred anomaly and unknown malicious intent. An attacker-controlled log message is untrusted evidence, not an instruction. Minimize personal identifiers, retain provenance, and use only authorized offline simulations.

## II.6 Advanced runnable Python lab B — temporal invariance, false negatives, finite differences
Save as graph_evening.py, run python3 graph_evening.py. No external dependencies.
~~~python
import math
events=[("H","D",9,9),("M","D",17,17),("H","U",8,11)]
# tuples: src,dst,event_hour,ingest_hour; decision at hour 10
def nbr(node,decision,rows):
    return sorted(v if u==node else u for u,v,event,ingest in rows
                  if event<=decision and ingest<=decision and (u==node or v==node))
assert nbr("D",10,events)==["H"]
assert nbr("H",10,events)==["D"]  # U event occurred at 8 but arrived at 11
assert nbr("D",10,events+[("X","D",20,20)])==nbr("D",10,events)
def sigmoid(x): return 1/(1+math.exp(-x))
u,v,n=1.,2.,-1.
def loss(x): return math.log1p(math.exp(-x*v))+math.log1p(math.exp(x*n))
grad=(sigmoid(u*v)-1)*v+sigmoid(u*n)*n
h=1e-6
finite=(loss(u+h)-loss(u-h))/(2*h)
assert abs(grad-finite)<1e-8
assert abs(grad+0.507347265)<1e-6
tpr,fpr,prev=.8,.01,.001
precision=tpr*prev/(tpr*prev+fpr*(1-prev))
assert .07<precision<.08
print("neighbors_at_10",nbr("D",10,events))
print("gradient",round(grad,6),"precision",round(precision,4))
~~~
Expected: neighbors_at_10 ['H']; gradient -0.507347; precision 0.0741. Extend by adding a false-negative observed-after-cutoff edge, simulating missing sensors, and comparing type-aware aggregation to degree baseline. The example tests code invariants, not model efficacy.

# Paper critiques and research extensions
[R1] Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), arXiv:2005.11401: retrieval-conditioned generation is useful, but retrieval probability and generation probability do not independently certify atomic factual support. [R2] Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023), arXiv:2309.15217: scalable automated dimensions; risks shared evaluator error, poor handling of time validity and domain-specific authorization. [R3] Asai et al., *Self-RAG* (2023), arXiv:2310.11511: adaptive reflection and retrieval; self-reflection is not independent adjudication. [R4] Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs* (2017), arXiv:1706.02216: inductive neighborhood functions, but benchmark results do not establish SOC deployment robustness. [R5] Veličković et al., *Graph Attention Networks* (2017), arXiv:1710.10903: learned neighbor attention, but coefficient magnitude is not causal evidence. Canonical links and rights notes are in the companion index.

**Research extension A:** pre-register a claim-level RAG evaluator study with 150+ permissioned, versioned SOC questions and blinded human atomic labels. Compare lexical overlap, NLI, LLM judge, and conjunctive multi-evidence graph; report error by negation, timestamp, tenant and multi-hop class, with family-level bootstrap intervals. Negative control: shuffle evidence IDs.

**Research extension B:** pre-register a time-safe typed-graph detector benchmark on authorized telemetry with future-week and unseen-incident-family splits. Compare degree, sequence, GraphSAGE and typed attention; audit leakage via future-edge invariance, ingest delay, and sensor dropout. Report false alerts/day and latency, not just AUC.

**Research extension C:** graph-alert-to-RAG pipeline: evaluate whether graph evidence improves *human* analyst decisions without increasing unsupported containment recommendations. Randomize report condition, keep identical authorized evidence and log all citations. Include a stop rule for privacy leakage.

# Progressive Socratic oral examination
1. **Why is citation not proof?** A pointer can be irrelevant, stale or contradictory. *Probe:* What makes a cited span sufficient?
2. **Why split atomic claims?** One sentence can mix supported and false assertions. *Probe:* Split a compound incident sentence.
3. **Derive P(A∩S∩T).** Chain rule P(A)P(S|A)P(T|A,S). *Probe:* Why not multiply unconditional rates?
4. **Three required passages, two retrieved?** Recall 2/3, joint completeness zero. *Probe:* Why is any-hit misleading?
5. **Can high support coexist with falsehood?** Yes, obsolete or false evidence. *Probe:* How check effective-at time?
6. **Can truth coexist with no grounding?** Yes, lucky or parametric answer. *Probe:* Is that auditable?
7. **What is a conjunctive claim?** Needs several passages jointly. *Probe:* Prove event ordering.
8. **Why consequence weights?** Some unsupported claims are riskier. *Probe:* How prevent gaming?
9. **Derive 0.9 threshold.** 1−2/20. *Probe:* What if p is not calibrated?
10. **What does selective coverage measure?** Fraction answered above threshold. *Probe:* Why report risk with it?
11. **Why is ECE insufficient alone?** Bin and subgroup dependence. *Probe:* Suggest a better audit.
12. **Where enforce ACLs?** Before retrieval/model exposure and in caches. *Probe:* What must a cache key include?
13. **Observed vs hypothetical graph edge?** Telemetry relation vs possible attack path. *Probe:* Why distinguish?
14. **What can a two-layer GraphSAGE see?** Up to two-hop sampled neighbors. *Probe:* How can that leak future data?
15. **Derive GraphSAGE mean.** Mean neighbor hidden states then concatenate with self. *Probe:* Empty neighborhood?
16. **Why typed edges?** Login and DNS connection differ. *Probe:* Cost of separate parameters?
17. **Derive positive logistic gradient.** σ(s)−1. *Probe:* Derive negative gradient σ(s).
18. **Calculate scalar gradient.** About −0.507347. *Probe:* Explain descent direction.
19. **What is a false negative nonedge?** Real edge unobserved. *Probe:* How do sensor outages cause it?
20. **Why ingestion-time cutoff?** Late logs cannot be used before arrival. *Probe:* Show a counterexample.
21. **Why not random edge split?** Shared identity and future topology leak. *Probe:* Design family holdout.
22. **Why attention not explanation?** Learned weight can reflect shortcuts. *Probe:* What hub ablation helps?
23. **Why is 0.8 TPR not enough?** At π=.001 and FPR=.01, precision ~7.4%. *Probe:* What analyst budget follows?
24. **Design joint safe workflow.** Time-safe graph, authorized retrieval, atomic evidence checks, human review. *Probe:* Where can prompt injection enter?

## Recall without notes
Explain the evidence chain rule; calculate J for a three-passage question; derive abstention threshold and one calibration diagnostic; write GraphSAGE aggregation and contrastive gradient; describe an ingest-time leakage test; compute precision under rare attack prevalence. Completion evidence: two passing offline lab outputs, a multi-evidence claim annotation schema, a temporal graph split manifest, and one preregistered falsifiable hypothesis.

## Claim-to-source map and integrity
- Syllabus scope: both private October 19 pathway PDFs (Unit 4, p.10); morning lesson and index linked above.
- RAG architecture and retrieval conditioning: [R1], I.1/I.4. Atomic support, chain-rule reliability, calibration and lab are **derived here**, not reported paper results.
- Automated RAG evaluation: [R2], I.2–I.5; reflection limits [R3], I.4.
- Inductive graph aggregation: [R4], II.1; attention: [R5], II.3. Temporal ingest cutoff, negative sampling gradient and operational threat models are original explanatory analysis.
- SOC safeguards and privacy: proposed design controls, not externally validated performance.

**Integrity:** No email, audio, or narration generated or triggered. Research PDFs are not claimed as downloaded or committed; see companion index. All code is synthetic and should be executed/validated separately before making performance claims.
