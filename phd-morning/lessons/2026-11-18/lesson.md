# Morning PhD lesson — 18 November 2026 (Asia/Kolkata)
**Generative AI Unit 4 Session 3: Grounding and RAG evaluation. Cybersecurity Unit 4 Session 2: Graph learning and self-supervision.**

**Syllabus provenance:** Original *Advanced Generative AI and Foundation Models Learning Pathway*, p.10 (Unit 4 Session 3), and *AI for Cybersecurity Learning Pathway*, p.10 (Unit 4 Session 2), attached to the 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*. Both read this run. **Continuity:** November 17 morning/evening covered query processing, chunking, reranking and Transformers for security-event sequences. Today we advance to assessing grounded claims and representing relationships among security entities, without conflating evidence quality with model confidence. All numbers in labs are synthetic, not operational performance.

## Spoken interactive routes
**10 min:** State the difference among evidence recall, claim support, and correctness; calculate the two-evidence example; explain why host-domain edges matter and why two-hop aggregation can leak future evidence; ask Q1–Q4 and Q13–Q15.
**30 min:** Derive the claim-support metrics, selective answering and Bayesian decision threshold; build the graph and derive message passing/contrastive loss; ask Q1–Q10 and Q13–Q20.
**Extended 90–120 min:** Work all derivations and counterexamples, execute labs when stationary, critique papers, formulate research design, answer Q1–Q24. Spoken tutor: ask, wait, invite a counterexample, correct, and deepen. Do not read Python aloud during the walk.

## Prerequisites and vocabulary
Prerequisites: sets and conditional probabilities; precision/recall; binary classification; dot products and normalized vectors; gradient of log-sigmoid; event timestamps; access-control lists (ACLs); temporal train/test splitting. **Evidence** is a retrievable, versioned source passage. **Claim** is an independently checkable proposition in an answer. **Entailment** means evidence logically supports a claim (not merely shares words). **Groundedness/faithfulness** measures evidence support; **correctness** concerns external truth or a gold answer; **relevance** concerns whether the question was addressed. **Abstention** means declining an unsupported answer. **Graph** G=(V,E) has entities V and observed relationships E. **Message passing** aggregates neighboring node features. **Self-supervision** constructs a training target from the data without manual incident labels. **Contrastive learning** brings selected positive representations closer and negatives farther apart. **Leakage** means training or inference uses information unavailable at decision time.

# Part A — Grounding and evaluation of RAG

## A1. Mental model and mechanism
Imagine a SOC assistant asked: “Which approved runbook applied when account X was locked, and was escalation required?” A retrieval system returns the lockout procedure and an obsolete draft. The answer confidently asserts that no escalation was required and cites the draft. **A citation is a pointer, not proof.** We must separately test: (1) did retrieval include both required current passages? (2) did generation use them correctly? (3) is each material claim entailed by the cited span? (4) did the answer resolve the user's question? (5) should the system have abstained?

A trustworthy evaluation pipeline:
~~~text
signed/versioned corpus + ACL + validity dates
    -> authorized retrieval (BM25/dense/hybrid)
    -> evidence IDs and immutable source offsets
    -> generation with citation IDs
    -> sentence/atomic-claim segmentation
    -> claim-to-evidence alignment + entailment review
    -> independent gold-answer correctness + abstention audit
    -> per-query trace, stratified metrics, human adjudication
~~~
Enforce tenant permissions in code *before* model exposure; do not allow a language model to override ACLs. Retrieved instructions are untrusted text. Record query, corpus hash, time, policy decision, model/retriever versions and passage offsets. A supported answer can still be wrong if the source is obsolete, inaccurate or out of scope; a correct answer may be unsupported by the supplied citations.

## A2. Formal metrics, denominators and a worked example
For query q, let E_q be the set of gold required evidence IDs, R_q@k the top-k retrieved authorized IDs, and C_q the set of material atomic answer claims. Define evidence recall@k = |E_q ∩ R_q@k|/|E_q|, with denominator |E_q|>0. Define strict joint evidence completeness J_q = 1[E_q ⊆ R_q@k]. Define claim support precision F_q = (# claims entailed by cited authorized valid evidence)/|C_q| for |C_q|>0. For abstentions, evaluate separately; do not silently score empty claims as perfectly faithful. **Citation precision** = supported citation links / assessed citation links; **citation coverage** = claims with at least one sufficient citation / all material claims. One claim may require two sources; sentence-level scoring may conceal a mixed supported/unsupported sentence.

Worked example: E={policy-v3§4, incident-7§2}, retrieved R={policy-v3§4, obsolete-policy§4}; evidence recall=1/2 and J=0. Answer has three atomic claims: “lockout occurred” (supported), “escalation not needed” (contradicted), “policy v3 applies” (supported). Claim support precision=2/3, **but the answer fails the multi-evidence task**. A query-level “any hit” metric would be 1, misleadingly. If the model abstains instead, it may be safer even though it gives no full answer.

Across queries, macro recall = (1/N) Σ_q recall_q. Micro recall = Σ_q |E_q∩R_q| / Σ_q |E_q|. Macro weights each question equally; micro weights evidence-heavy questions more. Always stratify answerable/unanswerable, tenant, source freshness, table reconstruction, negation, and multi-hop questions.

## A3. Why abstention can be rational: selective risk
Let p be the calibrated probability that an answer is correct **and adequately supported** conditional on available evidence. Let cost of wrong/unsupported answer C_w, cost of abstention C_a, and cost of correct answer zero. Expected answer loss is (1-p)C_w; abstention loss is C_a. Answer only if (1-p)C_w ≤ C_a, equivalently p ≥ 1-C_a/C_w. For C_w=20 and C_a=2, threshold p≥0.9. This is a decision-theoretic illustration, not a calibrated real-world rule. A model's verbal “90% confidence” is not a calibrated p. Evaluate risk versus coverage using held-out questions and human evidence labels; set thresholds on development data, not test. In regulated response contexts, some claims require hard policy gates regardless of p.

## A4. Why and when automated judges fail
RAGAS (Es et al., 2023, arXiv:2309.15217) proposes reference-free evaluation dimensions; it is useful for fast development but LLM-based judgments can share errors with the generator. Self-RAG (Asai et al., 2023, arXiv:2310.11511) trains reflection-token behavior to condition retrieval and critique; self-critique is not an independent verifier. Gold labels and human review remain necessary for consequential claims. A lexical “entailment” heuristic may accept “must escalate” as support for “must NOT escalate” because it ignores negation. An NLI model may mishandle dates, quantities, tables, or source authority.

**Counterexamples:** (a) cited span states “approval pending”; answer says “approved” — overlapping words, opposite status. (b) a 2024 policy supports a claim historically but not at a 2026 event date. (c) answer cites the correct procedure but attributes it to another tenant. (d) the right citation accompanies a wrong arithmetic calculation. (e) an answer contains two claims in one sentence; only one is supported. (f) the evidence is adversarially injected with “ignore prior instructions”; quoting it does not authorize obeying it.

**Operational trade-offs:** human review is costly but resolves difficult ambiguity; automated evaluators scale but require calibration and audit; longer context may include more evidence yet increase distraction, cost and stale-version risk; strict abstention reduces unsupported answers but may deny useful service. Report answer accuracy, claim support, abstention precision/recall, retrieval completeness, p95 latency, cost and unauthorized exposure separately.

## A5. Python lab: deterministic evidence audit (Python 3.10+, stdlib)
This lab deliberately uses **human-annotated support labels**, not an invented entailment model. Run: python3 rag_audit.py.
~~~python
from statistics import mean
queries = [
 {"id":"q1","gold":{"p3","i7"},"got":{"p3","old"},
  "claims":[("lockout occurred",True),("no escalation",False),("policy v3 applies",True)]},
 {"id":"q2","gold":{"r2"},"got":{"r2"},
  "claims":[("rotate key",True)]},
]
recalls=[]; complete=[]; faith=[]
for q in queries:
    recall=len(q["gold"] & q["got"])/len(q["gold"])
    joint=int(q["gold"] <= q["got"])
    support=sum(int(ok) for _,ok in q["claims"])/len(q["claims"])
    recalls.append(recall); complete.append(joint); faith.append(support)
    print(q["id"],"recall",round(recall,3),"joint",joint,"support",round(support,3))
assert recalls==[0.5,1.0]
assert complete==[0,1]
assert abs(mean(faith)-5/6)<1e-12
wrong_cost, abstain_cost=20,2
threshold=1-abstain_cost/wrong_cost
assert abs(threshold-0.9)<1e-12
print("macro_recall",round(mean(recalls),3),"macro_support",round(mean(faith),3),
      "answer_threshold",round(threshold,3))
~~~
Expected:
~~~text
q1 recall 0.5 joint 0 support 0.667
q2 recall 1.0 joint 1 support 1.0
macro_recall 0.75 macro_support 0.833 answer_threshold 0.9
~~~
Extension: add unanswerable queries, split claims atomically, check authorized source version, calculate risk-coverage curves, bootstrap confidence intervals by question family. **Do not infer empirical system quality from this tiny toy.**

# Part B — Graph learning and self-supervision for cybersecurity

## B1. Why relationships add information
A security sequence model sees one host's ordered events. A graph connects host H, user U, process P, domain D, and file F through typed, timestamped observations: U logged into H; P ran on H; P connected to D; P wrote F. The same suspicious domain contacted by many unrelated hosts may be more important than a single event; conversely a popular update domain may create benign high-degree structure. Graph evidence is relational, not automatic proof of compromise.

Distinguish **observed telemetry graph** (edges that actually occurred) from an **attack graph** (hypothesized possible exploit paths). Do not convert a possible path into a claimed observed incident. Identity resolution, clock skew, missing telemetry and NAT/shared IPs can create spurious edges.

~~~text
signed event sources -> normalize entity IDs + timestamps + edge types
   -> enforce tenant/time visibility -> construct time-windowed graph
   -> feature baselines: degree, distinct domains, rare edges
   -> GraphSAGE mean aggregator / optional typed message passing
   -> self-supervised edge or masked-event task
   -> labeled downstream alert evaluation + analyst evidence
   -> bounded human-reviewed investigation (not auto-block)
~~~

## B2. GraphSAGE message passing derived
Let x_v∈R^d be node v's features. Initialize h_v^(0)=x_v. For layer l, aggregate m_v^(l)=mean_{u∈N(v)} h_u^(l-1), then h_v^(l)=σ(W_l [h_v^(l-1) || m_v^(l)] + b_l), where N(v) are observed neighbors available at prediction time, || concatenates vectors, W_l is trainable, b_l bias, σ activation. Two layers can incorporate two-hop context, but high-degree hubs amplify sampling cost and oversmoothing may make distinct nodes similar. If a node has no neighbors, define a zero-vector aggregator or learned missing-neighbor representation explicitly. **Inductive** means learned functions can process unseen nodes using their features and neighborhoods; it does not mean guaranteed transfer to new attack families.

Worked scalar example (without learned W or σ): host A feature 2, neighbors domain D feature 4 and user U feature 0. Neighbor mean=(4+0)/2=2. Concatenated representation [2,2]. A second host B feature 2 with only D neighbor has [2,4]; equal own features but different context. This is *representation*, not a calibrated threat score.

Graph attention alternative: α_vu = softmax over u∈N(v) of e_vu, where e_vu=aᵀ[Wh_v||Wh_u] passed through a nonlinearity. Attention weights can prioritize neighbors but may latch onto identity shortcuts; weights are not causal explanations. Typed edges (login vs connection) and timestamps should be retained, rather than mixing all neighbors indiscriminately.

## B3. Self-supervised objectives and false negatives
For edge-prediction embeddings z_u,z_v, positive pair (u,v) and sampled negatives n_i, one logistic loss is L=-log σ(z_uᵀz_v)-Σ_i log σ(-z_uᵀz_{n_i}). The first term rewards observed edges; the second discourages sampled nonedges. Derive: d[-log σ(s)]/ds=σ(s)-1, and d[-log σ(-s)]/ds=σ(s). Therefore ∂L/∂z_u=(σ(z_uᵀz_v)-1)z_v + Σ_i σ(z_uᵀz_{n_i})z_{n_i}. **Critical assumption:** “not observed” is not equivalent to “truly absent.” Sensor outages, delayed logs, or sparse sampling create false negatives. A negative drawn from another tenant may also create a shortcut unrelated to maliciousness.

Masked-event prediction hides a type and predicts it from allowed context; ensure the mask removes the label from *all* features, and never use future neighbors for online detection. Contrastive positives may be two augmented views of the same authorized host-time window; augmentations must preserve security semantics (dropping the only privilege escalation edge is not innocuous). Time-respecting negative sampling and incident-family holdouts are essential.

## B4. Temporal leakage and threat models
If we score an edge at 09:00 using a graph built through 18:00, the model may “predict” with future evidence. Build G_{≤t} for each decision t and restrict all feature aggregations, embeddings and negative pools to that cutoff. Split by time and incident/host family; train-only fitting for entity encoders. A graph learner can also be poisoned through attacker-controlled edges, aliases or compromised telemetry; provenance checks, rate limits, robust aggregation and analyst review reduce but do not eliminate risk. Do not test on live third-party hosts.

**When it works:** stable entity identifiers, adequate relational coverage, multi-hop behavior that adds signal beyond counts, bounded latency and validated time splits. **When it fails:** sparse sensors, rapidly rotating identifiers, giant shared infrastructure hubs, high class imbalance, privacy restrictions, identity leakage, graph drift, and benign fan-out that resembles lateral movement. Benchmark degree/count features and the prior sequence baseline before adding a GNN.

## B5. Python lab: time-safe graph and contrastive gradient (stdlib)
Run: python3 graph_lab.py. This demonstrates aggregation and time cutoff; **not** a trained GNN.
~~~python
import math
features={"hostA":2.0,"hostB":2.0,"domain":4.0,"user":0.0}
edges=[("hostA","domain",1),("hostA","user",2),("hostB","domain",1),
       ("hostA","hostB",9)]
def neighbors(node,cutoff):
    return [v if u==node else u for u,v,t in edges
            if t<=cutoff and (u==node or v==node)]
def representation(node,cutoff):
    ns=neighbors(node,cutoff)
    return (features[node],sum(features[n] for n in ns)/len(ns) if ns else 0.0)
assert representation("hostA",2)==(2.0,2.0)
assert representation("hostB",2)==(2.0,4.0)
assert "hostB" not in neighbors("hostA",2)
assert "hostB" in neighbors("hostA",9)
def sigmoid(x): return 1/(1+math.exp(-x))
# scalar embeddings: positive pair (u,v), negative (u,n)
u,v,n=1.0,2.0,-1.0
grad=(sigmoid(u*v)-1)*v+sigmoid(u*n)*n
eps=1e-6
def loss(u):
    return math.log1p(math.exp(-u*v))+math.log1p(math.exp(u*n))
numeric=(loss(u+eps)-loss(u-eps))/(2*eps)
assert abs(grad-numeric)<1e-8
print("A",representation("hostA",2),"B",representation("hostB",2))
print("gradient",round(grad,4),"future_edge_excluded",True)
~~~
Expected: A (2.0, 2.0), B (2.0, 4.0); gradient approximately -0.5073; future_edge_excluded True. Extension: add a new unseen host, use time-restricted neighbor sampling, evaluate edge prediction against a degree baseline, inject missing edges and show degradation. Do not label a synthetic high score an attack.

## B6. Real-world-inspired SOC case
A SOC observes three hosts communicating with a newly seen domain after credential use. A graph can connect those hosts, users and domain; a sequence detector can examine per-host order; RAG can retrieve the *current authorized* investigation runbook. A useful analyst report separates **observed**: timestamps, entities, edges; **inferred**: unusual shared destination; **unknown**: malicious intent, data exfiltration; **action**: request additional evidence. High-degree common infrastructure can yield false positives. Forensic integrity requires source hashes, access audit, time zones, evidence retention and privacy minimization. No destructive action should be triggered solely by the graph embedding or fluent RAG summary.

# Research critique and doctoral extensions
**RAGAS** (Es et al., 2023, arXiv:2309.15217): strength is decomposition of reference-free evaluation; limitation is judge/model dependence and sensitivity to source quality. **Self-RAG** (Asai et al., 2023, arXiv:2310.11511): strength is adaptive retrieval/reflection; limitation is critique-token reliability and cost, and it does not replace external authorization or independent evidence checking. **GraphSAGE** (Hamilton, Ying, Leskovec, 2017, arXiv:1706.02216): strength is inductive neighborhood aggregation; limitation is benchmark-to-SOC transfer, edge provenance, and time-aware leakage not solved by architecture alone. **GAT** (Veličković et al., 2017, arXiv:1710.10903): learned neighbor weighting; attention is not a verified causal explanation. See companion index for canonical links, access notes and claim mapping.

**Research proposal R1:** Does independently adjudicated claim-level support predict harmful SOC recommendation errors better than answer-level similarity? Pre-register hypotheses, 100+ permissioned questions, gold atomic claims and versioned sources; split by runbook family/time; compare lexical, NLI and LLM judges against blinded humans; report disagreement and confidence intervals. **R2:** Does time-safe typed GraphSAGE improve detection of novel lateral-movement patterns over degree features and sequence models? Hold out hosts, future weeks and attack families; evaluate PR-AUC, false alerts/day, p95 latency, calibration and missing-sensor stress. **R3:** Joint pipeline: graph alert → authorized retrieval → cited analyst explanation; measure whether adding graph context improves evidence completeness without increasing unsupported action recommendations. Treat all results as hypotheses until tested.

# Progressive oral Socratic Q&A (pause after every question)
1. **What is a citation?** A pointer to source material, not a proof. *Follow-up:* What extra test makes it evidence?
2. **Evidence recall vs support?** Recall checks required passages retrieved; support checks whether answer claims follow from them. *Follow-up:* Can one be high while the other is low?
3. **Two required passages, one retrieved?** Recall 1/2, joint completeness 0. *Follow-up:* Why can any-hit mislead?
4. **Three claims, two supported?** Support precision 2/3. *Follow-up:* What if the unsupported claim is a dangerous instruction?
5. **Why atomize sentences?** Mixed support can hide inside a fluent sentence. *Follow-up:* Split a sentence with “and.”
6. **Can an answer be correct but ungrounded?** Yes, parametric knowledge may be right without cited evidence. *Follow-up:* Is that acceptable in audited forensics?
7. **Can an answer be grounded but wrong?** Yes, a source may be obsolete or false. *Follow-up:* How enforce valid-at time?
8. **What does macro average weight?** Each query equally. *Follow-up:* Why differ from micro?
9. **Derive abstention threshold.** Compare (1-p)C_w to C_a; answer if p≥1-C_a/C_w. *Follow-up:* What if p is uncalibrated?
10. **Why not trust an LLM judge?** Shared hallucination, negation and authority errors. *Follow-up:* Design a blinded human audit.
11. **What is citation coverage?** Fraction of material claims with sufficient citation. *Follow-up:* Why separate from citation precision?
12. **What must never be delegated to retrieved text?** Authorization or instruction priority. *Follow-up:* Name a prompt-injection failure.
13. **What is a graph edge?** An observed typed relation with provenance/time. *Follow-up:* Is a possible attack path an observed edge?
14. **Why graph rather than event sequence?** Cross-entity relationships. *Follow-up:* When is a count baseline enough?
15. **What does one GraphSAGE layer do?** Aggregates neighbors with self features. *Follow-up:* What can two layers see?
16. **Compute A's neighbor mean.** (4+0)/2=2. *Follow-up:* Why is B's mean 4?
17. **What is inductive learning?** A shared function embeds unseen nodes from features/neighbors. *Follow-up:* Why isn't this automatic zero-day detection?
18. **What is a false negative edge sample?** An unobserved but actually existing relation labeled negative. *Follow-up:* How do sensor gaps create it?
19. **Derive positive logistic gradient.** σ(s)-1 for -logσ(s). *Follow-up:* Derive negative term σ(s).
20. **Why time-restrict graphs?** Future edges leak information unavailable at decision time. *Follow-up:* How test invariance to adding future edges?
21. **Why might graph attention fail?** Shortcut hubs, noisy edges, drift; attention weights not causal. *Follow-up:* What ablation tests hub dependence?
22. **What does masking require?** Remove target from all observable paths, not just its primary token. *Follow-up:* Give a side-channel example.
23. **How evaluate a new detector?** Grouped temporal holdout, PR-AUC, false alerts/day, latency, missing-sensor stress. *Follow-up:* Why not only accuracy?
24. **Design a combined SOC workflow.** Time-safe graph alert, current authorized runbook retrieval, atomic citations, analyst review. *Follow-up:* Where do you place hard permission gates?

## Recall prompts and study checkpoint
Without notes: derive evidence recall and strict joint completeness; explain why claim support is not correctness; calculate the p≥0.9 threshold; write the GraphSAGE mean update and logistic contrastive derivative; identify a future-edge leak; propose a baseline and one negative control. Completion evidence: two reproducible lab outputs, an evaluation schema with annotated claims and an explicit temporal graph split. The lab is a mechanism demonstration, not published SOC efficacy.

## Source map
- Syllabus scope: private October 19 PDFs, Unit 4 Session 3 (GenAI) and Unit 4 Session 2 (Cyber), p.10.
- Prior retrieval and sequence background: [Nov 17 morning](../../../phd-morning/lessons/2026-11-17/lesson.md) and [evening](../../../phd-evening/lessons/2026-11-17/lesson.md).
- Retrieval vs generation and evidence requirements: Lewis et al. 2020 (arXiv:2005.11401), A1–A2 derivations.
- Evaluation dimensions: Es et al. 2023 (arXiv:2309.15217), A2–A5 original synthetic audit.
- Adaptive retrieval and reflection: Asai et al. 2023 (arXiv:2310.11511), A4 critique.
- Graph inductive aggregation: Hamilton et al. 2017 (arXiv:1706.02216), B2/B5 original calculations.
- Attention over graph neighborhoods: Veličković et al. 2017 (arXiv:1710.10903), B2 critique.
- Security leakage and governance: B4–B6 original threat-model analysis and both pathway syllabus instructions. No paper is claimed to establish this toy system's deployment performance.

**Integrity note:** No email sent, no audio generated, no narration triggered. Research PDFs not included unless license and repository rights can be independently confirmed; see references index.
