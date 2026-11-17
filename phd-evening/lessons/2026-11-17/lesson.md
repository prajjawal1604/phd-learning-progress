# Evening PhD deep dive — 17 November 2026 (Asia/Kolkata)
**Dependency verified:** read committed morning [lesson](../../../phd-morning/lessons/2026-11-17/lesson.md) and [reference index](../../../phd-morning/lessons/2026-11-17/references/index.md) on main before writing. **Syllabus:** both original PDFs attached to the 19 October 2026 Gmail message, *Advanced Generative AI and Foundation Models Learning Pathway*, Unit 4 Session 2 (p.10), and *AI for Cybersecurity Learning Pathway*, Unit 4 Session 1 (p.10). This deep dive extends only (G) query processing/chunking/reranking and (C) Transformers for security-event sequences. Do not interpret synthetic lab scores as deployment evidence.

## Spoken tutor protocol and routes
**10 minutes:** Explain candidate-recall ceiling, derive pairwise ranking loss and nDCG failure, then contrast causal event prediction with bidirectional masking; discuss Q1–Q5, Q13–Q16. **30 minutes:** Work query expansion counterexample, authorization-safe architecture, LambdaRank-style swap weighting, temporal leakage, transition surprisal and deployment drift; Q1–Q10, Q13–Q21. **90–120 minutes:** Derive all equations, execute both labs after the walk, critique papers, design a paired ablation and answer all 26 questions. Ask, pause, request a counterexample, explain, and probe again. Speak formulas verbally before symbols; do not read code aloud while walking.

**Prerequisites:** log-likelihood and gradient chain rule; logistic sigmoid; ranking labels and nDCG; Bayes rule; matrices/softmax; causality versus association; time-based evaluation; tenant ACLs. **Definitions:** candidate recall = fraction of required relevant passages retrieved before reranking; pairwise margin = score difference between relevant and irrelevant candidates; listwise metric = evaluates ordering jointly; false negative = relevant evidence excluded; causal attention = can attend only to observed history; event surprisal = negative log conditional likelihood; concept drift = change in relationship between input and target; covariate shift = change in input distribution.

# G. Advanced query processing, chunking, reranking

## G1. Why the system fails even with a strong reranker
Consider an analyst asking "Which approved procedure governed account lockout on 2026-10-01?" Candidate generator retrieves ten 2025 versions and excludes the correct 2026 procedure. The cross encoder can only permute those ten. If relevant set R(q) and candidate set C_K(q), evidence coverage is |R ∩ C_K|/|R|. Any downstream ranking of C_K has coverage at most that fraction. For a question requiring two independent sources, answerability may be zero even when one source is present. This is a **set containment bound**, not an empirical claim that generation must always fail: a generator might answer from parametric memory, but then it is not grounded in the missing evidence.

Mechanism: parse immutable document version -> preserve table headers and row relationships -> attach source offset, valid-from/valid-to, tenant, classification and status -> apply trusted authorization/time filters -> generate sparse+dense candidates -> merge and deduplicate by parent+version -> rerank only eligible passages -> enforce evidence diversity -> verify claim entailment and citations -> abstain if evidence insufficient. Filtering must be enforced by service-side code **before** model exposure; a model cannot safely interpret a retrieved instruction to broaden access. The audit log records corpus hash, retrieval and reranker revisions, policy decision, chosen offsets, latency and abstention reason.

Architecture:
~~~text
signed source + version + trusted ACL
    | parse headings/tables; parent-child chunks; offset mapping
    v
authorized candidate service -- BM25 / dense / exact-ID --> union
    | de-duplicate by document+version; preserve independent sources
    v
cross encoder [query, candidate] -> relevance score
    | validity/ACL still enforced by code; diversity constraint
    v
evidence bundle + provenance -> grounded answer / abstain
    | independent entailment audit; immutable trace
~~~
A policy label such as "approved" is a hard eligibility constraint, not merely +2 relevance points. Likewise the parent document and the child passage must share compatible ACLs and validity windows.

## G2. Pairwise objective, derivatives, and a ranking metric trap
Let s_+=f_\theta(q,d_+) and s_-=f_\theta(q,d_-). Define margin m=s_+-s_- and sigmoid \sigma(m)=1/(1+e^{-m}). Assuming a logistic comparison model, P(d_+\succ d_-)=\sigma(m). The negative log likelihood is
L=\log(1+e^{-m}); dL/dm=\sigma(m)-1.
By chain rule, \nabla_\theta L=(\sigma(m)-1)(\nabla_\theta s_+-\nabla_\theta s_-).
At m=0, L=ln2 and derivative -0.5; at m=ln3, derivative -0.25. A large positive margin produces a small gradient. With noisy preferences, training can push obsolete documents above current documents; the loss does not know policy validity.

**Why pairwise loss is not nDCG.** Pairwise loss counts comparisons, while nDCG weights positions and graded gains. DCG@k = sum_{i=1}^k (2^{r_i}-1)/log_2(i+1); nDCG=DCG/IDCG (when IDCG>0). Grades [2,0,1] have DCG=3+1/2=3.5; ideal [2,1,0] has 3+1/log_2 3=3.63093, giving nDCG~0.96394. A model can improve average pairwise classification yet worsen top-rank relevance if its errors concentrate on high-gain documents. A LambdaRank-style heuristic weights a pair's gradient by |\Delta nDCG| from swapping their positions. This aligns optimization pressure with a ranking metric, but is not a proof of globally optimizing nDCG.

Derive swap impact: at positions i,j with grades a,b, gain G(r)=2^r-1 and discount D(p)=1/log_2(p+1). Swapping changes DCG by [G(a)-G(b)][D(j)-D(i)]. Absolute normalized change is its magnitude divided by IDCG. For grades 2 at rank 1 and 0 at rank 2, swapping loses 3(1-1/log_2 3)~1.1072 DCG. For two grade-0 items, impact is zero. This explains why pair weighting can focus on top-of-list mistakes.

**Trade-offs:** more candidates increase recall and inference cost; larger chunks retain context but cause truncation and distractors; overlap reduces boundary loss but crowds the shortlist; cross encoders gain joint interactions but require O(K) inference calls and may truncate long passages; query expansion increases recall yet risks topic drift. Version-aware evaluation must compare at equal eligibility rules and p95 latency.

## G3. Counterexamples and advanced evaluation design
**Counterexample A — expansion:** Query "lock account" is expanded to "disable account" and retrieves irreversible disable instructions. Even if lexical overlap increases, intent is changed. Preserve the original query; log every rewrite; compare expansion on an independently labeled test set including near-synonym traps.

**Counterexample B — chunking:** A row "reset token" appears alone while its header says "DEPRECATED, DO NOT USE." Overlapping windows can preserve the header, but semantic splitting may still separate it. Store table-row+header as one logical retrieval unit; test reconstruction from source offsets.

**Counterexample C — metric:** A reranker increases nDCG@10 but hides the only second required citation. Evaluate *joint evidence completeness*, not only single-passage ranking. For a two-evidence query, define complete(q)=1 if both required evidence IDs are retrieved and cited, otherwise 0. Macro-average over queries; report unauthorized exposures as separate zero-tolerance events.

**Experimental design:** 100–200 hand-labeled synthetic or permissioned questions stratified by version, tenant, table, ambiguity and two-source requirement. Split by document family and creation date, not overlapping chunks. Compare lexical-only, hybrid, hybrid+cross encoder, and hybrid+cross encoder+diversity. Freeze corpus and ACLs; report recall@20, nDCG@10, joint completeness, unsupported claim rate, p50/p95 latency, compute and unauthorized retrieval count. Bootstrap by **query family**, not individual duplicated chunks, for uncertainty intervals. No claim of measured improvements is made here.

## G4. Lab G — rank gradients and metric-aware swap (Python 3.10+, standard library)
Save as g_rank_lab.py. This is a numerical demonstration, **not** a trained neural reranker.
~~~python
import math
def sigmoid(x): return 1/(1+math.exp(-x))
def loss(m): return math.log1p(math.exp(-m))
m=math.log(3)
eps=1e-6
analytic=sigmoid(m)-1
numeric=(loss(m+eps)-loss(m-eps))/(2*eps)
assert abs(analytic+0.25)<1e-12
assert abs(analytic-numeric)<1e-8
def dcg(grades):
    return sum((2**r-1)/math.log2(i+2) for i,r in enumerate(grades))
def ndcg(grades):
    ideal=dcg(sorted(grades,reverse=True))
    return dcg(grades)/ideal if ideal else 0.0
grades=[2,0,1]
assert abs(ndcg(grades)-3.5/(3+1/math.log2(3)))<1e-12
def swap_delta(grades,i,j):
    other=grades.copy()
    other[i],other[j]=other[j],other[i]
    return abs(ndcg(other)-ndcg(grades))
assert swap_delta([2,0,1],0,1)>swap_delta([2,0,1],1,2)
eligible={"A":("approved",2026),"B":("draft",2026),"C":("approved",2025)}
safe=[k for k,(status,year) in eligible.items() if status=="approved" and year==2026]
assert safe==["A"]
print("gradient",round(analytic,2),"nDCG",round(ndcg(grades),4),"eligible",safe)
~~~
Expected: gradient -0.25, nDCG approximately 0.9639, eligible ['A']. **Extension:** vary grades and K; demonstrate that improving average pairwise accuracy can reduce nDCG. Add a rank-12 relevant passage and show that reranking only top-10 cannot recover it. In production, replace hard-coded metadata with a policy engine and preserve its decision log.

## G5. Paper critique
Nogueira & Cho (2019), *Passage Re-ranking with BERT*, arXiv:1901.04085, https://arxiv.org/abs/1901.04085, demonstrates joint BERT passage reranking on retrieval benchmarks. **Strength:** task-specific query-passage interactions; **limitation:** ranking benchmarks do not establish ACL enforcement, temporal validity, or grounded citation correctness. Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv:2005.11401, https://arxiv.org/abs/2005.11401, combines retrieval and generation. **Strength:** explicit nonparametric memory; **limitation:** retrieval-conditioned likelihood does not guarantee faithful claim-level citations. Research extension: evaluate whether version-aware retrieval improves *joint evidence completeness* at fixed p95 latency without increasing unauthorized candidate exposure.

# C. Advanced Transformers for security-event sequences

## C1. Threat model and operational intuition
A sequence is a time-ordered set of typed events from one entity/session. A login, privilege escalation and outbound connection can suggest a chain; the same three tokens in reverse order are different evidence. But neither order alone proves maliciousness. Attackers may suppress sensors or blend into common patterns; a patch rollout may introduce benign unseen templates. Separate anomaly score, attack probability, analyst confirmation and automated response.

Architecture:
~~~text
raw signed telemetry -> redaction + normalization + provenance
    -> host/session grouping -> time ordering + clock skew policy
    -> train-only template vocabulary + UNK + missingness flags
    -> causal Transformer / Markov baseline
    -> next-event NLL + auxiliary metadata
    -> validation-calibrated threshold + abstention
    -> evidence-linked analyst queue (no destructive auto-action)
~~~
Event timestamps may arrive out of order. Use ingestion and event time separately, record clock uncertainty, avoid filling missing logs with assumed benign events, and maintain tenant isolation. Train/test windows from the same incident must never cross partitions.

## C2. Attention scaling and causal masking derived
For sequence matrix X in R^{T x d}, W_Q,W_K in R^{d x d_k}, W_V in R^{d x d_v}, define Q=XW_Q, K=XW_K, V=XW_V. Attention output is softmax(QK^T/sqrt(d_k)+M)V, with M_{ij}=-infinity if j>i for causal prediction. Without a mask, position t may inspect a future event and yield deceptively low evaluation loss.

Why sqrt(d_k)? Under an illustrative independence/zero-mean/unit-variance assumption for query and key coordinates, dot product z=sum_{h=1}^{d_k} q_h k_h has Var(z)=d_k. Dividing by sqrt(d_k) makes Var(z/sqrt(d_k)) approximately 1, reducing extreme softmax saturation. This is not a guarantee under correlated learned representations and does not calibrate attack probabilities.

Worked example: q=(1,0), k1=(1,0), k2=(0,1), d_k=2. Logits are (1/sqrt2,0); softmax weights approximately (0.6698,0.3302). With values v1=(2,0), v2=(0,4), output is (1.3396,1.3208). If k2 is future and forbidden, output is (2,0). Attention weights are not causal explanations.

**Compute:** T=1000 events gives T²=1,000,000 attention score entries per head, ~4MB at float32 before gradients and activations. A causal model can stream history with caching, but long histories, sensor missingness and state changes still require explicit policy. A shorter window may miss slow campaigns; a longer window increases cost and distractors.

## C3. Conditional surprisal, Bayes and threshold failure
A causal model estimates p_\theta(e_t | e_{<t}), and average negative log likelihood S=-1/n sum_t log p_\theta(e_t|e_{<t}). A high S means improbable under this model, **not** necessarily malicious. For two hypotheses H_A (attack) and H_B (benign), posterior odds equal prior odds times likelihood ratio:
P(H_A|x)/P(H_B|x)=[P(x|H_A)/P(x|H_B)] [P(H_A)/P(H_B)].
A next-event model trained only on normal logs approximates P(x|H_B); it does not supply P(x|H_A), so S alone cannot produce calibrated attack posterior.

Base-rate example: prevalence 0.001, sensitivity 0.90, false-positive rate 0.01. Precision = (0.90*0.001)/(0.90*0.001+0.01*0.999) ≈ 0.08264. Thus ~8.3% of alerts would be true attacks under these assumptions. An apparent 90% recall is not enough for autonomous blocking. Validate thresholds on future-like data and report alert counts per day, not only AUC.

**Counterexample — sequence leakage:** random splitting overlapping windows from the same host/day lets near-identical incident sequences appear in train and test. A model learns capture identity, not general attack behavior. Hold out hosts and later periods, and separate attack-family novelty from ordinary temporal shift. **Counterexample — missing sensor:** if a malicious event is never logged, the sequence model cannot infer its exact content from nothing; uncertainty and missingness indicators are essential. **Counterexample — template drift:** a routine deployment changes template IDs, raising surprisal without compromise. Include change-calendar evidence and compare against a simple Markov baseline.

## C4. Lab C — causal transition model and evaluation leakage (Python 3.10+, standard library)
Save as c_sequence_lab.py. Synthetic toy, **not** a Transformer or validated intrusion detector.
~~~python
import math
from collections import Counter
train=[
 ["login","ok","logout"],
 ["login","ok","logout"],
 ["login","fail","lockout"]
]
vocab=sorted({t for seq in train for t in seq}|{"<END>","<START>","<UNK>"})
counts=Counter()
totals=Counter()
for seq in train:
    s=["<START>"]+seq+["<END>"]
    for a,b in zip(s,s[1:]):
        counts[a,b]+=1
        totals[a]+=1
V=len(vocab)
def score(seq):
    s=["<START>"]+[t if t in vocab else "<UNK>" for t in seq]+["<END>"]
    return sum(-math.log((counts[a,b]+1)/(totals[a]+V))
               for a,b in zip(s,s[1:]))/(len(s)-1)
clean=["login","ok","logout"]
reverse=list(reversed(clean))
unknown=["login","new_template","logout"]
assert score(clean)<score(reverse)
assert score(clean)<score(unknown)
# Explicitly forbid target-position peeking:
def visible_history(seq,t): return seq[:t]
assert visible_history(clean,1)==["login"]
assert "ok" not in visible_history(clean,1)
print("clean",round(score(clean),3))
print("reverse",round(score(reverse),3))
print("unknown",round(score(unknown),3))
~~~
Expected: clean score lower than reverse and unknown; exact scores printed deterministically. A unigram model assigns equal scores to clean and reverse, unlike this transition model. Extension: implement a 2-head causal attention model in PyTorch with an upper-triangular mask, train only on normal sessions, verify that perturbing future events cannot change earlier logits, and compare calibration against this Markov baseline on grouped temporal splits. Do not use raw host IDs as prediction shortcuts.

## C5. Real-world SOC case and security/ethics
Imagine a multi-tenant SOC ingesting authentication, process and network events. A burst of failed logins followed by a privileged operation may trigger a detector; an authorized retrieval system can fetch the currently approved runbook and attach evidence offsets. Neither a high surprisal score nor a fluent RAG explanation proves account compromise. Separate observed facts, inferred hypotheses and analyst-approved actions. Redact personal data, minimize retention, encrypt tenant-specific records, restrict model/tool permissions, audit decisions and prohibit autonomous destructive containment on anomaly score alone. Offline synthetic red-teaming must not create live account lockouts or probe systems without authorization.

## C6. Paper critique and research design
Vaswani et al. (2017), *Attention Is All You Need*, arXiv:1706.03762, https://arxiv.org/abs/1706.03762: scaled attention and masks are foundational, but original translation results do not establish security-event effectiveness. Du et al. (2017), *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*, arXiv:1709.09338, https://arxiv.org/abs/1709.09338: sequence-based anomaly modeling motivates a log baseline; evaluate template stability, contamination and generalization to new software versions rather than assuming benchmark metrics transfer. **Hypothesis:** missingness-aware causal sequence modeling improves recall at a fixed 0.5% FPR over a Markov baseline on later unseen hosts. Pre-register the FPR, threshold selection, time/group splits and confidence intervals; publish negative results and the analyst review burden.

# Cross-track synthesis and doctoral extension
A trustworthy SOC assistant requires two independent validity checks: *did the detector observe enough evidence to justify investigation?* and *did retrieval provide authorized, temporally valid support for every claim?* A strong ranking metric cannot compensate for a detector's false positives; a strong detector cannot compensate for an obsolete runbook. Design a 2x2 ablation of (baseline vs causal detector) and (lexical vs version-aware reranked retrieval). Outcomes: detection recall at fixed FPR, joint citation completeness, abstention, privacy/ACL violations, analyst minutes and p95 end-to-end latency. Use incident-family clustered bootstrap confidence intervals; keep all incident-derived windows together. This is a proposal, not a measured result.

# Progressive oral Socratic examination — ask, wait, explain, challenge
1. **What limits reranking?** Its input candidate set. Follow-up: Can parametric recall count as evidence-grounded retrieval?
2. **Define candidate recall.** Required relevant documents present before rerank / total required. Follow-up: What if two are jointly needed?
3. **Why filter ACL first?** Prevent unauthorized exposure to any model stage. Follow-up: What about cached embeddings?
4. **Why is status a hard constraint?** Relevance cannot authorize a draft procedure. Follow-up: How represent policy provenance?
5. **Why keep table headers?** Row meaning depends on them. Follow-up: Show a deprecated-row failure.
6. **Derive pairwise probability.** Sigmoid of score gap. Follow-up: Why isn't it truth probability?
7. **Derive logistic loss.** log(1+exp(-margin)). Follow-up: Derivative at zero?
8. **Derivative at zero?** -0.5. Follow-up: Effect of gradient descent?
9. **What does nDCG weight?** Graded relevance and rank discount. Follow-up: Why can pairwise accuracy mislead?
10. **Why swap weighting?** Larger metric-impact errors get larger gradient magnitude. Follow-up: What about ties?
11. **What is expansion drift?** Rewrites change intent. Follow-up: Give a security procedure example.
12. **How split retrieval evaluation?** By document family/time. Follow-up: Why not by chunk?
13. **Why sequence order?** Different order can imply different process. Follow-up: Which model loses order?
14. **What is causal masking?** Prevent attention to future tokens. Follow-up: How test it?
15. **What is d_k?** Query/key vector dimension. Follow-up: Why divide by its square root?
16. **Why scale logits?** Reduce variance growth under assumptions. Follow-up: Does it calibrate alerts?
17. **What does surprisal mean?** Negative conditional log likelihood. Follow-up: Why not attack probability?
18. **What is a prior?** Pre-observation probability of attack. Follow-up: How does low prevalence affect precision?
19. **Compute alert precision.** Use sensitivity*prevalence divided by that plus FPR*(1-prevalence). Follow-up: Interpret 8.3%.
20. **Why compare Markov?** Simpler baseline tests value of attention. Follow-up: Which distribution shifts matter?
21. **What is template drift?** Changed log message patterns. Follow-up: How use deployment records?
22. **Why hold out hosts/time?** Avoid incident overlap and identity shortcuts. Follow-up: What about unseen attacks?
23. **How represent missing logs?** Explicit missingness/coverage flags, not benign assumptions. Follow-up: Can unobserved events be reconstructed?
24. **Why not autonomous blocking?** Scores are uncertain and costly false positives occur. Follow-up: Which human approval?
25. **How join both tracks?** Evidence-linked alerts with authorized current runbooks. Follow-up: Distinguish hypotheses from facts.
26. **Propose falsification.** Compare models on held-out future hosts at fixed FPR and latency; reject if no improvement. Follow-up: Which negative result would you publish?

## Recall and source map
Recall without looking: write the pairwise derivative, nDCG formula, attention mask, surprisal equation, Bayes precision calculation and two-source completeness rule. Then explain one failure of each model to a nontechnical analyst.
- G1–G4 and Q1–Q12: [morning G sections](../../../phd-morning/lessons/2026-11-17/lesson.md), Nogueira & Cho 2019, Lewis et al. 2020; original derivations and synthetic lab here.
- C1–C4 and Q13–Q24: [morning C sections](../../../phd-morning/lessons/2026-11-17/lesson.md), Vaswani et al. 2017, Du et al. 2017; synthetic Markov demonstration here.
- G5/C5/C6, Q25–Q26: both pathway PDFs (private Gmail attachments, p.10), papers and explicit proposed evaluation designs; not measured deployment results.
- All source links, access notes and paper critiques: [evening references](references/index.md).

**Integrity and limitations:** No paper PDFs claimed downloaded or committed. No email, audio or narration requested or produced. No full neural-model training or operational attack. Reproduce labs locally and compare on permissioned data before making empirical claims.
