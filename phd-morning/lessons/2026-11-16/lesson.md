# Morning PhD teaching package — 16 November 2026 (Asia/Kolkata)
**Syllabus progression:** Generative AI, Unit 4 Session 1, *Retrieval architecture* (pathway PDF p.10); AI for Cybersecurity, Unit 3 Session 6, *Robustness and red teaming* (pathway PDF p.8). Both original PDFs were retrieved from the 19 October 2026 Gmail message. Prior 15 November sessions: continual learning/merging and attacks across learning settings. The transition is deliberate: retrieval is an alternative to changing model weights, while security evaluation asks whether a defense survives adaptive attacks. No live target systems are used.

## Spoken routes, prerequisites and teaching protocol
**10 min:** Define parametric/nonparametric memory; trace RAG pipeline; explain robust min–max; solve retrieval and perturbation counterexamples; answer Q1–Q5.
**30 min:** Add BM25 derivation, hybrid retrieval, metadata access control, robust optimization, constrained security features; answer Q1–Q13.
**Extended 90–120 min:** Run both labs, critique primary papers, design experiments, answer Q1–Q24.
**Tutor protocol:** Ask each oral question without displaying the answer; wait for learner reasoning; request a numerical or operational counterexample; reveal answer; ask the follow-up. When walking, describe formulas aloud before symbolic notation.

Prerequisites: conditional probability, log, dot products, cosine similarity, cross entropy, gradients, confusion matrices, attack capability constraints, prior LoRA/forgetting and prompt-injection lessons.
Definitions: **RAG** retrieval-augmented generation: selecting external evidence before answering; **chunk** independently indexed passage; **lexical retrieval** exact-term weighted matching; **dense retrieval** similarity of learned vectors; **hybrid** combination of lexical and dense candidate sets; **reranker** rescoring shortlisted documents; **grounding** support of answer claims by evidence; **adaptive attacker** adversary who knows the defense design; **robust risk** expected worst-case loss over allowed input changes; **abstention** declining an automated decision when evidence or confidence is inadequate.

# Track G — Retrieval architecture: when external evidence beats another adapter

## G1. Motivation, mechanism and trust boundary
Yesterday we saw why adapting weights can forget old tasks and mix tenants. A new SOC runbook revision is a poor reason to retrain an entire language model: retrieval can select the version authorized for a tenant and date. But RAG is not magically correct; retrieval can miss the right chunk, and the generator can misread or ignore it. Think of the retriever as a librarian, not a truth oracle.

Architecture (each arrow is a distinct testable step):
~~~text
approved sources -> parse/OCR -> chunk+version+tenant+ACL -> lexical index + embedding index
                                                             |
user+tenant+time -> authentication -> query normalization -> ACL-filtered candidates
                                                             |
                                         lexical/dense fusion -> optional reranker
                                                             |
                                       selected evidence+source IDs -> answer generator
                                                             |
                                  claim-to-citation check -> answer/abstain + trace
~~~
At ingestion, preserve document ID, title, section, publication/validity dates, checksum, license, language, tenant and access policy. Treat every retrieved chunk as untrusted *data*; it cannot grant tool permissions. An ACL (access-control list) must be enforced before sensitive text reaches the generator, not merely by telling the model to ignore unauthorized passages.

Step-by-step: (1) audit source rights and authority; (2) extract tables with headings and row context; (3) chunk by semantic boundaries with modest overlap; (4) attach metadata and version; (5) index tokens and embeddings; (6) filter by authorization and validity; (7) retrieve candidates; (8) optionally rerank; (9) supply citations and a refusal condition; (10) verify support, log corpus/model revisions, and monitor drift. Never silently replace a superseded manual with a similarly named document.

## G2. Lexical ranking mathematics and worked example
Let N be document count, df(t) number of documents containing term t, tf(t,d) occurrences of t in document d, |d| length, and avgdl average document length. A common BM25 variant scores:
IDF(t)=ln[1+(N-df(t)+0.5)/(df(t)+0.5)].
BM25(q,d)=sum over unique query terms t of IDF(t)*[tf(t,d)*(k1+1)]/[tf(t,d)+k1*(1-b+b*|d|/avgdl)].
k1>0 controls term-frequency saturation; b in [0,1] controls length normalization. This is a ranking heuristic informed by probabilistic retrieval, not a calibrated relevance probability.
Example N=10, df=2 gives IDF=ln(1+8.5/2.5)=ln(4.4)=1.4816. With tf=2, k1=1.2, b=.75, and |d|=avgdl, multiplier 2*2.2/(2+1.2)=1.375, term contribution ≈2.0372. If tf doubles, score increases sublinearly. Exact strings such as CVE IDs, hostnames, and error codes favor lexical matching; synonyms and paraphrases may favor dense retrieval. A dense encoder maps q,d to vectors e_q,e_d; cosine(q,d)=(e_q·e_d)/(|e_q||e_d|). It is similarity, not proof of relevance or truth.

A practical hybrid fusion is reciprocal rank fusion (RRF):
RRF(d)=sum over retrieval systems s of 1/(k+rank_s(d)), with k often chosen as 60. RRF combines ranks without assuming lexical and dense raw scores are comparable. Example dA ranked 1 and 10: score=1/61+1/70≈.03068; dB ranked 3 and 3: 2/63≈.03175, so dB wins. Rank fusion may bury a rare exact identifier; add hard filters and report retrieval@k separately. RRF is a heuristic, not an inference that document B is factually superior.

## G3. Original RAG probability model and a key distinction
Lewis et al. (2020) combine a learned retriever p_eta(z|x) and generator p_theta(y|x,z), where x is query, z a retrieved passage and y answer. For RAG-Sequence:
p(y|x)≈sum_{z in topK} p_eta(z|x) * product_{i=1}^{|y|} p_theta(y_i|x,z,y_{<i}).
The same z conditions an entire answer sequence; RAG-Token can marginalize different z at each output position. Here eta and theta are parameters, and y_<i are preceding output tokens. Taking negative log gives training loss -ln p(y|x), which propagates pressure toward passages useful for the answer, under the approximation and architecture. **Not every production RAG system jointly trains retrieval and generation.** Many systems use fixed embeddings, independent vector search and a prompt; don't claim they reproduce the paper's probabilistic training.

Failure analysis: a correct passage outside topK is unrecoverable by generation; chunk splits may lose table headers; near-duplicate obsolete revisions crowd results; dense similarity can confuse semantically related but incorrect versions; prompt injection can exploit the retrieval-to-tool boundary. More chunks increase recall but consume context, latency and attention; larger overlap increases redundancy and cost. Measure evidence recall@k, MRR, authorization violations, citation precision, answer correctness, abstention, latency and cost separately.

## G4. Worked SOC example
Question: “For tenant A, what is the 2026-11 approved procedure after a high-severity failed login alert?” Suppose corpus has (a) tenant A current policy, (b) tenant A obsolete 2025 policy, (c) tenant B 2026 policy. A naive dense search may rank B first because wording is similar. Correct pipeline first filters tenant=A and valid_at=2026-11, then ranks current A. The generator should cite the section, state unknown steps if absent, and never invoke a disable-user tool merely because the retrieved runbook says so. User authorization and explicit confirmation govern actions independently.

## G5. Lab 1 — runnable lexical retrieval, RRF and ACL (Python 3.10+, standard library)
~~~python
import math, re
docs=[
 {"id":"A-current","tenant":"A","year":2026,"text":"failed login alert verify identity preserve logs escalate incident"},
 {"id":"A-old","tenant":"A","year":2025,"text":"failed login alert immediately disable account"},
 {"id":"B-current","tenant":"B","year":2026,"text":"failed login alert disable account escalate"},
]
query="failed login alert preserve logs"
tok=lambda s: re.findall(r"[a-z0-9]+",s.lower())
eligible=[d for d in docs if d["tenant"]=="A" and d["year"]==2026]
assert [d["id"] for d in eligible]==["A-current"]
N=len(eligible); avgdl=sum(len(tok(d["text"])) for d in eligible)/N
def score(d):
    words=tok(d["text"]); total=0
    for t in set(tok(query)):
        df=sum(t in tok(x["text"]) for x in eligible)
        tf=words.count(t)
        idf=math.log(1+(N-df+.5)/(df+.5))
        total+=idf*tf*2.2/(tf+1.2*(.25+.75*len(words)/avgdl)) if tf else 0
    return total
ranked=sorted(eligible,key=score,reverse=True)
assert ranked[0]["id"]=="A-current"
print("authorized top",ranked[0]["id"],"score",round(score(ranked[0]),3))
# Rank-fusion counterexample, not an ACL mechanism:
a=1/61+1/70; b=2/63
assert b>a
print("RRF B beats A:",round(b,5),">",round(a,5))
~~~
Expected: authorized top A-current; RRF B beats A: 0.03175 > 0.03068 (BM25 score is deterministic). Extension: remove the tenant filter and demonstrate that ranking cannot guarantee confidentiality; add five authorized documents and test recall@2. This lab tests a ranking mechanism, **not** an LLM's factuality or a production ACL implementation.

## G6. Paper critique and research extension
Lewis et al. (NeurIPS 2020, arXiv:2005.11401) established retrieval-conditioned generation on selected knowledge-intensive tasks with Wikipedia passages. Its reported improvements are not evidence that every corporate vector-search chatbot is grounded; domain access rules, evolving documents, table parsing and human verification require independent study. Robertson & Zaragoza (2009, DOI:10.1561/1500000019) explain BM25's probabilistic relevance foundations, but its lexical assumptions miss paraphrases. Falsifiable study: compare BM25, dense and RRF on 100 hand-labeled versioned SOC questions; hold tenant and time filters constant; use bootstrap CIs by document family; measure recall@5, citation support, unauthorized retrieval rate and p95 latency. Pre-register what constitutes a correct citation. Don't tune on the final test set.

# Track C — Robustness and adaptive red teaming

## C1. Defining the actual security objective
Yesterday's attacks involved poisoned baselines, malicious FL clients and prompt injection. Today the question is whether controls still work when the attacker understands them. A threat model must specify assets (credentials, incident integrity, model decision), attacker control (which inputs/updates/documents), knowledge, budget, constraints and success condition. An “adversarial example” that breaks protocol semantics or changes a feature unavailable to the attacker is not a valid security test.

~~~text
owned synthetic telemetry -> clean train/validation/test split -> baseline detector
                 |                                          |
       declared feasible attacker set                  defensive controls
                 |                                          |
      adaptive attack attempts ----------------------> clean + robust evaluation
                                                            |
                                      abstain / investigate / authorized action
~~~

## C2. Robust optimization derivation
Ordinary empirical risk: R(θ)=E_(x,y)~D[L(f_θ(x),y)], with model parameters θ, data distribution D, loss L. Robust risk:
R_rob(θ)=E_(x,y)~D[max_{δ in S(x)} L(f_θ(x+δ),y)].
S(x) is the *feasible* attacker perturbation set, which may be discrete, causal and feature-dependent. Defender solves min_θ R_rob(θ). In the differentiable L_infinity idealization S={δ:||δ||∞≤ε}, a first-order Taylor approximation L(x+δ)≈L(x)+∇_x L·δ is maximized by δ_i=ε sign(∂L/∂x_i); this yields FGSM. Repeating projected ascent gives PGD:
δ_{j+1}=Π_S[δ_j+α sign(∇_δ L(f_θ(x+δ_j),y))].
Π_S projects into feasible S; α is attack step. Neither FGSM nor PGD guarantees a global maximum on nonconvex losses. In network security, S must also preserve packet counts, timing relations, valid ranges, protocol and attack functionality; Euclidean clipping alone is inadequate.

Worked counterexample: model score s(x)=2x1−x2, decision malicious if s≥1. At x=(.6,.1), score=1.1. With independent |δ_i|≤.1, minimum score=1.1−.2−.1=.8, so label can flip. But if feasible changes require δ1=δ2 (two features derived from same event), minimum change is δ, bounded by -.1; score minimum=1.0, so no flip under a strict s<1 benign rule. This demonstrates why ignoring feature dependencies overstates attacker capability. A different feasible set can still permit attacks; constraints must be justified, not chosen to make defense look strong.

## C3. Defensive controls and their limits
Adversarial training uses adversarial examples during fitting; may improve robustness within the evaluated perturbation class, but often raises compute cost and reduces clean utility. Input validation checks schemas and invariants but may reject legitimate rare behavior; abstention sends uncertain decisions to analysts, increasing review workload. Rate limiting constrains query budgets but does not prevent a single malicious document. Signed provenance detects certain modifications but does not prove a signed source is trustworthy. Least-privilege tools and confirmation gates limit damage even if generation is compromised. For retrieval, **filter tenant permissions before ranking and generation**; for detectors, protect baseline updates from unauthenticated observations.

Use a release scorecard: clean recall, attack-conditioned recall, false positive rate at fixed prevalence, analyst queue size, abstention rate, attack success under multiple budgets, latency, group disparities, and rollback time. Avoid cherry-picking one blocked prompt or one weak FGSM run. An adaptive attacker should test the actual defense, including transformations, thresholds, rate limits and retrieval pipeline, in an authorized sandbox.

## C4. Lab 2 — constrained adversarial toy and abstention (Python 3.10+, standard library)
~~~python
from itertools import product
w=(2.,-1.); x=(.6,.1); eps=.1
score=lambda p:sum(a*b for a,b in zip(w,p))
grid=(-eps,0.,eps)
unconstrained=list(product(grid,repeat=2))
coupled=[d for d in unconstrained if abs(d[0]-d[1])<1e-12]
def minimum(ds):return min(score((x[0]+d[0],x[1]+d[1])) for d in ds)
a,b=minimum(unconstrained),minimum(coupled)
assert abs(a-.8)<1e-9 and abs(b-1.0)<1e-9
print("unconstrained min",round(a,2),"coupled min",round(b,2))
def decision(s):
    if .95<=s<=1.05:return "review" # human escalation, not a guaranteed defense
    return "alert" if s>1.05 else "benign"
assert decision(b)=="review"
print("coupled decision",decision(b))
~~~
Expected: unconstrained min 0.8, coupled min 1.0, coupled decision review. The grid is exhaustive only over three allowed values per coordinate; it is **not** an exhaustive continuous attack search. In this linear example the extrema occur at corners, making the calculation valid for the stated box and coupling. Extensions: add discrete packet-count invariants, adversary budgets, different decision thresholds, and count review workload. Never run against third-party systems.

## C5. Real-world framing, paper critique, ethics
A SOC anomaly detector could appear robust against naive feature noise yet fail when an adversary controls a valid sequence of logins. Conversely, arbitrary feature changes can create physically impossible flows and overstate risk. Madry et al. (ICLR 2018, arXiv:1706.06083) popularized the min–max first-order adversarial training framework on image benchmarks; its norm-ball threat model does not automatically transfer to network protocols. Vassilev et al. (NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025) provide taxonomy and mitigation concepts, not a certification of any particular model. Test in owned/synthetic environments, do not use real customer credentials or unauthorized endpoints, log attack assumptions and preserve clean baselines. A failure under one attack is evidence of vulnerability; passing one test is not proof of security.

## C6. Research questions
(1) Does protocol-constrained red teaming change the ranking of defenses compared with unconstrained L∞ perturbations? (2) At a fixed analyst review budget, does abstention improve incident recall without disproportionate minority-environment false alerts? (3) Does combining ACL-first RAG and action confirmation reduce indirect prompt-injection harm at matched legitimate task success? Pre-register datasets, seeds, attack budgets, holdout groups and uncertainty intervals; distinguish observed outcomes from hypotheses.

# Cross-track synthesis
RAG reduces the need for repeated weight adaptation but creates a new external-evidence trust boundary. Robustness is end-to-end: authorized retrieval, version selection, correct citations, tool permissions and response checks. The same evaluation discipline applies to both: keep clean utility, adversarial failure, leakage and operational cost separate. An accurate answer citing an unauthorized tenant document is a security failure.

# Progressive oral Socratic Q&A (ask, wait, reveal, probe)
1. **Why use RAG after continual learning?** Update facts without modifying weights. *Probe:* Which risks shift to retrieval?
2. **What is a chunk?** An indexed passage with metadata. *Probe:* How can chunking break a table?
3. **What is lexical retrieval?** Ranking exact terms/statistics. *Probe:* Why good for CVE IDs?
4. **What is dense retrieval?** Learned-vector similarity. *Probe:* Why similarity is not truth?
5. **What does BM25 IDF measure?** Term rarity across corpus. *Probe:* What if every document contains term?
6. **Why saturate term frequency?** Repetition gives diminishing relevance gains. *Probe:* What controls saturation?
7. **What does b adjust?** Length normalization. *Probe:* What if b=0?
8. **Why combine retrievers?** Complementary exact-term and semantic matches. *Probe:* Why raw scores cannot simply add?
9. **What is RRF?** Sum of inverse rank offsets. *Probe:* Compute the rank (3,3) score.
10. **What must happen before retrieval reaches generator?** Tenant/time authorization. *Probe:* Why prompt-only filtering fails?
11. **What is RAG-Sequence?** Marginalizes passages for entire output. *Probe:* Difference from RAG-Token?
12. **Is every RAG system jointly trained?** No. *Probe:* Name fixed-index architecture.
13. **How evaluate grounding?** Claim-level citation support and answer accuracy. *Probe:* What if passage is wrong?
14. **What is a threat model?** Asset, control, knowledge, budget, constraints, objective. *Probe:* Which feature can attacker change?
15. **What is robust risk?** Expected worst-case loss over feasible perturbations. *Probe:* Why not only clean loss?
16. **What is FGSM?** First-order sign-gradient perturbation. *Probe:* What assumption limits it?
17. **What is PGD?** Iterative projected attack. *Probe:* Does it prove global optimality?
18. **Why enforce protocol invariants?** Otherwise attacks may be impossible. *Probe:* Give correlated features.
19. **Calculate original score.** 2(.6)-.1=1.1. *Probe:* Minimum unconstrained score?
20. **Why coupled minimum differs?** δ1=δ2 makes score change δ, not 2δ1−δ2 independently. *Probe:* What if threshold changes?
21. **Does abstention eliminate risk?** No, it moves work to review. *Probe:* Measure queue load.
22. **What does adversarial training optimize?** Approximate min–max robust loss. *Probe:* Clean-accuracy trade-off?
23. **Why red-team adaptively?** Defenses may block only known attacks. *Probe:* What is held constant for fairness?
24. **Propose one falsifiable integrated study.** ACL-first versioned RAG plus tool gate versus prompt-only filtering, matched clean success and adversarial budget. *Probe:* What counts as unauthorized exposure?

# Recall and checkpoint
Without notes: (i) write BM25 and explain each symbol; (ii) derive RRF (1,10) vs (3,3); (iii) trace an ACL-safe RAG request; (iv) derive FGSM from Taylor expansion; (v) explain why coupled features change robust evaluation; (vi) specify five clean/adversarial operational metrics. A correct answer must name assumptions, not just formulas.

# Source map (offline teaching independence)
G1 architecture, tenant/time counterexample: local G1/G4 and syllabus GenAI Unit4 Session1. G2 BM25: Robertson & Zaragoza 2009 and local G2/G5; RRF example is local arithmetic. G3 RAG-Sequence/Token: Lewis et al. 2020 and local derivation. C1–C3 threat model, min–max, FGSM/PGD: Madry et al. 2018, NIST 2025 and local C2/C4; constrained-feature example is synthetic local derivation. C5 ethics and taxonomy: NIST 2025. Research questions are proposals, not verified results. Companion: [references/index.md](references/index.md).

# Scope and reproducibility
Python 3.10+ standard library, synthetic text/features only. The labs have deterministic assertions; a small isolated run is recommended before reusing them. No live intrusion, user secrets, email, audio, narration or GitHub Actions are invoked. No claim of latest 2026 breakthrough is made without verified primary publication.
