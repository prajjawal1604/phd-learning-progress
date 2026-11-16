# Evening PhD deep dive — 16 November 2026 (Asia/Kolkata)

**Dependency verified:** Read committed morning [lesson](../../../../phd-morning/lessons/2026-11-16/lesson.md) and its [references](../../../../phd-morning/lessons/2026-11-16/references/index.md) on \`main\` before drafting. Same subjects only: **GenAI Unit 4 Session 1 — retrieval architecture**, and **AI for Cybersecurity Unit 3 Session 6 — robustness and red teaming**. Both original pathway PDFs were fetched from the 19 October 2026 Gmail message (GenAI p.10; cybersecurity p.8); private syllabus PDFs are not copied into the repository. This lesson is a new, harder treatment, not a replacement for the morning foundation.

## Tutor route and prerequisites

**10 minutes:** Ask why ACL filtering precedes retrieval; derive the effect of duplicate passages on RAG marginalization; solve the coupled-feature adversarial example; use Q1–Q6. **30 minutes:** Include retrieval metrics, BM25/RRF calibration failure, robust-risk dual norm, the feasibility geometry, and Q1–Q14. **Extended 90–120 minutes:** Run both executable labs, audit results, discuss the primary papers and all Q1–Q24. Spoken protocol: present a scenario, wait for a verbal hypothesis, request an explicit assumption, then work through the mathematics aloud before revealing the result. Pause for learner calculations; avoid reading raw code aloud during a walk.

Prerequisites: conditional probability, natural logs, gradients, Lagrange/Hölder inequality, rank-based metrics, train/validation/test separation, identity authorization, and the morning BM25, RAG-Sequence, FGSM and PGD material. **Definitions:** candidate recall = fraction of questions whose relevant evidence appears in retrieved top-k; MRR = mean reciprocal rank of first relevant result; *support precision* = fraction of asserted claims entailed by cited evidence; *threat-model validity* = whether proposed attacker modifications can occur in the actual system; *adaptive evaluation* = attacks that account for deployed defenses; *selective risk* = error rate among decisions not abstained.

# Part G — Retrieval as a probabilistic and security-sensitive system

## G1. Why retrieval accuracy alone is insufficient

Imagine a SOC analyst asks which containment procedure applies to a specific tenant and incident date. A top-1 retriever finds a beautifully relevant paragraph from another tenant. Relevance is high, **system correctness is zero**: unauthorized exposure is itself a failure. Another retriever finds an authorized but superseded procedure. Again semantic similarity is high, but validity is wrong. Therefore the desired event is a conjunction:
\[
S=A\cap V\cap R\cap C,
\]
where A = authorized, V = version-valid, R = relevant evidence retrieved, C = answer claims supported. By the chain rule, not an independence assumption,
\[
P(S)=P(A)P(V|A)P(R|A,V)P(C|A,V,R).
\]
For illustrative conditional rates 1.0, .98, .90, .85, success is .7497, **not** .90. The factors are not universally independent, and the illustrative numbers are not measured production results. A system with perfect recall can still have zero authorized-answer quality.

**Architecture and trust boundaries**
~~~text
[authenticated user + tenant + as-of date]
           |
[policy engine: allowed document IDs / version range] <--- signed policy configuration
           |
[ACL and validity filter] <--- immutable document metadata, version history
           |
[BM25 + dense candidate retrieval] -> [deduplicate] -> [rank fusion/rerank]
           |                                          |
     [evidence IDs, hashes, offsets, scores, provenance]
           |
[untrusted evidence in isolated context] -> [answer with per-claim citations]
           |
[deterministic support/authorization checks] -> [abstain or release]
           |
[optional action request] -> [separate permission check + human approval]
~~~
The retrieved text is data, not a policy or tool instruction. A document that says “ignore prior rules” is not allowed to cross the evidence-to-control boundary. Access checks must also cover caches, reranker inputs, observability logs and cited snippets. Post-generation redaction cannot undo an exposure to an external model provider.

## G2. RAG-Sequence gradients and a counterexample

Morning introduced Lewis et al.'s RAG-Sequence marginalization. For one training pair (x,y), let retrieved documents be z_j, retriever weights p_j=p_eta(z_j|x) with sum p_j=1 over the considered set, and sequence likelihoods \ell_j=p_theta(y|x,z_j). Then
\[
P(y|x)=\sum_j p_j\ell_j,\qquad L=-\log\sum_jp_j\ell_j.
\]
The *posterior responsibility* of passage j given the observed answer is
\[
r_j=\frac{p_j\ell_j}{\sum_h p_h\ell_h}.
\]
If p_j=softmax(a_j) for retrieval logits a_j, differentiating log-sum-exp gives
\[
\frac{\partial L}{\partial a_j}=p_j-r_j.
\]
Derivation: \(\partial p_i/\partial a_j=p_i(\mathbf 1_{i=j}-p_j)\); substitute into \(-\partial\log\sum_i p_i\ell_i\). Thus training raises the logit of a passage when its answer-conditioned responsibility exceeds its retrieval prior. For p=(.5,.5), \ell=(.9,.1), r=(.9,.1), gradients=(-.4,+.4). A descent step favors the first passage. **Caveats:** this derivative assumes a fixed candidate set and differentiable softmax over it. Top-k selection is discrete; corpus changes, access restrictions and unnormalized retrieval scores complicate training. The likelihood of a fluent answer is not evidence of source truth.

**Duplicate-document failure:** if three near-identical obsolete chunks occupy top-k, they can dominate the retriever's probability mass or context even when a single authoritative current chunk exists. Naively deduplicating by identical text can also be wrong when identical words belong to distinct tenants or versions; deduplicate only within the authorized set while retaining provenance. A generator trained on answer likelihood can reward a passage that enables memorized answers rather than genuinely supporting them. Distinguish answer likelihood from claim-level entailment.

## G3. BM25, dense retrieval, rank fusion and threshold failure

BM25 from morning uses inverse document frequency and saturated term frequency; its IDF depends on **which corpus** defines N and df. If authorization filtering changes the candidate universe, decide explicitly whether IDF comes from a shared index or per-tenant corpus. Global IDF can encode broad corpus statistics without exposing text, but may rank tenant-specific jargon poorly. A tenant-local index improves isolation but costs memory and management. Exact identifiers (CVE, ticket, runbook version) often deserve deterministic constraints rather than semantic approximation.

RRF for two lists is \(s(d)=1/(k+r_L(d))+1/(k+r_D(d))\). With k=60, ranks (1,10) give .030679; (3,3) give .031746, so consistency wins over one first-place hit. But **a high fused score is not a calibrated probability** of relevance, and absent documents need an explicit missing-rank policy (usually zero contribution). Reranking may recover relevance but cannot authorize an unauthorized candidate. Always filter first.

**Evaluation:** for questions q with gold relevant set G_q and retrieved top-k R_q, evidence recall@k can be \(\frac1Q\sum_q 1[G_q\cap R_q\ne\emptyset]\) for at-least-one-evidence success; if multiple facts are required, use \(\frac1Q\sum_q |G_q\cap R_q|/|G_q|\) and state the definition. MRR is \(\frac1Q\sum_q 1/r_q\) with 0 if no relevant item is found. These two measures can disagree: system A may find one item early but miss necessary second evidence. Evaluate citation entailment, wrong-version rate, tenant leakage, answer abstention and p95 latency separately. Group bootstrap by document family to avoid pretending overlapping questions are independent.

## G4. Worked failure and recovery

A query asks for the November 2026 incident procedure for tenant A. Corpus: A/2026/current; A/2025/obsolete; B/2026/current; A/2026/draft. A semantic retriever might favor B/2026 because of matching language. The valid policy must require tenant A, status approved and effective date containing the query's as-of date. If no approved version exists, **abstain**; never silently substitute draft or B. When an authorized chunk includes “call disable_user immediately,” the generator may summarize that instruction as evidence but must not execute the tool. Tool execution requires a distinct authorized workflow. Failure modes: metadata spoofing at ingestion, stale ACL caches, missing date ranges, cross-tenant shared embeddings, contradictory evidence, OCR corruption, and unsupported citations.

## G5. Advanced lab 1: RAG responsibility, provenance and evaluation

Python 3.10+, standard library, offline synthetic data. Save as \`rag_deep_lab.py\` and run \`python rag_deep_lab.py\`.
~~~python
import math
from collections import defaultdict

p=[.5,.5]; likelihood=[.9,.1]
evidence=sum(a*b for a,b in zip(p,likelihood))
r=[a*b/evidence for a,b in zip(p,likelihood)]
grad=[a-b for a,b in zip(p,r)]
assert all(abs(a-b)<1e-12 for a,b in zip(r,[.9,.1]))
assert all(abs(a-b)<1e-12 for a,b in zip(grad,[-.4,.4]))
print("posterior",r,"retriever gradient",grad)

docs=[
 {"id":"A26","tenant":"A","approved":True,"start":20260101,"end":20261231,"text":"preserve logs"},
 {"id":"A25","tenant":"A","approved":True,"start":20250101,"end":20251231,"text":"disable immediately"},
 {"id":"B26","tenant":"B","approved":True,"start":20260101,"end":20261231,"text":"disable immediately"},
 {"id":"AD","tenant":"A","approved":False,"start":20260101,"end":20261231,"text":"disable immediately"},
]
asof=20261116
allowed=[d for d in docs if d["tenant"]=="A" and d["approved"] and d["start"]<=asof<=d["end"]]
assert [d["id"] for d in allowed]==["A26"]
print("eligible evidence", [d["id"] for d in allowed])

# Two queries: one answerable, one with no evidence. Report coverage explicitly.
rankings=[["A26"],[]]
gold=[{"A26"},{"A26"}]
recall=sum(bool(set(a)&g) for a,g in zip(rankings,gold))/len(gold)
mrr=sum(next((1/(i+1) for i,d in enumerate(a) if d in g),0)
        for a,g in zip(rankings,gold))/len(gold)
assert recall==mrr==.5
print("recall@1",recall,"MRR",mrr)
~~~ 
Expected: posterior [0.9, 0.1], gradient approximately [-0.4, 0.4], eligible evidence ['A26'], recall@1 0.5, MRR 0.5. The empty second result is a deliberate abstention example, not a failed data fetch. Extensions: compare exact-match constraints with RRF; inject a forged ACL field and test that only a trusted ingestion policy can set it; add citation-support labels; run 100 grouped synthetic queries with confidence intervals. **Do not** equate this toy retrieval evaluation with actual LLM answer accuracy.

## G6. Primary-paper critique

Lewis et al. (2020, arXiv:2005.11401) propose a retriever-generator probabilistic model and compare RAG-Sequence with RAG-Token on knowledge-intensive tasks. Its benchmark design does not establish tenant isolation, correct document validity, or claim-level citation support. Robertson & Zaragoza (2009, DOI:10.1561/1500000019) give BM25's probabilistic relevance foundation; exact lexical features are useful but not sufficient for paraphrase. Cormack, Clarke & Buettcher (2009, DOI:10.1145/1571941.1572114) investigate reciprocal rank fusion; rank-level complementarity does not establish relevance calibration or security. **Research design:** 100 versioned SOC questions, three corpora with explicit authorization, pre-registered splits by runbook family, compare BM25/dense/RRF and reranking at fixed token/latency budget; report confidence intervals and zero-tolerance unauthorized exposure separately from answer quality.

# Part C — Robustness: what changes when feasibility is modeled correctly?

## C1. Derive robust risk using dual norms

For a differentiable loss \(L(x)\), Taylor's approximation gives \(L(x+\delta)\approx L(x)+g^\top\delta\), where \(g=\nabla_xL(x)\). For a p-norm constraint \(\|\delta\|_p\le\epsilon\), Hölder's inequality yields
\[
\max_{\|\delta\|_p\le\epsilon}g^\top\delta=\epsilon\|g\|_q,\quad 1/p+1/q=1.
\]
Thus \(p=\infty\) gives \(\epsilon\|g\|_1\), attained by FGSM's sign vector; \(p=2\) gives \(\epsilon\|g\|_2\), attained by \(\delta=\epsilon g/\|g\|_2\) for nonzero g; \(p=1\) gives \(\epsilon\|g\|_\infty\), attained by spending budget on the coordinate with largest absolute gradient. This is a **local first-order statement**, not a guarantee about the nonlinear loss or a realistic attacker.

If feasible changes satisfy linear constraints \(A\delta=0\), then for L2 budget the attacker can use only the component of g in the nullspace of A. With orthogonal projector \(P=I-A^\top(AA^\top)^{-1}A\) when rows are independent, the maximum first-order increase is \(\epsilon\|Pg\|_2\). If Pg=0, there is no first-order attack in the feasible tangent space, but second-order changes or discrete attacks can still matter. Nonlinear invariants and integer packet counts require different solvers; projection onto an Lp ball alone is insufficient.

## C2. Counterexamples and calibrated abstention

Morning score \(s=2x_1-x_2\) at x=(.6,.1) is 1.1. Independent box changes \(|\delta_i|\le .1\) can reduce it to .8. Coupled \(\delta_1=\delta_2\) only reduce to 1.0. Under strict alert if s>=1, the coupled attacker cannot flip it; under a **different** threshold 1.05, it can. Robustness claims must include the threshold and attack objective. A detector's abstention band [0.95,1.05] sends score 1.0 to review; it is not an automated benign decision, but it consumes analyst capacity.

For selective classification let coverage \(c=P(\text{not abstain})\) and selective risk \(R_{\rm sel}=P(\text{wrong}\mid\text{not abstain})\). Reducing c can trivially reduce R_sel by rejecting hard cases; report **both**, plus missed attacks among abstentions and review queue size. If prevalence is 1%, recall .9 and FPR .01, precision \(=.009/(.009+.0099)\approx .476\). At prevalence .1%, with same rates, precision falls to \(.0009/(.0009+.00999)\approx .0826\). These are hypothetical rates, not measured deployments. An adversarially robust detector that floods analysts may still be operationally unsafe.

## C3. Red-team experiment design and failure modes

An adaptive red team must specify attack goal (missed alert, tool misuse, information exposure), control (which log fields or retrieved documents), budget (query count, modifications, time), physical/protocol feasibility, defense knowledge and stopping rule. Split entities and capture periods before fitting scalers or attack constraints. Keep a clean holdout untouched. Attack the **deployed composite system** (normalization, detector, thresholds, abstention, analyst routing) in a sandbox, not merely a surrogate model. Use several seeds and budgets; a single failed PGD run may reflect gradient masking, not robustness. Include simple baselines, random search and exact enumeration where feasible. Compare fixed false-positive and review budgets. Do not train on the final test attacks.

**Defense-in-depth architecture**
~~~text
[owned synthetic events] -> [schema + causal invariants] -> [frozen preprocessing]
                 |                                   |
          [feasible attacker] -> [detector + threshold + abstention]
                                                     |
                              [analyst review / controlled incident action]
                                                     |
                              [versioned audit, calibration and rollback]
~~~
A signed dataset can still contain poisoned observations; a signed runbook can still be malicious. Cryptographic integrity and semantic trust are different. For cross-track RAG, test prompt injection only in synthetic retrieved text and enforce tool permission boundaries.

## C4. Advanced lab 2: feasible gradients, exact grid, prevalence

Python 3.10+, standard library. Save as \`robust_deep_lab.py\`.
~~~python
import math
from itertools import product
g=(2.,-1.)
eps=.1
l1=eps*sum(abs(v) for v in g)
l2=eps*math.sqrt(sum(v*v for v in g))
linf=eps*max(abs(v) for v in g)
assert all(abs(a-b)<1e-12 for a,b in zip((l1,l2,linf),(.3,.1*math.sqrt(5),.2)))
print("box/L2/L1 first-order maxima",round(l1,4),round(l2,4),round(linf,4))

x=(.6,.1)
score=lambda y:2*y[0]-y[1]
grid=(-eps,0.,eps)
all_d=list(product(grid,repeat=2))
coupled=[d for d in all_d if d[0]==d[1]]
minimum=lambda ds:min(score((x[0]+d[0],x[1]+d[1])) for d in ds)
assert abs(minimum(all_d)-.8)<1e-12
assert abs(minimum(coupled)-1.0)<1e-12
print("minimum scores",minimum(all_d),minimum(coupled))

def precision(prevalence,recall=.9,fpr=.01):
    return prevalence*recall/(prevalence*recall+(1-prevalence)*fpr)
assert abs(precision(.01)-(.009/.0189))<1e-12
assert precision(.001)<precision(.01)
print("precision at 1% and 0.1%",round(precision(.01),4),round(precision(.001),4))
~~~ 
Expected first-order maxima 0.3, 0.2236, 0.2; minimum scores 0.8 and 1.0; precision 0.4762 and 0.0826. The grid is exhaustive for the **declared discrete set**, not all real-valued perturbations; for this linear score its corners also establish the continuous box minimum. Extensions: replace the coupling by a conservation constraint \(\delta_1+\delta_2=0\); test integer packet counts; compare abstention risk/coverage curves; use an owned simulated stream to estimate analyst load. No live targets.

## C5. Critique, ethics and doctoral extensions

Madry et al. (2018, arXiv:1706.06083) frame robust training as min-max optimization and demonstrate strong first-order adversaries on vision benchmarks. Image Lp-balls do not automatically represent what an attacker can modify in telemetry. NIST AI 100-2e2025 (Vassilev et al., 2025, DOI:10.6028/NIST.AI.100-2e2025) offers threat taxonomy and mitigations, not a certification. **Research hypothesis 1:** rankings of detectors by robust recall change under protocol-constrained versus unconstrained perturbations. **Hypothesis 2:** abstention lowers unattended error but raises analyst burden nonlinearly at low prevalence. **Hypothesis 3:** ACL-first versioned RAG plus action gating reduces injection-induced tool misuse at matched legitimate completion rate. Each requires pre-registered controls, uncertainty intervals and failure-case reporting. Test only owned/synthetic data; avoid real credentials, customer logs, unauthorized endpoints and offensive deployment.

# Progressive oral examination: ask, wait, answer, probe

1. **What must be checked before vector search?** Tenant and date authorization. *Probe:* What about caches?
2. **Why isn't high recall enough?** It ignores version, access and grounding. *Probe:* Give a perfect-recall privacy failure.
3. **What is RAG's latent variable?** Retrieved passage z. *Probe:* What does marginalization average?
4. **Define p and r in the gradient.** Prior retrieval weight and answer-conditioned responsibility. *Probe:* Which is larger for useful evidence?
5. **Derive dL/da.** Softmax derivative yields p-r. *Probe:* Sign under gradient descent?
6. **Why can duplicates hurt?** They crowd candidate mass/context. *Probe:* When should identical text remain separate?
7. **Does BM25 return probabilities?** No, relevance ranking scores. *Probe:* What changes under per-tenant IDF?
8. **Why use RRF?** Rank-based fusion avoids incomparable score scales. *Probe:* Can it authorize evidence?
9. **Calculate (1,10) versus (3,3).** .030679 versus .031746. *Probe:* Which is truly relevant?
10. **Recall@k versus MRR?** Presence versus earliest relevant rank. *Probe:* Multiple necessary passages?
11. **Why is citation presence insufficient?** A cited passage may not entail the claim. *Probe:* Define support precision.
12. **What if no approved policy exists?** Abstain and identify missing evidence. *Probe:* Why not use draft?
13. **What is the dual norm of L-infinity?** L1. *Probe:* Derive sign-gradient optimum.
14. **For L2, which perturbation?** Epsilon times normalized gradient. *Probe:* What if gradient is zero?
15. **What does Aδ=0 mean?** Feasible perturbations obey invariants. *Probe:* Give a telemetry example.
16. **What is Pg?** Gradient projected into feasible nullspace. *Probe:* Does Pg=0 prove security?
17. **Why does coupling change score minimum?** Independent coordinate movement is unavailable. *Probe:* Derive 1.0.
18. **Can threshold changes invalidate robustness?** Yes. *Probe:* Compare 1.0 and 1.05.
19. **What is selective coverage?** Fraction of decisions not abstained. *Probe:* Why report risk jointly?
20. **How does prevalence affect precision?** Lower prevalence lowers PPV at fixed TPR/FPR. *Probe:* Recompute at 0.1%.
21. **Why is weak PGD evidence insufficient?** Optimization can fail or mask gradients. *Probe:* Alternative attack?
22. **Why evaluate the full pipeline?** Preprocessing and response gates alter feasible success. *Probe:* Where can leakage occur?
23. **What distinguishes integrity from trust?** Signatures attest origin/unchanged bytes, not benign content. *Probe:* Signed malicious runbook?
24. **Propose a falsifiable dissertation study.** Compare versioned ACL-first RAG and constrained robust detector at fixed task/analyst budgets with grouped held-out tests. *Probe:* What failure would refute the hypothesis?

## Recall test and source map

Without notes, derive the RAG retriever gradient; explain why duplicates distort evidence selection; compare recall@k and MRR; derive the dual-norm adversarial bound; calculate the coupled-feature minimum; calculate precision at 0.1% prevalence; articulate a defensible end-to-end red-team protocol. For each, state assumptions and a counterexample.

**Source map:** G1/G4 ACL and temporal-validity design: morning G1/G4, original syllabus GenAI Unit4 Session1, and local engineering argument; G2 probability and gradient: Lewis 2020 plus explicit local derivation; G3 lexical retrieval: Robertson & Zaragoza 2009; rank fusion: Cormack et al. 2009 plus local arithmetic; C1–C3 robust objective: Madry 2018 and local Hölder/nullspace derivation; C3/C5 attacker taxonomy and safety: NIST 2025; both labs and numeric scenarios are **synthetic local calculations**, not empirical paper findings. Complete paper index: [references/index.md](references/index.md).

## Reproducibility and limitations

Both labs are Python 3.10+ standard library, deterministic, offline and safe to rerun. Paper PDFs are not claimed present unless verified in GitHub; see companion index. No model training, production RAG evaluation, live intrusion or operational robustness claim. No email, audio, narration or workflow trigger.