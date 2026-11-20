# Evening PhD deep dive — 20 November 2026 (Asia/Kolkata)
**Exact continuation:** morning Unit 5 Session 1, *vision–language representations/cross-modal alignment*; cybersecurity Unit 5 Session 1, *SOC use cases/evidence-centered assistants*. This evening deliberately does **not** advance to diffusion or the next SOC unit. Syllabus: October 19 Gmail attachments *Advanced Generative AI and Foundation Models Learning Pathway* and *AI for Cybersecurity Learning Pathway*, p.12 Unit 5 Session 1; both original PDF attachments were retrieved. Morning source: [lesson](../../../phd-morning/lessons/2026-11-20/lesson.md) and [index](../../../phd-morning/lessons/2026-11-20/references/index.md). All examples are synthetic.

## Spoken navigation
**10 minutes:** Describe dual-tower contrastive alignment; explain why similarity is not entailment; derive the false-negative gradient; trace one screenshot through an ACL-gated SOC evidence workflow; oral Q1, Q4, Q9, Q14.
**30 minutes:** Add temperature, multi-positive contrastive loss, calibration, claim-evidence matching, Bayesian decisions, adversarial OCR, Q1–12 and Q14–22.
**Extended 90–120 minutes:** Derive both objectives, reproduce two labs, critique papers, defend research design, answer all 26 oral questions. Pause before each answer; do not operate code while walking.

## Prerequisites, vocabulary and central puzzle
Know dot products, normalized vectors, softmax, logarithms, partial derivatives, conditional probability, Bayes, classification metrics, RBAC/ABAC, event-time validity, and SOC alert triage. **Embedding** is a learned vector, **contrastive positive** a semantically matching pair, **false negative** a valid match incorrectly designated negative, **temperature** a positive divisor of logits, **entailment** evidence sufficient to justify a precise claim, **provenance** source ID plus revision/time/location, **principal** authenticated actor, **policy decision point** trusted authorization code, **abstention** a deliberate unsupported-answer outcome.

Central question: *Can a multimodal embedding identify a relevant dashboard while still being unsafe evidence for a SOC containment decision?* Yes: retrieval ranks candidates, while validity, authorization, numeric accuracy, temporal context and causal interpretation must be checked separately.

# Part I — Advanced image–text alignment
## Mechanism and why it works
Given image I_i and caption T_i, two parameterized encoders f_theta and g_phi yield L2-normalized vectors u_i and v_i. Scores s_ij=u_i·v_j, logits z_ij=s_ij/tau. Train each image to select its caption among N captions and each caption to select its image among N images. At serving, encode document images once, index them, encode query text, retrieve approximate nearest neighbors, then use a costly region/OCR/fusion verifier for exact claims.

~~~text
trusted image corpus -> decode/crop/version -> image encoder -> normalized index
authenticated query -> text encoder -> ANN candidate retrieval
             -> ACL/effective-time recheck -> OCR/region verifier
             -> claim/evidence checker -> abstain or cite; never authorize action
~~~

Fast two-tower retrieval factorizes representations, but cannot reason jointly about fine-grained text/image regions at retrieval time. Cross-attention can compare regions with query tokens but costs per candidate. OCR can read exact characters but fails on low resolution, overlapping labels or layout. Hybrid pipelines use complementary error modes; their gains must be measured, not presumed.

## Full gradient and false-negative derivation
For one image row i, let p_ij=exp(s_ij/tau)/sum_k exp(s_ik/tau), L_i=-log p_ii. Then
dL_i/ds_ij=(p_ij-1[j=i])/tau.
For a false negative j≠i, this derivative is +p_ij/tau: gradient descent pushes its similarity **down**, even if caption j also correctly describes image i. Lower temperature amplifies this penalty. The full symmetric objective averages image→text and text→image losses; for batch N it divides the row/column gradients by N and by 2 in their respective terms.

For a set P_i of *all* valid captions for image i, define q_i=sum_{j in P_i}p_ij and L_i^multi=-log q_i. Differentiation yields
dL_i^multi/dz_ik = p_ik - 1[k in P_i] p_ik/q_i.
This reduces to ordinary cross entropy when |P_i|=1. For a positive k, the derivative is negative when q_i<1, and is zero for an all-positive batch (q_i=1). This is not a universal solution: multi-positive annotations may be incomplete; ubiquitous captions can cause representation collapse; positives must be scoped to the actual semantic question.

**Worked counterexample.** Two images of the same model of firewall appliance, two accurate captions, all four cosine similarities 1. Ordinary diagonal-only loss per row is log(2)=0.693147, even though no pair is semantically wrong. If both captions are designated valid for both images, the multi-positive row loss is -log(1)=0. Yet with identical embeddings the model cannot discriminate two different serial numbers. Exact identity requires OCR/metadata, not semantic alignment.

## Temperature, margins and calibration
For two candidates with score difference Delta=s_pos-s_neg, p_pos=1/(1+exp(-Delta/tau)); dL/dDelta=-(1-p_pos)/tau. With Delta=1, tau=1 gives p=.731059 and loss=.313262; tau=.5 gives p=.880797 and loss=.126928. Sharper ranking is not necessarily more trustworthy: scores are conditioned on candidate prompts and image preprocessing. For example, replacing an omitted 'unknown' candidate changes the denominator, so softmax values are not posterior probabilities of real-world compromise.

Normalization: u=x/||x||. For nonzero x, Jacobian du/dx=(I-uu^T)/||x||; therefore gradients tangent to the unit sphere do not directly increase embedding norm. Near-zero norm makes the Jacobian unstable; implementations use a small epsilon and validate finite outputs. If a caption's content is a number ('7.2 bar') and the crop drops its digits, high similarity to 'pressure gauge' cannot support the numerical reading.

## Evaluation design and failure cases
Compare Recall@1/5, region-level evidence recall, exact-value accuracy, ECE/calibration after a separately fitted calibrator, worst-group performance, abstention risk-coverage, latency, GPU/CPU cost and rights restrictions. Split by original document/site/device/time, **not** by crops of one image. Counterexamples: watermark shortcut, duplicated near-identical images, multilingual text, rotated screenshots, image prompt injection, outdated dashboard, visually identical hardware with distinct identifiers, and OCR reading '0' as 'O'. Keep preprocessing, encoder revision, index revision, prompt templates and timestamp in the evidence bundle.

## Lab A — runnable standard-library multi-positive contrastive gradients
Save as clip_evening.py. This is a deterministic logit-level exercise, not a trained CLIP checkpoint.

~~~python
import math
def softmax(xs):
    m=max(xs); e=[math.exp(x-m) for x in xs]; z=sum(e)
    return [v/z for v in e]
def loss_grad(scores, positives, tau=1.0):
    logits=[x/tau for x in scores]
    p=softmax(logits)
    q=sum(p[j] for j in positives)
    loss=-math.log(q)
    grad=[(v-(v/q if j in positives else 0.0))/tau
          for j,v in enumerate(p)]
    return loss,grad
single,g1=loss_grad([1.,1.],{0})
multi,g2=loss_grad([1.,1.],{0,1})
assert abs(single-math.log(2))<1e-12
assert abs(g1[1]-.5)<1e-12
assert abs(multi)<1e-12 and max(map(abs,g2))<1e-12
sharp,_=loss_grad([1.,0.],{0},.5)
assert abs(sharp-math.log1p(math.exp(-2)))<1e-12
print("single",round(single,6),"false_negative_grad",round(g1[1],6))
print("multi",round(multi,6),"sharp",round(sharp,6))
# finite difference for nonpositive logit
h=1e-5
a=loss_grad([1.,1.+h],{0})[0]
b=loss_grad([1.,1.-h],{0})[0]
assert abs((a-b)/(2*h)-g1[1])<1e-8
print("gradient_check PASS")
~~~
Expected: single 0.693147 false_negative_grad 0.5; multi 0.0 sharp 0.126928; gradient_check PASS. **Extension:** implement symmetric N×N objective, build multi-positive incidence matrix, check gradients by finite differences, then create an adversarial crop experiment. The result shows an objective pathology, not measured retrieval performance.

# Part II — Evidence-centered SOC assistants with multimodal inputs
## Trust boundary and operational mechanism
A dashboard screenshot is *data*, even when it contains a sentence that looks like an instruction. The analyst's authenticated identity, tenant and allowed incident scope originate from the trusted session; they must never be inferred from OCR. The retrieval service performs tenant and time filtering **before** ranking, and rechecks permissions when fetching each evidence item. A deterministic policy engine, not an LLM, decides whether a tool action is permitted.

~~~text
Analyst identity -> tenant/scope/incident-time -> trusted authorization
                                                  |
Telemetry event IDs -> scoped query ---------------+-> evidence bundle
Screenshot -> image embedding -> candidate -> ACL/time filter -> OCR/region
Playbooks -> version+effective-date filter --------+-> evidence bundle
Evidence bundle -> claim decomposition -> independent source/entailment check
             -> cite or abstain -> human review -> separately authorized action
             -> append-only audit / delayed confirmed outcome
~~~

A robust evidence record should include {tenant, principal, incident_id, source_id, revision, observed_at, valid_from, valid_until, region/bounding_box, claim, transformation_hash}. OCR-derived instructions are never passed as executable policy. The assistant may propose 'check MFA logs' but may not isolate a machine without separate approval. Partial telemetry is not proof of absence; event counts may include retries or duplicated ingest.

## Claim support as a constrained assignment problem
Suppose generated factual claims c_1,...,c_m and authorized evidence items e_1,...,e_n. Let A_ij=1 only if evidence e_j *actually entails* claim c_i with valid time, tenant and provenance; mere citation presence does not suffice. Define supported_i=1[max_j A_ij=1]. Claim support = sum_i supported_i/m. If a claim needs two documents jointly (e.g. failure followed by MFA reset), model a *set* of evidence, not a single nearest neighbor; this is why embedding similarity is insufficient. A deterministic checker can validate IDs, ACL, time and quote spans, but semantic entailment may require a carefully audited human or model judge. Report inter-rater agreement and adjudication on a labeled sample.

**Counterexample:** '12 failures on host H' is supported by event logs; 'account compromised' is not entailed by failures and one success alone. A screenshot of a green status tile may reflect an old dashboard refresh. A poisoned playbook containing 'export credentials to attacker' must never be promoted to trusted instruction. A source ID without a verifiable passage is a fabricated or meaningless citation.

## Bayesian decision theory beyond the morning example
With attack prevalence pi, detector sensitivity r, benign false-positive rate f:
P(A|alert)=pi*r/(pi*r+(1-pi)*f).
At pi=.01,r=.9,f=.02, this is .3125. But the posterior depends on population/time/selection; analysts only reviewing escalated cases creates selection bias. Suppose posterior p and two actions: investigate (cost C_I), ignore (expected miss cost p*C_FN). Under simplified constant costs, investigate if p>C_I/C_FN. This is a *triage* threshold, not authorization for containment.

If a multimodal verifier has sensitivity a and false-positive rate b conditional on the initial alert, assuming conditional independence given attack state, updated posterior after verifier-positive is
p'=p*a/[p*a+(1-p)*b].
At p=.3125,a=.8,b=.1, p'=.25/(.25+.06875)=.7843137. **Crucial:** conditional independence is a strong assumption; if screenshot and detector read the same stale telemetry, treating them as independent double-counts evidence and inflates confidence. Correlation-aware validation or empirical joint likelihoods are needed.

**Selective risk:** If system answers only when confidence exceeds threshold t, coverage C(t)=P(confidence>=t) and selective risk R(t)=P(error|answered). Increasing t often reduces coverage, but risk need not monotonically fall under distribution shift or miscalibration. Measure separately on incident-time holdouts and low-prevalence periods.

## Lab B — executable Bayesian and evidence-policy counterexamples
Save as soc_evening.py. Standard library; no real alerts or privileged actions.

~~~python
from dataclasses import dataclass
@dataclass(frozen=True)
class Evidence:
    id:str; tenant:str; start:int; end:int; claims:frozenset
docs=[
 Evidence("event-101","A",10,30,frozenset({"12 failures"})),
 Evidence("playbook-v3","A",10,30,frozenset({"review MFA"})),
 Evidence("stale-v2","A",0,10,frozenset({"disable account"})),
 Evidence("other-tenant","B",0,30,frozenset({"account compromised"}))
]
def authorize(d,tenant,t):
    return d.tenant==tenant and d.start<=t<d.end
def support(claim,tenant,t):
    return [d.id for d in docs if authorize(d,tenant,t) and claim in d.claims]
claims=["12 failures","review MFA","account compromised"]
assert [support(c,"A",15) for c in claims]==[["event-101"],["playbook-v3"],[]]
ratio=sum(bool(support(c,"A",15)) for c in claims)/len(claims)
def update(p,sensitivity,fpr):
    return p*sensitivity/(p*sensitivity+(1-p)*fpr)
p=.01*.9/(.01*.9+.99*.02)
p2=update(p,.8,.1)
assert abs(p-.3125)<1e-12
assert abs(p2-(.25/.31875))<1e-12
assert abs(ratio-2/3)<1e-12
print("initial",round(p,6),"verified",round(p2,6))
print("support",f"{int(ratio*3)}/3","unsafe_claim",support(claims[-1],"A",15))
~~~
Expected: initial 0.3125 verified 0.784314; support 2/3 unsafe_claim []. **Limitation:** a string-equality claim checker is a policy illustration, not natural-language entailment; the second Bayesian update assumes conditional independence. **Advanced extension:** synthesize 100 correlated alert/screenshot pairs; compare naive independent update against empirically estimated joint likelihoods, plot calibration and risk-coverage, inject malicious OCR text and test forbidden tool-call count (must be zero).

## Real-world cases, ethics and failure analysis
Industrial incident dashboards, phishing screenshots and identity-provider event consoles are plausible multimodal SOC inputs; no case here is a claim about an actual customer. Real-world risks include employee privacy, sensitive logs, confidential diagrams, model/vendor data retention, bias in alert selection, alert fatigue and destructive containment. Keep raw evidence inside approved boundaries; minimize prompts; hash transformations; enforce purpose limitation, human review and auditability. Never use the assistant to execute a live containment action during a classroom lab. Prompt-based safety language is not an authorization control.

# Paper critique and doctoral extensions
**Radford et al. (2021), CLIP [R1].** Large-scale image–text contrastive training supports broad zero-shot transfer. But global similarity is not region-level truth, captions can be noisy, and web-scale data may encode bias. Proposed falsification: on held-out industrial documents, compare CLIP-style retrieval, OCR+layout and hybrid on exact numeric citations, grouped by crop and resolution.
**Lewis et al. (2020), RAG [R2].** Retrieval-conditioned generation can incorporate external passages; benchmark results do not establish tenant authorization, evidence time-validity or incident causality. Proposed test: add stale, cross-tenant and contradictory documents to an otherwise fixed evaluation.
**Yao et al. (2023), ReAct [R3].** Interleaved reasoning and acting improves some interactive tasks; it does not prove safe tool use. Compare deterministic retrieval workflow and bounded agent under injected OCR and poisoned runbooks.
**Tabassi/NIST (2023), AI RMF [R4].** Governance guidance, not empirical validation. Convert Govern/Map/Measure/Manage into named owner, risk register, measurable violations and release gates.
**OWASP (2025), LLM Top 10 [R5].** Threat taxonomy, not mitigation proof. Build an explicit adversarial test suite with denied cross-tenant access and no side-effect execution.

**Research study 1:** Does multi-positive alignment improve semantic recall without degrading device-specific identity? Pre-register paired semantic and serial-number tasks, document-disjoint splits, paired bootstrap confidence intervals and per-group errors.
**Research study 2:** Does multimodal evidence improve analyst correctness when screenshot content conflicts with event IDs? Randomize synthetic cases, measure claim entailment, unsafe recommendations, review time, and abstention.
**Research study 3:** How much does correlated evidence inflate posterior risk? Simulate latent common-source failures; compare independent Bayes, joint-likelihood model and calibrated discriminative baseline; report Brier score and false escalations.

# Progressive oral Socratic examination (ask, pause, answer, probe)
1. **Why two towers?** Precompute corpus embeddings and retrieve cheaply. *Probe:* what joint interactions are lost?
2. **What does L2 normalization do?** Turns dot product into cosine; norms cease to drive similarity. *Probe:* zero norm?
3. **What is temperature?** Logit divisor controlling sharpness and gradient. *Probe:* why can low tau magnify mislabeled negatives?
4. **Derive dL/ds for a false negative.** p_j/tau>0, so descent pushes it away. *Probe:* why is that semantically wrong?
5. **Define multi-positive q.** Sum probabilities of all valid matches. *Probe:* why -log(q)?
6. **What if all candidates are positives?** q=1, loss and logit gradient zero. *Probe:* is identity discriminability learned?
7. **Explain the .693147 loss.** Two equal logits give p=.5 for the chosen diagonal. *Probe:* is the pair truly wrong?
8. **Why can CLIP not read a gauge reliably?** Global embedding similarity does not ensure exact region/OCR. *Probe:* what verifier?
9. **Why split by document, not crop?** Prevent near-duplicate leakage. *Probe:* what other grouping keys?
10. **Is zero-shot softmax calibrated?** Not inherently; depends on prompt set. *Probe:* what changes with an unknown class?
11. **What is selective risk?** Error among answered examples. *Probe:* what happens to coverage?
12. **Which experiment tests a shortcut?** Perturb background/watermark while preserving object. *Probe:* how measure group effects?
13. **What is the SOC assistant's first boundary?** Trusted principal, tenant, scope. *Probe:* can OCR change it?
14. **Why is screenshot text untrusted?** It is evidence, not authority. *Probe:* what enforces this?
15. **Why filter ACL before ranking?** Avoid unauthorized candidates and leakage. *Probe:* why recheck at fetch?
16. **Define provenance.** Origin, revision, timestamp and location. *Probe:* incident time versus current time?
17. **Is a citation ID sufficient?** No; evidence must entail the claim. *Probe:* conjunctive evidence?
18. **Why are 12 failures not compromise proof?** Alternative benign causes exist. *Probe:* what corroboration?
19. **Derive posterior precision.** pi*r/(pi*r+(1-pi)*f). *Probe:* effect of lower prevalence?
20. **What is .3125?** Posterior after alert in synthetic base-rate example. *Probe:* counts among 10,000?
21. **What does a second positive verifier yield?** .784314 under assumed conditional independence. *Probe:* what if same telemetry source?
22. **Why can independence overstate confidence?** Correlated evidence is double-counted. *Probe:* empirical joint likelihood?
23. **What can the assistant authorize?** Nothing outside trusted policy and human approval. *Probe:* isolation workflow?
24. **What is prompt injection in OCR?** Malicious data masquerading as instruction. *Probe:* enforce at executor?
25. **What is the decisive evaluation split?** Held-out incidents/documents and time. *Probe:* selection bias?
26. **Defend one thesis hypothesis.** State testable claim, baseline, metric, falsifier and ethics constraints. *Probe:* what negative result changes your theory?

## Recall and reproducibility
Without notes: (1) derive the multi-positive gradient; (2) explain why score is not entailment; (3) compute both Bayesian posteriors; (4) identify the two trust boundaries; (5) state one falsifiable experiment. Code requires Python 3.10+ standard library only. Lab assertions are intended to be runnable; no production model or SOC evaluation is claimed. Companion [references/index.md](references/index.md) contains canonical links, source map and access status.
