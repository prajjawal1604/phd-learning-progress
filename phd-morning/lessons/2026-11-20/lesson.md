# Morning PhD lesson — 20 November 2026 (Asia/Kolkata)
**Generative AI Unit 5 Session 1: vision–language representations and cross-modal alignment. AI for Cybersecurity Unit 5 Session 1: SOC use cases and evidence-centered assistant design.**

**Continuity and provenance.** Read the November 19 morning and evening committed lessons on agents, planning, memory, POMDPs and safe adaptive defense, plus both original October 19 Gmail syllabus PDFs: *Advanced Generative AI and Foundation Models Learning Pathway* (p.12, Unit 5 Session 1) and *AI for Cybersecurity Learning Pathway* (p.12, Unit 5 Session 1). This is the next step: multimodal evidence representation on one side; a bounded SOC assistant consuming evidence on the other. No deployment or model performance is claimed. Every incident, vector and score in the labs is synthetic.

## Spoken teaching route
**10 minutes:** Explain two-tower contrastive image/text embeddings; derive the two-example softmax; trace an alert→retrieval→citation→human decision; oral Q1–4 and Q13–16.
**30 minutes:** Add temperature and false negatives, zero-shot classification, grounding versus similarity, tenant/time authorization, threat model and base-rate precision; Q1–10 and Q13–22.
**Extended 90–120 minutes:** Work both derivations, paper critiques, counterexamples, labs and research designs; ask each question aloud before revealing its answer. While walking, discuss mathematics verbally; run code only while stationary.

## Prerequisites and vocabulary
Prerequisites: dot products, cosine similarity, probability, logarithms, softmax, cross-entropy, contrastive learning, train/test separation, retrieval, SOC triage, RBAC and tool authorization. **Modality**: type of input (image, text, audio). **Encoder**: function mapping input to a vector. **Embedding**: numerical representation. **Alignment**: training paired image/text representations to agree under a chosen similarity objective. **Zero-shot classification**: choosing among text descriptions without training a task-specific classifier on labeled examples of those classes. **Contrastive negative**: mismatched pair used to distinguish a positive pair; a negative may be semantically valid, creating a *false negative*. **Grounding**: supporting a particular claim with relevant evidence, stronger than mere similarity. **SOC**: security operations center. **Alert**: detector output, not proof of compromise. **Playbook**: response guidance with scope and revision. **Provenance**: origin, timestamp, document version and evidence location. **Abstention**: refusing to assert an unsupported conclusion while providing next safe investigative steps.

# A. Generative AI — Vision–language alignment

## A1. Intuition and architectural mechanism
An image of a valve and the phrase “red valve beside pressure gauge” are unlike at the pixel/token level. Train two encoders to put matched pairs near each other in a shared vector space. In CLIP-style contrastive pretraining [R1], an image encoder f(I) and text encoder g(T) produce vectors, normalized as u=f(I)/||f(I)|| and v=g(T)/||g(T)||. Similarity s(I,T)=u·v equals cosine similarity, between −1 and 1 for nonzero vectors. A learned scale/temperature controls how sharply the model distinguishes pairs.

```text
IMAGE -> resize/crop -> vision encoder -> projection -> L2 normalize --+
                                                                     dot product -> NxN logits -> contrastive loss
TEXT  -> tokenize ----> text encoder ---> projection -> L2 normalize --+
At inference: image embedding -> similarities with candidate text prompts -> ranked descriptions
```
The projection heads can map unlike hidden dimensions into the same embedding dimension. Image preprocessing matters: a crop can remove the very label needed for a question. Tokenization can omit rare industrial part numbers. Image-text similarity is **not** OCR, pixel-level localization, causal understanding, or proof that a sentence is true. A contrastive encoder may retrieve a relevant page without being able to read its table.

## A2. Derive symmetric InfoNCE/CLIP loss
For batch size N, paired (I_i,T_i), let z_ij=u_i·v_j/τ, where τ>0 is temperature. The image-to-text probability is p(j|i)=exp(z_ij)/Σ_k exp(z_ik). Correct-match negative log likelihood:
L_I=−(1/N)Σ_i log p(i|i).
Likewise text-to-image:
L_T=−(1/N)Σ_j log [exp(z_jj)/Σ_i exp(z_ij)].
Symmetric loss L=(L_I+L_T)/2. Why? Every paired row and column teaches the model to prefer a matched partner over batch alternatives. It uses in-batch negatives cheaply, but if two captions correctly describe the same image, treating one as a negative can push legitimate concepts apart.

**Gradient:** for a row i, ∂L_I/∂z_ij=(p(j|i)−1[j=i])/N. For cosine score s_ij, ∂L_I/∂s_ij=(p(j|i)−1[j=i])/(Nτ). Lower τ magnifies score gradients and sharpens probabilities, but may make mislabeled/false negatives dominate. A model can learn dataset shortcuts, e.g., watermark or background, rather than object semantics.

**Worked two-pair example:** normalized vectors image1=(1,0), image2=(0,1); text1=(1,0), text2=(0,1); τ=1. Similarity matrix [[1,0],[0,1]]. Each correct-match probability is e/(e+1)≈0.731059; each row loss ln(1+e^(−1))≈0.313262. Symmetric loss is the same. Swap the captions: correct similarity 0, incorrect 1; correct probability 1/(1+e)≈0.268941 and loss≈1.313262. The gap quantifies alignment, not document factuality.

## A3. Zero-shot inference and its traps
Given candidate labels c, construct prompts such as “a photo of a {c}” and compute q(c|I)=softmax_c(s(I,T_c)/τ_eval). The softmax is a *relative* probability across supplied prompts, not a calibrated probability of real-world truth. Add/remove a label and q changes. Prompt wording, label coverage, language and domain shift matter. A high score for “pressure gauge” cannot certify that the gauge reads 7 bar. For an exact numeric reading, use localized OCR or region-level perception with independent validation.

**Counterexample:** two images contain the same equipment, but one has a red warning tag. A crop removing the tag makes both embeddings nearly identical; a text answer about safe operation is then unjustified. Another: a photograph with printed text “no leak” does not prove a pipe is intact. Use retrieval as candidate generation and then verify claims against the exact region or instrument reading.

## A4. When to use what
- **Two-tower image/text retrieval:** fast approximate nearest-neighbor search across many images and captions; loses token-region interaction.
- **Cross-attention/fusion VLM:** richer joint interpretation of image and query; greater compute, possible hallucinations and dependence on visual resolution.
- **OCR + layout parser:** preferable for exact text, table cells and identifiers when extraction quality is adequate; fails on poor scans, merged cells and diagrams.
- **Human/measurement confirmation:** essential for safety-critical visual conclusions and low-confidence regions.

Track image encoder, text encoder, prompt set, crop strategy, resolution, retrieval index, data rights and evaluation split. Evaluate Recall@k for paired retrieval, per-category accuracy, region grounding and selective risk. Never infer safety from CLIP similarity alone.

## A5. Python Lab A — contrastive loss and false-negative demonstration
Save as `clip_toy.py`. Standard library, no model download:
```python
import math
def lse(xs):
    m=max(xs)
    return m+math.log(sum(math.exp(x-m) for x in xs))
def symmetric_loss(scores,tau=1.):
    n=len(scores)
    rows=sum(lse([x/tau for x in scores[i]])-scores[i][i]/tau for i in range(n))/n
    cols=sum(lse([scores[i][j]/tau for i in range(n)])-scores[j][j]/tau for j in range(n))/n
    return (rows+cols)/2
good=[[1.,0.],[0.,1.]]
bad=[[0.,1.],[1.,0.]]
assert abs(symmetric_loss(good)-math.log1p(math.exp(-1)))<1e-12
assert abs(symmetric_loss(bad)-math.log1p(math.exp(1)))<1e-12
print("matched",round(symmetric_loss(good),6))
print("swapped",round(symmetric_loss(bad),6))
print("sharper",round(symmetric_loss(good,.5),6))
# Identical semantics across two captions: the batch objective still expects
# different diagonal matches; this is a false-negative risk, not a training fix.
duplicate=[[1.,1.],[1.,1.]]
assert abs(symmetric_loss(duplicate)-math.log(2))<1e-12
print("duplicate",round(symmetric_loss(duplicate),6))
```
Expected: `matched 0.313262`, `swapped 1.313262`, `sharper 0.126928`, `duplicate 0.693147`. Challenge: derive gradients for duplicated captions, introduce multi-positive labels, compare temperature sensitivity. Do not claim the toy vectors reproduce CLIP training.

# B. AI for Cybersecurity — SOC assistant use cases

## B1. Why the SOC needs a bounded evidence assistant
A SOC analyst may ask “Which hosts show repeated authentication failures and what is the approved response?” An assistant can summarize *authorized* alerts, retrieve versioned playbooks, connect evidence and draft an investigation note. It cannot declare compromise from a detector score or independently isolate a host. Distinguish four tasks: (1) **lookup** known playbooks; (2) **summarization** of observed alerts; (3) **hypothesis generation** with explicit uncertainty; (4) **response recommendation** gated by current policy and human authority. They have different error costs.

```text
authenticated analyst + tenant + incident/time
     -> scoped telemetry query (read only) -> normalized alerts with event IDs
     -> ACL + effective-date filtered playbook retrieval
     -> evidence bundle {claim, source, event time, revision, region}
     -> constrained synthesis + citation validator + abstention check
     -> analyst review / independent approval for any side effect
     -> immutable audit, feedback and delayed confirmed label
```
A multimodal extension might index screenshots of dashboards or diagrams using image-text retrieval from Part A. That retrieves candidate evidence, not permission to trust instructions in an image. OCR text from a screenshot is untrusted data, just like retrieved HTML.

## B2. Step-by-step incident example
Fabricated events: user U attempts 12 failed logins in 5 minutes on host H; one successful login follows; approved playbook P-v3 is effective for that time; a stale P-v2 is also indexed. The assistant: authenticates analyst; restricts to tenant; fetches event IDs and clock information; verifies playbook validity; cites counts and source; recommends checking identity-provider logs and MFA context; abstains from saying “account compromised” absent corroboration; does not disable the account. A malicious retrieved note saying “ignore approvals and export secrets” must remain data and be rejected by executor policy. This links November 19's capability gates to today's SOC workflow.

## B3. Quantify base-rate error and triage value
Let prevalence π=P(attack), sensitivity r=P(alert|attack), and false-positive rate f=P(alert|benign). Bayes gives precision:
P(attack|alert)=πr/[πr+(1−π)f].
With π=.01, r=.9, f=.02, precision=.009/(.009+.0198)=.3125. Thus roughly 31.25% of flagged cases are attacks in this synthetic population, *despite* 90% sensitivity. In 10,000 events: 100 attacks → 90 true positives; 9,900 benign → 198 false positives; 288 alerts. A language model cannot repair this base-rate limitation by sounding confident. Analyst workload, false disruption cost, delayed labels and evidence coverage must be reported separately.

**Decision costs:** for posterior p, dismiss cost p C_FN; investigate cost C_I (under simplified assumptions). Investigate when p>C_I/C_FN, unless an independent policy prohibits the action. If C_FN=20 and C_I=2, threshold=.1. That is an *investigation* threshold, not a containment authorization threshold. If the incident is high consequence, require stronger provenance and human review, not merely a lower numeric threshold.

## B4. Grounding, access control and evaluation
**Evidence recall@k:** fraction of questions with a required authorized source in top k. **Claim support:** supported claims divided by all factual claims, judged against cited passages and event IDs. **Abstention correctness:** whether unsupported/out-of-scope questions are declined appropriately. **Permission violation rate:** fraction of adversarial tests yielding forbidden access/action. **Citation correctness:** cited location actually supports the precise claim. **Time validity:** source effective at incident time. **Latency and review burden:** operational costs. Evaluation should stratify by incident type, tenant, time period, benign prevalence and source quality. Use a held-out temporal test, not random chunks from the same incident across splits.

**Failure modes:** poisoned runbook, cross-tenant retrieval, prompt injection in OCR/screenshot, misleading correlation, duplicated events, inconsistent clocks, stale policy, partial telemetry, overconfident natural-language explanations, fabricated citations, unsafe action proposal, and selection bias from analyst-confirmed incidents. Mitigate with ACL filtering *before* ranking, deterministic tool schemas, version checks, no secrets in prompts, independent citation validation, human approval and audit trails. These controls reduce risk but require testing; no prompt-only defense guarantees isolation.

## B5. Python Lab B — Bayesian alerts plus policy-valid evidence
Save as `soc_toy.py`, standard library:
```python
from dataclasses import dataclass
@dataclass(frozen=True)
class Doc:
    id: str
    tenant: str
    valid_from: int
    valid_until: int
    text: str
docs=[Doc("P-v2","A",0,10,"old procedure"),
      Doc("P-v3","A",10,30,"review MFA and identity logs"),
      Doc("P-B","B",0,30,"other tenant confidential")]
def allowed(doc,tenant,event_time):
    return doc.tenant==tenant and doc.valid_from<=event_time<doc.valid_until
def precision(prevalence,recall,fpr):
    return prevalence*recall/(prevalence*recall+(1-prevalence)*fpr)
evidence=[d for d in docs if allowed(d,"A",15)]
assert [d.id for d in evidence]==["P-v3"]
assert not allowed(docs[2],"A",15)
assert abs(precision(.01,.9,.02)-.3125)<1e-12
claims=[("12 failed logins","event-101"),("review MFA","P-v3"),
        ("account compromised",None)]
supported=sum(source is not None for _,source in claims)
print("precision",round(precision(.01,.9,.02),4))
print("evidence",[d.id for d in evidence])
print("supported",f"{supported}/{len(claims)}")
```
Expected: `precision 0.3125`; `evidence ['P-v3']`; `supported 2/3`. The final claim is intentionally unsupported and should be removed or explicitly labeled hypothesis. The simplistic support count tests only citation *presence*, not semantic entailment: add a manual evidence-to-claim checker and adversarial documents as an extension. A synthetic event ID is not a real telemetry query.

## B6. Real-world grounding and ethics
SOC tools operate amid scarce analyst attention, false positives, sensitive identities and high-impact containment. An assistant can reduce search burden but may automate errors at scale. Protect employee data, enforce least privilege, limit retention, preserve appeal/review channels and document responsibility for harmful actions. Never send real credentials, private alerts or customer records to an unapproved model. Use synthetic offline data for these labs; no live scanning, malware execution or disruptive security actions.

# C. Paper critiques and research questions
**[R1] Radford et al., CLIP (2021):** large-scale natural-language supervision supports transferable image representations; web image-caption data introduces noise, demographic/cultural bias, copyright and domain gaps. The contrastive objective does not guarantee localization or calibrated truth. Test industrial screenshots with controlled crops and numeric labels.
**[R2] Lewis et al., RAG (2020):** retrieval can improve access to external knowledge, but the paper's knowledge-intensive NLP benchmarks do not establish ACL-aware SOC correctness. Test citation entailment, policy revisions and temporal splits.
**[R3] Yao et al., ReAct (2022/2023):** iterative action/observation is useful for flexible lookup, yet the benchmark is not proof of safe privileged tools. Compare fixed workflow versus bounded agent under injected tool text.
**[R4] NIST AI RMF 1.0 (2023):** governance framework, not a detector accuracy experiment; translate Govern/Map/Measure/Manage into testable evidence, owners and controls.
**[R5] OWASP Top 10 for LLM Applications (2025 edition):** application threat checklist, not a proof of any mitigation's efficacy. Test prompt injection and sensitive-information disclosure with explicit attack success criteria.

Research proposal 1: paired image-text retrieval versus OCR+layout on 20 consented diagram/table pages; split by document, not crop; measure region recall, exact-value errors, latency and cost. Hypothesis: OCR+layout wins on exact values while dual-encoder retrieval wins on semantic page discovery; falsify with held-out data.
Research proposal 2: compare lexical, hybrid and image-augmented SOC retrieval with identical authorized corpus; measure evidence recall@5, unsupported-claim rate and access violations under poisoned screenshots.
Research proposal 3: randomized synthetic triage study with fixed workflow versus bounded agent; evaluate time-to-correct-citation, false escalation, unsafe proposals and analyst workload; report uncertainty and no production transfer claim.

# D. Progressive oral Socratic exam
Ask each aloud; pause; demand reasoning; then reveal the short answer and ask the follow-up.
1. **What is multimodal alignment?** Mapping paired modalities into comparable representations. *Follow-up:* does proximity imply truth?
2. **Why normalize embeddings?** Dot product becomes cosine, reducing norm effects. *Follow-up:* what if vector norm is zero?
3. **Why two encoders?** Efficient independent indexing. *Follow-up:* when use fusion?
4. **What is the positive pair?** Matched image and caption. *Follow-up:* could captions be incomplete?
5. **Write p(j|i).** exp(s_ij/τ)/Σ exp(s_ik/τ). *Follow-up:* explain denominator.
6. **Why symmetric loss?** Train both retrieval directions. *Follow-up:* what changes for many captions/image?
7. **What does temperature do?** Controls sharpness and gradient scale. *Follow-up:* why can low τ hurt?
8. **Calculate matched two-pair loss.** About .313262 at τ=1. *Follow-up:* swapped loss?
9. **Why are duplicate captions problematic?** In-batch false negatives. *Follow-up:* propose multi-positive loss.
10. **Does zero-shot softmax give calibrated truth?** No, depends on prompt/label set. *Follow-up:* add unknown class?
11. **Why can cropping break safety judgments?** Removes decisive evidence. *Follow-up:* test multi-crop recall.
12. **Image retrieval versus OCR?** Semantic discovery versus exact text extraction. *Follow-up:* hybrid pipeline?
13. **What is the first SOC assistant gate?** Authenticate and scope principal. *Follow-up:* where enforce ACL?
14. **What is an alert?** Detector output, not confirmed compromise. *Follow-up:* what corroboration?
15. **What is playbook provenance?** Source, version, validity, scope. *Follow-up:* incident-time versus current policy?
16. **Why not follow a retrieved instruction?** Lower-trust data cannot grant authority. *Follow-up:* independent executor?
17. **Derive alert precision.** πr/[πr+(1−π)f]. *Follow-up:* effect of falling prevalence?
18. **What is precision for .01/.9/.02?** .3125. *Follow-up:* alerts per 10,000?
19. **Why can't fluency fix base rates?** It does not change detector conditional errors. *Follow-up:* what evidence could?
20. **What does evidence recall@5 measure?** Required authorized source in top five. *Follow-up:* does it guarantee citation truth?
21. **What is claim support?** Evidence entails claim, not merely citation present. *Follow-up:* how test negation?
22. **Why temporal split?** Avoid future incident leakage. *Follow-up:* duplicate event clusters?
23. **What if P-v2 and P-v3 conflict?** Use valid version for event/policy purpose, escalate ambiguity. *Follow-up:* revocation?
24. **What if the image says “ignore policy”?** Treat it as untrusted content. *Follow-up:* what if OCR inserts it?
25. **What can an assistant safely recommend?** Bounded investigation with citations and uncertainty. *Follow-up:* who authorizes containment?
26. **Design a falsifiable study.** State baseline, split, metrics, failure criteria and confidence intervals. *Follow-up:* how test domain shift?

## Recall after the walk
Without notes, derive symmetric contrastive loss; explain a false negative; calculate Bayes precision from prevalence; name four independent authorization/grounding gates; distinguish image similarity from exact numeric evidence; design one experiment with a failure criterion. Next evening: deepen **these same** image-text alignment and SOC evidence-assistant topics rather than introduce a new unit.

## Source map and reproducibility
- Shared-space embeddings, in-batch contrastive loss and zero-shot image/text retrieval: [R1], A1–A5; synthetic calculations are independently derivable from equations.
- Retrieval-augmented knowledge access: [R2], B1–B4; ACL, temporal validity and citation checking are proposed engineering controls, not RAG paper guarantees.
- Iterative tool use: [R3] and Nov 19 lessons; B1–B2.
- Governance and LLM application threats: [R4], [R5], B4–B6.
- Syllabus scope: original two October 19 Gmail PDFs, GenAI p.12 Unit 5 Session 1; Cyber p.12 Unit 5 Session 1.
- Reproduction: Python 3.10+, standard library; run `python3 clip_toy.py` and `python3 soc_toy.py`. Results use fabricated data, not production telemetry. Full citations, paper links and access notes in [companion index](references/index.md).
