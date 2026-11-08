# Evening PhD deep dive — 8 November 2026 (Asia/Kolkata)
**Dependency checked:** Read committed [morning lesson](../../../phd-morning/lessons/2026-11-08/lesson.md) and its [reference index](../../../phd-morning/lessons/2026-11-08/references/index.md) on main before writing. Both October 19 syllabus attachments were identified in the Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*: *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 2, Session 5) and *AI for Cybersecurity Learning Pathway* (Unit 2, Session 3). This evening remains on **prompting/SFT/structured evidence** and **static/dynamic malware analysis**. The attachments' full contents were not independently re-extracted in this run; session mapping is grounded in the committed morning package.

**Mode:** Spoken interactive teaching text, not an outline. All numerical examples are constructed, not reported field measurements. **Safety:** synthetic data only; never execute suspicious binaries.

## Walking routes
**10 minutes:** Tell the distinction between conditional prompting and gradient updates; calculate two-token NLL; explain why an allowed evidence ID does not entail a claim; derive why malware precision collapses at low prevalence; answer Q1–Q5. **30 minutes:** Work through the logit gradient, counterexample to prompt grounding, Bayes likelihood-ratio test, entropy caveats, and Q1–Q10. **Extended 90–150 minutes:** run both labs, audit counterexamples, compare research designs and finish Q1–Q18. Tutor: ask each question first, wait for the learner's explanation, then reveal the answer and probe the follow-up.

## Prerequisites and precise vocabulary
Know conditional probability, softmax, logarithms, confusion matrices, Python 3.10+, and the morning lesson's static/dynamic distinction. A **token** is a model prediction unit; **logit** is an unnormalized score; **cross entropy** penalizes low probability on the target; **teacher forcing** trains on ground-truth previous tokens; **distribution shift** changes the data generating process; **calibration** means predicted probability approximately matches empirical frequency within appropriate groups; **likelihood ratio** compares how probable an observation is under two hypotheses; **evidence entailment** asks whether cited observations actually support a claim; **packer** wraps or transforms executable content, sometimes benignly; **sandbox evasion** means behavior depends on the analysis environment. A static score is not a probability until empirically calibrated.

# Track G: prompting, SFT, and grounded structured outputs — deeper mechanisms

## G1. Why examples in a prompt differ from supervised updates
At inference, the decoder computes p_θ(y_t|x,y_{<t}). θ is fixed; x is the trusted instruction plus examples and bounded untrusted evidence; y_{<t} is generated prefix. Few-shot examples modify x and hence attention activations and next-token probabilities, but do not update θ. During supervised fine-tuning, with examples (x_i,y_i), the loss is L(θ)=−(1/M) Σ_i Σ_t log p_θ(y_{it}|x_i,y_{i,<t}); M is number of target tokens (normalization choices vary). Optimizer updates θ. This distinction matters for governance: prompt revision is reversible per request, SFT affects future requests to that checkpoint.

**Derive the gradient.** For vocabulary logits z_j, softmax p_j=e^{z_j}/Σ_k e^{z_k}. For target index c, ℓ=−log p_c=−z_c+log Σ_k e^{z_k}. Therefore ∂ℓ/∂z_j=p_j−1[j=c]. Example p=[0.7,0.2,0.1], c=1 (zero-indexed): gradient [0.7,−0.8,0.1]. A gradient descent step in logit space increases the target logit and decreases the others. This is not an actual weight update: logits depend on a large neural network and parameter gradients use the chain rule. If a false but fluent target is labeled correct, this objective still rewards it.

**Two-token worked case:** p(y1)=0.8, p(y2|y1)=0.25, joint=0.2, sequence NLL=−ln(0.2)=1.60944 nats; mean NLL=0.80472; token perplexity exp(0.80472)=2.23607. Raising the second conditional probability to 0.5 changes joint to 0.4, NLL to 0.91629. Yet likelihood is about imitation, not the truth of the sentence. At inference, an incorrect first token changes the conditioning prefix: this is exposure mismatch. Scheduled sampling is one proposed response but has objective and consistency caveats; do not assume it is a universal fix.

**When each approach works:** zero-shot for clear standardized tasks; few-shot for unfamiliar formats and consistent conventions; SFT when enough rights-cleared representative examples exist and repeated prompting costs dominate. **When it fails:** demonstrations bias toward a spurious label; SFT memorizes a contaminated split; output JSON passes schema but invents a causal link; an injected log line is promoted into a system instruction. Measure factual support and authorization separately from format accuracy.

## G2. Evidence-grounding is a two-stage verification problem
Imagine E1: failed authentication for account A at 10:01; E2: successful authentication for A at 10:03; E3: process P started at 10:04 on host H. The statement "E1 preceded E2" is directly supported. "E1 proves credential theft" is **not** entailed. "E3 is malicious" is **not** entailed. Even if a JSON object's evidence_ids=["E1","E2"], it can assert an unsupported conclusion. A citation allowlist checks *referential integrity*, not *semantic entailment*. The trusted policy must govern data access and actions outside the language model.

Architecture:
```text
identity / policy -> authorized evidence fetch -> immutable event IDs + provenance
                                                  |
untrusted strings -> inert quoted fields ---------+--> constrained model generation
                                                         |
                                        JSON/schema + allowlist + exact-claim verifier
                                                         |
                                        calibrated abstention / analyst adjudication
                                                         |
                                        audited downstream action (separate authority)
```
A structured decoder may reduce malformed syntax; an external schema validator still checks types and IDs. A claim verifier can be a curated rule set, a separate human annotation, or a carefully tested entailment model, each with distinct failure modes. A model judging itself is not independent ground truth.

**Research protocol:** 2×2 factorial design: prompt zero-shot/two-shot × evidence complete/one critical event omitted; optional SFT third arm only if training data are truly independent. Hold model, decoding settings, event inventory and split fixed. Report (i) valid JSON fraction; (ii) valid evidence IDs fraction; (iii) supported-claim precision = supported asserted claims / asserted factual claims; (iv) required-fact recall; (v) unsupported-claim count; (vi) abstention appropriateness; (vii) latency and tokens. Split by incident and organization to prevent paraphrase leakage. Blind adjudicators to prompt arm. Threat-model injection embedded in log strings, but never permit log contents to issue privileged commands.

### Advanced Lab G: explicit claim entailment gate and controlled counterexamples
**Python 3.10+, standard library; offline.** This toy verifier supports only *pre-registered exact claims*, not natural-language semantic inference.
```python
import json, math
EVENTS = {
 "E1": {"kind":"auth_fail","minute":1,"actor":"A"},
 "E2": {"kind":"auth_success","minute":3,"actor":"A"},
 "E3": {"kind":"process_start","minute":4,"actor":"H"},
}
ALLOWED = {
 "E1": {"Account A failed authentication at minute 1"},
 "E2": {"Account A succeeded authentication at minute 3"},
 "E3": {"A process started on H at minute 4"},
}
def gate(raw):
    try: obj=json.loads(raw)
    except (ValueError,TypeError): return False,"parse"
    if not isinstance(obj,dict) or set(obj)!={"claims","unknowns"}: return False,"schema"
    if not isinstance(obj["claims"],list) or not isinstance(obj["unknowns"],list): return False,"types"
    if not all(isinstance(s,str) for s in obj["unknowns"]): return False,"types"
    for claim in obj["claims"]:
        if not isinstance(claim,dict) or set(claim)!={"text","evidence_ids"}: return False,"schema"
        ids=claim["evidence_ids"]
        if not isinstance(ids,list) or not ids or not all(isinstance(i,str) and i in EVENTS for i in ids):
            return False,"reference"
        if not isinstance(claim["text"],str): return False,"types"
        if not any(claim["text"] in ALLOWED[i] for i in ids): return False,"unsupported"
    return True,"supported by toy rulebook"
def payload(text,ids):
    return json.dumps({"claims":[{"text":text,"evidence_ids":ids}],"unknowns":["intent"]})
assert gate(payload("Account A failed authentication at minute 1",["E1"]))[0]
assert gate(payload("Credential theft proven",["E1","E2"]))==(False,"unsupported")
assert gate(payload("Account A failed authentication at minute 1",["E99"]))==(False,"reference")
assert gate("not json")==(False,"parse")
print("G PASS",gate(payload("Credential theft proven",["E1"])))
```
**Expected:** `G PASS (False, 'unsupported')`. **Counterexample:** "A process started on H at minute 4" paired with E1 alone is rejected even though E3 exists: attribution matters. **Limitation:** exact-string allowlisting cannot handle valid paraphrases or claims combining evidence; its false-negative rate could be high. **Extension:** build a versioned claim graph with temporal edges and independently labeled entailment pairs; estimate adjudicator agreement and test adversarial paraphrases.

## G3. Paper critique
Brown et al. (2020, *Language Models are Few-Shot Learners*, arXiv:2005.14165) document in-context performance at scale; no inference-time gradient step follows. Critique: benchmark selection, task-specific prompting and scaling confound attribution to a single mechanism. Wei et al. (2022, *Finetuned Language Models Are Zero-Shot Learners*, arXiv:2109.01652) demonstrate instruction-tuning transfer; critique task overlap, model size dependence and prompt sensitivity. Ouyang et al. (2022, *Training language models to follow instructions with human feedback*, arXiv:2203.02155) combine supervised and preference-based steps; their reported preferences are not proof of security or source entailment. A rigorous extension evaluates **unsupported claims per 100 generated factual claims** and shifts in abstention under missing evidence, not merely user preference.

# Track C: static and dynamic malware triage — deeper statistical decisions

## C1. Feature semantics and what an analyst can infer
Static observations include file magic/header consistency, section sizes, import features, strings, cryptographic hash and byte entropy. Dynamic observations, from a properly authorized isolated environment, may include process ancestry, attempted connections, persistence changes and timing. Neither is a universal oracle: compressed legitimate software may look packed; malicious scripts may have low entropy; a sleeping sample may show no behavior in a short sandbox run. A missing sandbox result must be encoded **missing**, not benign.

**Entropy derivation:** For byte probabilities p_b, H(X)=−Σ_{b=0}^{255} p_b log₂ p_b, with 0 log 0 defined as 0. A fair mixture of 0x00 and 0xFF yields 1 bit per byte; all 256 values equally probable yields 8. Entropy alone is invariant to many byte permutations and loses program semantics. Two files with identical byte histograms can have very different execution behavior.

**Bayes threshold derivation:** Let M denote malware, B benign, observation x, prior π=P(M), sensitivity distribution p(x|M), benign distribution p(x|B). Posterior odds:
P(M|x)/P(B|x)=[p(x|M)/p(x|B)]·[π/(1−π)].
Under a two-action cost model, quarantine has expected loss C_FP·P(B|x); allow has expected loss C_FN·P(M|x). Quarantine iff posterior q>P threshold=C_FP/(C_FP+C_FN), equivalently likelihood ratio Λ(x)>[C_FP/C_FN]·[(1−π)/π]. For π=.001, C_FP=2 and C_FN=98, LR threshold ≈20.39. These costs are stylized and omit investigation capacity, incident severity and legal impacts. A score trained on a balanced dataset is **not** automatically a deployment posterior.

**Base-rate counterexample:** At π=.001, sensitivity=.90 and specificity=.99, in one million files TP=900, FP=9990, precision=900/10890≈.08264. Improving specificity to .999 gives FP=999 and precision=900/1899≈.47393. In other words, improving specificity by 0.9 percentage points changes alert precision dramatically at low prevalence. Always include absolute queue volume and uncertainty intervals; don't present a synthetic calculation as a measured system.

## C2. Controlled fusion and missingness
A conceptual pipeline:
```text
authorized sample intake -> immutable hash + provenance -> static features
                                          |                    |
                                          +--> isolated sandbox + observed/not-observed/missing
                                                               |
                                 versioned fusion + calibration + threshold by cost
                                                               |
                                  human analyst -> evidence ledger -> audited decision
```
If static and dynamic observations are conditionally independent given class, their likelihood ratios multiply; in practice shared packer artifacts and sandbox evasion can make that assumption false. Naively multiplying correlated evidence overstates confidence. Test fusion on temporally held-out families and on cases where dynamic telemetry is deliberately absent. Never treat "no observed connection" as "no connection possible". Separate isolation failures from behavior absence. Real-world defensive scenario: a signed, compressed enterprise installer triggers high entropy and an unfamiliar network endpoint; a legitimate update can explain both. Conversely, a low-entropy script with delayed execution may evade a short sandbox run. These are constructed case studies, not specific attributed incidents.

### Advanced Lab C: prevalence shift, calibration and missing telemetry
**Python 3.10+, standard library; synthetic counts, no malware.**
```python
import math
def metrics(n, prevalence, sensitivity, specificity):
    tp=n*prevalence*sensitivity
    fp=n*(1-prevalence)*(1-specificity)
    return tp,fp,tp/(tp+fp)
tp,fp,ppv=metrics(1_000_000,.001,.9,.99)
assert abs(tp-900)<1e-9 and abs(fp-9990)<1e-6
assert abs(ppv-900/10890)<1e-12
_,fp2,ppv2=metrics(1_000_000,.001,.9,.999)
assert abs(fp2-999)<1e-6
assert .47<ppv2<.48
def posterior(prior,lr):
    odds=(prior/(1-prior))*lr
    return odds/(1+odds)
assert abs(posterior(.001,1)-.001)<1e-12
assert posterior(.001,30)>.02
def decide(q,cfp=2,cfn=98):
    return "quarantine" if (1-q)*cfp < q*cfn else "review/allow"
assert decide(posterior(.001,30))=="quarantine"
assert decide(posterior(.001,10))=="review/allow"
# Missing telemetry is not a negative observation:
def combine(static_lr,dynamic_lr=None):
    return static_lr if dynamic_lr is None else static_lr*dynamic_lr
assert combine(5,None)==5
print("C PASS",round(ppv,4),round(ppv2,4),round(posterior(.001,30),4))
```
**Expected:** `C PASS 0.0826 0.4739 0.0292`. **Failure exercise:** set static_lr=5 and dynamic_lr=6 but suppose both arise from the same packed feature. Multiplication produces 30 without independent evidence: an invalid confidence boost. **Ablations:** change π from .001 to .01; stratify by signed software and packer presence; represent missing sandbox result as an explicit category and compare calibration error. **Evaluation:** chronological train/test split, malware-family group holdout, exact hashes deduplicated, confusion counts, calibration curves, analyst workload and model version logged. This is a synthetic probability lab, not executable malware analysis.

## C3. Paper critique and research extension
Anderson & Roth (2018, *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637) provide a reproducible static Windows PE feature benchmark; critique: scope limited to PE, collection/label bias, drift, potential family or time leakage. Saxe & Berlin (2015, *Deep Neural Network Based Malware Detection Using Two Dimensional Binary Program Features*, arXiv:1508.03096) motivate learned static representations; critique: robustness to packing and unseen families requires dedicated testing. MITRE ATT&CK's software catalog is a descriptive taxonomy, not a prevalence estimate or a universal ground-truth label. **Doctoral hypothesis:** calibrated static+dynamic fusion improves family-held-out precision at fixed analyst capacity relative to static-only, but improvement shrinks under realistic sandbox missingness. Pre-register thresholds, costs, split manifests, confidence intervals, signed benign controls, and the exact missing-data mechanism.

# Cross-track research: LLM-assisted triage without decision laundering
An LLM receives only authorized redacted static features and sandbox observations with provenance IDs. The LLM generates an evidence-linked **hypothesis**, not a malware verdict. An external verifier checks IDs and support; a separate calibrated classifier or analyst applies quarantine policy. Malicious strings embedded in samples are untrusted content. The LLM must not receive production credentials or permission to run the sample. Research question: does two-shot evidence-grounded prompting improve analyst utility without increasing unsupported assertions, especially when sandbox data are missing? Randomize prompt arm across matched incidents; use blinded adjudication and report abstention, false certainty, time saved and errors by subgroup.

## Misconceptions and ethics
A citation is not proof; valid JSON is not correct analysis; lower cross-entropy is not higher factuality; high entropy is not a malware posterior; no observed sandbox behavior is not a clean bill of health; a SHA-256 hash identifies bytes, not intent. Obtain permission and rights before handling malware or sharing telemetry. Keep raw sensitive logs out of external LLMs, retain only necessary data, and require human authorization for disruptive actions. Do not test unknown binaries outside controlled approved infrastructure.

## Progressive Socratic dialogue (ask first, then reveal)
1. **Does few-shot update θ?** No, it changes context x. *Probe:* what would make θ change?
2. **Derive ∂ℓ/∂z_j.** p_j−1[j=c]. *Probe:* why does log-sum-exp appear?
3. **Compute NLL of .8×.25.** 1.60944 nats. *Probe:* why not call it factuality?
4. **What is exposure mismatch?** Training sees true prefixes; inference sees sampled prefixes. *Probe:* give an error cascade.
5. **What does a JSON schema guarantee?** Structure, not truth. *Probe:* invent valid unsupported text.
6. **What does an evidence-ID allowlist guarantee?** Valid references, not entailment. *Probe:* how verify a claim?
7. **Why can two-shot underperform zero-shot?** Wrong demonstrations, anchoring, length. *Probe:* design an ablation.
8. **What does SFT optimize?** Teacher-forced token likelihood. *Probe:* how could contamination inflate scores?
9. **Why split by incident?** Avoid near-duplicate leakage. *Probe:* what about organization shift?
10. **Why abstain?** Evidence may be insufficient; confidence is not proof. *Probe:* measure abstention quality.
11. **What does 1 bit/byte mean?** Two equally probable byte values. *Probe:* is the file malicious?
12. **Why is 8 bits/byte not a verdict?** Benign compressed/encrypted data can be uniform. *Probe:* give a control.
13. **Write posterior odds.** Likelihood ratio × prior odds. *Probe:* where does prior come from?
14. **Derive cost threshold.** q>C_FP/(C_FP+C_FN). *Probe:* what if costs vary by host?
15. **Explain 900 TP versus 9990 FP.** Low base rate dominates alert queue. *Probe:* improve specificity.
16. **Why not multiply two LRs blindly?** Features may be dependent. *Probe:* show double counting.
17. **What does missing sandbox data mean?** Unknown observation, not negative. *Probe:* what if missingness is adversarial?
18. **Design a publishable study.** Chronological family-aware split, static-only vs fusion, calibrated thresholds, independent evidence grading, missingness and subgroup ablations. *Probe:* which result would falsify your hypothesis?

## Reading, tutorial, listening and source map
Primary references: [G1] Brown et al. (2020), *Language Models are Few-Shot Learners*, arXiv:2005.14165, https://arxiv.org/abs/2005.14165; [G2] Wei et al. (2022), *Finetuned Language Models Are Zero-Shot Learners*, arXiv:2109.01652, https://arxiv.org/abs/2109.01652; [G3] Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155, https://arxiv.org/abs/2203.02155; [C1] Anderson & Roth (2018), *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637, https://arxiv.org/abs/1804.04637; [C2] Saxe & Berlin (2015), *Deep Neural Network Based Malware Detection Using Two Dimensional Binary Program Features*, arXiv:1508.03096, https://arxiv.org/abs/1508.03096. Official taxonomy: [C3] MITRE ATT&CK software https://attack.mitre.org/software/. Tutorials: https://d2l.ai/ (softmax, cross entropy); https://docs.python.org/3/library/json.html (JSON parser); https://scikit-learn.org/stable/modules/calibration.html (calibration). Listening directories, no episode claimed verified: https://twimlai.com/podcast/ and https://isc.sans.edu/podcast.html.

**Source map:** prompting vs weight updates → G1 and [G1,G2]; SFT gradient → G1 local algebra and [G2,G3]; claim verification limitations → G2 + Lab G (toy demonstration); malware features and dataset scope → C1/C3 and [C1,C2]; entropy and posterior decisions → C1 local derivation and Lab C; technique taxonomy → [C3]; cross-track pipeline → proposed threat model, **not** empirical validation. Companion [references index](references/index.md) lists stable links, access notes and local paper status. Research log should record date, sample provenance, split manifest, parameters, version, adjudicators, confusion counts, calibration and unresolved limitations. No specific current-news claim is made without live verification.

**Validation status:** Code contains assertions and expected outputs; execution status is documented separately. No research PDF is claimed downloaded. No email, narration, or audio is part of this preparation.
