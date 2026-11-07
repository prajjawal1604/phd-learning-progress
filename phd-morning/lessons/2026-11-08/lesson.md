# Morning PhD teaching package — 8 November 2026 (Asia/Kolkata)
**Syllabus provenance:** October 19, 2026 email attachments *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 2, Session 5: prompting and instruction tuning) and *AI for Cybersecurity Learning Pathway* (Unit 2, Session 3: malware analysis). This advances November 7's decoding and log-sequence topics by one session each. It is a teaching text for conversation, not a syllabus outline. All numerical examples are illustrative unless identified as empirical research.

## Spoken routes and prerequisites
**10 minutes:** Explain prompt versus model weights; derive cross-entropy on two instruction responses; distinguish static from dynamic malware evidence; answer oral Q1–Q5. **30 minutes:** Add structured-output validation, leakage and threat models, entropy and Bayesian alert thresholds, labs' worked examples, Q1–Q10. **Extended 90–150 minutes:** run both labs, critique papers, discuss Q1–Q16, draft the experimental protocol. Ask the tutor to pause after each question and probe for a counterexample.

You should know conditional probability, log likelihood, cross entropy, the autoregressive decoder and sampling (November 7); Python dictionaries, confusion matrices, event timestamps, safe handling of untrusted telemetry. Define: **prompt** = inference-time context/instructions; **few-shot** = demonstrations inside that context; **SFT** = supervised fine-tuning that changes model parameters; **schema validation** = deterministic check of generated structure; **static malware analysis** = inspection without execution; **dynamic analysis** = observation during controlled execution; **packing** = transformation/encapsulation that hides original file content; **sandbox** = isolated analysis environment with controlled resources. None of these terms imply a verdict by themselves.

# Track G — Prompting, instruction tuning and evidence-grounded output

## G1. Mechanism: what changes at inference and what changes at training?
A decoder represents P_θ(y_t | x, y_<t), where θ denotes learned weights, x includes system policy, user request, retrieved evidence and few-shot examples, and y_<t are previously generated tokens. A zero-shot instruction changes x, **not θ**. Few-shot examples also change x; they can prime output conventions or task mappings but cannot be assumed to teach a permanent capability. SFT constructs pairs (x_i,y_i) and updates θ by minimizing negative log likelihood: L(θ)=−(1/N) Σ_i Σ_t log P_θ(y_{i,t}|x_i,y_{i,<t}). Gradient step θ_new=θ−η∇_θL; η is learning rate. Teacher forcing supplies ground-truth prefix tokens during training; inference consumes model-generated prefixes, so errors can propagate. This is an exposure mismatch, not proof that SFT is harmful.

**Worked derivation:** For a correct response of two tokens with conditional probabilities .8 and .25, joint likelihood=.2; sequence NLL=−ln(.8)−ln(.25)=1.6094 nats; mean per-token NLL=.8047 and perplexity exp(.8047)=2.236. If training raises the second probability to .5 while holding the first fixed, joint likelihood=.4, NLL=.9163, mean=.4581. This does not establish better factuality: the probabilities describe the observed response, not independent evidence. At each token, cross entropy for a one-hot target k is −log softmax(z)_k; derivative with respect to logit z_j is p_j−1[j=k], where 1 is an indicator. For target k, the gradient lowers loss by increasing its relative logit. A dataset containing unsupported confident answers can train confident unsupported answers.

## G2. Zero-shot, few-shot and structured outputs
Consider an incident summarizer receiving three **trusted, verified** events: E1: host H authentication failed at 10:01; E2: H authentication succeeded at 10:03; E3: process P launched at 10:04. A weak prompt asks, "Was this an intrusion?" and may elicit certainty unsupported by the record. A bounded prompt requests: "Return JSON with evidence_ids, observed_facts, hypotheses, unknowns, confidence_limitations. Only cite E1–E3. Do not infer malicious intent from event sequence alone." The model can still hallucinate an ID or invent a claim. A parser can enforce field types; an evidence-ID allowlist can enforce references; **neither proves semantic entailment**. Human review or an independent claim-evidence verifier remains necessary.

**Mechanism:** (1) authorize evidence retrieval; (2) separate trusted instructions from untrusted log content; (3) provide stable evidence IDs and timestamps; (4) choose deterministic decoding for repeatable tests (even then runtime differences may exist); (5) parse and validate; (6) compare claims with source spans; (7) abstain/escalate when sources are incomplete. Prompt injection in log messages is data, not policy. Use schema-constrained decoding when available, but still validate after generation.

```text
trusted task + approved evidence IDs ----> prompt assembler
untrusted log text ----------------------> quoted evidence segment
                                               |
                                     decoder + fixed settings
                                               |
                                     JSON parse + schema checks
                                               |
                                evidence ID and claim verification
                                               |
                                  analyst review / abstention
```

**Failure modes and trade-offs:** longer prompts consume context and prefill compute; few-shot demonstrations may anchor to wrong examples; strict JSON improves integration but can hide semantic errors; SFT can improve instruction-following on representative tasks but requires curated rights-cleared data, training resources and evaluation against forgetting and contamination. A prompt-only approach is cheap to iterate and easy to audit; SFT amortizes task behavior but risks regressions. Neither substitutes for retrieval authorization or a reference evaluator.

### Lab G — local, runnable prompt-output validation (Python 3.10+, standard library)
This lab tests a deterministic output gate, **not** a language model. It demonstrates the boundary between syntactic validity, allowed citations and factual verification.
```python
import json
ALLOWED = {"E1", "E2", "E3"}
REQUIRED = {"evidence_ids", "observed_facts", "hypotheses", "unknowns"}
def validate(raw):
    try: obj = json.loads(raw)
    except json.JSONDecodeError: return False, "invalid JSON"
    if not isinstance(obj, dict) or set(obj) != REQUIRED:
        return False, "schema mismatch"
    if not isinstance(obj["evidence_ids"], list) or not all(
        isinstance(x,str) and x in ALLOWED for x in obj["evidence_ids"]
    ): return False, "invalid evidence ID"
    if not all(isinstance(obj[k], list) and all(isinstance(v,str) for v in obj[k])
               for k in ("observed_facts","hypotheses","unknowns")):
        return False, "invalid field types"
    return True, "structure and references valid; facts NOT verified"
good = json.dumps({"evidence_ids":["E1","E2"],"observed_facts":["A failed login preceded a success"],
                   "hypotheses":["May be benign"],"unknowns":["Whether the actor was authorized"]})
bad = good.replace('"E2"', '"E99"')
assert validate(good)[0] is True
assert validate(bad) == (False,"invalid evidence ID")
assert validate('not json') == (False,"invalid JSON")
print(validate(good)); print("PASS")
```
**Expected:** `(True, 'structure and references valid; facts NOT verified')`, then `PASS`. Counterexample: replacing observed_facts with "Malware confirmed" passes this gate but is unsupported. **Extension:** build a separate gold claim-to-evidence table and compute claim precision = supported claims / all asserted claims, coverage = required supported facts stated / all required facts. Do not conflate these with string-format compliance. Compare zero-shot, two-shot and SFT on the same held-out incidents with blinded adjudication; stratify by missing telemetry.

## G3. Paper critique and research design
Ouyang et al. (2022, arXiv:2203.02155) study training language models to follow instructions with human feedback; reported improvements are conditioned on their data, evaluation and model setup and are not proof of universal safety. Wei et al. (2022, arXiv:2109.01652) examine instruction-finetuned models and zero-shot generalization; benchmark gains need scrutiny for task overlap and prompt sensitivity. Brown et al. (2020, arXiv:2005.14165) document in-context learning at scale, not gradient updates caused by examples. **Falsifiable hypothesis:** at fixed model and evidence, two-shot examples improve evidence-ID correctness over zero-shot, but may increase unsupported assertions if examples contain confident interpretations. Pre-register evidence correctness, unsupported-claim rate, latency, token cost, and subgroup shifts; sample by incident, not repeated paraphrases.

# Track C — Static versus dynamic malware analysis

## C1. What each observation can and cannot establish
Static analysis inspects bytes without executing the sample: cryptographic hashes for identity, file type and header consistency, import tables, embedded strings, sections and byte entropy. A SHA-256 match can identify an exact previously known file but says nothing about an unseen variant; byte entropy can indicate compression, encryption or packed data but cannot distinguish benign archives from malicious packing. Dynamic analysis executes a sample only in a controlled, authorized sandbox and observes process tree, file changes, registry/config changes and network attempts. It reveals behavior triggered on that run, not every possible behavior: malware may sleep, require a trigger, check environment fingerprints or avoid instrumented sandboxes. Static analysis is fast, scalable and non-executing but vulnerable to obfuscation; dynamic analysis is more behavior-rich but costly, risky and coverage-limited.

**Worked example:** A file has 2,048 bytes, with 1,024 zeros and 1,024 bytes of value 255. P(0)=P(255)=.5, Shannon byte entropy H=−Σ_b p(b)log₂p(b)=−2(.5log₂.5)=1 bit/byte, not 8. If all 256 byte values appear equally, H=8 bits/byte. For a 90%-zero and 10%-255 sample, H≈.469 bits/byte. High entropy is **not** a calibrated probability of malware. A benign compressed executable may be high entropy, and a malicious script may be low entropy.

**Worked decision economics:** Suppose a *calibrated* posterior q=P(malware|evidence), false-positive quarantine cost C_FP=2, missed-malware cost C_FN=98. Quarantine when (1−q)2 < q98, or q>2/100=.02 under a simplified two-action loss model. If q is merely a raw anomaly score, this calculation is invalid. For a base rate of 1 malicious file in 1,000, sensitivity=.9 and specificity=.99: expected per 1,000,000 files =900 true positives, 9,990 false positives, precision=900/(900+9990)≈8.26%. A high sensitivity does not ensure a useful alert queue at low prevalence.

## C2. Operational architecture and failure analysis
```text
authorized file intake -> quarantine / immutable hash / metadata
                              |
                   static features + signatures
                              |
                risk triage + provenance ledger
                              |
              isolated sandbox (no production credentials)
                              |
          process / filesystem / network observations
                              |
                 correlation + calibrated score
                              |
              analyst adjudication + safe retention
```
Do not execute unknown binaries on a personal laptop or corporate workstation for this course. Use only synthetic byte arrays and published, rights-cleared metadata. A production sandbox needs isolation, egress policy, disposable images, monitored timeouts, legal authorization, and incident procedures. Preserve hashes, timestamps, versions and data provenance; do not collect private user documents without a lawful purpose. A sample that evades dynamic execution is *unknown*, not benign.

### Lab C — static byte entropy and alert precision (safe synthetic data)
```python
import math
from collections import Counter
def entropy(blob):
    if not blob: return 0.
    counts=Counter(blob); n=len(blob)
    return -sum((c/n)*math.log2(c/n) for c in counts.values())
assert abs(entropy(bytes([0]*1024+[255]*1024))-1.)<1e-12
assert abs(entropy(bytes(range(256)))-8.)<1e-12
assert entropy(bytes([0]*2048))==0.
N=1_000_000; prevalence=.001; sensitivity=.9; specificity=.99
tp=N*prevalence*sensitivity
fp=N*(1-prevalence)*(1-specificity)
precision=tp/(tp+fp)
assert tp==900
assert abs(fp-9990)<1e-7
assert .082<precision<.083
print("entropy",entropy(bytes([0]*1024+[255]*1024)))
print("TP",int(tp),"FP",round(fp),"precision",round(precision,4))
print("PASS")
```
**Expected:** `entropy 1.0`; `TP 900 FP 9990 precision 0.0826`; `PASS`. This is a synthetic thought experiment, not a validated detector. **Ablations:** increase prevalence tenfold; vary specificity from .99 to .999; compare benign high-entropy compressed data to malicious low-entropy scripts. **Failure test:** a malicious packed file and a benign archive may share the same entropy; no scalar threshold can reliably separate them without additional evidence.

## C3. Papers, critique and doctoral extension
Anderson & Roth (2018, arXiv:1804.04637) introduce EMBER, a labeled dataset for static PE malware detection and baseline models; it supports reproducible feature studies but labels, temporal shifts and executable format scope limit generalization. Saxe & Berlin (2015, arXiv:1508.03096) examine deep neural detection from binary features; evaluate feature robustness and data leakage before generalizing. MITRE ATT&CK's Malware knowledge base catalogs observed techniques, not a prevalence-calibrated detector or ground truth for arbitrary samples. **Research question:** Does adding controlled dynamic behavioral metadata improve time-split detection beyond static features when packed benign software is common? Evaluate by file-family grouping and chronological holdout, count analyst reviews, record sandbox failures as missing observations, and assess false positives for signed software. Do not claim dynamic observations were collected in this lesson.

# Cross-track case: LLM-assisted malware triage
An analyst has a static scan (hash, import list, entropy), a sandbox report (attempted outbound connection), and a log sequence. The assistant must produce a *hypothesis with cited evidence*, not a binary verdict. A malicious sample may contain strings like "ignore previous instructions" that are **untrusted file contents**. The retrieval layer strips secrets, preserves source IDs and rejects unsupported citations; the decision engine keeps quarantine policy separate from the LLM. If sandbox telemetry is missing, say "not observed", not "did not happen". Pair the G validator with the C evidence ledger and evaluate unsupported-claim rate under adversarial strings.

# Misconceptions, ethics and practical boundaries
Prompting is not weight training; syntactically valid JSON is not factually valid; temperature does not calibrate truth; high byte entropy is not malware probability; a clean sandbox trace is not proof of safety; a hash is an identity fingerprint, not a behavior classifier. Secure file custody and authorization come before experimentation. Protect sensitive indicators, avoid unauthorized sample acquisition, and use redacted examples in third-party models. Human review is required for disruptive actions.

# Progressive oral Socratic questions — pause before answers
1. **What does few-shot change?** Input context, not weights. *Follow-up:* when would weights change?
2. **Write the SFT objective.** Sum negative log conditional token probabilities over examples. *Follow-up:* explain teacher forcing.
3. **Calculate .8×.25.** .2 joint probability. *Follow-up:* NLL in nats? 1.6094.
4. **Why can SFT improve fluency but not truth?** It imitates training targets, including mistakes. *Follow-up:* propose a claim-level metric.
5. **What is a schema validator good for?** Types, fields, permitted IDs. *Follow-up:* give a semantically false valid JSON example.
6. **What is prompt injection here?** Instructions embedded in untrusted logs or file strings. *Follow-up:* where should authorization be enforced? Outside the model.
7. **Contrast zero-shot and few-shot.** Instruction alone versus instruction plus demonstrations. *Follow-up:* what confounds comparisons? Prompt length, examples.
8. **What is static analysis?** Non-executing byte/metadata inspection. *Follow-up:* name two features and limitations.
9. **What is dynamic analysis?** Controlled observation during execution. *Follow-up:* why might behavior be absent? Evasion, trigger, timeout.
10. **Derive byte entropy for two equally likely values.** 1 bit/byte. *Follow-up:* can it diagnose maliciousness? No.
11. **Why can packed benign software fool a detector?** High entropy and atypical structure are nonspecific. *Follow-up:* how test? Benign packed controls.
12. **What does a SHA-256 match prove?** Byte identity to a reference sample (subject to practical collision assumptions), not intent. *Follow-up:* how handle variants?
13. **Derive the quarantine threshold.** Compare (1−q)C_FP with q C_FN, yielding q>C_FP/(C_FP+C_FN). *Follow-up:* what if q uncalibrated?
14. **Why is precision only ~8.26%?** Very low prevalence produces many false positives despite 99% specificity. *Follow-up:* change specificity to 99.9%.
15. **Why chronological, family-aware splits?** Reduce future leakage and near-duplicate memorization. *Follow-up:* what remains confounded?
16. **Design a joint study.** Compare prompting regimes on fixed evidence from static-only vs static+dynamic triage; evaluate support, abstention, precision and latency. *Follow-up:* how blind human reviewers?

# Reading, listening, source map and reproducibility
**Primary research:** [G1] Brown et al., *Language Models are Few-Shot Learners* (2020), arXiv:2005.14165, https://arxiv.org/abs/2005.14165; [G2] Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (2022), arXiv:2109.01652, https://arxiv.org/abs/2109.01652; [G3] Ouyang et al., *Training language models to follow instructions with human feedback* (2022), arXiv:2203.02155, https://arxiv.org/abs/2203.02155; [C1] Anderson & Roth, *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models* (2018), arXiv:1804.04637, https://arxiv.org/abs/1804.04637; [C2] Saxe & Berlin, *Deep Neural Network Based Malware Detection Using Two Dimensional Binary Program Features* (2015), arXiv:1508.03096, https://arxiv.org/abs/1508.03096; [C3] MITRE ATT&CK Malware knowledge base, https://attack.mitre.org/software/ (living official reference, version/date should be recorded in any experiment).
**Tutorial:** Dive into Deep Learning (https://d2l.ai/) for gradients and cross-entropy; Python `json` documentation (https://docs.python.org/3/library/json.html) for schema parsing limitations; scikit-learn precision/recall guide (https://scikit-learn.org/stable/modules/model_evaluation.html) for evaluation. **Listening:** The TWIML AI Podcast (https://twimlai.com/podcast/) and SANS Internet Storm Center podcast (https://isc.sans.edu/podcast.html); choose relevant episodes and verify availability, rather than claiming a specific episode was reviewed.

**Claim-to-source map:** autoregressive/SFT math → derivation G1 and [G2,G3]; in-context learning → G1 and [G1]; structured-output and ID validation → Lab G (local demonstration, not an external efficacy claim); malware static features and dataset limitations → C1/C3 and [C1,C2]; entropy calculations and base-rate precision → Lab C; malware technique taxonomy → [C3]; cross-track LLM triage → proposed architecture and threat-model reasoning, not a published evaluation. Both private syllabus PDFs determine session selection, not empirical claims. The [companion reference index](references/index.md) gives paper summaries, rights/access status and stable URLs.

**Research log template:** date; hypothesis; sample provenance; split manifest; model/checkpoint; prompts and decoding parameters; Python version; code revision; labels and adjudicator; raw confusion counts; failures; next test. No external research PDFs are claimed as downloaded. No fresh November 2026 news is asserted without independent verification. No email, audio, or narration requested.
