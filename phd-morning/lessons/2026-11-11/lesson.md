# Morning PhD teaching package — 11 November 2026 (Asia/Kolkata)

**Syllabus provenance:** The two PDFs attached to the 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027* were read. *Advanced Generative AI and Foundation Models Learning Pathway*, p.8, Unit 3 Session 2, calls for supervised adaptation, instruction tuning, target masks, and comparison with retrieval. *AI for Cybersecurity Learning Pathway*, p.6, Unit 2 Session 6, calls for cross-task cases and benchmark validity; p.7 specifies detector comparison, inference latency, and structured IOC extraction. **Continuity:** 10 November covered pretraining corpus quality and threat-intelligence graphs. Today advances to **supervised fine-tuning (SFT) and instruction datasets**, and **cross-task security benchmarking and dataset shift**. Examples below are synthetic; no experimental effectiveness claims are made.

## Spoken teaching route
**10-minute walk:** define SFT versus pretraining and retrieval; explain masking; compare network/malware/phishing detection units; calculate one precision and latency example; answer oral Q1–Q5. **30-minute walk:** derive token loss and gradient, walk through the data and evaluation pipeline, analyze temporal/entity leakage, and answer Q1–Q11. **Extended 90–150 minutes:** implement both labs, critique the primary sources, complete the full Socratic exam, and preregister two research studies. A tutor should ask questions one at a time, pause for an answer, then reveal the worked answer and follow-up. Read equations verbally; do not dictate code character by character.

## Prerequisites and glossary
Recall autoregressive factorization, cross-entropy, softmax, gradients, document-group splitting, precision, recall, FPR, calibration, and threat intelligence provenance. **Instruction example** = input instruction plus context and desired response. **SFT** = gradient updates on labeled input/output examples. **Teacher forcing** = conditioning on reference preceding tokens during training. **Target mask** = loss indicator deciding which token positions count. **RAG** = retrieval-augmented generation; inference-time external evidence without necessarily updating weights. **Domain shift** = train and deployment distributions differ. **Unit of analysis** = what a prediction describes (flow, file, URL, host-window, incident). **Macro-F1** = unweighted mean of per-class F1. **Point-in-time join** = features available no later than the decision timestamp. **Selective prediction** = abstaining when evidence or confidence is inadequate.

# Track G — Supervised adaptation and instruction tuning

## G1. Why SFT changes behavior rather than supplying guaranteed facts
Pretraining learns broad token regularities from unlabeled sequences. SFT uses carefully curated examples such as: user asks “Give a JSON incident summary with cited evidence”; assistant responds with the desired schema, a cautious evidence interpretation, and abstention when evidence is absent. The same base model can learn to follow a schema, but a fluent output may still be factually unsupported. SFT modifies parameters; RAG adds documents at answer time. SFT is appropriate for stable response behavior and recurring tasks with quality labels; RAG is appropriate for volatile manuals, revocable documents and citation-grounded facts. Both may be needed. Fine-tuning cannot repair absent ground truth, poor labels, unauthorized data or an evaluation contaminated by paraphrases.

**End-to-end architecture:**
```text
authorized task spec + answer rubric
  -> provenance/consent + redact secrets -> group/temporal split
  -> instruction template + tokenizer -> input/response boundary
  -> causal shift + response-only target mask -> token cross entropy
  -> optimizer + checkpoint -> dev-set schema/evidence/error audit
  -> sealed test: accuracy, abstention, general retention, latency
  -> deploy with retrieval + access controls + drift monitoring
```
For a chat format, system/developer/user and assistant boundaries must be encoded consistently with the model's expected chat template. An incorrectly inserted EOS or role token changes training behavior. For a supervised **response-only** objective, prompt tokens provide conditioning context but do not contribute to the supervised loss. Alternative full-sequence losses exist; state the choice and its effect. A target mask is distinct from a causal attention mask: causal attention blocks future information; target masking excludes specified positions from loss.

## G2. Derivation from likelihood to gradients
Let prompt tokens be x=(x_1,…,x_m), response tokens y=(y_1,…,y_n), parameters θ, and reference response y. Under autoregressive teacher forcing,
```math
p_θ(y|x) = ∏_{t=1}^n p_θ(y_t | x,y_{<t})
L_SFT(θ) = -(1/Z) Σ_{i=1}^N Σ_{t=1}^{n_i} w_i m_{it} log p_θ(y_{it}|x_i,y_{i,<t})
Z = Σ_{i,t} w_i m_{it}
```
Here N = examples, n_i = response length, w_i = example weight, m_it ∈ {0,1} = included-token indicator, and Z normalizes by included weight. The loss is undefined if Z=0: reject all-masked batches. With vocabulary logits z_j and target class k, p_j=exp(z_j)/Σ_l exp(z_l), and ℓ=−log p_k. Differentiate: ∂ℓ/∂z_j = p_j−1[j=k]. The model is pushed toward the reference next token even if another answer is also valid; this is a central limitation of single-reference SFT.

**Numerical worked example:** one prompt has three tokens (masked), and two response tokens whose reference probabilities are 0.8 and 0.25. Response-only loss = −(ln 0.8+ln 0.25)/2 = **0.804719 nats/token**; perplexity exp(loss)=**2.23607**. If the three prompt positions were accidentally scored with probabilities 0.9, 0.9, 0.9, the five-token loss would be [2×0.1053605+0.2231436+1.3862944]/5 = **0.385104** nats/token; this smaller number is not evidence of better response learning, merely a different denominator/objective. Verify shift: at position t the logits must predict the token at t+1, not copy the token already visible at t.

**Optimization:** θ_{s+1}=θ_s−η_s ∇_θ L_B, where s is optimizer step, η_s learning rate and L_B minibatch loss. With D workers, microbatch b per worker and a accumulation steps, global batch B=Dba, provided each step includes the same examples per worker. Gradient clipping controls unstable updates but does not ensure truthfulness. SFT can cause catastrophic forgetting: retain a general-task holdout, not only domain validation.

**Why it can fail:** inconsistent annotation rubric, class imbalance, duplicated paraphrases, train/test answer leakage, narrow tasks, noisy refusals, model-specific template mismatch, prompt-only loss, overlong truncation, evaluation on teacher-forced reference tokens rather than generated answers, and overfitting. Low training loss is not proof of grounded factuality. Do not put secrets, personal records or proprietary data in a public training corpus.

## G3. Lab G — executable response masking and a toy conditional model
Python 3.10+, standard library only. This is a mechanism demonstration, not neural-network fine-tuning.
```python
import math
# Three prompt tokens are excluded; two response tokens are scored.
tokens = ["<user>", "classify", "alert", "<assistant>", "review", "now"]
targets = ["classify", "alert", "<assistant>", "review", "now"]
p_ref = [0.9, 0.9, 0.9, 0.8, 0.25]
mask = [0, 0, 0, 1, 1]
def masked_nll(probabilities, mask):
    assert len(probabilities) == len(mask) and sum(mask) > 0
    assert all(0 < p <= 1 for p in probabilities)
    return -sum(m * math.log(p) for p,m in zip(probabilities,mask))/sum(mask)
response_loss = masked_nll(p_ref,mask)
all_loss = masked_nll(p_ref,[1]*len(mask))
assert abs(response_loss - 0.8047189562) < 1e-9
assert abs(all_loss - 0.3851038919) < 1e-9
print("response-only",round(response_loss,6),"ppl",round(math.exp(response_loss),6))
print("all-token",round(all_loss,6))
# Single-token softmax gradient p - one_hot(label)
logits = [0.0, math.log(3)]
probs = [math.exp(z)/sum(math.exp(v) for v in logits) for z in logits]
grad = [probs[0]-1,probs[1]]
assert all(abs(a-b)<1e-12 for a,b in zip(grad,[-.75,.75]))
print("gradient",grad)
```
Expected: `response-only 0.804719 ppl 2.236068`; `all-token 0.385104`; `gradient [-0.75, 0.75]` (floating-point representation may vary). **Extension:** implement a tiny bigram conditional model with SGD; compare including versus excluding prompt positions and test on disjoint paraphrase groups. Then create 50 synthetic instruction records with a data card, stable schema and 10 sealed held-out tasks. Report schema validity, exact correctness, evidence entailment and abstention separately. Avoid calling this an LLM SFT benchmark.

## G4. Concrete operational case and alternatives
An industrial maintenance assistant must produce `{"component":...,"action":...,"source":...}`. Fine-tuning on 100 labeled examples may improve JSON consistency, but the replacement interval changes monthly. Retrieve the current approved manual by version and user permissions, verify cited span, then render the trained format. If the manual is missing, abstain. Test four conditions: base+prompt, base+RAG, SFT only, SFT+RAG. Hold the model revision, decoding, evaluation set and retrieval corpus version fixed. Report both correct decisions and unsupported claims. An injected document saying “ignore your rules” is data, not authority.

**Paper critique:** Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155, mixes SFT with preference/reward stages: do not attribute all reported gains to SFT alone. Wei et al. (2022), *Finetuned Language Models Are Zero-Shot Learners*, arXiv:2109.01652, explores instruction tuning transfer; scrutinize instruction dataset/task overlap and model scale before generalizing. Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, JMLR 21, arXiv:1910.10683, motivates a unified text-to-text format but does not establish universal domain factuality. Research hypothesis: group-split SFT improves schema validity over prompting without increasing unsupported factual claims; preregister annotation rubric, confidence intervals by task family, and a general-task forgetting control.

# Track C — Comparative security cases and benchmarks

## C1. Why “best accuracy” across papers is not a meaningful comparison
Network IDS classifies flows or host-windows; malware detection classifies files or behavior traces; phishing detection classifies URLs, pages or messages. These targets, prevalence, feature availability, and response latencies differ. A 99% file accuracy cannot be ranked against a 97% flow accuracy. Operational questions: what is the event, when must a decision occur, which features were observable then, what action is authorized, and how many alerts can humans inspect?

```text
flows / static files / URLs -> separate provenance and label audits
      -> unit-specific temporal + entity/family split
      -> training-only preprocessing -> rule / logistic / tree baseline
      -> validation threshold + calibrated score + evidence references
      -> sealed chronological test -> per-class, per-incident, latency
      -> analyst queue -> investigation -> delayed ground-truth feedback
```
Network case: encrypted TLS flow metadata is observable, plaintext content may not be. Malware case: EMBER-style static PE features can be extracted without executing malware, but packed and novel samples shift the distribution. Phishing case: lexical URL features are fast; page fetch and reputation checks add delay and can be unavailable or poisoned. A graph correlation feature may not exist until *after* the initial alert. Never train using a reputation verdict published after the prediction timestamp.

## C2. Derive operational metrics, including a counterexample
For attack prevalence π=P(Y=1), sensitivity r=P(alert|attack), false-positive rate f=P(alert|benign), Bayes gives:
```math
P(attack|alert) = (πr)/(πr+(1−π)f).
```
At π=.001, r=.90, f=.01, precision=.0009/(.0009+.00999)=**0.082645**. Among 100,000 events, expected TP=90, FP=999, alerts=1089; only ~8.26% are attacks. An “all benign” model scores 99.9% accuracy but misses all attacks. Accuracy alone is inappropriate.

For class c, F1_c=2TP_c/(2TP_c+FP_c+FN_c); macro-F1=(1/K)Σ_c F1_c for K classes. Macro-F1 emphasizes rare classes, but classes absent in a test period require an explicit undefined/zero policy and counts. For a detector that takes 2 ms preprocessing + 3 ms model inference + 40 ms reputation lookup + 50 ms queue wait, end-to-end latency is **95 ms** in this illustrative serial path. Report p50/p95/p99 and cold-start effects, not just model inference. A missing reputation response may force abstention or a documented fallback.

**Counterexample to random split:** 100 domains each contribute 100 nearly identical phishing URLs. Randomly splitting individual URLs makes the test set share domain templates with training; a domain-group split tests unseen domains and may perform worse. Neither protocol is universally right: deployment to known domains versus new campaigns defines the estimand. For malware, split by family/campaign and time where feasible; for flows, prevent same connection or capture session leaking into train and test.

**Cost-sensitive decision:** let C_FP be loss from unnecessary containment and C_FN loss from missed attack. With calibrated posterior p=P(attack|evidence), block if C_FP(1−p) < C_FN p, i.e. **p > C_FP/(C_FP+C_FN)**. Example C_FP=100, C_FN=900 -> threshold .10. This assumes the two-action cost model and trustworthy calibration; irreversible action may require human approval regardless.

## C3. Lab C — metrics, grouped evaluation, point-in-time intelligence
Python 3.10+ standard library, synthetic only, no attack traffic or malware.
```python
import math
pi, recall, fpr = .001, .9, .01
precision = pi*recall/(pi*recall+(1-pi)*fpr)
assert abs(precision-0.0826446281)<1e-9
N=100000
tp=N*pi*recall
fp=N*(1-pi)*fpr
assert (tp,fp)==(90.0,999.0)
print("alerts",int(tp+fp),"precision",round(precision,4))
# An IOC known after the event is not a valid feature at event time.
events = [("u1","d1",10),("u2","d2",20),("u3","d1",30)]
intel = {"d1":{"published":25,"expires":40}, "d2":{"published":5,"expires":15}}
def known_at(domain,t):
    x=intel.get(domain)
    return bool(x and x["published"]<=t<x["expires"])
flags=[known_at(d,t) for _,d,t in events]
assert flags == [False,False,True]
print("point-in-time matches",flags)
# Group leakage: train and test must not share an entity group.
train=[("d1",1),("d2",0)]
test=[("d3",1),("d4",0)]
assert {d for d,_ in train}.isdisjoint({d for d,_ in test})
threshold=100/(100+900)
assert abs(threshold-.1)<1e-12
print("cost threshold",round(threshold,2))
```
Expected: `alerts 1089 precision 0.0826`; `point-in-time matches [False, False, True]`; `cost threshold 0.1`. **Extension:** generate synthetic predictions for benign, phishing, malware and botnet classes; compute per-class confusion and macro-F1; simulate missing sensors and p95 latency; select thresholds on validation only. For an actual CICIDS2017 or EMBER benchmark, follow its licensing, published label definitions, capture period, and official split; never claim this synthetic exercise reproduces published scores.

## C4. Cases, critique, failure analysis, ethics
**Case A—flow IDS:** bursts of connections may be backup traffic; feature shortcuts (IP, timestamp, capture day) inflate performance. Compare a rule and logistic model with tree ensemble under the same chronological split. **Case B—static malware:** packed legitimate installers can look high-entropy; packed malware may evade static cues. Use only authorized prepared features, never execute unknown samples on a personal machine. **Case C—phishing:** shared hosting and compromised legitimate domains make reputation noisy; lexical-only classifiers can fail on novel campaigns. Include analyst review and safe URL handling.

**Dataset/paper critiques:** Sharafaldin, Lashkari & Ghorbani (2018), *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, DOI:10.5220/0006639801080116, documents CICIDS2017 construction; critique capture realism and labeling artifacts. Anderson & Roth (2018), *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637; distinguish feature-only dataset and malware family inference, examine temporal labels and split. Du et al. (2017), *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*, DOI:10.1145/3133956.3134015; a log-sequence model's normal-behavior assumptions do not automatically transfer to file or URL classification. Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, DOI:10.1371/journal.pone.0118432; read prevalence and evaluation design before treating PR curves as deployment utility.

**Ethics and security:** respect dataset terms, protect identities, avoid auto-quarantining on one unverified signal, guard against malicious feed poisoning, document false-positive impact and analyst appeals. Red-team only authorized offline data. **Research question:** at a fixed analyst alert budget, how does performance differ under chronological, entity-grouped and random splits across three distinct tasks? Preregister data/label audit, test period, incident-level unit, thresholds, missing-data strategy and bootstrapped confidence intervals. Compare within-task baselines; do not pool unrelated labels into a single headline score.

# Cross-track synthesis
SFT data and security benchmarks share a deeper invariant: **an evaluation item must be independent of training examples, and every feature or document must be available at decision time**. A SOC summarizer trained on synthetic incident explanations should cite source events, distinguish observation from inference and refuse to invent missing telemetry. Train it to format uncertainty; do not teach it to manufacture confidence. Keep training, retrieval and analyst-feedback permissions separate.

## Progressive Socratic oral exam (ask first; reveal answer only after attempt)
1. **Pretraining vs SFT?** Pretraining learns broad token statistics; SFT updates parameters toward labeled responses. **Follow-up:** Which is better for a manual revised yesterday, and why?
2. **What is response-only masking?** Prompt conditions generation but contributes zero supervised loss. **Follow-up:** How is it different from causal masking?
3. **Write conditional response likelihood.** Product over response tokens conditioned on prompt and earlier reference response tokens. **Follow-up:** What does teacher forcing hide about inference errors?
4. **Why is Z required?** Normalize by included token weights; reject Z=0. **Follow-up:** What changes if you average per sequence instead?
5. **Derive ∂ℓ/∂z_j.** Softmax probability minus one-hot target. **Follow-up:** What if two answers are valid?
6. **Compute two-token loss for .8 and .25.** .804719 nats/token. **Follow-up:** Why not include prompt positions?
7. **Can low SFT loss prove factuality?** No; it measures reference-token fit. **Follow-up:** Design an entailment test.
8. **SFT or RAG for changing procedures?** RAG supplies fresh authorized evidence; SFT may improve stable format. **Follow-up:** How do you validate document version?
9. **Why retain a general holdout?** Detect forgetting. **Follow-up:** Which tasks should be frozen before adaptation?
10. **What is a security unit of analysis?** The predicted object/time horizon. **Follow-up:** Why is flow recall not incident recall?
11. **Calculate precision at π=.001, r=.9, f=.01.** ~8.26%. **Follow-up:** What happens when prevalence halves?
12. **Why not compare 99% malware and 97% IDS accuracy?** Different tasks, prevalence, splits and costs. **Follow-up:** Define a within-task baseline.
13. **Explain a point-in-time join.** Exclude information published after decision time. **Follow-up:** What about revised labels?
14. **How does domain-group splitting change phishing evaluation?** Tests generalization to unseen domains. **Follow-up:** When is known-domain deployment a valid target?
15. **Compute cost threshold for FP=100, FN=900.** 0.1, assuming calibrated posterior and binary costs. **Follow-up:** Why might human approval remain mandatory?
16. **What is macro-F1?** Mean class-specific F1; rare classes count equally. **Follow-up:** How handle absent classes?
17. **Why measure p95 end-to-end latency?** Queue, lookup and preprocessing dominate in some systems. **Follow-up:** Which component fails closed?
18. **Design a falsifiable cross-track experiment.** Compare SFT+RAG versus prompting on sealed incident cases while withholding future intelligence and measuring supported-claim rate. **Follow-up:** Which result would reject the hypothesis?

## Recall and research deliverables
Without notes: explain target mask versus causal mask, derive response NLL, state the 8.26% precision counterexample, distinguish file/URL/flow units, and describe a valid point-in-time split. Extended study: run both scripts, store stdout, draft an instruction data card and a three-task benchmark protocol, write one paper critique per track, and preregister negative controls.

## Source map
- Syllabus progression: GenAI pathway p.8 Unit 3 Session 2; Cyber pathway pp.6–7 Unit 2 Session 6 and lab.
- Conditional likelihood, response masking, teacher forcing and SFT evaluation: G1–G3; Ouyang 2022, Wei 2022, Raffel 2020; mathematical steps explicitly derived here.
- Retrieval-versus-parameter-update distinction and deployment controls: G1, G4; syllabus p.8 and the proposed operational architecture.
- Benchmark provenance and task definitions: C1–C4; Sharafaldin 2018, Anderson & Roth 2018, Du 2017.
- Imbalance/precision mathematics: C2–C3; Saito & Rehmsmeier 2015; numeric results are synthetic calculations, not paper findings.
- Ethical limits, future-information leakage, analyst oversight: C1–C4 and cross-track synthesis; recommendations, not claimed experimental results.
- Full canonical identifiers, source/access notes, and companion summaries: [references/index.md](references/index.md).

**Scope and verification:** This lesson is an instructional artifact, not a completed pretrained-model experiment. The small code examples contain assertions and deterministic expected outputs. No narration, audio, email, or workflow trigger is part of this lesson.
