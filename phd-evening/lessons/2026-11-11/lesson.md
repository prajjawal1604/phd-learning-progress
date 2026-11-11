# Evening PhD deep dive — 11 November 2026 (Asia/Kolkata)
**Dependency confirmed:** Read committed morning lesson and references on `main` before writing. This lesson deepens precisely (G) supervised fine-tuning/instruction adaptation and (C) comparative cybersecurity benchmark validity. Syllabus: 19 October 2026 email, *Advanced Generative AI and Foundation Models Learning Pathway* p.8 Unit 3 Session 2 and *AI for Cybersecurity Learning Pathway* pp.6–7 Unit 2 Session 6. Both original attachments were retrieved and inspected. All measurements below are constructed examples, not published experimental results.

## Tutor route and prerequisites
**10 min:** ask why response-only SFT loss can improve while real outputs worsen; derive token weighting; ask why a random security benchmark split is optimistic; compute posterior precision. **30 min:** add gradients, packing, LoRA comparison, shift and calibration, paired group bootstrap, and questions 1–10. **Extended (90–150 min):** run both labs, critique the studies, design ablations and answer all questions. For walking, say the intuition first, read one equation verbally, pause for the learner's response, then disclose the answer.

Prerequisites: softmax, conditional likelihood, cross-entropy, teacher forcing, stochastic gradient descent, precision/recall, conditional probability, grouped and temporal train/test splits. **Exposure bias** means training conditions on gold previous tokens but deployment conditions on generated previous tokens. **Loss mask** selects supervised targets; **causal mask** prevents attention to future tokens. **Calibration** means predicted risk p agrees with observed event frequency within appropriate bins. **Distribution shift** means the joint distribution of features/labels changes between training and deployment. **Estimand** means the precise population-level quantity the evaluation aims to measure. **Group bootstrap** resamples independent entities rather than correlated records.

# G. SFT: objective design, gradient weighting, packing, and evidence-grounded adaptation

## G1. What the objective actually optimizes
Given prompt x, gold response y=(y1,...,yT), parameters θ, teacher-forced likelihood is pθ(y|x)=product_t pθ(yt|x,y<t). For examples i with token mask m_it in {0,1} and example weight w_i, weighted response-only negative log-likelihood is
```text
L(θ) = - Σ_i Σ_t w_i m_it log pθ(y_it | x_i,y_i,<t) / Σ_i Σ_t w_i m_it.
```
Every symbol: i indexes examples; t response-token position; θ model weights; m excludes prompt/padding tokens; w controls example importance. This **token-average** weights long responses more heavily than short ones. An alternative **sequence-average** is L_seq=(1/N)Σ_i[-Σ_t m_it log p_it / Σ_t m_it]. Neither is universally correct. For one 2-token response with token losses [0.1,0.3] and one 8-token response with losses all 1.0, token-average=(0.4+8)/10=0.84, sequence-average=(0.2+1)/2=0.60. If the research question concerns equal importance per instruction, sequence averaging may be better; if token prediction throughput matters, token averaging is natural. A length-correlated safety class can otherwise be underrepresented.

Let logits z_j, softmax p_j=exp(z_j)/Σ_k exp(z_k), and target class q. For a scored token, ∂(-log p_q)/∂z_j=p_j-1[j=q]. For a masked token the direct loss gradient is zero, but its prompt embedding can still receive gradient through attention from later scored tokens; “prompt masked” does **not** mean prompt information is unused. With label smoothing ε over V classes, target distribution r_j=(1-ε)1[j=q]+ε/V, giving ∂L/∂z_j=p_j-r_j. This may reduce overconfidence but can harm exact-format tokens; test, don't assume.

**Teacher-forcing counterexample:** model predicts reference token A with .95 conditional on gold prefix, but at inference first generates B with probability .05; its continuation from B was never evaluated by the reference-only loss. Thus low teacher-forced perplexity cannot establish multi-turn reliability. Evaluate generated responses under frozen decoding and evidence rubrics, including abstention and unsupported claims.

## G2. Implementation mechanism and hazards
```text
consented instruction records + immutable source IDs
 -> near-duplicate grouping + train/dev/test manifests
 -> role-aware chat template -> tokenize -> response boundary
 -> shift logits by one -> target labels; prompt/pad=-100
 -> packed batches with explicit example boundaries
 -> forward -> masked CE -> optimizer/LoRA -> checkpoint
 -> generate on sealed test -> schema, evidence, refusal, forgetting
 -> controlled retrieval with source ACL -> human review
```
A common Hugging Face/PyTorch convention uses ignore_index=-100 for unscored labels; inspect the trainer's shift behavior rather than shifting twice. With packing, concatenating unrelated conversations without block-diagonal attention permits example B to see example A; causal attention alone does not guarantee sample isolation. EOS placement and chat-template role delimiters must match the model. Dynamic padding and truncation can silently delete the answer; assert each batch has at least one scored token and log truncation rates.

**Parameter-efficient adaptation:** for frozen W∈R^(d_out×d_in), LoRA learns ΔW=(α/r)BA with B∈R^(d_out×r), A∈R^(r×d_in), rank r, scale α/r. Trainable parameters r(d_in+d_out), versus d_out d_in for dense W; at d_in=d_out=4096, r=16 gives 131,072 versus 16,777,216, about 0.78% for that matrix. This is not a claim about whole-model memory: activations, optimizer, adapters and inference KV cache still cost memory. Low rank can constrain updates and may fail under large domain shifts. See Hu et al. (2021), arXiv:2106.09685.

**SFT versus RAG:** SFT is for stable policy/format and repeated annotated tasks; retrieval is for changing authorized facts. In a maintenance assistant, a tuned JSON schema is useful, but current repair intervals must come from versioned manuals. An untrusted retrieved paragraph saying “ignore system rules” is content, not instruction. Compare base+prompt, base+RAG, SFT, SFT+RAG on the same sealed grouped test. Measure exact schema validity, answer correctness, evidence entailment, unsupported assertion rate, abstention accuracy, general-task retention, latency and cost; preregister acceptance criteria.

## G3. Worked example and executable advanced lab
A response with reference probabilities .8 and .25 has average NLL -[ln(.8)+ln(.25)]/2=.804719 nats/token. The first probability contributes .223144 and the second 1.386294; the hard token dominates. If one task family has long outputs, global token-averaging disproportionately emphasizes it. **Lab:** compare objectives and gradient numerically with Python 3.10+ standard library only:
```python
import math
examples = [[.8,.25],[math.exp(-1)]*8]
def seq_loss(xs): return sum(-math.log(p) for p in xs)/len(xs)
token = sum(len(x)*seq_loss(x) for x in examples)/sum(map(len,examples))
sequence = sum(map(seq_loss,examples))/len(examples)
assert abs(token-(1.6094379124341003+8)/10)<1e-10
assert abs(sequence-(.8047189562170501+1)/2)<1e-10
print(round(token,6),round(sequence,6))
def softmax(z):
    m=max(z); e=[math.exp(v-m) for v in z]; return [v/sum(e) for v in e]
z=[0.,math.log(3)]
p=softmax(z); analytic=[p[0]-1,p[1]]
def loss(a): return -math.log(softmax(a)[0])
h=1e-5
numeric=[(loss([z[k]+(h if k==j else 0) for k in range(2)])-
          loss([z[k]-(h if k==j else 0) for k in range(2)]))/(2*h)
         for j in range(2)]
assert max(abs(a-b) for a,b in zip(analytic,numeric))<1e-8
print([round(v,2) for v in analytic])
```
Expected: `0.960944 0.902359` and `[-0.75, 0.75]`. (The first example's NLL sum is 1.6094379.) **Lab extensions:** add per-example weights and padding masks; simulate an all-masked batch; build a packing boundary unit test; compare a two-class conditional model's gold-prefix loss against free-running error; on a permitted tiny pretrained model, measure adapter parameter counts, throughput and catastrophic forgetting. Save random seeds, package versions, data card and hashes.

## G4. Paper critique
Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155: SFT is one stage of a multi-stage pipeline; improvements cannot be assigned solely to demonstrations. Wei et al. (2022), *Finetuned Language Models Are Zero-Shot Learners*, arXiv:2109.01652: task transfer depends on instruction distribution and model size; check overlap. Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685: parameter-efficiency does not guarantee adaptation quality or privacy. Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv:1910.10683: unified task formatting does not ensure evidence grounding. Research extension: factorial study of masking objective × adapter rank × RAG, with task-family held-out groups, generated-answer audits, forgetting and cost; report confidence intervals and negative results.

# C. Cybersecurity benchmark validity: temporal leakage, costs, calibration and uncertainty

## C1. Define the estimand before computing metrics
A flow IDS predicts a flow or host-time-window; a static malware model predicts a file; a phishing classifier predicts a URL/message. Do not compare headline accuracies across different units, base rates, horizons or action costs. In a SOC, detection usefulness also includes event-to-alert delay, missing telemetry, analyst capacity and false-containment harm.

```text
timestamped flows / file features / URL observations
 -> source + label provenance audit -> entity/family + chronological split
 -> training-only preprocessing -> rule and ML baselines
 -> validation-only calibration and threshold selection
 -> sealed future test (including unseen groups)
 -> alert queue, p95 end-to-end latency, analyst decisions
 -> delayed labels and shift monitoring (never backfill future intel)
```
A **random record split** estimates generalization to more records resembling already-observed entities; an **entity-held-out chronological split** estimates performance on future unseen entities. Both can be legitimate for different deployments, but mixing them conceals the actual question. Avoid same campaign, capture session, binary family, domain or near-duplicate across train/test. Fit scaling, imputation, feature selection, calibration and threshold only on training/validation partitions.

## C2. Bayesian precision and decision theory
Let π=P(attack) prevalence, r=P(alert|attack) recall, f=P(alert|benign) FPR. Bayes gives precision=πr/[πr+(1-π)f]. At π=.001, r=.9, f=.01, precision=.082645; for 100,000 events expect TP=90, FP=999. At π=.0001 with the same rates, precision=.008921, demonstrating base-rate sensitivity. An “all benign” detector scores 99.9% accuracy at π=.001 yet detects nothing.

If posterior p=P(attack|evidence) is calibrated, false-positive action cost C_FP and miss cost C_FN, blocking has expected cost C_FP(1-p), not blocking C_FN p; block when p>C_FP/(C_FP+C_FN). With C_FP=100 and C_FN=900, threshold=.1. **Important:** calibrated probabilities on the test prevalence may not remain calibrated after deployment prevalence shifts. If training prevalence π_train differs from deployment π_deploy under pure prior shift, posterior odds can be adjusted:
```text
odds_deploy = odds_train × [π_deploy/(1-π_deploy)] / [π_train/(1-π_train)].
```
This assumes class-conditional feature distributions stay fixed; under adversarial feature shift it may be wrong. Automatic quarantine may still require human approval regardless of cost threshold.

**Latency:** serial path 2 ms preprocessing + 3 ms inference + 40 ms lookup + 50 ms queue =95 ms, but p95 of a sum is not generally the sum of p95s. Measure end-to-end event timestamps on real workloads; include timeouts, abstentions and delayed labels. For alert budget B, select a validation threshold such that expected alerts/day≤B, then evaluate on untouched future data.

## C3. Why uncertainty intervals need groups
Suppose 10,000 test flows arise from only 20 capture sessions. Treating flows as 10,000 independent samples underestimates uncertainty if same-session errors correlate. A **cluster bootstrap** samples 20 sessions with replacement, keeping all flows in each selected session; repeat and compute percentile intervals for precision/recall and alert counts. An ordinary record bootstrap incorrectly assumes independence. A paired comparison must resample the same sessions for both detectors to estimate their performance difference.

**Lab: prior shift, costs and group bootstrap** (Python 3.10+, standard library, synthetic data):
```python
import math,random
def precision(pi,r=.9,f=.01): return pi*r/(pi*r+(1-pi)*f)
assert abs(precision(.001)-.08264462809917356)<1e-12
print(round(precision(.001),6),round(precision(.0001),6))
def adjust(p,pi_train,pi_new):
    odds=p/(1-p)*(pi_new/(1-pi_new))/(pi_train/(1-pi_train))
    return odds/(1+odds)
assert abs(adjust(.5,.5,.1)-.1)<1e-12
print("adjusted",round(adjust(.5,.5,.1),3))
# Each tuple is one correlated incident group: TP, FP, FN for detector A and B.
groups=[((9,1,1),(8,0,2)),((3,4,2),(4,2,1)),
        ((8,2,1),(7,1,2)),((2,6,3),(3,4,2)),
        ((7,1,1),(6,1,2)),((4,3,2),(5,2,1))]
def f1(counts):
    tp=sum(c[0] for c in counts);fp=sum(c[1] for c in counts);fn=sum(c[2] for c in counts)
    return 2*tp/(2*tp+fp+fn)
rng=random.Random(42); diffs=[]
for _ in range(2000):
    ids=[rng.randrange(len(groups)) for _ in groups]
    diffs.append(f1([groups[i][0] for i in ids])-f1([groups[i][1] for i in ids]))
diffs.sort()
print("paired CI",round(diffs[50],3),round(diffs[1949],3))
assert len(diffs)==2000 and all(-1<=d<=1 for d in diffs)
```
Expected first line: `0.082645 0.008921`; second `adjusted 0.1`. Final paired CI is deterministic with Python's seeded random generator but is a **toy demonstration**, not an empirical confidence claim. **Extensions:** simulate 1,000 entities with a time trend, compare record-random versus entity-time splits, calculate per-task macro-F1 and abstention coverage, then compare record versus group bootstrap interval width. Never run unknown malware; use public prepared features or synthetic examples.

## C4. Counterexamples, paper critiques, ethics
**Network:** TLS metadata cannot reveal encrypted payload; capture-day shortcuts can make random-split IDS appear excellent. **Malware:** high-entropy packed benign installers can be false positives; static PE features cannot replace dynamic behavior analysis. **Phishing:** a trusted domain can be compromised and a new domain can lack reputation; later threat-feed verdicts are not available at decision time. A single feature can be correlated with campaign identity, not malice. Analyst review and provenance are essential.

Sharafaldin, Lashkari & Ghorbani (2018), *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, DOI:10.5220/0006639801080116: examine capture artifacts and labels before generalization. Anderson & Roth (2018), *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models*, arXiv:1804.04637: prepared static features, not a proof of dynamic efficacy. Du et al. (2017), *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*, DOI:10.1145/3133956.3134015: log sequence assumptions need not transfer to files/URLs. Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, DOI:10.1371/journal.pone.0118432: PR curves expose imbalance effects, but utility still depends on prevalence and costs. Research extension: preregister group/temporal split, label audit, alert budget, point-in-time intelligence, drift scenarios, per-incident bootstrap, and human-review constraints. Never deploy automatic containment from synthetic scores. Respect dataset licenses, privacy, auditability and appeal processes.

# Cross-track research insight
SFT and security benchmark evaluation share the same failure: **the learner can memorize correlated or future information and look excellent on a contaminated test**. In an SOC assistant, SFT should teach cautious structured explanation; retrieved logs must be access-controlled and timestamp-valid; the assistant must cite events and abstain on missing telemetry. An untrusted log line containing imperative text is not a higher-priority instruction. Evaluate the complete chain: detection → evidence → explanation → human action.

## Progressive oral examination (answer after learner attempts; follow-up extends difficulty)
1. What differs between causal mask and loss mask? **A:** Future visibility versus scored positions. **Follow-up:** Can prompt embeddings get gradient?
2. Derive conditional SFT likelihood. **A:** Product of next-response-token conditionals. **Follow-up:** Why does teacher forcing hide recovery errors?
3. Why does global token-average overweight long responses? **A:** Each token contributes to denominator. **Follow-up:** Derive sequence-average.
4. Compute NLL of .8 and .25. **A:** .804719 nats/token. **Follow-up:** What happens if prompt tokens are included?
5. Derive softmax CE gradient. **A:** p_j−one_hot_j. **Follow-up:** Generalize to label smoothing.
6. What is LoRA's trainable parameter count? **A:** r(d_in+d_out) for one matrix. **Follow-up:** Which memory terms remain?
7. Why can packed training leak across examples? **A:** Causal attention can still see previous packed sample. **Follow-up:** Design block isolation.
8. SFT or RAG for a changing manual? **A:** Retrieval for current facts, SFT for stable behavior. **Follow-up:** How would you test injection?
9. Why isn't lower teacher-forced loss enough? **A:** Generated prefixes differ and evidence may be absent. **Follow-up:** Name four generation metrics.
10. What is the estimand for an unseen-campaign detector? **A:** Future new campaign performance. **Follow-up:** Choose split units.
11. Derive precision from π,r,f. **A:** πr/(πr+(1−π)f). **Follow-up:** Evaluate π=.001.
12. Why can accuracy 99.9% be useless? **A:** All-benign misses rare attacks. **Follow-up:** Compute analyst workload.
13. Derive cost threshold. **A:** C_FP/(C_FP+C_FN). **Follow-up:** What if p is uncalibrated?
14. What does prior-shift correction assume? **A:** Stable class-conditional features. **Follow-up:** Give an adversarial violation.
15. Why not bootstrap individual flows? **A:** Correlation within capture sessions. **Follow-up:** Describe paired group bootstrap.
16. Why is p95 inference latency insufficient? **A:** Queues, lookups and timeouts dominate. **Follow-up:** Instrument event-to-action.
17. Can a later IOC verdict be a training feature? **A:** Not for historical decision-time evaluation. **Follow-up:** Implement an as-of join.
18. Design one integrated SOC-SFT experiment. **A:** group/time split; frozen retrieval; grounded explanations; abstention; alert-budget evaluation. **Follow-up:** What would falsify your hypothesis?

## Recall cards and research deliverables
Without notes: (a) write both SFT loss normalizations, (b) explain prompt gradient despite zero prompt loss, (c) derive LoRA count, (d) compute precision for rare attacks, (e) explain group bootstrap and prior-shift caveat, (f) state two paper limitations. Deliverable: two-page preregistration of objectives, dataset provenance, frozen splits, baselines, ablations, metrics, uncertainty intervals, ethical approval, negative-result plan. For any future lab, log code revision, random seed, data license and source hashes.

## Source map
Morning lesson `../../../../phd-morning/lessons/2026-11-11/lesson.md` and morning references index are the direct syllabus continuity baseline. G1–G4: Ouyang 2022, Wei 2022, Hu 2021, Raffel 2020; mathematical gradient and numerical examples are derived here, not attributed as empirical paper findings. C1–C4: Sharafaldin 2018, Anderson/Roth 2018, Du 2017, Saito/Rehmsmeier 2015; cost, shift and bootstrap derivations are explicit reasoning and synthetic labs. Canonical identifiers, URLs, access notes and summaries: [companion references](references/index.md). No claim of live paper PDF redistribution or full-model training.
