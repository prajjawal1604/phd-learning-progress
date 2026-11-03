# PhD Morning Lesson — Day 4 — Wednesday, 4 November 2026

**Tracks:** Advanced Generative AI and Foundation Models; AI for Cybersecurity. **Progression:** Unit 1, Session 4 in both pathways, following the November 3 foundation-model transfer and cybersecurity feature-engineering sessions. This is accelerated independent study, not a replacement for university contact-hour requirements.

**Today's focus:** Transformer attention and positional information; cybersecurity detection pipelines, imbalanced evaluation, and operational alert thresholds. The evening lesson must deepen these same topics.

## Learning outcomes and timetable
By the end, derive scaled dot-product attention, explain multi-head attention and positional embeddings, identify causal masking, and distinguish architectural computation from model capability. For cybersecurity, design a leakage-resistant detector, calculate precision/recall/false-positive rate, and select a threshold from validation data under an alert budget.

Suggested 3-hour block: 35 min transformer theory; 25 min derivations; 35 min attention Python lab; 40 min cybersecurity lab; 25 min evaluation and paper reading; 20 min quiz and research notebook.

## Part A — Transformers, attention, and position

### 1. Why attention?
Recurrent networks process sequences through state updates, while self-attention allows every token to incorporate information from permitted context positions. A Transformer layer combines attention, residual connections, normalization and a position-wise feed-forward network. Attention alone does not encode order: positional information must enter explicitly or through architecture.

For input matrix X of shape n × d_model, trainable projections give Q=XW_Q, K=XW_K, V=XW_V. For head dimension d_k:

**Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k) + M)V**,

where M contains 0 for permitted positions and a large negative value for disallowed positions. In decoder-only causal language models, token i must not attend to later token j>i. Encoder-style bidirectional attention permits both directions unless restricted by padding or another mask.

```text
token IDs -> embeddings + position information
                       |
                  layer norm*
                       |
          Q projection K projection V projection
                \          |         /
                   scaled dot-product
                   + causal/padding mask
                          |
                        softmax
                          |
                     weighted V
                          |
               concat heads + output
                          |
               residual + feed-forward
                          |
                      next layer
* pre-norm and post-norm variants differ
```

### 2. Worked example: two-token attention
Let Q for the first token be [1,0], K rows [1,0] and [0,1], and V rows [2,0] and [0,4]. With d_k=2, logits are [1/sqrt(2),0] = [0.7071,0]. Their softmax weights are approximately [0.6698,0.3302]. The output is [1.3396,1.3208]. This is a convex combination of value vectors. The attention weight is **not** a calibrated explanation of why a model produced a final answer.

If a causal mask is applied to the first token in a two-token sequence, its future-token logit is suppressed and the output becomes [2,0]. The second token may attend to both.

**Complexity:** A dense n×n attention matrix takes O(n²) pairwise scores per head and O(n²) memory when materialized. This is not the entire model's runtime complexity; projection and feed-forward costs matter too. FlashAttention changes memory traffic and avoids storing the full attention matrix but does not simply make arbitrary dense attention linear-time.

### 3. Positional information
Absolute learned position vectors add a trainable vector p_i to token embedding e_i. Sinusoidal positions use sine and cosine frequencies to provide deterministic features. Rotary position embeddings rotate query/key coordinates according to position, so dot products reflect relative offsets. Each has extrapolation and implementation tradeoffs; no positional scheme guarantees long-context reasoning. Test length generalization with controlled sequence lengths and tasks.

### 4. Multi-head attention
For h heads, each head attends in its own projected subspace. Concatenate head outputs and multiply by W_O. With d_model=512 and 8 equally sized heads, d_k=64. The factor sqrt(64)=8 scales dot products. Heads are not guaranteed to learn human-readable syntactic roles; empirical interpretation requires intervention, not visualization alone.

### 5. Lab A: inspect causal versus unmasked attention
Python 3.11+; only numpy required: `python -m pip install numpy`.

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
Q = np.array([[1.,0.],[0.,1.]])
K = Q.copy()
V = np.array([[2.,0.],[0.,4.]])
logits = Q @ K.T / np.sqrt(2.)
def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
for name, mask in [
    ("bidirectional", np.zeros((2,2))),
    ("causal", np.triu(np.full((2,2), -1e9), 1)),
]:
    weights = softmax(logits + mask)
    out = weights @ V
    print(name, "weights:", weights, "output:", out)
    assert np.allclose(weights.sum(axis=1), 1.)
```

**Expected check:** the first causal row is [1,0], its output [2,0]; the bidirectional first row is approximately [0.6698,0.3302]. Change the key vectors, rerun, and explain how similarity changes the weighting. Repeat with three tokens and verify the upper triangle of the causal weight matrix is zero.

**Research extension:** compare learned absolute, sinusoidal and rotary positions on a synthetic copy/reverse task at lengths 16, 32, 64; train only on lengths ≤32. Report accuracy by length, seeds, parameter counts, and runtime. A result on this synthetic task does not establish real-world long-context ability.

## Part B — Cybersecurity detection and evaluation

### 6. From features to an operational decision
Yesterday's session built as-of-time features from authentication/network telemetry. Today, use those features in a detector with a stated **prediction unit**, **label horizon**, **review capacity**, **base rate**, and **time-aware evaluation**. The label "failed login" is not the same as a confirmed compromise. The target must be defined independently of the features used to predict it.

```text
timestamped authorized events
       |
as-of feature aggregation (no future ingestion)
       |
chronological train / validation / future test
       |
train-only imputer, encoder, scaler + classifier
       |
validation score calibration and threshold selection
       |
frozen test evaluation at fixed analyst budget
       |
analyst triage -> feedback -> delayed verified labels
```

A random row split may leak entity-specific behavior across partitions. If a user appears repeatedly, choose forward-time and optionally group-disjoint evaluation to match the deployment question. Prevent target leakage from incident-close timestamps, analyst decisions and retrospective enrichment.

### 7. Confusion matrix and rare-event base rates
Define TP: alert on a real incident; FP: alert on a benign case; FN: missed incident; TN: benign correctly unalerted.

- Precision = TP/(TP+FP): proportion of alerts worth investigating.
- Recall = TP/(TP+FN): proportion of incidents detected.
- FPR = FP/(FP+TN): fraction of benign events alerted.
- F1 = 2PR/(P+R), defined when denominators permit.
- PR-AUC is often informative for imbalanced incident labels; always disclose prevalence.

**Worked example:** In 10,000 cases, 100 are truly incidents (1% prevalence). A detector catches 80 (recall 80%) and alerts on 198 of the 9,900 benign cases (FPR 2%). Precision = 80/(80+198)=28.78%, meaning ~71.22% of alerts are false positives despite 98% specificity. There are 278 alerts; if analysts can review only 100, the operating threshold must be adjusted using validation data, not silently evaluated on a different budget.

A high ROC-AUC does not ensure useful precision at a low base rate. Calibration and precision-at-k are distinct from ranking metrics.

### 8. Thresholds, cost and drift
Given a validation score s(x), choose threshold t to maximize an explicitly defined utility or satisfy `alerts_per_day <= capacity`. Never tune t on the final test. If labels arrive after a delay, avoid treating unlabeled events as negatives. Monitor alert volume, score distributions, feature missingness, delayed precision and time-to-triage. Drift in telemetry format may look like malicious behavior.

**Decision cost example:** Suppose each false alert costs 15 analyst minutes and a missed incident incurs an estimated 600-minute response cost. A simple expected cost is `15*FP + 600*FN`. These hypothetical costs do not encode real severity; use stakeholder-approved values and sensitivity analyses before deployment.

### 9. Lab B: leakage-safe, imbalanced toy detector
Python 3.11+; `python -m pip install numpy scikit-learn`. The data are synthetic; no production conclusions.

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, average_precision_score
rng = np.random.default_rng(42)
n = 3000
# Rows are in temporal order; changing prevalence is deliberate.
failed = rng.poisson(2.0, n)
new_device = rng.binomial(1, .15, n)
night = rng.binomial(1, .25, n)
risk_logit = -5.0 + .42*failed + 1.1*new_device + .55*night
prob = 1/(1+np.exp(-risk_logit))
y = rng.binomial(1, prob)
X = np.column_stack([failed, new_device, night])
train, valid, test = slice(0,1800), slice(1800,2400), slice(2400,3000)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X[train], y[train])
val_score = model.predict_proba(X[valid])[:,1]
# Select a validation alert budget of 5% (30 of 600).
budget = 30
threshold = np.sort(val_score)[-budget]
test_score = model.predict_proba(X[test])[:,1]
pred = (test_score >= threshold).astype(int)
print("test prevalence", y[test].mean())
print("validation budget", budget, "test alerts", pred.sum())
print("precision", precision_score(y[test], pred, zero_division=0))
print("recall", recall_score(y[test], pred, zero_division=0))
print("average precision", average_precision_score(y[test], test_score))
```

**Interpretation:** The validation threshold can yield a different number of test alerts if score distributions drift; it is not a guarantee of 30 alerts on future data. The synthetic label is generated from the same features and therefore favors this model. For a credible experiment, introduce delayed labels, new entities, a changed base rate and a baseline of simple threshold rules. Report confidence intervals from repeated independent simulations; never use this toy as a claim of production detection efficacy.

### 10. Worked research protocol
**Question:** Does a pretrained sequence representation improve precision-at-100 over hand-built authentication counts under temporal distribution shift?

**Hypotheses:** H0: no improvement; H1: improvement under fixed analyst budget. Define observation window and prediction horizon, train on month 1, validate on month 2, test on month 3, and use a strict event-availability cutoff. Compare rule baseline, logistic regression and frozen sequence representation with linear head. Fix incident definition, label latency, alert capacity and compute. Report PR-AUC, precision@100, recall, FPR, calibration, per-entity slices and failure cases. Avoid claiming causality from observational performance.

**Threats to validity:** shared accounts across splits, duplicate alerts, post-incident labels, selection bias from investigated-only incidents, incomplete ground truth, class prevalence changes, time-zone shifts and adversarial adaptation.

## 11. Global AI and cybersecurity news — source-verified as of 4 November 2026
1. **2 November — OpenAI Grid Protectors Program:** OpenAI announced plans to make future compute campuses more flexible under grid stress. This is an infrastructure and governance announcement, not proof that all AI data centers already provide grid balancing. Source: https://openai.com/index/grid-protectors-program/ . Research prompt: compare latency-sensitive training and inference workloads under demand-response constraints.
2. **30 October — AI training preference signals:** OpenAI described support for a common robots.txt signal expressing publishers' preferences about foundational-model training. This concerns governance and data provenance, not an automatic copyright determination. Source: https://openai.com/index/helping-websites-express-ai-training-preferences/ . Research prompt: what metadata should a dataset manifest preserve?
3. **November — CISA President's Cup Game of the Month:** CISA's official practice page lists the November 2026 challenge, “Inception,” centered on software-security assessment and vulnerability analysis. This is a training resource, not an active threat advisory. Source: https://presidentscup.cisa.gov/game-of-the-month/ . Research prompt: translate a defensive lab finding into an auditable detection feature.

**Verification boundary:** No unverified zero-day, exploitation or breach claims are included. News is educational context, not evidence for model effectiveness.

## 12. Learning resources and reading order
**Foundational papers:** Vaswani et al. (2017), *Attention Is All You Need*, https://arxiv.org/abs/1706.03762 ; Dao et al. (2022), *FlashAttention*, https://arxiv.org/abs/2205.14135 ; Bommasani et al. (2021), *On the Opportunities and Risks of Foundation Models*, https://arxiv.org/abs/2108.07258 .

**Tutorials:** Harvard NLP *Annotated Transformer*, https://nlp.seas.harvard.edu/annotated-transformer/ ; Hugging Face NLP course, https://huggingface.co/learn/nlp-course/ ; scikit-learn model evaluation guide, https://scikit-learn.org/stable/modules/model_evaluation.html ; Google ML Crash Course classification metrics, https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall .

**Lecture/video:** Stanford CS25 Transformers United, https://web.stanford.edu/class/cs25/ ; 3Blue1Brown's attention explainer, https://www.3blue1brown.com/ .

**Podcasts:** *Practical AI* (https://changelog.com/practicalai) for applied model deployment discussions; *SANS Internet Storm Center Stormcast* (https://isc.sans.edu/podcast.html) for security operations context. These are series recommendations, not claims about a particular episode.

## 13. Quiz — answer before reading solutions
1. Why divide attention logits by sqrt(d_k)?
2. What changes when causal masking is applied?
3. Why are attention weights not automatically faithful explanations?
4. With d_model=512 and eight equal heads, what is d_k?
5. If TP=30, FP=70 and FN=20, calculate precision and recall.
6. Why does random splitting authentication rows risk optimistic results?
7. What is the difference between selecting a threshold on validation and test?
8. Why can a detector have 98% specificity but low precision?

**Answers:** 1. To temper dot-product magnitude as dimensions grow and reduce softmax saturation. 2. Tokens cannot attend to future positions. 3. Later layers, value vectors and nonlinearities contribute; attention is a computational weighting, not a causal attribution. 4. 64. 5. Precision 30%; recall 60%. 6. Entity/time overlap and future information can leak. 7. Test-based selection biases the reported result; validation selection preserves an untouched evaluation. 8. With rare positives, even a small benign false-positive fraction can outnumber true alerts.

## 14. Research notebook deliverables
Write one page on transformer architecture with annotated shapes for Q/K/V and a causal mask; save Lab A outputs with changed keys; save Lab B seed, data generator, threshold, confusion matrix and metric results; write a 250-word critique of the limits of attention visualization; and formulate one falsifiable hypothesis about precision-at-k under drift.

**For evening continuation:** deepen attention backpropagation, head dimensions, positional encoding tradeoffs, rare-event calibration, confidence intervals and incident triage economics. Do not advance to Session 5.

## Narration handoff
This file is the canonical narration source for `phd-morning/lessons/2026-11-04/lesson.md`. GitHub Actions should generate and publish audio after this commit. Audio is **not** considered available until a completed workflow and accessible MP3 are verified. The email delivery is a separate later task.
