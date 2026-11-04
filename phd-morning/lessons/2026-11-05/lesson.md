# PhD Morning Lesson — Day 5 — Thursday, 5 November 2026

**Pathways:** Advanced Generative AI and Foundation Models; AI for Cybersecurity (Marwadi University, 19 October 2026). **Accelerated progression:** yesterday covered scaled attention and security evaluation. Today consolidates **Generative AI Unit 2, Session 2: position, padding/causal masks and Transformer families** and **Cybersecurity Unit 1, Session 5: drift and experimental methodology**. The university's 40 contact hours and 60 independent-study hours remain distinct; this independent accelerated schedule does not change credit requirements.

**Duration:** ~3 hours core, 1 hour optional research. Outcomes: implement correct attention masks, explain encoder/decoder visibility, audit a temporal split, distinguish covariate drift from concept drift, and document a reproducible test protocol.

## A. Generative AI — mask correctness, position and model families

### 1. Three architectures, three information contracts

An **encoder-only** Transformer can attend bidirectionally to visible input tokens and often learns contextual representations using masked-token objectives. A **decoder-only** Transformer generates autoregressively, restricting each query position to earlier positions and itself. An **encoder-decoder** model combines bidirectional source encoding with causal target self-attention and target-to-source cross-attention. These are architectural patterns, not guarantees about model quality.

```text
Encoder-only:       [t1] <--> [t2] <--> [t3]   (padding excluded)
Decoder-only:       [t1] ----> [t2] ----> [t3] (no future visibility)
Encoder-decoder:    source -> bidirectional encoder
                                | | |
                    target -> causal decoder -> cross-attend source
```

The Transformer attention equation for a single head is

`A(Q,K,V) = softmax(Q K^T / sqrt(d_k) + M) V`.

For a sequence of length L, the score tensor is L×L per head. A **causal mask** forbids key index j>query index i. A **key-padding mask** forbids attention to padding keys, even if the query is valid. Masking padded *query outputs* is a separate concern: they may still produce values and must be excluded from loss or downstream pooling. Implementations vary in whether boolean True means keep or block; verify the API instead of assuming.

### 2. Worked numerical example: combined masks

Suppose a three-token sequence consists of two real tokens followed by padding. Let all unmasked scaled logits be zero. The attention distributions are:

- Bidirectional, no padding mask: each query sees [1/3, 1/3, 1/3] — incorrect for real queries because padding contributes.
- Bidirectional plus padding mask: each real query sees [1/2, 1/2, 0].
- Causal plus padding mask: query 0 sees [1,0,0]; query 1 sees [1/2,1/2,0]. Query 2 is padding and should not contribute to loss.

With values v0=[2,0], v1=[0,4], v2=[100,100], query 1's correctly masked output is [1,2]. Without padding exclusion it would be [34,34.67] approximately. This illustrates how a seemingly harmless padded vector can contaminate representations.

**Position:** without positional signals, unmasked self-attention is permutation-equivariant under consistent token permutation. Learned absolute positions, sinusoidal encodings, and rotary embeddings introduce order differently. A causal mask itself introduces directional position constraints, but does not substitute for all positional information.

### 3. Lab A — robust attention and mask assertions (NumPy)

Environment: Python 3.11+, `python -m pip install numpy`. Save as `attention_masks.py`.

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
L = 3
Q = np.zeros((L, 2))
K = np.zeros((L, 2))
V = np.array([[2.,0.],[0.,4.],[100.,100.]])
is_real = np.array([True, True, False])
scores = Q @ K.T / np.sqrt(2.0)
causal_block = np.triu(np.ones((L,L), dtype=bool), 1)
padding_block = np.broadcast_to(~is_real[None,:], (L,L))
def attention(block):
    logits = np.where(block, -np.inf, scores)
    # Rows for padded queries are not used in loss; they still have valid keys here.
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    w = exp / exp.sum(axis=1, keepdims=True)
    return w, w @ V
for label, block in [
    ("unmasked", np.zeros((L,L), dtype=bool)),
    ("padding", padding_block),
    ("causal+padding", causal_block | padding_block)
]:
    w, out = attention(block)
    print(label, "\nweights", w, "\noutput", out)
    assert np.allclose(w.sum(axis=1), 1)
    assert np.all(w[block] == 0)
w, out = attention(causal_block | padding_block)
assert np.allclose(w[0], [1,0,0])
assert np.allclose(w[1], [0.5,0.5,0])
assert np.allclose(out[1], [1,2])
print("PASS")
```

**Important failure case:** a row with *all* keys masked yields `-inf - (-inf)`, i.e. NaNs. Real implementations should explicitly handle fully masked rows and define their output policy. Avoid asserting a softmax exists over an empty support.

**Extension:** reproduce this with PyTorch `scaled_dot_product_attention` after checking the installed version's mask polarity. Compare against a manual implementation, using a fixed random seed and tolerance. Explain why causal masking is necessary for teacher-forced next-token training.

### 4. Model-family comparison exercise

For three tasks—classifying an incident description, writing a structured incident summary from a source document, and predicting the next log template—identify the suitable attention visibility, target, and leakage risk. There may be multiple workable architectures. Distinguish an architectural preference from a measured result.

## B. Cybersecurity — drift, validation and defensible methodology

### 5. What changes over time?

**Covariate shift:** the feature distribution P(X) changes, potentially with stable P(Y|X). Example: a new identity provider changes the distribution of failed-login counts. **Concept drift:** P(Y|X) changes; for instance, a behavior previously associated with attacks becomes normal after a business process change. **Prior probability shift:** the prevalence P(Y) changes. In practice, mixtures occur, and delayed/incomplete incident labels make attribution difficult.

A detector's score distribution moving is evidence of change, not proof of an attack or a specific drift type.

```text
raw events -> timestamp + source + label availability audit
             -> chronological train | validation | untouched future test
             -> fit preprocessors on train only
             -> train detector -> select threshold on validation
             -> lock protocol -> test once
             -> monitor feature drift + delayed ground-truth quality
```

### 6. Worked example: why prevalence changes precision

Suppose sensitivity/recall=0.80 and FPR=0.02. At 1% incident prevalence, Bayes' rule gives

`PPV = (0.80*0.01) / (0.80*0.01 + 0.02*0.99) = 0.2878`.

At 0.1% prevalence, with the same conditional rates,

`PPV = (0.80*0.001) / (0.80*0.001 + 0.02*0.999) = 0.0385`.

Only ~3.85% of alerts are true incidents under these assumptions. A published recall/FPR pair cannot be interpreted without the deployment base rate. The assumed rates themselves may drift, so these are illustrations, not deployment forecasts.

### 7. Lab B — synthetic time shift, fixed validation threshold

Python: `python -m pip install numpy scikit-learn`. Run offline only. Labels are simulated and not representative of actual incidents.

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, average_precision_score
rng = np.random.default_rng(20261105)
n = 3600
time = np.arange(n)
# Feature prevalence shifts after the training period.
night = rng.binomial(1, np.where(time < 2400, .20, .55))
failures = rng.poisson(np.where(time < 2400, 2.0, 3.5))
new_device = rng.binomial(1, .12, n)
X = np.column_stack((night, failures, new_device))
# Stable conditional relationship; changing X distribution is covariate shift.
logit = -5.5 + .6*night + .45*failures + 1.0*new_device
p = 1/(1+np.exp(-logit))
y = rng.binomial(1,p)
train, val, test = slice(0,1800), slice(1800,2400), slice(2400,n)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X[train], y[train])
s_val = model.predict_proba(X[val])[:,1]
s_test = model.predict_proba(X[test])[:,1]
# A deterministic validation cutoff, not a guarantee of future alert volume.
threshold = np.sort(s_val)[-30]
for name, target, scores in [("validation",y[val],s_val),("future",y[test],s_test)]:
    pred = scores >= threshold
    print(name, "n", len(target), "prevalence", round(target.mean(),4),
          "alerts", int(pred.sum()),
          "precision", round(precision_score(target,pred,zero_division=0),4),
          "recall", round(recall_score(target,pred,zero_division=0),4),
          "AP", round(average_precision_score(target,scores),4))
print("feature mean failures train/future:", X[train,1].mean(), X[test,1].mean())
assert X[test,1].mean() > X[train,1].mean()
```

**Interpretation:** the changed input distribution alters alert counts even when the conditional data-generating mechanism is stable. The validation threshold may select more or fewer than 30 alerts in future. Ties can also cause the validation alert count to differ from 30. A deployment with a hard review capacity needs a documented queue/ranking policy rather than an assumed static-threshold guarantee.

**Controlled extension:** change the post-2400 label-generating coefficient on `failures` from +0.45 to -0.10 to simulate concept drift, without leaking that coefficient into training. Repeat 20 seeds; compare AP, precision, recall, alert count and confidence intervals. Report both drift scenarios separately.

### 8. Experimental design checklist

State prediction unit (one login, host-day, or incident); feature availability cutoff; label observation window; entity overlap; duplicate policy; missing values; class prevalence; split dates; training-only preprocessing; validation-selected thresholds; untouched final test; delayed labels; alert review budget; model/data versions; random seeds. A random split of duplicated network flows can inflate metrics through capture artifacts. Do not resample before splitting.

**Research question:** Does forward-time evaluation reduce apparent performance compared with random splitting of the same authentication dataset? Compare fixed model families and metrics, then inspect whether repeated entities and post-event enrichment explain the difference. A score difference alone does not prove leakage.

## C. Cross-track synthesis

A decoder's future-token leakage and a security detector's future-event leakage are structurally similar: both let the model access information unavailable at intended inference time. In each case, document the **information boundary** and test it. For language models, assert causal/padding masks. For cyber detection, enforce event-time feature cutoffs and chronological splits.

## D. Global AI and cybersecurity developments — checked 5 November 2026

1. **4 November — OpenAI announced an internal security sprint** with external security researchers to test and harden systems. Research angle: define scope, attacker assumptions, severity rubric and evidence of remediation; announcement is not evidence of a measured security improvement. Official source: https://openai.com/index/preparing-our-next-security-sprint-to-harden-openai/
2. **4 November — OpenAI announced a planned acquisition of Pierre Computer Company**, describing developer code browsing and diff tooling. Research angle: study how reliable code context and provenance affect coding-agent evaluations; an announced acquisition is not proof of a completed integration. Official source: https://openai.com/index/openai-to-acquire-pierre/
3. **2 November — Grid Protectors Program:** OpenAI described planned flexible demand arrangements for future compute campuses. Research angle: quantify the latency/availability trade-off of workload curtailment, rather than assuming that all AI compute is interruptible. Official source: https://openai.com/index/grid-protectors-program/
4. **November — CISA President's Cup Game of the Month, “Inception”**: an official security skills exercise in software security assessment and vulnerability analysis. This is training content, **not** an active vulnerability advisory. Work only in the authorized challenge environment. Source: https://presidentscup.cisa.gov/game-of-the-month/

**News methodology:** Official publisher pages were consulted for these dated items. Announcements and training exercises are distinguished from independently measured deployment outcomes. This selection is global-topic coverage, not an exhaustive incident feed.

## E. Learning resources (credible, reusable)

- Vaswani et al., *Attention Is All You Need* (2017), https://arxiv.org/abs/1706.03762 — read masking and encoder/decoder sections.
- Dive into Deep Learning, attention chapter: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html — tutorial and exercises.
- PyTorch scaled dot product attention reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html — verify current mask conventions.
- Hugging Face NLP course: https://huggingface.co/learn/llm-course/chapter1/1 — encoder vs decoder overview.
- Google *Rules of Machine Learning*: https://developers.google.com/machine-learning/guides/rules-of-ml — practical evaluation and monitoring.
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework — governance and evaluation framing.
- DeepLog paper: https://dl.acm.org/doi/10.1145/3133956.3134015 — security log-sequence modeling; note normal-behavior assumptions.
- **Audio/podcast learning:** TWIML AI Podcast https://twimlai.com/podcast/ (find episodes on Transformers/evaluation); Risky Business https://risky.biz/ (security news and practitioner analysis). These are series recommendations, not claims about a specific newly released episode.
- **Video lectures:** Stanford CS25 Transformers seminar archive https://web.stanford.edu/class/cs25/ ; Stanford CS229 lectures https://cs229.stanford.edu/ . Verify a particular lecture's availability before citing it as completed viewing.

## F. Knowledge check — answer before opening the key

1. Why is a padding mask not interchangeable with a causal mask?
2. What happens when every key in a softmax row is masked with -infinity?
3. Can an encoder-only model see later unmasked input positions?
4. Does an attention map alone prove why a prediction was made?
5. Define P(X) drift versus P(Y|X) drift.
6. With recall .8, FPR .02 and prevalence .001, estimate precision.
7. Why must a threshold be selected before looking at the final test?
8. Can a validation top-30 threshold guarantee 30 future alerts?
9. Give one concrete example of target leakage in incident telemetry.
10. What connects causal attention and chronological security evaluation?

**Answer key:** 1. Padding excludes nonexistent input tokens; causal excludes future positions. 2. NaN unless explicitly handled. 3. Yes, absent another mask. 4. No. 5. Covariate versus concept drift. 6. About 3.85%. 7. To avoid optimistic selection bias. 8. No, drift and ties alter counts. 9. A feature populated only after analyst incident closure. 10. Both enforce inference-time information availability.

## G. Deliverables and evening handoff

Save: (a) attention mask output and three assertions; (b) synthetic drift experiment metrics; (c) a two-paragraph methods critique; (d) one research hypothesis and a failure-analysis note. **Evening deep dive should stay on today's same topics:** attention mask edge cases, positional encoding comparison, temporal/entity splits, drift diagnostics and statistical uncertainty. Do not advance to a new session in the evening.

**Narration handoff:** this Markdown is the canonical lesson. Audio is a separate GitHub Actions product; no MP3 existence or delivery is asserted here. No email should be sent by the preparation step.
