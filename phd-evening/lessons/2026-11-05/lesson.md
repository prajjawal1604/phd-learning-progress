# PhD Evening Deep Dive — Thursday, 5 November 2026
**Tracks:** Advanced Generative AI and Foundation Models; AI for Cybersecurity. **Same-day continuation:** morning Unit 2 Session 2 (attention masks, position, Transformer families) and Cybersecurity Unit 1 Session 5 (drift, temporal evaluation). This is an independent accelerated study session, not a substitute for university contact hours.

**Time budget:** 45 min theory, 65 min generative-AI lab, 65 min cybersecurity lab, 35 min research and quiz. **Learning outcomes:** derive attention-mask behavior and rotary position effects; implement safe masking including empty-support rows; distinguish covariate/concept/prior drift; design temporal and entity-held-out evaluations with uncertainty; defend an inference-time information boundary.

## Part I — Transformer information boundaries

### 1. From logits to legal attention

For query q_i and key k_j, the scaled logit is s_ij = (q_i·k_j)/sqrt(d_k). Define legal support S_i = {j: key j is nonpadding AND (not causal OR j<=i)}. The attention probability is exp(s_ij)/sum_{k in S_i} exp(s_ik) for j in S_i and zero elsewhere. If S_i is empty, the ordinary softmax is **undefined**; choose an explicit zero-output policy for padded query positions, or omit those queries from loss. Never assume a framework automatically protects against every fully masked row.

**Numerical worked example:** q=[1,0], keys k0=[1,0], k1=[0,1], d_k=2. Scaled scores are [0.7071,0]. With both keys legal, probabilities are exp(.7071)/(exp(.7071)+1)=.6698 and .3302. Values v0=[2,0], v1=[0,4] give output [.6698×2,.3302×4]=[1.3396,1.3208]. With a causal mask at position 0, key1 is forbidden and the output is exactly [2,0]. If key0 is padding too, there is no legal key: do not produce a fictitious probability distribution.

**Mask polarity warning:** In PyTorch scaled_dot_product_attention, a boolean attention mask's True means an element **participates** in attention; in other APIs boolean masks may mean **blocked**. Test against a manual oracle for your installed version. Float masks are additive logit biases.

```text
INPUT tokens:      [START] [LOGIN] [FAIL] [PAD]
KEY PADDING:          ✓      ✓      ✓     ×
CAUSAL visibility:
query START           ✓      ×      ×     ×
query LOGIN           ✓      ✓      ×     ×
query FAIL            ✓      ✓      ✓     ×
query PAD             ignored in training loss
                      ↓
legal logits → stable softmax → weighted values → projection
```

### 2. Position is not the same as visibility

Self-attention without positional information treats consistently permuted token order equivariantly. Causal masking imposes directionality but does not give every architecture a full representation of relative distance. Sinusoidal absolute embeddings add a position vector to each token; learned embeddings optimize a finite table; RoPE rotates pairs of query/key coordinates as a function of token position. For a 2D pair, R(theta)=[ [cos theta,-sin theta],[sin theta,cos theta] ]. Orthogonality implies (R(i theta)q)·(R(j theta)k)=q·R((j-i)theta)k: the dot product depends on relative offset. RoPE alone does not guarantee extrapolation beyond training lengths.

**Worked RoPE calculation:** q=k=[1,0], theta=pi/2 per position. At positions i=0,j=1, the rotated dot product is cos(pi/2)=0. At i=j, it is 1. A change in relative position changes similarity even with identical unrotated vectors. Real heads use multiple frequencies.

**Research caveat:** attention weights are not a faithful causal explanation by themselves. Evaluate model behavior with interventions, ablations and controlled counterfactuals, not colorful attention maps alone.

### 3. Lab A — masked attention oracle and RoPE checks

Prerequisites: Python 3.11+, `pip install numpy`. Save as `evening_attention.py`. CPU-only, deterministic; expected terminal result: `ALL CHECKS PASSED`.

```python
import numpy as np
rng = np.random.default_rng(20261105)
def safe_attention(q, k, v, allowed):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    out = np.zeros((q.shape[0], v.shape[1]))
    weights = np.zeros((q.shape[0], k.shape[0]))
    for i in range(q.shape[0]):
        idx = np.flatnonzero(allowed[i])
        if len(idx) == 0:  # explicit empty-support policy
            continue
        z = scores[i, idx]
        p = np.exp(z - z.max())
        p /= p.sum()
        weights[i, idx] = p
        out[i] = p @ v[idx]
    return weights, out
q = np.zeros((4,2)); k = np.zeros((4,2))
v = np.array([[2.,0.],[0.,4.],[6.,6.],[999.,999.]])
real = np.array([True,True,True,False])
allowed = (np.arange(4)[:,None] >= np.arange(4)[None,:]) & real[None,:]
allowed[~real,:] = False
w, y = safe_attention(q,k,v,allowed)
assert np.allclose(w[0],[1,0,0,0])
assert np.allclose(w[1],[.5,.5,0,0])
assert np.allclose(y[1],[1,2])
assert np.allclose(w[3],0) and np.allclose(y[3],0)
assert np.isfinite(y).all()
def rotate(x, angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([c*x[0]-s*x[1],s*x[0]+c*x[1]])
base=np.array([1.,0.])
assert np.isclose(rotate(base,0) @ rotate(base,np.pi/2),0,atol=1e-12)
assert np.isclose(rotate(base,np.pi/2) @ rotate(base,np.pi/2),1)
print("weights:\n",w,"\noutputs:\n",y)
print("ALL CHECKS PASSED")
```

**Ablations:** (a) remove key-padding exclusion and observe contamination by [999,999]; (b) remove causal restriction and measure future-token access; (c) test an all-masked row; (d) compare two identical token embeddings at different RoPE offsets. Explain why zeroing the padded query output and excluding its loss are separate choices. **Stretch:** implement the same with PyTorch's functional attention and compare to this oracle at tolerance 1e-5; use explicit mask polarity.

## Part II — Security detection under distribution shift

### 4. Three kinds of drift, three different diagnoses

Covariate drift means P_t(X) changes. Concept drift means P_t(Y|X) changes. Label/prior shift means P_t(Y) changes (strict label-shift assumptions also require stable P(X|Y)). In a live SOC, delayed incident adjudication means observed Y can lag X by days. Monitoring only feature distributions cannot prove concept drift. A change in alert volume may reflect an upstream log-source migration, attack activity, or a threshold bug.

```text
Event time → ingestion → feature cutoff → model score → triage
     |                       |                   |
     |                       |                   └→ delayed analyst label
     |                       └→ freeze what was known at decision time
     └→ chronological train / validation / future test
                                  ↓
              compare feature shift + alert rate + eventual labels
                                  ↓
              investigate → calibrate/retrain only on approved data
```

**Worked base-rate example:** suppose recall=.8, false-positive rate=.02 and incident prevalence=.001. Precision = .8×.001/(.8×.001+.02×.999)=.0385. In 100,000 events, expected true positives=80 and false positives=1,998; approximately 2,078 alerts for 80 true incidents. At a review capacity of 100 alerts/day, the fixed threshold is operationally infeasible. Raising threshold may reduce FPR but typically changes recall; both must be remeasured on representative future data. Do not report an unmeasured performance improvement.

### 5. Time splits, entities and uncertainty

A random split can put near-identical records from one attack campaign in both train and test. A forward-time split protects chronology, but the same hosts can still appear across periods; an entity-held-out test measures generalization to unseen hosts/users. These test different questions. Define feature timestamps (what was known at prediction time), label maturity, and threshold selection **before** looking at final test results.

For alert precision, a naive Wald interval is unstable with rare positives. Bootstrap by **independent entity or time block**, not by highly correlated individual events. Even bootstrap intervals cannot repair biased or immature labels.

### 6. Lab B — covariate vs concept shift, temporal evaluation

Prerequisites: `pip install numpy scikit-learn`. Save as `evening_drift.py`. Uses synthetic authentication events; no production data, no offensive activity.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score
rng=np.random.default_rng(20261105)
n=6000
t=np.arange(n)
future=t>=4000
fail=rng.poisson(np.where(future,4.0,1.5))
new=rng.binomial(1,.15,n)
night=rng.binomial(1,np.where(future,.5,.2))
X=np.column_stack([fail,new,night])
sigmoid=lambda z: 1/(1+np.exp(-z))
# Shared conditional mechanism: covariate shift in the future.
logit=-5+.45*fail+.8*new+.4*night
y_cov=rng.binomial(1,sigmoid(logit))
# Alternative concept shift only for future labels.
logit_con=logit.copy()
logit_con[future]=-5-.15*fail[future]+.8*new[future]+.4*night[future]
y_con=y_cov.copy()
y_con[future]=rng.binomial(1,sigmoid(logit_con[future]))
train=np.arange(0,3000); val=np.arange(3000,4000); test=np.arange(4000,n)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
model.fit(X[train],y_cov[train])
sv=model.predict_proba(X[val])[:,1]
st=model.predict_proba(X[test])[:,1]
# Select on validation only. Quantile is not a future alert-capacity guarantee.
threshold=np.quantile(sv,.95)
def report(label,y):
    pred=st>=threshold
    return dict(scenario=label,prevalence=round(y.mean(),4),
        alerts=int(pred.sum()),precision=round(precision_score(y,pred,zero_division=0),4),
        recall=round(recall_score(y,pred,zero_division=0),4),
        average_precision=round(average_precision_score(y,st),4))
print("validation threshold",threshold)
print("mean fail train/future",X[train,0].mean(),X[test,0].mean())
print(report("covariate",y_cov[test]))
print(report("concept",y_con[test]))
assert X[test,0].mean()>X[train,0].mean()
assert 0<=threshold<=1
```

**Interpretation assignment:** Why can the same model scores have different precision/recall when only the future label mechanism changes? Why does comparing average precision across datasets with different prevalence require care? Why must validation not be reselected after seeing test labels? Run 30 seeds, report mean and 5th/95th percentile for alert counts and average precision. Label this as **simulation variability**, not a population confidence interval.

**Extension design:** introduce host IDs with repeated correlated events. Compare chronological split and unseen-host split. Keep the last two weeks as untouched final test, allow a label maturation lag, and document entity-level bootstrap units. If labels are unavailable, report feature/score drift only; do not infer concept drift.

## Part III — Research synthesis and reproducibility

### 7. One unified idea: inference-time information availability

Causal attention prevents token t from using token t+1 during autoregressive prediction. Event-time feature governance prevents an incident detector from using analyst notes created after the event. Both are **information-set constraints**. A model may look excellent offline while violating the real prediction boundary. Write a one-page threat-to-validity statement covering leakage, dataset shift, correlated samples, delayed labels, threshold tuning, and multiple comparisons.

**Mini research proposal:** "How do attention-mask implementation errors and temporal feature leakage distort offline evaluation?" Hypotheses: (H1) allowing future information artificially improves measured predictive accuracy; (H2) chronology-aware evaluation reduces optimism under drift; (H3) entity-level uncertainty intervals are wider than event-level intervals when observations cluster. Avoid asserting these hypotheses are universally true. Specify two synthetic datasets, an ablation grid, metrics, seeds, and pre-registered success criteria.

**Deliverables tonight:** lab outputs and assertions; two drift scenarios; a diagram of the information boundary; one table of validation/test policies; a 250-word research critique; three next-day questions. Commit lab files only if separately requested; this lesson Markdown is the canonical narration input.

### 8. Quiz (attempt without notes)

1. What is the legal support set S_i in causal padded attention?
2. Why does softmax of an all -infinity row fail?
3. Calculate the two-key attention output for weights [.75,.25] and values [2,0],[0,4].
4. What does the relative-angle identity in RoPE establish?
5. Why is a high attention weight not a causal explanation?
6. Distinguish P(X) from P(Y|X) drift.
7. With prevalence .001, recall .8 and FPR .02, approximately how many false alerts per 100,000 events?
8. Why is an entity-held-out split not equivalent to a chronological split?
9. What is the difference between an alert budget and a fixed score threshold?
10. How can delayed labels bias a current precision estimate?
11. Why should preprocessing fit only on training data?
12. Name one failure mode common to causal language modeling and event-time incident detection.

**Answer key:** 1. Nonpadding keys at positions j<=i. 2. Normalization has zero valid mass and numerical NaNs. 3. [1.5,1]. 4. Rotation of Q/K makes their dot product sensitive to relative offset. 5. Attention is a mechanism, not proof of counterfactual causal importance. 6. Input marginal vs conditional target relation. 7. About 1,998. 8. One tests time generalization, the other new entities. 9. Capacity is a count constraint; threshold maps scores to decisions and counts may drift. 10. Recent unresolved positives may be mislabeled or missing. 11. Prevent leakage of validation/test distribution. 12. Access to information not available at actual inference.

### 9. Curated reading, tutorials, lectures and audio

**Primary research:** Vaswani et al. (2017), *Attention Is All You Need* https://arxiv.org/abs/1706.03762 ; Su et al., *RoFormer* https://arxiv.org/abs/2104.09864 ; Lipton et al., *Detecting and Correcting for Label Shift* https://arxiv.org/abs/1802.03916 ; Gama et al., *A Survey on Concept Drift Adaptation* https://doi.org/10.1145/2523813 .

**Implementation tutorials:** PyTorch scaled-dot-product attention https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html ; Dive into Deep Learning attention and Transformers https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html ; scikit-learn time-series cross-validation https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html ; NIST AI Risk Management Framework https://www.nist.gov/itl/ai-risk-management-framework .

**Lectures:** Stanford CS25 Transformers https://web.stanford.edu/class/cs25/ ; Stanford CS229 https://cs229.stanford.edu/ . **Podcasts:** TWIML AI https://twimlai.com/podcast/ for foundation-model/evaluation interviews; Risky Business https://risky.biz/ for operational security discussion. These are series and course catalogs, not assertions that a specific episode was published today.

### 10. Evening reflection

Explain in your own words why a causal mask and a chronological split are both tests of what a system is *allowed to know*. Record one unexpected lab output and whether it reflects a bug, a sampling artifact or an expected mathematical result. Prepare tomorrow's study notes without advancing tonight's unit.

**Narration and delivery boundary:** this Markdown is preparation-only. A push to main is intended to trigger GitHub Actions narration. The MP3 is not presumed to exist until a completed workflow and published file are verified. Do not send email from this preparation step.
