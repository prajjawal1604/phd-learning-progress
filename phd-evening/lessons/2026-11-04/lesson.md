# Evening PhD Deep Dive — 4 November 2026 (Wednesday)
**Pathways:** Advanced Generative AI and Foundation Models; AI for Cybersecurity (Marwadi University). **Progress:** Same-day expansion of morning Unit 1 Session 4 (attention/position and detection/evaluation). The generative pathway formally introduces Transformers in Unit 2; this accelerated bridge is preparatory, not a replacement for the Unit 1 tokenization checkpoint. No progression to Session 5 today.

## Outcomes and plan (about 3 hours)
Derive attention with explicit tensor shapes and masks; compare absolute and rotary position schemes; compute gradients for a toy attention head; audit rare-event precision, calibration and threshold economics; run two reproducible Python labs; design a falsifiable research comparison. Allocate 45 minutes to derivations, 45 to Lab A, 40 to Lab B, 25 to research reading and 25 to quiz/review.

## A. Attention from first principles

For batch B, sequence length T, model width D and H heads with head width d=D/H:
- Input X: (B,T,D); learned Wq,Wk,Wv: (D,D).
- Reshape projected Q,K,V from (B,T,D) into (B,H,T,d).
- Scores S=Q Kᵀ / sqrt(d): (B,H,T,T). Add a mask **before** softmax.
- A=softmax(S+M) across key positions, O=A V: (B,H,T,d).
- Transpose/concatenate heads to (B,T,D), then multiply by Wo: (D,D).

The attention output is a weighted combination of value vectors, not a probability that an answer is correct. Row-wise normalization means weights sum to one over **allowed** keys. A causal mask forbids j>i. A padding mask forbids invalid key positions; fully masked query rows need explicit handling, otherwise softmax can produce undefined or misleading values.

```text
input tokens (B,T) -> embeddings (B,T,D) + position
                                 |
                         Q,K,V projections
                                 |
                     split H heads (B,H,T,d)
                                 |
                     QK^T / sqrt(d) + mask
                                 |
                     softmax -> weights A
                                 |
                         A @ V -> concat
                                 |
                        output projection
                                 |
                   residual + norm + FFN
```

### Worked numerical example: masking and gradients
For one query q=[1,0], keys k1=[1,0], k2=[0,1], values v1=[2,0], v2=[0,4], d=2: scores=[0.707107,0], weights=[0.669762,0.330238]. Output=[1.339524,1.320952]. If key 2 is future information, its logit is masked and output=[2,0].

Let loss L be the **first coordinate** of the unmasked output. L=2a1. Since a1=exp(s1)/(exp(s1)+exp(s2)), ∂a1/∂s1=a1(1−a1)=0.22118, so ∂L/∂s1=0.44236. Since s1=q·k1/sqrt(2), ∂L/∂q1=0.44236/sqrt(2)=0.3128. This is a local derivative holding keys/values fixed, not the gradient through an entire Transformer. Masking key 2 gives a1=1 and this particular gradient becomes zero: the normalized attention has no competing choice.

**Why scale?** If independent zero-mean query/key components each have variance one, dot-product variance grows approximately with d; dividing by sqrt(d) stabilizes its scale. Real learned features are not necessarily independent.

### Position representations
**Learned absolute embeddings:** add p_i to x_i; flexible at seen positions but extrapolation beyond trained positions is not guaranteed. **Sinusoidal:** deterministic sine/cosine functions with multiple frequencies; positions are distinguishable without a learned lookup table. **Rotary (RoPE):** rotate pairs of query/key coordinates by position-dependent angles, giving relative-offset structure in their dot products. None alone proves robust long-context retrieval.

**Complexity caveat:** Dense attention forms O(T²) score interactions per head; implementations such as FlashAttention reduce memory traffic and avoid materializing the full attention matrix, but do not magically eliminate all quadratic arithmetic. Feed-forward blocks and projection costs also matter.

## Lab A — masked multi-head attention with checks
Environment: Python 3.11+, NumPy. Install: `python -m pip install numpy`. Save as `attention_evening.py`.

```python
import numpy as np
rng = np.random.default_rng(7)
B,T,D,H = 2,4,8,2
d = D//H
x = rng.normal(size=(B,T,D))
wq,wk,wv,wo = [rng.normal(scale=1/np.sqrt(D),size=(D,D)) for _ in range(4)]
def heads(z,w):
    return (z@w).reshape(B,T,H,d).transpose(0,2,1,3)
q,k,v = heads(x,wq),heads(x,wk),heads(x,wv)
scores = q @ k.transpose(0,1,3,2) / np.sqrt(d)
mask = np.triu(np.ones((T,T),dtype=bool),1)
scores = np.where(mask[None,None,:,:], -np.inf, scores)
z = scores - np.max(scores,axis=-1,keepdims=True)
p = np.exp(z); p /= p.sum(axis=-1,keepdims=True)
out = (p@v).transpose(0,2,1,3).reshape(B,T,D)@wo
assert p.shape==(B,H,T,T)
assert out.shape==(B,T,D)
assert np.allclose(p.sum(axis=-1),1)
assert np.all(p[:,:,mask]==0)
assert np.allclose(p[:,:,0,0],1)
print("attention shapes:",p.shape,out.shape)
print("first head weights:\n",np.round(p[0,0],3))
# No future leakage: altering last token must not affect earlier causal outputs.
x2=x.copy(); x2[:, -1, :]+=100
q2,k2,v2=heads(x2,wq),heads(x2,wk),heads(x2,wv)
s2=np.where(mask[None,None,:,:],-np.inf,q2@k2.transpose(0,1,3,2)/np.sqrt(d))
e=np.exp(s2-np.max(s2,axis=-1,keepdims=True)); p2=e/e.sum(axis=-1,keepdims=True)
o2=(p2@v2).transpose(0,2,1,3).reshape(B,T,D)@wo
assert np.allclose(out[:,:-1],o2[:,:-1],atol=1e-10)
print("causal future-token invariance: PASS")
```

**Ablations:** (1) remove the mask and demonstrate future-token dependence; (2) change T from 4 to 16 and report score-tensor elements; (3) add padding on the final token, masking its **key column** and excluding padded queries from loss; (4) compare softmax entropy with and without sqrt(d) scaling over 20 random seeds. Record shape traces, exact seeds, means and variance. Avoid claiming attention heatmaps provide causal explanations.

## B. Cybersecurity: rare-event evaluation beyond accuracy

The morning lesson established time-aware splits and precision/recall. Tonight: calibrate and quantify uncertainty before selecting operational thresholds.

### Worked base-rate shift
Suppose sensitivity=0.80 and FPR=0.02. At prevalence π=0.01:
PPV = 0.80×0.01 / (0.80×0.01 + 0.02×0.99) = 0.28777.
At π=0.001, PPV = 0.0008/(0.0008+0.01998) ≈ 0.03850.
**Same detector operating characteristics, dramatically different analyst experience.** The second computation assumes sensitivity and FPR remain stable despite the population change; in practice they may drift too.

### Calibration versus discrimination
Ranking asks whether incidents score above benign events; calibration asks whether events assigned 0.2 risk occur about 20% of the time in an appropriate evaluation slice. Average precision is a ranking summary, not a calibration metric. Brier score is mean (predicted probability−label)². Calibration curves require enough positives per bin; with very rare incidents, report bin counts and uncertainty. Fit any calibration mapping on a separate validation segment, not the final test.

### Triage economics
Assume 20,000 daily events, 0.5% incidents (100). At 80% recall and 1% FPR: TP=80, FN=20, FP≈199, total alerts≈279. At 12 minutes per alert, daily review demand≈55.8 analyst-hours. If the team can handle 100 alerts/day, select a threshold using prior validation data and report recall at that capacity. Avoid using a fixed percentage threshold as if it guaranteed a fixed future volume.

```text
event ingestion -> as-of-time features -> chronological split
                                           |
train -> fit model -> validation -> choose calibration and threshold
                                           |
                                  locked future test
                                           |
                    precision@budget, recall, FP/day, Brier
                                           |
                          monitored analyst feedback
```

## Lab B — imbalanced detection, capacity and uncertainty
Python 3.11+, NumPy and scikit-learn: `python -m pip install numpy scikit-learn`. Synthetic data are for method validation, not operational efficacy.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             average_precision_score, brier_score_loss)
rng=np.random.default_rng(42)
n=6000
failed=rng.poisson(2,size=n)
new_device=rng.binomial(1,.15,size=n)
off_hours=rng.binomial(1,.25,size=n)
# Deliberate temporal shift in baseline incident probability.
logit=-5.5+.5*failed+1.0*new_device+.6*off_hours+(.35*(np.arange(n)>=4000))
p=1/(1+np.exp(-logit))
y=rng.binomial(1,p)
X=np.column_stack([failed,new_device,off_hours])
tr,va,te=slice(0,3500),slice(3500,4500),slice(4500,6000)
clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
clf.fit(X[tr],y[tr])
sv=clf.predict_proba(X[va])[:,1]
st=clf.predict_proba(X[te])[:,1]
# Capacity selection by validation quantile; ties can exceed the budget.
capacity=50
threshold=np.sort(sv)[-capacity]
pred=st>=threshold
print("threshold",round(threshold,5))
print("validation alerts",int((sv>=threshold).sum()),"test alerts",int(pred.sum()))
print("test prevalence",round(y[te].mean(),4))
print("precision",precision_score(y[te],pred,zero_division=0))
print("recall",recall_score(y[te],pred,zero_division=0))
print("average precision",average_precision_score(y[te],st))
print("Brier",brier_score_loss(y[te],st))
# Bootstrap only the locked test predictions; intervals ignore training uncertainty.
ids=np.arange(len(st)); vals=[]
for _ in range(500):
    ix=rng.choice(ids,size=len(ids),replace=True)
    vals.append(precision_score(y[te][ix],pred[ix],zero_division=0))
print("bootstrap precision 95% percentile interval",np.quantile(vals,[.025,.975]))
```

**Interpretation:** A threshold selected for 50 alerts in 1,000 validation records need not produce 75 alerts in 1,500 future records, because prevalence and score distribution can change. Ties may exceed the validation budget. Bootstrap percentiles can be unstable with few positive alerts and do not fix biased labels or time dependence; consider day-block bootstrap for correlated deployment data. Never reselect the threshold on test results.

### Research-grade extension
Compare (A) simple failure-count rule, (B) logistic regression and (C) frozen sequence embeddings with a linear head. Train on earlier weeks, calibrate/choose thresholds on a later validation period, and evaluate on a subsequent untouched month. Keep label horizon and analyst budget identical. Report incident prevalence, PR-AUC, precision@100, recall@100, Brier score, false alerts/day, 95% intervals and failure categories. Ablate device features, late-arriving labels and duplicated entities. Track dataset and model versions, feature availability timestamps, compute and energy where measurable.

**Research question:** Does positional sequence modeling improve precision@100 under temporal shift over count-only features? **Null hypothesis:** no improvement. Use repeated temporal windows or independent deployments for stronger evidence; a single synthetic dataset cannot answer this question.

## Reading, tutorials, audio
- Vaswani et al., *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762 — trace attention equations and shapes.
- Dao et al., *FlashAttention* (2022): https://arxiv.org/abs/2205.14135 — distinguish IO optimization from architectural sparsity.
- Harvard NLP, *The Annotated Transformer*: https://nlp.seas.harvard.edu/annotated-transformer/ — compare code and equations.
- Dive into Deep Learning, attention chapter: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html — practice matrix calculations.
- scikit-learn, probability calibration: https://scikit-learn.org/stable/modules/calibration.html — calibration diagnostics and caveats.
- scikit-learn, precision-recall: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html — understand imbalanced ranking.
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework — assurance and monitoring context.
- Podcast series: *Practical AI* https://changelog.com/practicalai ; *SANS Internet Storm Center Stormcast* https://isc.sans.edu/podcast.html . Choose relevant recent episodes; these links are series, not verified episode claims.
- Video lecture: Stanford CS25 https://web.stanford.edu/class/cs25/ — transformer seminars.

## Quiz — attempt before answers
1. With B=3,T=12,D=384,H=6, what is the attention-score shape and d?
2. Why is a causal mask applied before softmax?
3. What happens if a padding key is not masked?
4. Can an attention map alone establish which input caused a final prediction?
5. Why might RoPE generalize differently from learned absolute embeddings?
6. Calculate precision with sensitivity 90%, FPR 1%, prevalence 0.1%.
7. Explain the distinction between average precision and Brier score.
8. Why must a threshold be chosen before the locked future test?
9. Why might an IID bootstrap interval be too narrow for daily correlated alerts?
10. What evidence would refute the sequence-embedding hypothesis?

**Answer key:** (1) (3,6,12,12), d=64. (2) Forbidden future positions must have zero probability. (3) Valid queries may attend to meaningless padding and change output. (4) No; attention weights alone are not causal attribution. (5) RoPE encodes relative offsets via rotations while a learned table has trained position entries; neither guarantees extrapolation. (6) 0.0009/(0.0009+0.00999)≈8.26%. (7) AP summarizes ranking on positive examples; Brier measures squared probability error. (8) Otherwise test leakage and optimistic reporting. (9) Temporal correlation reduces effective independent sample count. (10) Repeated held-out windows show no improvement or consistent degradation at fixed capacity, with uncertainty quantified.

## Research log and deliverables
Produce (i) annotated attention shapes and one causal-invariance assertion; (ii) one page comparing absolute/sinusoidal/rotary positions and limitations; (iii) Lab B metrics, threshold, confusion counts, uncertainty and calibration note; (iv) a 250-word paper critique; (v) a preregistered temporal evaluation plan with a baseline and explicit alert budget. Record negative findings and avoid overstating synthetic results.

## Narration and delivery handoff
Canonical Markdown path: `phd-evening/lessons/2026-11-04/lesson.md`. A push to main is intended to trigger GitHub Actions narration. Audio availability must be confirmed from a successful workflow and accessible MP3, never inferred from the commit. This is preparation only; do **not** send email.
