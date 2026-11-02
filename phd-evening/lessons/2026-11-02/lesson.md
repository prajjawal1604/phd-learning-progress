# PhD Evening Deep Dive — Day 2 — Monday, 2 November 2026

**Course pair:** Advanced Generative AI and Foundation Models + AI for Cybersecurity (Marwadi University PhD coursework).  
**Pathway alignment:** Both 19 October 2026 learning pathway PDFs, Unit 1 Session 2 and accompanying Unit 1 labs.  
**Morning source:** `phd-morning/lessons/2026-11-02/lesson.md`.  
**Evening focus:** **Stay on the same concepts**: autoregressive (AR) versus latent-variable models; supervised versus unsupervised versus semi-supervised cybersecurity learning. Deepen derivations, reproducibility, leakage control, decision thresholds, and the research hypothesis. Do **not** advance to the next syllabus session tonight.  
**Study allocation:** 20 min mathematical derivations, 30 min generative lab, 30 min security lab, 20 min paper critique, 15 min oral defense and research log. Longer optional experiments may take 1–2 additional hours.

> **Integrity rule.** Every number below is a hand-computable toy result or an explicitly described synthetic experiment, never a measured production result. Run the code and record actual outputs rather than copying expected findings. All security examples use synthetic, offline data; do not scan systems or process private logs without authorization.

## 1. The conceptual bridge from the morning

Morning distinguished `p(x_t | x_<t)` from a latent generative story `p(z)p(x|z)`; and supervised incident labels from unsupervised unusualness and semi-supervised provisional labels. Tonight ask a harder question: **What is identifiable from the evidence, what is merely an assumption, and what can be evaluated without looking into the future?**

| Question | AR model | Latent-variable model | Security analogue |
|---|---|---|---|
| What is observed? | Previous tokens and next token | Observation `x` | Time-valid event features |
| What is hidden? | Learned internal state, not necessarily explicit `z` | Explicit latent variable `z` | Unobserved operating regime, attacker intent |
| Training target | Conditional log likelihood | Marginal likelihood or a lower bound | Label likelihood, anomaly density, or pseudo-label objective |
| Output | Next-token distribution | Samples and approximate posterior | Risk score or anomaly score, **not** incident proof |
| Critical failure | Error propagation and exposure bias | Approximate inference error, posterior collapse | Base-rate error, label bias, drift and leakage |

### Learning objectives

By the end, derive an ELBO from a KL identity; compute a Gaussian KL by hand; demonstrate why `beta` changes latent usage; quantify how prevalence changes positive predictive value; build a time-ordered security evaluation with a fixed alert budget; and defend when pseudo-labeling can hurt rather than help.

---

# PART A — GENERATIVE MODEL THEORY AND EXPERIMENTS

## 2. Derive the ELBO without skipping the crucial equality

Start with a latent generative model: `p_theta(x,z)=p(z)p_theta(x|z)`. We want `log p_theta(x)`, where `p_theta(x)=integral p(z)p_theta(x|z) dz`. For a high-dimensional decoder the integral is often intractable. Introduce an approximate posterior `q_phi(z|x)`.

For any valid `q` whose support is compatible with the posterior:

`log p_theta(x) = E_q[log p_theta(x,z) - log q_phi(z|x)] + KL(q_phi(z|x) || p_theta(z|x)).`

**Proof sketch:** Expand the right-hand KL as `E_q[log q(z|x)-log p(z|x)]`. Bayes' rule gives `log p(z|x)=log p(x,z)-log p(x)`. Rearranging yields the equality. Since KL is nonnegative, the first term is a lower bound:

`ELBO(theta,phi;x) = E_q[log p_theta(x|z)] - KL(q_phi(z|x)||p(z)) <= log p_theta(x).`

There are **two different KLs** in this story. The training regularizer `KL(q(z|x)||p(z))` pushes the encoder toward the prior. The *gap* `KL(q(z|x)||p(z|x))` measures the approximation error between the chosen encoder and the true posterior. They are not interchangeable. A high ELBO can result from better reconstruction, a smaller regularizer, or both; it is not automatically a measure of semantic quality.

### Diagram A — what the model generates versus what it infers

```text
GENERATIVE PATH                          INFERENCE PATH
 prior p(z)                                observed x
     |                                         |
 sample z                                encoder q_phi(z|x)
     |                                         |
 decoder p_theta(x|z)                    mean mu, logvar
     |                                         |
 sample x'                               sample z ~ q
     |                                         |
 model distribution                      approximate posterior

Training connects both paths with reconstruction likelihood
and KL(q_phi(z|x) || p(z)); the inference path is not
the same as the generative prior.
```

## 3. Gaussian KL and the reparameterization trick

For one latent coordinate, let `q(z|x)=N(mu,sigma^2)` and `p(z)=N(0,1)`. The closed-form divergence is

`KL(q||p) = 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2)).`

**Worked example A1.** Set `mu=1`, `sigma=0.5`, so variance `0.25` and `log(0.25)=-1.386294`. Then

`KL = 0.5*(1 + 0.25 - 1 - (-1.386294)) = 0.818147 nats.`

If the decoder's expected log likelihood for this observation is `-1.20`, then the ELBO is `-1.20 - 0.818147 = -2.018147`. The true log evidence is **at least** this value, but cannot be determined from these two numbers alone. A lower bound is not an unbiased point estimate of evidence.

To optimize through sampling, write `z = mu_phi(x) + sigma_phi(x)*epsilon` where `epsilon ~ N(0,1)`. Sampling noise is then independent of the encoder parameters; gradients can flow through `mu` and `sigma`. This is the reparameterization trick in the original VAE paper. For discrete latent variables, the same pathwise Gaussian trick does not directly apply; alternative estimators or relaxations are needed.

**Worked example A2 — a three-dimensional encoder.** Suppose `mu=[1,0,-0.5]`, `variance=[0.25,1,4]`. Per-coordinate KL values are approximately `[0.8181,0,0.5*(0.25+4-1-ln4)=0.9319]`; total KL is about **1.7501 nats**. Independent diagonal-Gaussian coordinates make the KL additive; a full-covariance posterior needs a matrix expression.

### Why a VAE can ignore its latent variables

A sufficiently expressive decoder can explain `x` while making `q_phi(z|x)` almost equal to the prior for all inputs. Then the KL term is near zero, and different inputs no longer use distinguishable `z` values. This is **posterior collapse**. It can be especially relevant when a powerful autoregressive decoder already models local sequence structure well.

Watch for three signals together: (1) near-zero average KL per dimension; (2) reconstructions barely change when `z` is replaced with prior samples; (3) latent interpolation has little systematic effect. **Low KL alone is not proof** of collapse; simple data or an intentionally weak latent factor may justify low information use.

A beta-VAE changes the objective to `E_q[log p(x|z)] - beta*KL(q||p)`. When `beta>1`, regularization is stronger; when `beta<1`, latent usage may rise at the cost of weaker prior matching. Beta-VAE is not generally the same lower bound on `log p(x)` for arbitrary beta. A KL warm-up schedule can encourage the model to use `z` early, but it does not guarantee interpretable factors or solve all collapse mechanisms.

## 4. Autoregressive versus latent-variable generation: a careful comparison

The chain rule `p(x_1,...,x_T)=product_t p(x_t|x_<t)` is exact. An AR neural model learns each conditional. A latent-variable model instead integrates over a hidden state: `p(x)=integral p(z)p(x|z) dz`. These are **not mutually exclusive**. A decoder can itself be autoregressive: `p(x|z)=product_t p(x_t|x_<t,z)`.

Consider a maintenance-log sequence:

```text
08:00 motor_start -> 08:01 vibration_high -> 08:03 temperature_high
```

An AR model may assign high probability to the next event given preceding events. A latent regime model may infer `z=high-load` or `z=sensor-fault`, then generate plausible sequences under each regime. Neither probability alone tells us whether sabotage occurred. The same sequence can be probable under multiple causes; causal interpretation needs additional evidence or intervention.

**Doctoral question:** If an AR model gives lower held-out NLL than a VAE, does it make a better anomaly detector? **Not necessarily.** A density model can assign high likelihood to some out-of-distribution observations, and detection quality depends on the ranking, deployment population, alert budget and independent incident labels. Evaluate density and operational detection separately.

## 5. GenAI Python lab — exact posterior, KL and beta sensitivity (30 minutes)

This lab uses **only NumPy**, is fast on CPU, and produces numbers you can verify by hand. It complements the morning bigram model without requiring a GPU or pretending to train a large VAE.

Save as `day02_evening_latent.py`. Install with `python -m pip install numpy`.

```python
import numpy as np

# Toy binary latent variable: z=1 high-load; x=1 high-volume event.
prior = np.array([0.8, 0.2])       # P(z=0), P(z=1)
likelihood_x1 = np.array([0.1, 0.9])
joint = prior * likelihood_x1
evidence = joint.sum()
posterior = joint / evidence

assert np.isclose(evidence, 0.26)
assert np.allclose(posterior, [0.08 / 0.26, 0.18 / 0.26])
print("P(x=1):", evidence)
print("Exact posterior P(z|x=1):", posterior)

def discrete_kl(q, p):
    q, p = np.asarray(q, float), np.asarray(p, float)
    return np.sum(np.where(q > 0, q * np.log(q / p), 0.0))

def elbo(q):
    # q is an approximation to the exact posterior given x=1.
    return np.dot(q, np.log(likelihood_x1)) - discrete_kl(q, prior)

for q in [np.array([0.5, 0.5]), posterior, prior]:
    gap = discrete_kl(q, posterior)
    print("q:", q, "ELBO:", round(elbo(q), 6),
          "log evidence:", round(np.log(evidence), 6),
          "KL gap:", round(gap, 6))
    assert np.isclose(np.log(evidence) - elbo(q), gap)

def gaussian_kl(mu, logvar):
    mu, logvar = np.asarray(mu), np.asarray(logvar)
    return 0.5 * np.sum(mu**2 + np.exp(logvar) - 1 - logvar)

mu = np.array([1.0, 0.0, -0.5])
var = np.array([0.25, 1.0, 4.0])
print("Gaussian diagonal KL:", gaussian_kl(mu, np.log(var)))

# beta sensitivity: fix reconstruction term, vary beta.
reconstruction_loglik = -1.2
for beta in [0.25, 0.5, 1.0, 2.0]:
    objective = reconstruction_loglik - beta * gaussian_kl(mu, np.log(var))
    print("beta:", beta, "objective:", round(objective, 6))
```

**What this lab establishes:** The exact ELBO gap identity for a tractable discrete model, a diagonal-Gaussian KL, and the *numerical* effect of beta **holding model parameters fixed**. It does **not** establish that changing beta improves trained-model quality; that would require fitting models under matched budgets and seeds.

**Optional neural extension (PyTorch, 20–40 minutes):** Construct a two-dimensional synthetic mixture; train a small MLP encoder outputting `mu, logvar`, sample `z=mu+exp(0.5*logvar)*eps`, and decode a Gaussian mean. Train separate models for `beta=0.25,1,2` with identical seeds, data splits, optimizer, epochs and architecture. Plot held-out reconstruction loss and KL by epoch. Report whether one latent coordinate is unused; do not claim disentanglement from a scatterplot alone.

**Lab questions:** Why does the posterior equal `[0.3077,0.6923]`? Why does `q=posterior` close the ELBO gap exactly? What happens if you set `q=prior`? What extra experimental control is required before claiming beta prevents collapse?

---

# PART B — CYBERSECURITY LEARNING UNDER REALISTIC OPERATING CONSTRAINTS

## 6. A security classifier is a decision policy, not just a model

The morning example had 100 attacks in 10,000 sessions, 80 true positives and 198 false positives. It produced **278 alerts** and precision **80/278=28.78%**, even with 80% recall and 97.82% accuracy. Tonight derive the general base-rate effect.

Let `pi=P(attack)`, sensitivity `r=P(flag|attack)`, and false-positive rate `f=P(flag|benign)`. Bayes' theorem gives:

`P(attack|flag) = r*pi / (r*pi + f*(1-pi)).`

With `r=0.8`, `f=0.02` and `pi=0.01`, precision is `0.008/(0.008+0.0198)=0.28777`. If prevalence drops to `pi=0.001`, precision becomes `0.0008/(0.0008+0.01998)=0.03850`. **A classifier can retain the same conditional error rates while its alert queue becomes overwhelmingly benign.**

The base rate itself may change by tenant, time of day, attack campaign or labeling practice. Do not assume that a validation set with deliberately oversampled attacks has deployment prevalence. Probability calibration under one population may fail under another; a prior-probability correction is valid only under specific shift assumptions (for example, stable class-conditional feature distributions), which should be tested rather than presumed.

### Diagram B — defensible security evidence flow

```text
timestamped synthetic / authorized events
             |
       feature builder
    (prediction-time only)
             |
   chronological train period
       /       |       \
  supervised  anomaly  semi-supervised
  classifier  detector  pseudo-label path
       \       |       /
        validation period
   budget, threshold, calibration
             |
       FROZEN POLICY
             |
        future test period
             |
   TP/FP/FN/TN, recall@budget,
   precision, volume, drift
             |
 analyst evidence and review
             |
 delayed, audited outcome labels
```

**Never** let the future test period influence scaling, feature selection, pseudo-label selection, calibration, threshold choice or hyperparameter tuning. A label may arrive after an alert; its eventual value is valid for evaluation, but not for reconstructing what the detector knew at prediction time.

## 7. Three learning paradigms — the assumptions behind them

**Supervised:** Learn `p(y|x)` from reviewed incident labels. Logistic regression is a transparent baseline. Risks: rare positives, class imbalance, mislabeled incidents, time-dependent feature leakage, and training on alerts selected by an earlier detector. A probability score is meaningful as a risk estimate only after evaluating calibration and prevalence shift.

**Unsupervised:** Learn a reference distribution or isolation score from features without attack labels. An Isolation Forest score orders observations by unusualness; it is not `P(attack|x)`. The training reference may already contain attacks. Rare legitimate deployments can be more anomalous than a common attack. The sign of `score_samples` matters: lower raw scores mean more anomalous, so the lab negates it.

**Semi-supervised:** Use a small labeled subset plus unlabeled records. Pseudo-labeling accepts predictions as temporary training targets; it can reinforce the model's mistakes. Confidence thresholds are not guarantees of correctness, especially when the labeled set is enriched for incidents. Other semi-supervised approaches use consistency regularization or graph assumptions; each needs a justified relationship between unlabeled structure and the target labels.

**Positive-unlabeled nuance:** If you have only confirmed attacks and everything else is unlabeled, the unlabeled set is **not** a verified benign class. Treating it as negative induces label noise. The observed labeled-positive mechanism matters: some attacks are much more likely to be investigated and confirmed than others. A PU-learning method requires assumptions about the labeling process; no technique makes missing labels disappear.

## 8. Threshold selection, calibration and alert-budget math

A model's raw score can be excellent for ranking while poorly calibrated. **Discrimination** asks whether positives tend to rank above negatives (ROC-AUC, PR-AUC). **Calibration** asks whether events assigned probability `0.2` actually occur about 20% of the time in an appropriate population. **Decision utility** asks whether acting on a score helps given costs and a review budget. These are three different questions.

Suppose a SOC can review 30 of 1,000 sessions per day. A policy might set a score threshold at the 97th percentile of a **validation** period. That yields roughly 3% alerts on that period, subject to ties; it does not guarantee exactly 30 alerts tomorrow. Drift may raise or lower the number. A true top-30 policy enforces the budget on the day being scored but requires knowing all that day's scores before ranking, which introduces latency and cannot be used for immediate streaming decisions.

**Worked example B1 — compare under one budget.** Suppose on 1,000 held-out sessions there are 20 attacks. At 30 alerts, detector A captures 12 attacks, detector B captures 9. Then recall@30 is 60% for A and 45% for B, and precision@30 is 40% and 30%. A may be preferable for that budget even if B has a better ROC-AUC over all thresholds. The counts are illustrative; a credible study reports uncertainty across several time blocks and avoids selecting the winner on the final test.

**Worked example B2 — calibration error.** Suppose a score bin contains 100 validation sessions with mean predicted risk 0.20, but 8 confirmed attacks. The empirical event rate is 0.08, so the absolute bin calibration gap is 0.12. This does not prove the entire model is miscalibrated; the bin may be small or selectively labeled. Reliability plots should include sample counts and outcome completeness.

**Operational cost model:** Let `C_FP` be analyst time cost for a false alert and `C_FN` the cost of a missed incident. A simple expected loss is `C_FP*FP + C_FN*FN`, but real costs vary with severity and correlated incidents. State your utility assumptions before optimizing thresholds.

## 9. Cybersecurity Python lab — temporal split, three scores and a fixed budget

Save as `day02_evening_security.py`. Install `python -m pip install numpy scikit-learn`. The script generates synthetic sessions with time drift, uses **training-only** normalization, fits supervised and unsupervised baselines, builds a deliberately simple pseudo-label model, and evaluates all three using a **validation-derived** threshold. It uses no live telemetry.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, average_precision_score

rng = np.random.default_rng(20261102)
n = 12000
time_index = np.arange(n)
# Synthetic 1% attacks, with two overlapping features.
y = (rng.random(n) < 0.01).astype(int)
X = np.column_stack([
    rng.normal(size=n) + 1.8*y,
    rng.normal(size=n) + 1.3*y,
    rng.normal(size=n) + 0.5*y,
])
# Benign operating regime changes after training; not an attack.
X[7000:, 0] += 0.4

tr = np.arange(0, 7000)
va = np.arange(7000, 9500)
te = np.arange(9500, n)
assert tr.max() < va.min() < te.min()
scaler = StandardScaler().fit(X[tr])
Z = scaler.transform(X)
Ztr, Zva, Zte = Z[tr], Z[va], Z[te]

# Simulate a limited, positively enriched reviewed training set.
positive = tr[y[tr] == 1]
negative = tr[y[tr] == 0]
known = np.r_[rng.choice(positive, 35, replace=False),
              rng.choice(negative, 465, replace=False)]
unknown = np.setdiff1d(tr, known)
print("Reviewed positives / total:", y[known].sum(), len(known))
print("Population prevalence:", y[tr].mean())

sup = LogisticRegression(max_iter=2000, random_state=0)
sup.fit(Z[known], y[known])
score_sup_val = sup.predict_proba(Zva)[:, 1]
score_sup_test = sup.predict_proba(Zte)[:, 1]

iso = IsolationForest(n_estimators=150, random_state=0)
iso.fit(Ztr)
score_iso_val = -iso.score_samples(Zva)
score_iso_test = -iso.score_samples(Zte)

# Conservative pseudo-label experiment: add only extreme scores.
# The model is still affected by the enriched initial label selection.
candidate_scores = sup.predict_proba(Z[unknown])[:, 1]
selected = (candidate_scores >= 0.98) | (candidate_scores <= 0.02)
pseudo_idx = unknown[selected]
pseudo_y = (candidate_scores[selected] >= 0.5).astype(int)
semi = LogisticRegression(max_iter=2000, random_state=0)
semi.fit(np.vstack([Z[known], Z[pseudo_idx]]),
         np.r_[y[known], pseudo_y])
score_semi_val = semi.predict_proba(Zva)[:, 1]
score_semi_test = semi.predict_proba(Zte)[:, 1]
print("Pseudo-label count:", len(pseudo_idx),
      "pseudo-positives:", int(pseudo_y.sum()))

def report(name, validation_score, test_score, fraction=0.03):
    # Frozen validation-derived cutoff, never fitted on test labels.
    cutoff = np.quantile(validation_score, 1.0 - fraction)
    pred = (test_score >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0,1]).ravel()
    alerts = tp + fp
    print(name, "threshold", round(float(cutoff), 4),
          "TP", tp, "FP", fp, "FN", fn, "TN", tn,
          "alerts", alerts,
          "precision", round(tp/alerts, 3) if alerts else 0.0,
          "recall", round(tp/(tp+fn), 3) if tp+fn else 0.0,
          "AP", round(average_precision_score(y[te], test_score), 3))
    return pred

p1 = report("Supervised", score_sup_val, score_sup_test)
p2 = report("IsolationForest", score_iso_val, score_iso_test)
p3 = report("Pseudo-label", score_semi_val, score_semi_test)
print("Test attacks / sessions:", int(y[te].sum()), len(te))
print("Supervised vs anomaly disagreement:", int((p1 != p2).sum()))
```

**Reproducibility checks:** Record Python, NumPy and scikit-learn versions. Keep the seed fixed while comparing the three methods. The script's 35 reviewed positive examples are feasible at the expected 1% prevalence among 7,000 training records, but the exact count is random; if changing the seed or prevalence, guard the sampling requirement. The label-review mechanism is intentionally biased and must not be described as random sampling.

**Interpretation constraints:** The 3% validation quantile is an *alert-rate policy*, not probability calibration. Average precision is a ranking summary sensitive to prevalence. The pseudo-label model may be worse than the supervised model; that is a valid negative result. The synthetic drift starts in validation and continues into test, so compare results against a no-drift run before attributing a change to distribution shift. Do not interpret feature 0 as an actionable real-world attack technique.

### Required ablations and error analysis

1. Change prevalence from 1% to 0.5%; update the 35-positive selection safely, and compare precision and recall **with confidence intervals across at least five seeds**.
2. Change benign drift from +0.4 to +1.0; report alert volume and false-positive rate by temporal block.
3. Remove pseudo-labels and compare against the same supervised baseline; report the number and class mix of accepted pseudo-labels.
4. Fit the anomaly model on reviewed negatives only as an **experimental ablation**, not as an automatic improvement. Compare contamination sensitivity.
5. Plot score histograms for benign and attack sessions in the validation period and a reliability diagram for the supervised score **only if the validation labels are representative and sufficiently complete**.
6. For a true daily top-30 budget, group test sessions into time blocks, rank within each block, and report recall@30 per block; explain the latency penalty compared with a streaming threshold.

## 10. Common experimental traps and how to defend against them

**Temporal leakage:** A random row split of adjacent event windows can place nearly identical windows from one incident in training and test. Prefer chronology, plus entity/incident grouping when required. Fit scaler, imputer, feature selector and calibrator on permitted training/validation periods only.

**Post-event labels:** A ticket resolution code, analyst verdict or incident severity assigned tomorrow is not a feature available at detection time. Preserve a feature-availability timestamp separate from event timestamp.

**Selective labels:** Reviewed alerts are not representative of all traffic. Record why each example was labeled; use independent sampling or adjudication to estimate false negatives. Missing labels are not negative labels.

**Model selection leakage:** Repeatedly choosing the best approach after inspecting test scores turns the test set into another validation set. Freeze the protocol and use a new untouched future block for final claims.

**Metric mismatch:** Accuracy on 99% benign traffic is misleading; ROC-AUC can look good while the analyst queue is unusable. Report PR-AUC or average precision, recall@budget, precision@budget, FPR, volume, latency, and denominators.

**Confidence versus certainty:** A probability score, anomaly score and generated textual explanation are different artifacts. An LLM explanation must cite evidence, acknowledge missing data and allow an analyst to override the recommendation.

---

# PART C — INTEGRATED DOCTORAL RESEARCH

## 11. A unified hypothesis and falsification plan

**Hypothesis H1:** At a fixed analyst review budget and on strictly future time blocks, a combination of supervised risk and latent-sequence anomaly evidence increases recall for previously unseen attack patterns compared with either component alone, without increasing unsupported factual claims by a retrieval-grounded analyst assistant.

**Null H0:** The combined approach provides no reliable gain in recall@budget after accounting for temporal drift, attack-family imbalance and uncertainty, or its explanations create more unsupported claims.

**Research units:** One prediction per account/session over a defined interval. Outcomes are incident labels adjudicated independently of the evaluated model. Training uses past data; validation chooses the budget and any fusion weights; a frozen future test measures generalization. Separate known-family, unseen-family, tenant and time-of-day slices. If the data lacks reliable attack-family labels, do not claim unseen-family performance.

**Baselines:** (a) deterministic rules, (b) supervised logistic regression, (c) isolation-based anomaly detector, (d) calibrated or rank-normalized fusion, (e) fusion plus evidence-grounded language assistant. Compare equal data availability and compute budgets. Add a **random-score baseline** and a **review-all-if-budget-allows** upper-bound reference.

**Primary outcomes:** Recall@fixed analyst budget and unsupported-claim rate per assistant response. **Secondary:** precision, false positives per 1,000 events, mean analyst review time, calibration error, citation correctness, per-tenant disparity, end-to-end latency and cost.

**Statistical plan:** Pre-register the test period and review budget. Bootstrap at the incident or time-block level (not individual overlapping rows) for confidence intervals. Report absolute differences with uncertainty, not only relative percentage improvements. If the gain disappears on future blocks or citations are fabricated, reject the claim rather than moving the goalposts.

### Diagram C — evidence-grounded research architecture

```text
  authorized timestamped telemetry
                 |
        provenance + availability
                 |
        frozen feature snapshot
          /             \
 supervised classifier   AR/latent sequence score
          \             /
        validated score fusion
                 |
         fixed analyst budget
                 |
     retrieve original event evidence
                 |
    language model drafts with citations
                 |
   evidence verifier + analyst approval
                 |
  delayed outcome / correction / audit
```

**Threats to validity:** A generative score may simply reproduce features already used by the supervised model. Shared failures can make fusion worse. The analyst assistant may cite true but irrelevant evidence. Analysts may change their behavior after seeing model scores, altering future labels. Evaluate ablations, independent evidence quality and human-in-the-loop effects.

## 12. Research reading and critique: three claims to interrogate

**Paper 1 — Kingma & Welling, *Auto-Encoding Variational Bayes* (submitted 20 Dec 2013).** Read the introduction and the derivation of the stochastic gradient estimator: https://arxiv.org/abs/1312.6114 . Explain exactly why the Gaussian reparameterization permits backpropagation, how the approximate posterior changes computational cost, and what the paper does **not** claim about causal or semantic latent factors.

**Paper 2 — van Engelen & Hoos, *A survey on semi-supervised learning* (2020).** https://doi.org/10.1007/s10994-019-05855-6 . Identify the assumption under which unlabeled data helps; then construct a counterexample where the cluster structure follows device type while the attack label cuts across device types. Explain how confident pseudo-labels can worsen minority-attack recall.

**Paper 3 — NIST, *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations* (2025).** https://csrc.nist.gov/pubs/ai/100/2/e2025/final . Map the security experiment's trust boundaries: data source, labels, trained detector, retrieved evidence, and language-model output. Separate ordinary drift from attacker-controlled manipulation. No live attack execution is required.

### Evidence-critique exercise: a recent AI-cyber capability headline

The morning lesson referenced **Google's 28 October 2026 Gemini 4 Carbon announcement** and **OpenAI's 26 October 2026 GPT-6.1 Terra announcement**, including provider-reported cyber benchmark results and capability thresholds. Rather than re-reporting the scores as independently verified outcomes, complete this matrix: benchmark name; exact task; data contamination controls; allowed tools and time; evaluation population; independent replication; deployment permissions; and observed failure cases. Sources to examine are the primary announcements cited in the morning lesson, not social-media summaries. A capability benchmark does not establish authorized or reliable autonomous incident response.

## 13. Tutorials, lectures and audio resources

**Lecture/tutorial path:** Stanford CS236 Deep Generative Models course materials, especially autoregressive models and VAEs: https://deepgenerativemodels.github.io/ . Read the original VAE paper alongside Stanford's derivations: https://arxiv.org/abs/1312.6114 . For the security evaluation side, the scikit-learn probability-calibration guide provides reliability diagrams and the distinction between calibrated probability and score ranking: https://scikit-learn.org/stable/modules/calibration.html . Its anomaly-detection guide clarifies the sign and meaning of Isolation Forest scores: https://scikit-learn.org/stable/modules/outlier_detection.html .

**Foundational textbook:** *Dive into Deep Learning*, language-modeling chapter: https://d2l.ai/chapter_recurrent-neural-networks/language-model.html . Compare its token-level likelihood objective with the ELBO derivation; note which losses are directly comparable and which are not.

**Podcasts to listen to (feeds, not claimed same-day episodes):** SANS Internet Storm Center Stormcast for concise operational cyber context: https://isc.sans.edu/podcast.html . *Practical AI* for experiment design and applied model trade-offs: https://changelog.com/practicalai . While listening, write one operational question that a paper or experiment could test. These links are supplemental; a podcast is not primary evidence for a benchmark score.

**Real-data follow-up, deferred until authorization and audit:** CICIDS2017 from the University of New Brunswick, https://www.unb.ca/cic/datasets/ids-2017.html . Before any model training, create a dataset card recording capture days, split logic, feature timestamps, duplicates, label provenance, missing values and expected deployment differences. Do not claim the synthetic lab reproduces published CICIDS2017 results.

## 14. Doctoral viva: twelve questions with concise answers

1. **Why is the ELBO a lower bound?** Because `log p(x)-ELBO=KL(q(z|x)||p(z|x))>=0`.
2. **Which KL is the VAE regularizer?** `KL(q(z|x)||p(z))`, not the intractable posterior gap.
3. **What is the role of reparameterization?** Move sampling noise outside differentiable encoder parameters to obtain pathwise gradients.
4. **Why might a VAE decoder ignore z?** A strong decoder can model x without information from z while reducing KL cost.
5. **Does beta>1 always improve latent factors?** No; it strengthens regularization and can sacrifice reconstruction or latent usage.
6. **Can a model be AR and latent-variable simultaneously?** Yes; `p(x|z)` may itself be autoregressive.
7. **Why does a 2% false-positive rate matter at 0.1% attack prevalence?** False alerts can dominate the positive queue despite high sensitivity.
8. **Is Isolation Forest's anomaly score a calibrated attack probability?** No.
9. **Why might pseudo-labeling harm recall?** Confident mistakes and selective labeling can amplify systematic blind spots.
10. **How do you prevent future leakage?** Time-valid features, chronological/grouped splits, train-only transforms and frozen thresholds.
11. **What does recall@30 mean?** Fraction of actual attacks captured among a specified top-30 review policy, with an explicit population and time unit.
12. **What would falsify the integrated hypothesis?** No robust future-period recall gain at equal budget, or an unacceptable rise in unsupported assistant claims.

### Oral-defense prompts (answer without notes)

Explain why `log p(x)` cannot be read directly from an arbitrary ELBO. Derive the base-rate precision formula on a whiteboard. Defend the difference between an anomaly score and an incident verdict. Explain how you would detect test contamination in overlapping log windows. State one scenario in which the simplest logistic regression model should remain the production choice.

## 15. Quiz: calculate, do not guess

**Q1.** A Gaussian posterior has `mu=0`, `sigma^2=4`. Compute KL to a standard normal.  
**Q2.** If `ELBO=-3.2` and `KL(q||true posterior)=0.3`, what is `log p(x)`?  
**Q3.** A detector has recall 0.9, FPR 0.01 and attack prevalence 0.005. Estimate precision.  
**Q4.** Validation gives 25 alerts per 1,000; tomorrow's benign distribution shifts. Is 25 guaranteed tomorrow?  
**Q5.** Can an unlabeled event be safely treated as a confirmed benign negative?  
**Q6.** Which is stronger evidence of posterior collapse: low KL alone, or low KL plus insensitivity to replacing z?  
**Q7.** Why is random row splitting of overlapping session windows risky?  
**Q8.** If a model is selected after repeated test comparisons, what must be done for a defensible final evaluation?

**Answer key:** (1) `0.5*(4-1-ln4)=0.80685` nats. (2) `-2.9`. (3) `0.9*0.005/(0.9*0.005+0.01*0.995)≈31.14%`. (4) No, a fixed threshold can produce a different volume after drift. (5) No, unless separately adjudicated or an assumption is explicitly justified. (6) Both signals together. (7) Near-duplicate and incident-level leakage. (8) Freeze the model and protocol, then evaluate on a genuinely untouched future holdout.

## 16. Deliverables, self-assessment and tomorrow's handoff

Create a dated research-log entry with: (a) the handwritten ELBO identity and the 0.818147 Gaussian KL calculation; (b) `day02_evening_latent.py`, actual output and one paragraph on beta; (c) `day02_evening_security.py`, confusion matrices, review volumes, prevalence and drift ablations; (d) one table separating ranking, calibration and decision utility; (e) a three-paper literature matrix; (f) one falsifiable research hypothesis with baselines and limitations.

For each result record **data source, environment versions, seed, train/validation/test boundaries, assumptions, observed result, failure case, next experiment**. Rate your ability to defend the six morning outcomes on a 0–3 scale: 0 not yet; 1 with notes; 2 independently; 3 under skeptical questioning. A negative result is acceptable when its methodology is valid.

**Progression:** Tonight completes the **deeper treatment of Unit 1 Session 2** in both courses. **Next morning, 3 November 2026**, proceed to **foundation models and their evolution** (Generative AI Unit 1 Session 3) and **feature engineering / security representations** (Cybersecurity Unit 1 Session 3), contingent on understanding today's checks. The independent-study exercises complement, but do not modify, the university's stated 40 contact hours and 60 independent hours per pathway.

**Narration handoff:** This Markdown is intended as input to the separate GitHub Actions speech workflow when committed. A successful GitHub commit does **not** prove an MP3 was generated, published or delivered. Verify the actual run, published file and playback before claiming audio exists. **Preparation only: no email is sent by this task.**
