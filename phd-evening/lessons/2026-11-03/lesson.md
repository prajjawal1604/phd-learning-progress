# PhD Evening Deep Dive — Day 3 | Tuesday, 3 November 2026

**Pathways:** Advanced Generative AI and Foundation Models; AI for Cybersecurity (19 October 2026 pathway PDFs).  
**Position:** Unit 1, Session 3 in both. **Continuity:** This evening extends the 3 November morning lesson on foundation-model pretraining/transfer/adaptation and cybersecurity feature engineering/representations. **Do not advance to Session 4.**  
**Study time:** 2.5–3 hours core; 60–90 minutes optional research. **Evidence discipline:** All numbers below are worked or synthetic examples unless explicitly identified as published findings. Use no real credentials or unauthorized telemetry.

## Learning contract and morning retrieval

Before starting, explain without notes (a) why broad pretraining and adaptable reuse—not parameter count alone—characterize a foundation model, (b) the difference between a frozen encoder, a task-adapted model and retrieval at inference, and (c) why a feature's **availability at decision time** matters more than its presence in a CSV. Reopen [today's morning lesson](../../../../phd-morning/lessons/2026-11-03/lesson.md) and record which of its two labs you completed, actual library versions, seeds and results. Morning's toy TF–IDF/SVD transfer comparison is an *analogy*, not evidence about a real foundation model.

**Evening deliverables:** a transfer-learning experiment card; a leakage-safe authentication feature ledger; two reproducible Python scripts and result tables; a short research critique; quiz answers; a 200-word oral-defense summary.

## Part I — Foundation models: what transfers and why?

### 1. Representations, pretraining objectives and the adaptation boundary

A foundation model is broadly pretrained for adaptation to many downstream tasks. The pretrained objective shapes the representation: causal language modeling minimizes negative log likelihood of the next token; masked-language modeling predicts hidden tokens from context; contrastive training increases similarity of selected positive pairs relative to negatives. None of these objectives directly optimizes the precision of a security alert.

For causal pretraining, with token sequence `x_1,...,x_T`:

```text
L_pre(theta) = - sum_{t=1}^{T} log p_theta(x_t | x_1,...,x_{t-1})
```

A downstream classification task may use `h_theta(x)` and train only a head `W`:

```text
z = h_theta(x)                     # frozen or adapted representation
p(y=k|x) = softmax(W z + b)_k
L_task = -sum_i log p(y_i|x_i) + lambda * ||W||_F^2
```

**Key distinction:** lower pretraining perplexity on broad text does not entail better out-of-domain security classification. The pretraining distribution may lack rare commands, domain-specific abbreviations or event-order semantics. A frozen vector can preserve topical similarity while suppressing a single negation that flips an incident's meaning.

**Architecture diagram:**

```text
                broad corpus (permitted sources only)
                            |
                self-supervised pretraining
                            |
                  shared model parameters
                 /             |             \
        frozen embeddings   update weights   no weight update
          + linear probe     SFT / LoRA       prompt + retrieval
                 \             |             /
                 same held-out downstream task
                            |
            precision/recall, calibration, OOD,
            latency, memory, fairness, provenance
```

A RAG system is a *system-level* adaptation: retrieved documents enter the context; the base weights need not change. If a retrieval document says “ignore previous instructions,” treat that as untrusted data, not an instruction. Record document identifiers and whether claims are actually supported.

### 2. Worked derivation: adaptation budget is not inference budget

Assume a **hypothetical** 100-million-parameter encoder produces a 768-dimensional representation. A 3-class linear head has `768*3 + 3 = 2,307` trainable parameters, or `0.002307%` of the base parameter count. Yet inference still evaluates the 100-million-parameter encoder. Frozen weights save training gradients and optimizer state, **not** the base model's forward-pass compute.

For a `768 x 768` matrix, a rank-8 low-rank update uses `A in R^(768x8)`, `B in R^(8x768)`: `8*(768+768)=12,288` parameters versus `589,824` for a full update, a factor of 48 **for this one matrix only**. In LoRA notation, `W' = W + (alpha/r) A B` (factor order depends on convention). This is not a claim of 48x end-to-end speedup. Fine-tuning may also change representations in ways that harm previous tasks (forgetting).

**Oral challenge:** Why might a linear probe beat a fully fine-tuned model on a 40-example dataset? Discuss variance, optimization, pretraining alignment and validation selection. Why might the reverse occur after 40,000 high-quality examples?

### 3. Controlled transfer experiment design

Do not compare “model A on dataset A” against “model B on dataset B.” Define one target label, one frozen train/validation/test split, one preprocessing boundary and a fixed budget. Compare (i) majority-class baseline, (ii) lexical TF–IDF + logistic regression, (iii) pretrained frozen representation + logistic regression, (iv) optional adapter fine-tune. Report **macro-F1**, class-specific recall, calibration, runtime, trainable parameters and variance across at least five seeds.

If labeled data are scarce, draw **nested** training subsets from the same training pool (e.g., 10, 25, 50, 100 per class). Choose hyperparameters on validation only. Never tune on the test set. Use out-of-distribution testing: train on one documentation style and test on a later period or new source. A useful result is a **learning curve**, not a single lucky score.

#### Worked example: positive versus negative transfer

Suppose 100 test cases contain 20 true security incidents. A frozen transfer classifier flags 25, of which 15 are correct: `TP=15, FP=10, FN=5, TN=70`. Precision `15/25=0.60`; recall `15/20=0.75`; F1 `2*0.60*0.75/(0.60+0.75)=0.667`. A lexical baseline flags 12, with 10 correct: `TP=10, FP=2, FN=10, TN=78`; precision `0.833`, recall `0.50`, F1 `0.625`. Which is better depends on the cost of a missed incident, alert capacity and analyst burden. The frozen model's higher F1 does not establish its superiority at a fixed **10-alert-per-day** budget. Re-rank scores and compare at the same budget.

### 4. GenAI lab — representation compression, negation and transfer failure

**Purpose:** deepen the morning's SVD/linear-probe analogy by deliberately creating a semantic counterexample. No external checkpoint or private data is required.

**Setup:** Python 3.11; `pip install numpy scikit-learn`. Save `day03_evening_transfer.py`.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Self-authored, synthetic; labels mean "access denied" (1) vs allowed (0).
train = [
    "access allowed for user", "login permitted for account",
    "authorization granted for client", "token approved for device",
    "access denied for user", "login blocked for account",
    "authorization rejected for client", "token refused for device",
]
y_train = np.array([0]*4 + [1]*4)
test = [
    "access not allowed for user",   # Negation not represented explicitly in train.
    "login was not permitted for account",
    "authorization granted for client",
    "token refused for device",
]
y_test = np.array([1, 1, 0, 1])

vec = TfidfVectorizer(ngram_range=(1, 2))
X = vec.fit_transform(train)   # Train only
Xt = vec.transform(test)
for n in (2, 4):
    svd = TruncatedSVD(n_components=n, random_state=7)
    Z = svd.fit_transform(X)
    Zt = svd.transform(Xt)
    clf = LogisticRegression(max_iter=1000, random_state=7).fit(Z, y_train)
    pred = clf.predict(Zt)
    print("components", n, "predictions", pred.tolist())
    print(confusion_matrix(y_test, pred, labels=[0,1]))
    print(classification_report(y_test, pred, zero_division=0))
    print("explained variance sum", round(float(svd.explained_variance_ratio_.sum()), 3))

# Lexical baseline on identical split
lex = LogisticRegression(max_iter=1000, random_state=7).fit(X, y_train)
print("lexical baseline", lex.predict(Xt).tolist())
print("unseen tokens", sorted(set("not was".split()) - set(vec.vocabulary_)))
```

**Interpretation:** The unseen negation token “not” is ignored by this fitted vocabulary, so an embedding-like compression can mistake “allowed” for “not allowed.” Do not assert a specific classification outcome without executing the script: scores depend on the small sample and learned coefficients. The failure mechanism is visible regardless of the exact score.

**Ablations:** add *negated* training examples; switch `ngram_range` to (1,1); evaluate on a held-out paraphrase family; compare the learned decision boundary at 2 vs 4 components. Explain which experiment measures lexical coverage and which measures representation compression. A **real foundation-model extension** can use a pinned public sentence-transformer checkpoint on non-sensitive examples; report model card, license, tokenizer, hardware and reproducibility.

**Research question:** Can a model have high average embedding similarity for two texts with opposite security meaning? Design 20 minimal contrast pairs and report nearest-neighbor retrieval errors with confidence intervals, not just anecdotes.

## Part II — Cybersecurity: prediction-time-valid features and embeddings

### 5. Build a feature-availability ledger before modeling

Specify the prediction: “At the completion of an authentication attempt at time `t`, prioritize accounts for investigation using evidence **available by t**, with later-confirmed incident status as the delayed label.” The target is **not** “failed login,” which is already observed.

| Feature | Event-time rule | Ingestion-time rule | Leakage risk | Action |
|---|---|---|---|---|
| failures in prior 10 minutes | `t-10m < event_time <= t` | `ingested_at <= t` | late-arriving logs | as-of join |
| unique source IPs prior hour | trailing window | known by t | identity reuse | aggregate and document |
| region / device category | current event | present by t | unseen categories | train-fitted encoder |
| minutes since previous success | previous events only | ingested by t | future normalization | missing indicator |
| analyst ticket outcome | generally after t | not known by t | direct target leakage | **exclude** |
| incidents confirmed next 24h | after t | delayed | label used as feature | **label only** |

A timestamp alone is insufficient: `event_time < t` but `ingested_at > t` means the information was **not yet available**. Feature stores and offline replays must reproduce the production availability boundary. Clock skew, deduplication and entity merges also require explicit policy.

**Diagram — valid online/offline parity:**

```text
event_time ------> raw event --------> ingested_at
     |                                    |
     +------- eligible only if both <= t -+
                                          |
                                as-of feature builder
                                          |
                              train-only preprocessing
                                          |
                               classifier / risk score
                                          |
                             validation-set threshold
                                          |
                               frozen future test
                                          |
                           analyst review + delayed label
```

### 6. Worked leakage and scaling example

Training failures `[0,1,1,2]` have population mean `1` and standard deviation `sqrt(0.5)=0.7071`. A future count of `3` becomes `z=(3-1)/0.7071=2.828`. If the scaler uses the future row while computing training statistics, test information has crossed the boundary.

Consider a user whose failed logins occurred at 09:00, 09:03, 09:08 and 09:21. At 09:10, with window `(09:00,09:10]`, count is **2**, not 3: 09:00 is excluded by the open left boundary. With `[09:00,09:10]`, count is **3**. At 09:22, window `(09:12,09:22]` contains only 09:21, count **1**. **Boundary semantics are part of the model specification.**

For categorical variables, one-hot encoding is reasonable for region; it is often unsuitable for millions of IPs or account IDs. High-cardinality IDs can enable memorization. Target encoding must be cross-fitted and historical. An unseen region is not proof of an attack. Missing geolocation may reflect privacy settings or a sensor outage; distinguish missingness causes before calling them suspicious.

### 7. Alert economics and decision thresholds

Suppose 10,000 authentication events contain 50 confirmed incidents (prevalence 0.5%). A detector at 90% recall finds 45 incidents. At 95% specificity, it also flags `0.05*9950 ≈ 498` benign events. Precision is about `45/(45+498)=8.3%`, meaning approximately 92% of alerts are false positives. High recall and high specificity can coexist with low precision under low prevalence.

**What to report:** precision at fixed alert budget, recall at fixed analyst capacity, average precision (PR-AUC), calibration, detection delay and subgroup false-alert rates. Accuracy can be deceptively high: an always-benign classifier gets 99.5% here and catches nothing.

**Threat-model caveat:** If labels arrive only for investigated alerts, the training set suffers selection bias. “Not investigated” is not equivalent to “benign.” Discuss delayed-label and positive-unlabeled evaluation before interpreting a score.

### 8. Cybersecurity lab — as-of joins, ingestion delay and leakage stress test

**Setup:** `pip install numpy pandas scikit-learn`. Save `day03_evening_cyber.py`. Synthetic telemetry only.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score

rng = np.random.default_rng(20261103)
n = 900
t = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
failed = rng.poisson(0.8, n)
new_device = rng.binomial(1, .15, n)
region = rng.choice(["IN","US","EU"], n, p=[.6,.2,.2])
# Each record is scored at event time; late ingestion must not count as known.
lag_min = rng.choice([0, 0, 0, 3, 15], n)
events = pd.DataFrame({
    "event_time": t, "ingested_at": t + pd.to_timedelta(lag_min, unit="m"),
    "failed": failed, "new_device": new_device, "region": region,
})
# Deterministic synthetic label mechanism, not real incident truth.
p = 1/(1+np.exp(-(-4.3 + .8*failed + 1.2*new_device)))
events["label"] = rng.binomial(1, p)
# As-of valid 10-minute count; deliberately simple O(n^2) for small lab.
counts = []
for i, row in events.iterrows():
    known = events.loc[
        (events.event_time > row.event_time - pd.Timedelta(minutes=10)) &
        (events.event_time <= row.event_time) &
        (events.ingested_at <= row.event_time)
    ]
    counts.append(int(known.failed.sum()))
events["failed_10m_known"] = counts
events["hour"] = events.event_time.dt.hour
features = ["failed_10m_known", "new_device", "hour", "region"]
# Chronological train/validation/test, never random rows across time.
train, val, test = events.iloc[:540], events.iloc[540:720], events.iloc[720:]
prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), ["failed_10m_known", "new_device", "hour"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]),
])
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
model.fit(train[features], train.label)
val_scores = model.predict_proba(val[features])[:,1]
# Choose threshold using validation only; cap alert volume near 10%.
threshold = float(np.quantile(val_scores, .90))
test_scores = model.predict_proba(test[features])[:,1]
pred = (test_scores >= threshold).astype(int)
print("train/val/test", len(train), len(val), len(test))
print("threshold", round(threshold, 4), "test alerts", int(pred.sum()))
print("test precision", round(precision_score(test.label,pred,zero_division=0),3))
print("test recall", round(recall_score(test.label,pred,zero_division=0),3))
print("test average precision", round(average_precision_score(test.label,test_scores),3))
print("late events", int((events.ingested_at > events.event_time).sum()))
# Diagnostic only: future/late-aware count is prohibited for real scoring.
future_counts = []
for _, row in events.iterrows():
    window = events.loc[
        (events.event_time > row.event_time - pd.Timedelta(minutes=10)) &
        (events.event_time <= row.event_time)
    ]
    future_counts.append(int(window.failed.sum()))
print("rows changed by ingestion constraint",
      int((events.failed_10m_known != future_counts).sum()))
```

**Important interpretation:** This lab intentionally makes `failed` and `new_device` correlated with the synthetic label; its output measures code behavior under that assumption, not a field detector's effectiveness. The as-of count includes the current record **only if already ingested**. In a real request/response scoring service, the current event might be available directly even before pipeline ingestion; document which source the service actually sees. The O(n²) window loop is pedagogical; use a tested streaming/window implementation for scale.

**Required ablations:** (1) compare the valid count against `future_counts`, which improperly ignores ingestion delay; (2) insert a new region in the test set and confirm encoding does not crash; (3) fit the scaler on all rows and label the experiment *intentionally invalid*, then compare the scores; (4) replace the time split with a random split and explain why this can exaggerate generalization; (5) repeat five random seeds for the synthetic label mechanism, keeping the split fixed. Never present the invalid model as a production improvement.

### 9. Sequence representations and the limits of contrastive learning

Compare these ordered event sequences:

```text
A: fail -> fail -> success -> token_use
B: success -> token_use -> fail -> fail
```

A bag-of-events representation sees identical counts; a sequence model can distinguish order. But if logs are batched or clocks skewed, the sequence itself may be uncertain. A learned embedding is only as trustworthy as its temporal provenance and pretraining task.

In contrastive learning, treating all events from one host as positives can teach **host identity**, not attack behavior; treating all different-host events as negatives can push the same attack pattern apart. Document the positive/negative sampling rule, hard-negative strategy and temporal leakage risk. Test transfer to unseen hosts, new devices and a later month.

**Mini research proposal:** Compare a 10-feature transparent baseline with a small event-sequence encoder at the same daily alert budget. Keep labels and evaluation horizon identical; report bootstrap confidence intervals by account or day (not by correlated individual event), latency and investigator-readable evidence. Hypothesis: sequence order improves detection of account-takeover patterns not recoverable from counts. Falsifier: no statistically meaningful improvement on unseen entities at the fixed budget.

## Part III — Reading, tutorials, podcast and research discipline

**Foundational reading (official or primary):**

1. Bommasani et al. (2021), *On the Opportunities and Risks of Foundation Models*: https://arxiv.org/abs/2108.07258 — broad pretraining, adaptation, downstream risks.
2. Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*: https://arxiv.org/abs/2106.09685 — parameter-efficient adaptation, what is and isn't trained.
3. Gururangan et al. (2020), *Don't Stop Pretraining*: https://aclanthology.org/2020.acl-main.740/ — domain/task adaptive pretraining and evaluation design.
4. scikit-learn, *Common pitfalls and recommended practices*: https://scikit-learn.org/stable/common_pitfalls.html — train/test leakage and pipelines.
5. scikit-learn, *Precision-recall*: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html — rare-event evaluation.
6. NIST, *AI Risk Management Framework*: https://www.nist.gov/itl/ai-risk-management-framework — govern, map, measure, manage; evidence and oversight.
7. MITRE ATT&CK: https://attack.mitre.org/ — use technique descriptions to formulate **defensive** hypotheses, not as automatic ground-truth labels.

**Credible tutorials:** Hugging Face NLP course https://huggingface.co/learn/nlp-course/chapter1/1 (tokenization, transformers, fine-tuning); Stanford CS224N course materials https://web.stanford.edu/class/cs224n/ (representation learning); Google Machine Learning Crash Course https://developers.google.com/machine-learning/crash-course (classification, data handling). Check the specific lesson title and revision when citing material in a thesis.

**Audio / podcast study options:** *Practical AI* https://changelog.com/practicalai and *The TWIML AI Podcast* https://twimlai.com/ — select an episode on foundation-model adaptation or representation learning, note its publication date and distinguish expert opinion from experimental evidence. For security context, *SANS Internet Stormcenter Daily Network Security Podcast* https://isc.sans.edu/podcast.html — summarize one defensive telemetry case and identify which features would have been available at alert time. These are **series recommendations**, not a claim that a particular new episode was checked today.

### 10. Paper critique exercise (20 minutes)

Read the abstract, experimental setup and limitations of *Don't Stop Pretraining*. Write a one-page critique with: (a) source and target distribution definitions; (b) baseline fairness; (c) training compute and data availability; (d) whether performance differences survive multiple seeds; (e) whether the paper's findings can transfer to security logs without revalidation. Add one falsifiable hypothesis and one negative-control experiment.

For LoRA, draw the low-rank matrix shapes and calculate trainable parameters for rank 4, 8 and 16 at hidden size 768. Explain why changing rank is not the same as changing the full model parameter count. For the security lab, list **every feature** that would be missing or late if the pipeline experienced a 15-minute ingestion outage.

## Part IV — Assessment and oral defense

**Quiz (answer before reading the key):**

1. Does a 100M-parameter model become a foundation model solely because of its size?
2. Why can a frozen encoder require almost as much inference compute as a fine-tuned encoder?
3. In what sense is retrieval different from fine-tuning?
4. What does a rank-8 LoRA update change, and what stays frozen?
5. Why might a tiny SVD representation fail on negation?
6. At 09:10, does an event at 09:00 belong to `(09:00,09:10]`?
7. Can an event at 09:08 be used at 09:10 if it was ingested at 09:15?
8. Why is analyst ticket status a dangerous alert-time feature?
9. Why can 95% specificity still create a large false-alert workload?
10. How would you test a sequence model against a count baseline without giving either an unfair advantage?

**Answer key:** (1) No; broad pretraining and adaptable reuse matter. (2) Frozen parameters still participate in the forward pass. (3) Retrieval adds external evidence at inference; it need not update weights. (4) It learns a low-rank weight update; the base weight can remain frozen. (5) Compression and vocabulary coverage may suppress a decisive token. (6) No, left boundary open. (7) No, not through the ingested log stream; direct request-time access would need separate documentation. (8) It is determined after investigation and leaks the label. (9) The benign population is enormous when incidents are rare. (10) Same prediction horizon, features available at t, split, label source, validation protocol and fixed alert budget; compare on a future, unseen-entity test set.

**Oral defense prompts:** Define foundation model without using the word “large”; derive the linear-probe objective; explain positive and negative transfer; distinguish representation quality from operational alert utility; demonstrate an as-of join; identify two hidden leakage paths; justify precision at a fixed alert budget; state three limitations of the toy labs.

**Submission checklist:** `day03_evening_transfer.py` and `day03_evening_cyber.py`; saved console output with environment versions; a 2x2 confusion matrix; one feature ledger; an ablation table; one-page paper critique; a five-sentence thesis-relevance note. Record all failed runs and do not invent results.

## Narration and delivery handoff

This Markdown is the complete **evening preparation artifact**. The GitHub Actions push trigger should narrate `phd-evening/lessons/2026-11-03/lesson.md`. The MP3 is **not verified** until the workflow completes, publishes to `lesson-audio`, and its remote file is independently confirmed. The separate evening delivery task should email the full lesson and only a verified audio link; this preparation task sends no email.
