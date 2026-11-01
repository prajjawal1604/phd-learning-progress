# PhD Morning Learning — Day 2 — Monday, 2 November 2026

**Courses:** Advanced Generative AI and Foundation Models; AI for Cybersecurity  
**India date:** 2026-11-02 (Asia/Kolkata)  
**Pathway position:** Accelerated Unit 1, session 2 of both courses. Yesterday: generative versus discriminative objectives, likelihood, security telemetry and confusion matrices. Today: **autoregressive versus latent-variable generative models**, and **supervised versus unsupervised versus semi-supervised security learning**. Tonight's deep dive should extend these SAME topics, not jump to Unit 2.  
**Suggested morning study:** 90 minutes: 25 minutes GenAI theory and worked examples, 25 minutes cybersecurity theory, 25 minutes hands-on labs, 10 minutes primary-source reading, 5 minutes recall and research log. Additional coding and paper review can continue independently.

> **Research discipline:** Separate mathematical definitions, measured experimental results, provider claims and conjectures. All examples below are illustrative, not claims of performance on a real security dataset. Use only authorized or synthetic security data. Record versions, seeds, data provenance, splits and failed experiments.

## 1. Outcomes to demonstrate before advancing

1. Factor a joint sequence probability into conditional next-token probabilities, and calculate a complete short-sequence probability.
2. Explain the difference between observed variables, latent variables, a prior, a likelihood and a posterior.
3. Derive the basic evidence lower bound (ELBO) intuition for a variational autoencoder (VAE), including its reconstruction and KL terms.
4. Select supervised, unsupervised or semi-supervised learning for a security use case based on **label availability, deployment timing, prevalence and evidence quality**.
5. Calculate the operational cost of false alerts under a realistic base rate; distinguish anomaly from attack.
6. Produce two reproducible, small Python experiments and an evidence-backed research log.

---

# PART A — ADVANCED GENERATIVE AI AND FOUNDATION MODELS

## 2. Autoregressive models: a distribution built one decision at a time

An autoregressive model (AR model) generates a sequence by conditioning each next element on what came before. For tokens t1 through tN, the probability chain rule gives

**P(t1, ..., tN) = P(t1) × P(t2 | t1) × ... × P(tN | t1, ..., tN-1).**

This factorization is mathematically exact for any joint distribution with well-defined conditionals. A particular neural language model is only an *approximation* to those conditional distributions. It learns parameters theta by minimizing negative log likelihood on training sequences:

**L_AR(theta) = - sum over sequences and positions of log P_theta(t_i | t_<i).**

During training, a model commonly receives the true preceding tokens, called *teacher forcing*. During generation it conditions on its own earlier sampled outputs. The discrepancy matters: a mistaken generated token can change the entire future context.

### Worked example A1: sequence probability

Suppose a toy language model estimates:

- P("alert") = 0.60
- P("resolved" | "alert") = 0.80
- P("<end>" | "alert resolved") = 0.50

Then P("alert resolved <end>") = 0.60 × 0.80 × 0.50 = **0.24**.

Its negative log likelihood for this three-token sequence is -ln(0.24) ≈ **1.427 nats**; average token NLL ≈ **0.476 nats/token**; perplexity exp(0.476) ≈ **1.61**. Those numbers describe this exact toy tokenization and example, not factual correctness. The end token matters: a model that predicts words but never ends a sequence has not defined the same complete probability distribution.

Now consider two alternatives after "alert": "resolved" with probability 0.8 and "pending" with probability 0.2. Greedy decoding chooses "resolved"; stochastic sampling may choose either. Changing temperature changes the sampling distribution, not the learned evidence.

### Why autoregression is useful

Autoregression fits language and time-ordered events naturally: each next token or event has a conditional prediction target. A causal Transformer makes many training-position predictions in parallel while using a mask to prevent each position from reading future tokens. **Generation** remains sequential at the token level, although optimized implementations cache prior key/value representations. Predicting plausible next events does not mean a model understands why an incident happened.

### Diagram A: autoregressive generation

    Training: observed sequence
    [START] -> [alert] -> [resolved] -> [END]
       |         |            |
       v         v            v
    predict    predict      predict
     alert     resolved       END

    Inference: [START] -> sample t1 -> sample t2 -> ... -> END
                               ^           ^
                      earlier samples become context

A causal model does not get to read the real future during inference. If a benchmark inadvertently includes future log entries or a resolved ticket's final disposition in the input, it is measuring leakage rather than useful early prediction.

## 3. Latent-variable models: generate through an unobserved cause

A latent-variable model introduces an unobserved variable z. The generative story is:

1. Sample z from a **prior** P(z).
2. Sample observation x from a **likelihood** P(x | z).

The resulting marginal data distribution is

**P(x) = sum_z P(x | z) P(z)** for discrete z, or **P(x) = integral P(x | z) P(z) dz** for continuous z.

The **posterior** P(z | x) is different from the prior: it tells us which hidden states are plausible *after observing x*. By Bayes' rule,

**P(z | x) = P(x | z) P(z) / P(x).**

A latent state might encode a document topic, writing style or event regime, but a learned latent coordinate is not guaranteed to have a human-interpretable meaning.

### Worked example A2: infer a hidden operating regime

Imagine a synthetic server that is in "normal" mode z=0 on 80% of days and "maintenance" mode z=1 on 20% of days. An observed high-volume event x occurs on 10% of normal days and 90% of maintenance days.

- P(z=1)=0.20; P(z=0)=0.80.
- P(x | z=1)=0.90; P(x | z=0)=0.10.
- P(x)=0.90×0.20 + 0.10×0.80 = 0.26.
- **P(z=1 | x)=0.18/0.26 ≈ 0.6923.**

Even after a high-volume event, maintenance is only about 69.2% probable. This is an inference about a latent regime, not a security incident verdict. A prior with a different maintenance frequency would change the posterior.

### Diagram B: generative and inference directions

    Generative direction:  z ~ P(z) ------> x ~ P(x | z)
                         hidden regime      observed event

    Inference direction:  x --------------> estimate P(z | x)
                         observation         plausible regimes

    AR: [previous tokens] ----------------> [next token]
    Latent: [sample hidden z] ------------> [whole observation / sequence]

These are not mutually exclusive families. A sequence model may also have a global latent variable, and a decoder conditioned on z may still be autoregressive.

## 4. VAEs: when the posterior is difficult to calculate

A variational autoencoder uses a neural **encoder** q_phi(z | x) to approximate an often-intractable posterior, and a **decoder** p_theta(x | z) to reconstruct or generate observations. A prior p(z), commonly a standard normal distribution, supplies a source from which to sample at generation time.

The core identity is

**log p_theta(x) = ELBO(theta, phi; x) + KL[q_phi(z|x) || p_theta(z|x)].**

Since KL divergence is nonnegative, the evidence lower bound cannot exceed log p_theta(x). A convenient equivalent form is

**ELBO = E_{q_phi(z|x)}[log p_theta(x|z)] - KL[q_phi(z|x) || p(z)].**

Read the terms carefully:

- **Reconstruction term:** encourages a sampled z to retain information needed to explain x.
- **KL regularizer:** encourages the approximate posterior to remain compatible with the prior.
- **Reparameterization trick:** for a diagonal Gaussian, write z = mu_phi(x) + sigma_phi(x) ⊙ epsilon, epsilon ~ N(0,I), so gradients can flow through mu and sigma.

A low reconstruction error alone is not proof of good generation or anomaly detection. If the decoder ignores z, the model can exhibit *posterior collapse*. A latent dimension may encode nuisance factors rather than the desired semantic concept. Compare held-out reconstruction, generated-sample quality and latent-use diagnostics rather than declaring success from one loss curve.

### Worked example A3: ELBO arithmetic

Suppose the expected log reconstruction likelihood is -2.0 nats and the KL term is 0.3 nats. Then **ELBO = -2.0 - 0.3 = -2.3 nats**, and negative ELBO (a common minimization objective) is **2.3 nats**. If another configuration gives reconstruction -1.7 but KL 0.8, its ELBO is -2.5: better reconstruction by itself did **not** yield a better bound. These are illustrative scalar values; compare only like-for-like likelihood definitions and data.

### AR versus latent-variable model: choose by the question

| Dimension | Autoregressive model | Latent-variable model |
|---|---|---|
| Core factorization | P(x)=product_i P(x_i \| x_<i) | P(x)=integral P(x\|z)P(z)dz |
| Natural example | Next token in a support reply | Hidden operating regime generating observations |
| Generation | Usually sequential token sampling | Sample latent, then decode; decoder may itself be AR |
| Training | Next-token negative log likelihood | Often approximate likelihood via ELBO |
| Inference question | "What follows this context?" | "What hidden states explain this observation?" |
| Failure mode | Exposure bias, unsupported fluent text | Posterior collapse, uninterpretable latents |
| Security caution | Future-event leakage | Latent cluster is not an attack label |

## 5. GenAI lab — reproducible bigram and latent-posterior checks (25 minutes)

**Environment:** Python 3.10+; standard library only. Save as day02_genai.py. This is a pedagogical miniature, not a pretrained foundation model.

    import math
    from collections import Counter
    from random import Random

    train = [
        "alert resolved", "alert resolved", "alert pending",
        "ticket resolved", "ticket pending", "alert resolved",
    ]
    test = ["alert pending", "ticket resolved"]
    vocab = sorted(set(" ".join(train).split()) | {"<end>", "<unk>"})
    V = len(vocab)
    contexts = Counter()
    pairs = Counter()

    for sentence in train:
        tokens = ["<start>"] + sentence.split() + ["<end>"]
        for left, right in zip(tokens, tokens[1:]):
            contexts[left] += 1
            pairs[(left, right)] += 1

    def p(next_token, previous):
        next_token = next_token if next_token in vocab else "<unk>"
        return (pairs[(previous, next_token)] + 1) / (contexts[previous] + V)

    losses = []
    for sentence in test:
        tokens = ["<start>"] + sentence.split() + ["<end>"]
        for left, right in zip(tokens, tokens[1:]):
            losses.append(-math.log(p(right, left)))
    print("Held-out NLL:", round(sum(losses) / len(losses), 4))
    print("Held-out perplexity:", round(math.exp(sum(losses) / len(losses)), 4))

    rng = Random(7)
    for _ in range(3):
        token, output = "<start>", []
        for step in range(8):
            weights = [p(w, token) for w in vocab]
            token = rng.choices(vocab, weights=weights, k=1)[0]
            if token == "<end>":
                break
            output.append(token)
        print("Sample:", " ".join(output))

    prior_maintenance = 0.20
    likelihood_high_given_maintenance = 0.90
    likelihood_high_given_normal = 0.10
    evidence = (likelihood_high_given_maintenance * prior_maintenance
                + likelihood_high_given_normal * (1-prior_maintenance))
    posterior = likelihood_high_given_maintenance * prior_maintenance / evidence
    print("P(maintenance | high volume):", round(posterior, 4))

**Inspect, don't merely execute:** Verify that the posterior is 0.6923. Confirm that NLL and perplexity are computed only from test sequences, while counts and vocabulary come from training. The toy sampler may emit the unknown-token symbol or odd transitions because add-one smoothing gives every vocabulary token nonzero probability. That is a **modeling limitation to document**, not something to hide. In a better model, special-token constraints and a declared unknown-token policy are necessary.

**Controlled extension:** Replace the test sentences with held-out self-authored domain sentences, never train on them, and compare an unsmoothed model (which may assign zero probability) with add-one smoothing. Repeat using character versus word tokens, but do not compare their perplexities as though the token units were identical. Keep a fixed seed and corpus card.

**Research note:** Write three sentences distinguishing (i) the exact probability chain rule, (ii) an approximate learned conditional model, and (iii) an approximate variational posterior. These are different kinds of approximation.

---

# PART B — AI FOR CYBERSECURITY

## 6. Start with the decision and evidence, not the algorithm

Consider a SOC that receives authentication events and network flow summaries. Its goal is to flag suspicious account sessions within five minutes of a login. Before training anything, define:

- **Unit of prediction:** one account-session, not one packet or one complete incident.
- **Decision time:** five minutes after login; no later investigation notes can enter features.
- **Positive label:** a subsequently *confirmed* credential-abuse incident under a documented labeling policy.
- **Features available now:** recent failed logins, device novelty, session hour, geolocation consistency, network-flow aggregates where legitimately available.
- **Intervention:** analyst review, not automatic account deletion.
- **Deployment constraints:** rare attacks, limited analyst capacity, missing labels, changing user behavior.

Three learning paradigms answer different questions.

## 7. Supervised learning: learn from labeled examples

A supervised detector learns f_theta(x) ≈ P(y=attack | x) from labeled examples (x,y). Logistic regression, decision trees and ensembles are legitimate starting points. Their output is not automatically calibrated probability; calibration and threshold selection need separate validation.

A binary cross-entropy objective is

**L = -sum_i [y_i log p_i + (1-y_i) log(1-p_i)].**

For an event labeled attack (y=1), predicting p=0.8 incurs -ln(0.8)≈0.223. Predicting p=0.1 incurs -ln(0.1)≈2.303, a much larger penalty. For a benign event (y=0), the penalty is -ln(1-p). This explains why training uses both positive and negative examples, but does not guarantee that a deployed threshold of 0.5 is appropriate.

**Strength:** directly optimizes a known detection target. **Weakness:** verified incident labels are expensive, incomplete and sometimes delayed; labels can encode investigators' earlier rules or coverage gaps.

## 8. Unsupervised learning: unusual is not necessarily malicious

Unsupervised anomaly detection learns patterns in x without needing attack labels for training. Examples include robust statistics, clustering, density estimation and isolation forests. It returns an **anomaly score**, not a confirmed attack probability. Rare events include software deployments, seasonal spikes, new hires and genuine attacks.

Suppose normal login counts per hour are approximately centered around 4 with standard deviation 2. A count of 12 is (12-4)/2=**4 standard deviations** above that baseline. This is unusual *under the assumed reference distribution*, but a legitimate batch operation could produce it. If the distribution is skewed or changes by hour, a global z-score may be misleading.

**Operational question:** Are the unusual events relevant to the specific threat hypothesis, and can analysts review them? A detector that identifies every holiday shift is a poor security control even if its anomaly scores are mathematically valid.

## 9. Semi-supervised learning: exploit unlabeled data without pretending it is labeled

Semi-supervised learning uses a small labeled set L and a larger unlabeled set U. A common method is **pseudo-labeling**: train a supervised model on L, assign provisional labels only to very confident examples in U, then retrain. Other approaches include consistency regularization and graph-based propagation.

### Diagram C: three learning paths

    SUPERVISED
    labeled events + confirmed outcomes --> train --> attack score --> review

    UNSUPERVISED
    unlabeled historical events ---------> fit normality --> anomaly score --> review

    SEMI-SUPERVISED
    small labeled set -----> initial model
                                  |
    unlabeled set ------> confident pseudo-labels
                                  |
                           retrain cautiously --> evaluate on untouched labels

A high confidence score is not proof of a correct pseudo-label. Under domain shift, models can confidently mislabel novel attacks or benign maintenance. Self-training can amplify errors and class imbalance. Do not put validation or test examples into U when the evaluation protocol is meant to represent fully unseen future periods; at minimum, clearly declare any transductive setting.

**Terminology warning:** Some cybersecurity papers use "semi-supervised anomaly detection" to mean training only on verified normal data. That differs from the general ML meaning of jointly using labeled and unlabeled examples. State which definition you adopt.

## 10. Worked example B1: why prevalence overwhelms a good-looking detector

Imagine 10,000 account-sessions: 100 confirmed attacks and 9,900 benign sessions. At a chosen threshold, a supervised detector has **80% recall** and **2% false-positive rate**.

- True positives: 0.80 × 100 = **80**.
- False negatives: 100 - 80 = **20**.
- False positives: 0.02 × 9,900 = **198**.
- True negatives: 9,900 - 198 = **9,702**.
- Precision = 80 / (80+198) = **28.78%**.
- Recall = 80 / 100 = **80%**.
- Accuracy = (80+9,702)/10,000 = **97.82%**.
- Alert workload = **278 reviews per 10,000 sessions**.

If the analyst team can review only 50 alerts in that period, it cannot review all 278. Sorting by score may help, but a review budget alone does not prove which 50 include attacks. Evaluate **recall at a fixed alert budget**, not just a global ROC curve.

### Worked example B2: what happens if the base rate changes?

Keep recall 80% and false-positive rate 2%, but reduce attack prevalence to 0.1% in 100,000 sessions:

- Attacks = 100; benign = 99,900.
- TP = 80; FP = 1,998.
- Precision = 80/(80+1,998) ≈ **3.85%**.
- Total alerts = **2,078**.

The classifier's conditional rates did not change, but most alerts are now false. This is why a published benchmark precision cannot simply be transferred to a real organization.

### Worked example B3: threshold trade-off

An anomaly detector flags the 200 most unusual sessions among 10,000. Suppose 30 of those are attacks. Precision is **30/200=15%** and recall is **30/100=30%**. Its ranking may be useful for prioritization, but the top-200 policy cannot be described as "85% benign traffic detected" or as a probability of attack. Evaluate at the same review budget as the supervised system for a fair comparison.

## 11. Data leakage, selection bias and time

**Leakage:** A feature such as "ticket closed as malicious" is created after the decision time and must not be available to the predictor. Likewise, overlapping flow windows from one incident appearing in both train and test inflate results.

**Label selection:** Incidents that analysts investigated are more likely to have labels. If training labels come only from prior alerts, the model may learn the previous detector's blind spots. Track which events were reviewed and why.

**Temporal drift:** An employee's normal login pattern changes during travel or a new product launch. Measure score distributions and labeled outcomes across time; a shift in scores alone does not establish more attacks.

**Threshold policy:** Choose the threshold using validation data and an explicit operational constraint, such as no more than 20 alerts per 1,000 sessions. Freeze it before the held-out test. Report precision, recall, false-positive rate, PR-AUC when feasible, calibration if interpreting probabilities, and alert volume. Compare against majority-class and simple rule baselines.

## 12. Cybersecurity lab — compare three paradigms on synthetic events (30 minutes)

**Install:** Python 3.10+, numpy and scikit-learn. Suggested command: python -m pip install numpy scikit-learn. Save as day02_cyber.py. The synthetic data is deliberately simplified; results are *not* a cybersecurity benchmark.

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_score, recall_score, confusion_matrix

    rng = np.random.default_rng(42)
    n = 10000
    y = (rng.random(n) < 0.02).astype(int)
    # Two features: synthetic failure-count proxy and device novelty.
    X = np.column_stack([
        rng.normal(0, 1, n) + 1.6 * y,
        rng.normal(0, 1, n) + 1.2 * y
    ])
    X[7000:, 0] += 0.25  # Explicitly simulated benign distribution shift.
    train, val, test = slice(0, 6000), slice(6000, 8000), slice(8000, n)
    Xt, yt = X[train], y[train]
    Xv, yv = X[val], y[val]
    Xs, ys = X[test], y[test]

    # Limited confirmed labels: targeted review enriches positive examples.
    pos = np.flatnonzero(yt == 1)
    neg = np.flatnonzero(yt == 0)
    known = np.r_[rng.choice(pos, 40, replace=False),
                  rng.choice(neg, 460, replace=False)]
    unlabelled = np.setdiff1d(np.arange(len(yt)), known)
    supervised = LogisticRegression(max_iter=1000, random_state=42)
    supervised.fit(Xt[known], yt[known])
    sv = supervised.predict_proba(Xv)[:, 1]
    st = supervised.predict_proba(Xs)[:, 1]

    # Unsupervised: fit a reference distribution using only training features.
    anomaly = IsolationForest(n_estimators=100, random_state=42)
    anomaly.fit(Xt)
    av = -anomaly.score_samples(Xv)
    at = -anomaly.score_samples(Xs)

    # Pseudo-label only highly confident training-period events.
    pu = supervised.predict_proba(Xt[unlabelled])[:, 1]
    accept = (pu < 0.01) | (pu > 0.99)
    pseudo_ids = unlabelled[accept]
    pseudo_y = (pu[accept] > 0.5).astype(int)
    semi = LogisticRegression(max_iter=1000, random_state=42)
    semi.fit(np.vstack([Xt[known], Xt[pseudo_ids]]),
             np.r_[yt[known], pseudo_y])
    mv = semi.predict_proba(Xv)[:, 1]
    mt = semi.predict_proba(Xs)[:, 1]

    def threshold_for_budget(validation_scores, max_alert_fraction=0.03):
        # Threshold selected ONLY on validation scores, no test peeking.
        return np.quantile(validation_scores, 1-max_alert_fraction)

    for name, vs, ts in [
        ("Supervised", sv, st),
        ("Unsupervised anomaly", av, at),
        ("Semi-supervised", mv, mt),
    ]:
        cutoff = threshold_for_budget(vs)
        predicted = (ts >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(ys, predicted, labels=[0, 1]).ravel()
        print(name, "TP", tp, "FP", fp, "FN", fn, "TN", tn,
              "precision", round(precision_score(ys, predicted, zero_division=0), 3),
              "recall", round(recall_score(ys, predicted, zero_division=0), 3),
              "alerts", int(predicted.sum()))
    print("Known labels:", len(known),
          "Pseudo-labels accepted:", len(pseudo_ids))

**Interpretation:** Each method receives a validation-derived score cutoff intended to flag roughly 3% of validation sessions. A test distribution shift can change that fraction. The "known" labels were deliberately enriched for positives through targeted review; they are **not** a random representative sample. This makes the demonstration practical but means raw model probabilities are not necessarily calibrated to the true 2% prevalence. The unsupervised reference includes a small proportion of attacks because it has no labels. Pseudo-labeling may add few records or mostly negatives. Those outcomes are scientifically useful to record.

**Required checks:** (1) No test labels enter model fitting or threshold selection. (2) Compare alert counts, not just precision. (3) Rerun with y prevalence changed from 0.02 to 0.005 and note how the precision and class-count variability change. (4) Try shifting X[7000:,0] by 0.75 rather than 0.25 and record how normal-behavior drift affects the detectors. (5) Report the number of true positives in each split, since small positive counts make estimates noisy.

**Real-data follow-on, not for today's quick lab:** Use an authorized CICIDS2017 flow-feature subset. Record the official source, file hashes, capture days, label definitions, missing and infinite values, duplicates, entity IDs and split manifest. Fit all preprocessing on training only. Do not assume random row splitting gives an independent test; discuss unseen attack types under temporal splits. Never execute malware or probe live systems.

---

# PART C — INTEGRATED DOCTORAL RESEARCH

## 13. Connect the two courses: a latent event model is not a SOC verdict

A generative model might learn P(events) or P(events | latent operating regime). It can help summarize typical sequences or highlight improbable transitions. A supervised classifier instead targets P(confirmed incident | features). These are **different estimands**. An unusual sequence may be a deployment; a common-looking login may still be compromised. A retrieval-grounded language assistant can *explain* an alert using cited evidence but must not promote a model's anomaly score into a confirmed incident.

### Diagram D: evidence-grounded security research architecture

    Authorized logs + provenance
               |
        time-valid features
               |
       +-------+---------+
       |                 |
    classifier      sequence/latent model
       |                 |
    attack score      anomaly score
       +-------+---------+
               |
       calibrated triage policy
               |
    retrieve relevant evidence
               |
       LLM draft with citations
               |
      analyst approval / override
               |
       outcome + error log

The analyst's confirmed outcome may arrive hours or days later. Never train a model on future outcome data while claiming five-minute detection. Keep raw evidence, retrieved snippets, model versions, thresholds and overrides so another researcher can audit the conclusion.

**Proposed doctoral hypothesis:** "At a fixed analyst review budget, combining a calibrated supervised account-abuse detector with an independently trained event-sequence anomaly score improves recall of previously unseen abuse patterns, without increasing unsupported incident claims in an evidence-grounded language assistant."

**Falsifiable design:** Compare rules-only, supervised-only, anomaly-only, fused scores and fused scores plus evidence-grounded assistant. Use time-separated test blocks and attack-family slices. Hold alert budget constant. Primary metrics: recall at budget and unsupported-claim rate. Secondary metrics: precision, FPR, analyst review time, citation correctness, calibration, latency and subgroup performance. Pre-register thresholds and evaluation protocol. The hypothesis is false if fusion provides no robust gain, if it increases unsupported claims, or if the gain disappears under a realistic temporal split.

### Research-paper reading matrix for today

| Source | What to inspect | One skeptical question |
|---|---|---|
| Kingma & Welling, *Auto-Encoding Variational Bayes* (2013/2014), https://arxiv.org/abs/1312.6114 | ELBO, approximate posterior, reparameterization | What happens if the approximate posterior family is too restrictive? |
| Bommasani et al., *On the Opportunities and Risks of Foundation Models* (2021), https://arxiv.org/abs/2108.07258 | Broad pretraining, adaptation, evaluation risks | Which claims transfer to your actual security telemetry? |
| van Engelen & Hoos, *A survey on semi-supervised learning* (2020), https://doi.org/10.1007/s10994-019-05855-6 | Assumptions and families of semi-supervised methods | When do pseudo-labels reinforce existing bias? |
| CICIDS2017, University of New Brunswick, https://www.unb.ca/cic/datasets/ids-2017.html | Capture design, labels, attack days, feature provenance | Can the evaluation split be defended against leakage? |

For each paper, save: question, dataset/assumptions, method, baseline, evaluation split, result, one limitation, and one reproducible next experiment. A literature matrix is more useful than a list of paper titles.

## 14. Verified global AI and cybersecurity intelligence — as of early 2 November 2026 IST

**AI — 28 October 2026: Google introduced Gemini 4 Carbon.** Google's announcement reports 88.2% on SEC-bench Pro, 54.8% on ExploitGym and 71.0% on CWE-bench, and describes initially limited access for cyber defenders. These are *provider-reported benchmark scores*; they do not establish safe autonomous incident response. Research action: compare task definitions, permitted tools, evaluation environments and safety controls before using a cyber-capability headline. Primary source: https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/

**AI — 26 October 2026: OpenAI introduced GPT-6.1 Terra for computer use.** The official release describes improved computer-use and visual reasoning, and says the model meets its Preparedness Framework's Critical threshold for cybersecurity capability. A capability threshold is not permission to run unsupervised defensive or offensive actions. Research action: add authorization boundaries, evidence checks and human approvals to any computer-using SOC agent. Primary sources: https://openai.com/index/introducing-gpt-6-1-terra/ and https://help.openai.com/en/articles/9624314-model-release-notes

**AI — 6 October preview, with 29 October reporting: Mistral Large 4.** Mistral's primary release page describes a 1-trillion-parameter multimodal model with 52 billion active parameters and planned open weights; independent 29 October reporting says weights became available. Because the original primary preview page still describes a future weight release, treat availability as a separately reported update rather than infer it from the preview. Research action: distinguish open weights, open training data, benchmark validity, deployment control and dual-use risk. Primary overview: https://mistral.ai/news/mistral-large-4/ ; dated report: https://fortune.com/2026/10/29/mistral-releases-new-model-large-4-and-claims-top-open-model-for-agentic-coding-and-cyber-defense/

**Cybersecurity — 29 October 2026: active exploitation of PHP2Shell affecting exposed PAN-OS management interfaces.** Palo Alto Networks Unit 42 updated its advisory on 29 October to report exploitation involving CVE-2026-109618 and CVE-2026-109619. The UK's NCSC had issued a 21 October warning about critical PHP issues and urged patching. Research action: create a *defensive* evidence map linking asset inventory, software version, public exposure, patch state and host telemetry; do not equate a CVE match with confirmed compromise. Vendor advisory: https://unit42.paloaltonetworks.com/cve-2026-109618-cve-2026-109619/ ; government guidance: https://www.ncsc.gov.uk/news/critical-security-issue-affecting-php ; PHP project releases: https://www.php.net/archive/2026.php

**Evidence quality note:** These are the latest relevant items verified from accessible primary sources and clearly dated reporting during preparation, not a claim that no other global event occurred. Distinguish original publication date from later updates. Avoid repeating a vendor's marketing benchmark as independently replicated evidence.

## 15. Learning resources — watch first, listen second, then read

**Video-first learning (conceptual):**

1. 3Blue1Brown, *But what is a GPT?* — visualize autoregressive token prediction and Transformer behavior: https://www.youtube.com/watch?v=wjZofJX0v4M
2. StatQuest video channel — search for "variational autoencoders" and "precision recall" and use the visual derivations alongside today's equations: https://www.youtube.com/@statquest
3. Stanford CS236 deep generative models course materials and lecture recordings index: https://deepgenerativemodels.github.io/

**Audio / podcast follow-up:** Practical AI for ML experimentation and applied model decisions: https://changelog.com/practicalai ; SANS Internet Storm Center Stormcast for security operations and threat context: https://isc.sans.edu/podcast.html . These are ongoing feeds, not assertions that a particular episode was released today. A complete lesson narration, if generated by the separate GitHub workflow, must be verified before being represented as available.

**Primary readings and tools:** Dive into Deep Learning, language models https://d2l.ai/chapter_recurrent-neural-networks/language-model.html ; VAE paper https://arxiv.org/abs/1312.6114 ; scikit-learn novelty and outlier detection guide https://scikit-learn.org/stable/modules/outlier_detection.html ; NIST adversarial machine learning taxonomy https://csrc.nist.gov/pubs/ai/100/2/e2025/final ; CICIDS2017 official dataset https://www.unb.ca/cic/datasets/ids-2017.html .

**Visual memory map to sketch:** In the center write "What is observed? What is hidden? What is labeled?" Branch into AR next-token conditionals, latent prior/posterior, supervised incident labels, unsupervised anomaly scores and semi-supervised provisional labels. Draw a red boundary around *future data* to remind yourself it cannot cross into prediction-time features.

## 16. Ten-question doctoral recall test

1. Is autoregressive factorization an approximation? **No, the probability chain rule is exact; a learned conditional model is approximate.**
2. Why can teacher-forced training differ from generation? **At inference the model conditions on its own earlier outputs, including mistakes.**
3. What is the difference between p(z) and p(z|x)? **Prior before observing x versus posterior after observing x.**
4. In the maintenance example, why isn't the posterior 90%? **The 20% prior and 10% high-volume likelihood under normal operation matter.**
5. What two terms appear in the VAE ELBO? **Expected reconstruction log likelihood and KL regularization.**
6. Does a high anomaly score imply malicious intent? **No; unusual benign events are common.**
7. What does semi-supervised pseudo-labeling risk? **Amplifying confident errors, selection bias and class imbalance.**
8. Why might 97.82% accuracy be poor for a SOC? **278 alerts include 198 false positives and exceed a 50-alert review budget.**
9. Which split best approximates a future deployment? **A defensible temporal or entity-grouped split matching the actual decision setting; document unseen-class complications.**
10. What distinguishes an evidence-grounded explanation from a generated assertion? **Citable source events, explicit uncertainty and a human-verifiable link from claim to evidence.**

## 17. Submission checklist and progress record

Save a research log entry for 2026-11-02 with:

- **Theory:** one diagram contrasting AR and latent-variable generation; a handwritten derivation of the 0.6923 posterior and ELBO arithmetic.
- **GenAI experiment:** day02_genai.py, environment version, seed, NLL/perplexity, three samples and two failure observations.
- **Cyber experiment:** day02_cyber.py, train/validation/test boundaries, number of labeled and pseudo-labeled records, confusion matrices and alert counts for three approaches.
- **Research:** two entries in the literature matrix, one falsifiable hypothesis, one evidence-quality critique of a news benchmark.
- **Integrity:** no test leakage, no unauthorized scanning, no claim that anomaly means compromise, no claim that a future MP3 already exists.
- **Self-rating:** 0=not yet, 1=with notes, 2=independent explanation, 3=can defend under questioning. Rate each of the six outcomes from Section 1.

**Evening handoff:** Deepen today's autoregressive/latent-variable comparison with a tiny VAE derivation and posterior-collapse diagnostics; deepen supervised/unsupervised/semi-supervised security learning with calibrated thresholds, label bias, temporal validation and a stronger controlled experiment. Preserve these same topics. **Next morning's syllabus progression:** foundation-model evolution and transfer (GenAI Unit 1 Session 3), and feature engineering/representations for security telemetry (Cybersecurity Unit 1 Session 3).

**Source alignment:** Both attached 19 October 2026 PhD learning pathway PDFs: *Advanced Generative AI and Foundation Models* (Unit 1 Session 2 and Unit 1 lab) and *AI for Cybersecurity* (Unit 1 Session 2 and Unit 1 lab). This is independent study content; it does not alter the university's 40 contact-hour / 60 independent-hour course requirements.

**Audio status:** The Markdown is the source for the separate GitHub Actions narration workflow. The lesson's existence does **not** verify MP3 generation, publishing or playback.
