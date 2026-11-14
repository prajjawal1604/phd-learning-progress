# Evening PhD deep dive — Saturday 14 November 2026 (Asia/Kolkata)
**Dependency verified:** Read committed [morning lesson](../../../phd-morning/lessons/2026-11-14/lesson.md) and [morning references](../../../phd-morning/lessons/2026-11-14/references/index.md) on main before authoring. This is a deeper treatment of **GenAI Unit 3 Session 5 (RLHF, preferences, DPO)** and **Cybersecurity Unit 3 Session 4 (model extraction, membership and attribute inference)**, *not* a new unit. Original syllabus sources: 19 October 2026 Gmail attachments, *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 3) and *AI for Cybersecurity Learning Pathway* (Unit 3). All examples below are synthetic; never probe third-party systems.

## Voice-first teaching route and protocol
**10 min:** Tell the incident-response preference story, derive the DPO logit from two policy ratios, explain low-prevalence membership inference, ask Q1–Q5.
**30 min:** Work the KL constrained optimum, demonstrate label-noise bias and length effects, derive likelihood-ratio threshold and precision, ask Q1–Q12.
**Extended 90–150 min:** Execute both labs, prove the threshold and information arguments, critique primary papers, construct a preregistered privacy/utility protocol, ask all 24 questions. For each question, tutor waits for learner's attempt, asks "under what assumption?", then reveals the answer and follow-up. No spoken audio file is created.

## Prerequisites, precise definitions and intuition
Know log likelihood, softmax, conditional sequence probability, Bayes rule, gradients, KL divergence, ROC, train/test splits and LoRA/QLoRA. A **policy** πθ(y|x) is a distribution over whole responses y to prompt x. A **reference** π0 is a frozen policy. A **preference label** says one response was chosen over another, conditional on rubric and annotator. **Bradley–Terry** models pair choice by a logistic function of reward difference. **DPO** trains a policy using preference pairs and a reference rather than an explicit reward-model/PPO loop. **Fidelity** measures agreement of a surrogate with a target model; **accuracy** measures agreement with ground truth. **Membership inference** asks whether a specific record was in training; **attribute inference** predicts a hidden attribute, preferably measured as incremental improvement beyond publicly available features. **Likelihood ratio** compares the evidence probability under two hypotheses. **Calibration** means a probability forecast matches frequencies under a defined population. **DP** bounds changes in output distribution under neighboring datasets, subject to a precise adjacency definition.

Imagine a SOC copilot. A reviewer prefers "check process lineage before isolating" over "isolate immediately" for a weak alert. DPO can learn this preference; it cannot establish whether the answer is factually supported, authorized, or privacy-safe. Meanwhile, a query interface may leak information about the copilot's training records. The unifying question: **can preference gains coexist with defensible privacy guarantees?** That is a research hypothesis, not an established causal result.

# Part I — Advanced DPO and preference learning

## 1. Re-derive the optimum, with assumptions exposed
For fixed prompt x and finite response set Y, maximize
J(π)=Σ_y π(y)r(y)−βΣ_y π(y)log[π(y)/π0(y)], β>0,
subject to Σ_y π(y)=1 and π(y)≥0. Assume π0(y)>0 for all responses considered and finite rewards. Form Lagrangian J+λ(Σπ−1). The derivative wrt π(y) is r(y)−β(log[π(y)/π0(y)]+1)+λ=0, hence
π*(y)=π0(y)exp(r(y)/β)/Z, Z=Σ_z π0(z)exp(r(z)/β).
Equivalently r(y)=βlog[π*(y)/π0(y)]+βlog Z. The additive log Z cancels **only** when comparing responses for the same x. Under Bradley–Terry,
P(y+≻y−|x)=σ(r+−r−).
Replacing optimal policy with parameterized πθ gives
h=β{[logπθ(y+|x)−logπ0(y+|x)]−[logπθ(y−|x)−logπ0(y−|x)]},
L_DPO=softplus(−h)=log(1+exp(−h)).
DPO is a surrogate derived from idealized regularized reward optimization and a pairwise noise model, not a proof that every finite neural-network optimization recovers π*.

**Gradient:** dL/dh=σ(h)−1. Let g+=∇θlogπθ(y+|x), g−=∇θlogπθ(y−|x). Then ∇θL=β[σ(h)−1](g+−g−). This is a *difference of sequence score functions*, not two independent probability knobs. Shared parameters mean increasing the winner-to-loser odds need not increase the winner's absolute probability. At h=0, β=0.2, the scalar coefficient is −0.1. For β→0 with fixed log ratios, gradient vanishes; do not confuse this limit with how the optimum π* changes with β in the original reward problem.

**Two-response worked case:** π0=(0.5,0.5), πθ=(0.6,0.4), β=1. The odds ratio is (0.6/0.4)/(0.5/0.5)=1.5, h=ln1.5≈0.405465, modeled preference σ(h)=0.6, loss=−ln0.6≈0.510826. If the preference labels are wrong, the exact same gradient reinforces the wrong response.

## 2. Label-noise and disagreement: derive the optimum, not merely a warning
Suppose a fixed pair is labeled winner A with probability q across repeated independent annotations. Write h for the model's preference logit for A. Expected cross-entropy is
E[L]=−q logσ(h)−(1−q)log[1−σ(h)].
Derivative σ(h)−q; second derivative σ(h)[1−σ(h)]>0. Thus the unconstrained optimum is σ(h*)=q and h*=log[q/(1−q)] for 0<q<1. If true rubric-consistent choice is q=0.8, h*=ln4≈1.3863. Symmetric label flips at rate η transform observed q into q_obs=η+(1−2η)q. With η=.2, q_obs=.68 and h*=ln(.68/.32)≈0.7538. This is **attenuation**, not merely random training noise. With η>.5 the preference direction reverses. If annotator disagreement is systematic by group, a single scalar q hides different utility functions. Keep annotator/rubric metadata and test subgroup performance; avoid treating majority vote as universal truth.

**Counterexample:** 80% of raters prefer a short response omitting uncertainty; in safety-critical triage, that omission increases wrong isolation. Pairwise accuracy can improve while operational harm rises. Measure evidence fidelity, false containment recommendations, refusal rates and reviewer disagreement independently.

## 3. Sequence length, reference drift and optimizer mechanics
An autoregressive response probability is logπ(y|x)=Σ_{t=1}^{T_y} log p(y_t|x,y_<t). Prompt, padding and unrelated continuation tokens must be masked. A one-token response with log probability −0.2 and a four-token response with each token log probability −0.2 have sequence log probabilities −0.2 and −0.8; raw sequence comparisons have length dependence. Do **not** blindly divide by T and call it DPO: length normalization changes the objective and the derived interpretation. Instead report length-stratified performance and optionally compare explicitly named alternative objectives.

For the same (x,y), using a moving π0 invalidates the frozen-reference objective; reference version, tokenizer, template, special tokens and truncation rules must be pinned. If policy uses QLoRA, the frozen base and adapter must be unambiguously distinguished from the frozen reference; quantization changes log-probability numerics. Compute response-only logprobs in stable precision, use log-sigmoid/softplus for numerical stability, exclude empty or fully truncated responses, check train/validation split by incident family, and avoid evaluating on preference pairs already seen in training.

Architecture:
~~~text
permissioned SOC prompts -> two safe responses -> rubric + annotator IDs
          |                       |                    |
          +-------------- curated same-prompt pairs --+
                                  |
               fixed tokenizer + chat template + response mask
                         /                         \
                 frozen reference              trainable policy
                    logprobs                      logprobs
                         \                         /
                       beta * difference of log-ratios
                                      |
                              softplus(-logit)
                                      |
                             policy/adapter update
                                      |
             held-out incident-family factuality + safety + privacy audit
~~~
**When DPO works:** pair labels correlate with desired behavior, training and evaluation distributions match reasonably, and independent metrics catch reward misspecification. **When it fails:** inconsistent prompts, label bias, reference mismatch, length shortcuts, memorization, distribution shift, overrefusal and false confidence. Compared with PPO-based RLHF, basic DPO avoids online rollouts and explicit reward-model optimization, but cannot eliminate preference-data quality or distribution-shift problems.

## 4. Advanced runnable Python lab: noisy preferences and finite-difference gradient
Python 3.10+ standard library; save as dpo_evening.py. This is a scalar **preference logit** experiment, not LLM training.
~~~python
from math import exp, log, isclose
def sigmoid(z):
    return 1/(1+exp(-z)) if z>=0 else exp(z)/(1+exp(z))
def softplus(z):
    return max(z,0)+log(1+exp(-abs(z)))
def expected_loss(h,q):
    return q*softplus(-h)+(1-q)*softplus(h)
def grad(h,q): return sigmoid(h)-q
q_true, flip=.8,.2
q_seen=flip+(1-2*flip)*q_true
h_star=log(q_seen/(1-q_seen))
assert isclose(q_seen,.68)
assert isclose(h_star,.7537718023763802,abs_tol=1e-12)
assert abs(grad(h_star,q_seen))<1e-12
eps=1e-6
fd=(expected_loss(h_star+eps,q_seen)-expected_loss(h_star-eps,q_seen))/(2*eps)
assert abs(fd-grad(h_star,q_seen))<1e-8
assert expected_loss(h_star,q_seen)<expected_loss(0,q_seen)
# fixed-policy two-response DPO loss
h=log((.6/.4)/(.5/.5))
assert isclose(softplus(-h),-log(.6),abs_tol=1e-12)
print("observed q",round(q_seen,2),"optimal logit",round(h_star,6))
print("DPO toy loss",round(softplus(-h),6),"gradient check PASS")
~~~
Expected: observed q 0.68, optimal logit 0.753772; DPO toy loss 0.510826; gradient check PASS. **Extensions:** vary flip from 0 to .6 and plot optimum; split q by two annotator groups; compare β values by parameterizing h=βz; test finite-difference gradients near extreme logits; compare length distributions. **Interpretation caution:** h*=logit(q) does not imply the policy's calibrated probability of factual correctness equals q.

## 5. Critical reading and falsifiable extension
Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155, https://arxiv.org/abs/2203.02155: useful for demonstration→preference→reward→RL pipeline; human preference judgments and annotator selection constrain transferability. Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290, https://arxiv.org/abs/2305.18290: derives DPO and studies selected tasks; neither perfect labels nor universal safety follow. **Research hypothesis:** a privacy-aware preference rubric reduces accidental disclosure in a SOC assistant without materially worsening legitimate evidence-grounded answers. Compare SFT, ordinary DPO and privacy-rubric DPO at matched compute, fixed held-out incident families, multiple seeds, predeclared error bars, and no real personal data.

# Part II — Model extraction and inference: statistical depth

## 6. Distinguish target function, records and hidden properties
**Extraction:** from permitted queries x_i and target outputs f(x_i), fit g; evaluate fidelity P_D[g(x)=f(x)] on independently sampled D. If f is systematically wrong, high fidelity can copy wrongness. For ideal logistic f(x)=σ(w·x+b), exact probability outputs reveal logit(f(x))=w·x+b; a full-rank design could identify coefficients. Clipping, label-only output, nonidentifiability and shifted query distributions complicate this. This is an explanation of risk, **not** a recipe to target an external API.
**Membership:** hypothesis H1 that a record was used for training versus H0 not used; record definition, duplicate policy, population and access all matter.
**Attribute inference:** distinguish model-added information from public-feature baseline; a high attribute accuracy that is achievable without the model is not model-attributable leakage.

## 7. Derive the Bayes-optimal membership threshold
Let score s have density p1(s) under H1 and p0(s) under H0. Prior π=P(H1). Posterior odds are
P(H1|s)/P(H0|s)=[π/(1−π)]·[p1(s)/p0(s)].
The likelihood ratio Λ(s)=p1(s)/p0(s) is evidence; the posterior also depends on prior. With false-positive cost C_FP and false-negative cost C_FN, declaring member has conditional risk C_FP P(H0|s); declaring nonmember has risk C_FN P(H1|s). Declare member iff
Λ(s)>[C_FP(1−π)]/[C_FN π].
For equal costs and π=.01, threshold is 99. This is why a score that discriminates in a balanced test may yield poor positive predictive value in deployment.

**Threshold-event worked example:** π=.01, TPR=.8, FPR=.1. Bayes gives PPV=.008/(.008+.099)=.074766 (7.48%). A different threshold TPR=.4, FPR=.001 gives PPV=.004/(.004+.00099)=.801603 (80.16%) but misses more members. AUC alone hides this trade-off. These are hypothetical operating points, not empirical attacks.

**Caution:** a score threshold is Bayes-optimal only if the likelihood-ratio is monotonic in that score; arbitrary confidence scores need not have that property. In high-dimensional, nonstationary or adaptively selected queries, density estimation and the assumed prior may be wrong.

## 8. Information limits and privacy guarantees
The Neyman–Pearson principle states that for simple hypotheses and known densities, thresholding Λ is most powerful for a fixed false-positive rate (subject to randomization at ties). This is a statistical ideal; operational MI scores approximate it. If a query response is post-processed deterministically, data-processing inequality says it cannot increase *ideal* information about membership relative to the original response, but it can still reveal information and can improve a particular poorly designed attack's practical performance. Label-only interfaces reduce richness; they do not prove zero leakage.

Differential privacy: a randomized mechanism M is (ε,δ)-DP if for all neighboring datasets D,D' and measurable output events S,
P[M(D)∈S]≤exp(ε)P[M(D')∈S]+δ.
For pure DP (δ=0), with balanced neighboring-dataset prior and likelihood ratio bounded by exp(±ε), posterior membership odds are bounded by exp(±ε); at ε=ln2, posterior lies between 1/3 and 2/3. This bound concerns the defined adjacency and observation mechanism; it does not guarantee secrecy of already-public facts, absence of model extraction, or safety of external retrieval/logging. DP-SGD clips per-example gradients and adds noise, with a privacy accountant for repeated steps. Utility and rare-attack recall may suffer, so measure both.

**Counterexamples:** (a) train/test generalization gap is tiny yet a rare memorized string leaks; (b) two datasets have different source populations, making MI AUC high due to dataset shift rather than training membership; (c) surrogate fidelity 99% on benign traffic but near zero on rare incidents; (d) a privacy-preserving model still exposes sensitive content through unprotected RAG retrieval; (e) response throttling reduces volume but does not eliminate a single informative query.

Architecture:
~~~text
owned synthetic records -> group/time split -> owned target model
              |                                   |
         membership labels                   query interface
              |                            (fixed permissions)
              +------- privacy audit ------------+
                         |        |        |
                     score bins  prior   query ledger
                         |        |        |
                   ROC / low-FPR PPV / intervals
                                  |
                 compare public-feature baseline, utility, privacy
                                  |
                      reviewed research report only
~~~

## 9. Advanced runnable Python lab: prevalence and decision costs
Save as privacy_evening.py; no network, training records or external model.
~~~python
from math import isclose, log, exp
def ppv(prior,tpr,fpr):
    return prior*tpr/(prior*tpr+(1-prior)*fpr)
def posterior_odds(prior,lr):
    return (prior/(1-prior))*lr
p=.01
assert isclose(ppv(p,.8,.1),.07476635514018691)
assert isclose(ppv(p,.4,.001),.8016032064128257)
# Equal costs: a likelihood ratio of 50 is not enough at 1% prior.
assert posterior_odds(p,50)<1
assert posterior_odds(p,100)>1
# Unequal costs: missing a member costs 10x a false alarm.
threshold=(1-p)/(10*p)
assert isclose(threshold,9.9)
# Pure-DP toy posterior bounds for epsilon=ln 2 and prior 1/2
eps=log(2)
lower=exp(-eps)/(1+exp(-eps))
upper=exp(eps)/(1+exp(eps))
assert isclose(lower,1/3) and isclose(upper,2/3)
print("PPV loose threshold",round(ppv(p,.8,.1),6))
print("PPV strict threshold",round(ppv(p,.4,.001),6))
print("LR threshold with 10x FN cost",round(threshold,1))
print("pure DP balanced-prior bounds",round(lower,6),round(upper,6))
~~~
Expected: PPV loose threshold 0.074766; PPV strict threshold 0.801603; LR threshold with 10x FN cost 9.9; pure DP balanced-prior bounds 0.333333 0.666667. **Advanced extension:** synthesize scores from two Gaussian distributions and estimate low-FPR TPR using fixed seeds; stratify by rare/common records; group-bootstrap uncertainty by source; shift nonmember distribution and observe spurious AUC; compare against public-feature-only attribute inference. Do not report these simulations as real privacy vulnerabilities.

## 10. Paper critique and ethical methodology
Shokri et al. (2017), *Membership Inference Attacks Against Machine Learning Models*, DOI:10.1109/SP.2017.41, https://doi.org/10.1109/SP.2017.41: shadow-model methodology; evaluate whether attack knowledge, output richness and dataset matching hold in your SOC setting. Tramèr et al. (2016), *Stealing Machine Learning Models via Prediction APIs*, USENIX Security, https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer: extraction under studied API settings; fidelity is not proof of recovered weights or private records. Carlini et al. (2021), *Extracting Training Data from Large Language Models*, USENIX Security, https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting: studied memorization/extraction, not an inevitability for every model. Yeom et al. (2018), *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*, arXiv:1709.01604, https://arxiv.org/abs/1709.01604: analyzes overfitting, membership and attribute inference, including why generalization alone does not certify privacy. Abadi et al. (2016), *Deep Learning with Differential Privacy*, arXiv:1607.00133, DOI:10.1145/2976749.2978318, https://arxiv.org/abs/1607.00133: DP-SGD and accounting; empirical utility/privacy results are setup-specific.

Ethics: run only on self-owned or explicitly authorized data/models; do not infer real individuals' attributes, recover secrets or issue adaptive probes to third parties. Treat synthetic canaries as test artifacts, encrypt logs, minimize query retention, obtain institutional review before real-user studies and pre-register responsible disclosure. A successful privacy audit should identify risks and controls, not expose training records.

# Cross-track dissertation experiment: privacy-aware preference optimization
**Question:** Does changing preference labels/rubrics affect disclosure risk at fixed model capacity and data? Hold base checkpoint, corpus, tokenization, retrieval permissions, incident-family splits and query interface constant. Arms: SFT, standard DPO, privacy-aware DPO; optional BF16-LoRA versus QLoRA as a separately randomized factor. Collect blinded preference ratings, factuality, correct refusal, evidence citations, utility on rare incidents, member/nonmember low-FPR advantage, PPV under declared priors and model-extraction fidelity on permitted synthetic queries. Record seed, version, β, epochs, response length, privacy accountant if DP used, uncertainty and failure cases. Use group-level bootstrap over incident families; do not claim causality if only one seed or data sources differ. Negative findings (e.g., reduced disclosure but unsafe overrefusal) are publishable evidence when protocol is sound.

# Oral Socratic ladder — 24 questions with answers and probes
1. **Why does same-prompt comparison matter?** Shared partition term cancels. *Probe:* what breaks for different prompts?
2. **What is π0?** Frozen reference distribution. *Probe:* why log probabilities rather than raw logits?
3. **Derive π*.** π0 exp(r/β)/Z. *Probe:* where does Z come from?
4. **What is DPO's logit?** β times difference of policy/reference log-ratios. *Probe:* compute for 0.6/0.4 versus 0.5/0.5.
5. **What is loss at h=0?** ln2. *Probe:* derivative?
6. **What is the gradient wrt θ?** β(σ(h)−1)(g+−g−). *Probe:* why shared parameters matter?
7. **Can winner probability fall after a DPO update?** Yes, relative odds and normalization are coupled. *Probe:* construct a three-response counterexample.
8. **Why not normalize sequence likelihood silently?** It changes the objective. *Probe:* how evaluate length bias?
9. **What does q represent?** Annotation probability under defined rubric. *Probe:* is it factual correctness?
10. **What is optimal h under noisy labels?** logit(q_obs). *Probe:* derive q_obs under symmetric flips.
11. **If q=.8, η=.2, what is q_obs?** .68. *Probe:* logit? .7538.
12. **When can labels invert preference?** Flip rate >.5 under symmetric model. *Probe:* what about systematic bias?
13. **Define extraction fidelity.** Agreement with target over a specified test distribution. *Probe:* how differs from task accuracy?
14. **Can label-only access leak?** Yes, potentially. *Probe:* what additional assumptions govern risk?
15. **What is membership inference?** Test training inclusion of a record. *Probe:* define record identity with duplicates.
16. **What is attribute inference?** Predict a hidden property. *Probe:* why compare public-feature baseline?
17. **Write posterior odds.** Prior odds times likelihood ratio. *Probe:* explain prior dependence.
18. **Derive cost-sensitive LR threshold.** C_FP(1−π)/(C_FNπ). *Probe:* threshold for π=.01 and equal costs? 99.
19. **PPV at 1%, .8 TPR, .1 FPR?** 7.48%. *Probe:* why not 80%?
20. **What does AUC omit?** Prior, decision costs and low-FPR operating point. *Probe:* name better deployment reporting.
21. **What does Neyman–Pearson guarantee?** LR test optimal under simple known hypotheses at fixed FPR. *Probe:* what fails with distribution shift?
22. **State (ε,δ)-DP.** Neighboring-output probability bound with additive δ. *Probe:* what is the adjacency definition?
23. **Does DP guarantee no leakage via RAG?** No. *Probe:* trace retrieval and logs.
24. **Design a falsifiable dissertation test.** Pre-register data, arms, priors, low-FPR privacy metric, utility and uncertainty. *Probe:* what negative outcome refutes your hypothesis?

## Rapid recall and misconception audit
Recall without notes: derive π* and h; calculate q_obs and gradient; distinguish preference from truth; calculate posterior odds and PPV; define fidelity, membership, attribute inference and DP adjacency; explain why group/time splits matter. Reject: "DPO automatically improves truth"; "small training gap proves privacy"; "AUC=.5 guarantees privacy"; "label-only output is secret"; "DP protects arbitrary retrieval"; "quantization prevents leakage". The walk ends with a one-page hypothesis, two measurable outcomes, three controls, one falsification condition and a permissions checklist.

## Source map and reproducibility
- **Syllabus and continuity:** original 19 Oct Gmail PDFs; 14 Nov morning [lesson](../../../phd-morning/lessons/2026-11-14/lesson.md) and [index](../../../phd-morning/lessons/2026-11-14/references/index.md).
- **DPO derivation, annotation assumptions:** this document §§1–4; Rafailov 2023; Ouyang 2022. Numeric results are synthetic calculations, not paper outcomes.
- **Extraction and privacy definitions:** §§6–9; Tramèr 2016; Shokri 2017; Yeom 2018; Carlini 2021; Abadi 2016.
- **Research design and ethical boundaries:** §§5,10 and cross-track protocol; proposed hypotheses, not literature-established causal claims.
- **Local PDFs, access/license and detailed reference summaries:** [evening companion index](references/index.md). No paper binaries were assumed or fabricated. No email, audio or narration was generated.
