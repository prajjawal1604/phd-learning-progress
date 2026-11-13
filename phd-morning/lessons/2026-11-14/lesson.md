# Morning PhD teaching package — 14 November 2026 (Asia/Kolkata)
**Syllabus progression:** After the 13 November morning/evening sessions on quantization/QLoRA and poisoning/backdoors, today advances to **GenAI Unit 3 Session 5: preferences, RLHF and direct preference optimization** and **Cybersecurity Unit 3 Session 4: model extraction, membership inference and attribute inference**. Both scopes were checked against the original 19 October 2026 Gmail attachments, *Advanced Generative AI and Foundation Models Learning Pathway* (p. 8) and *AI for Cybersecurity Learning Pathway* (p. 8). Prior sessions are prerequisites, not repeated. This lesson is an offline, synthetic teaching artifact; it is not evidence of a live attack or a production benchmark.

## How to tutor this while walking
**10 minutes:** Explain the difference between SFT and pairwise preferences; derive the Bradley–Terry preference probability; distinguish extraction from membership inference; answer Q1–Q6. **30 minutes:** Trace SFT→reward model→PPO versus SFT→DPO; derive DPO loss and gradients; compute one membership-inference confusion matrix; answer Q1–Q14. **Extended 90–150 minutes:** Work both labs, challenge their assumptions, critique papers, design a privacy-preserving evaluation, and answer all Q1–Q22. Tutor protocol: ask one question, wait, request a counterexample, then reveal the answer and probe the next question. Speak “beta” as the preference strength/regularization coefficient, “sigma” as logistic sigmoid, and “pi” as a stochastic policy.

## Prerequisites and glossary
Know conditional language-model probabilities, token log likelihood, SFT, cross-entropy, LoRA/QLoRA, train/test separation, class prevalence, ROC and precision/recall. **Prompt x**: user input. **Response y**: candidate completion. **Policy πθ(y|x)**: probability assigned by trainable model θ to a complete response, product of its conditional token probabilities. **Reference πref**: frozen baseline policy, often SFT model. **Preference pair (y+,y−)**: two responses with a judged winner and loser for the same prompt. **Reward model rφ(x,y)**: learned scalar proxy for preferences, not objective truth. **KL divergence**: Eπ[log π/πref], penalizing departure from a reference. **DPO**: direct preference optimization using paired responses and a fixed reference. **Extraction**: copying aspects of a target model's functionality via queries. **Membership inference (MI)**: deciding whether a specific record belonged to a target's training set. **Attribute inference**: estimating an unknown sensitive attribute from permitted observations. **Shadow model**: surrogate trained to study attack/defense behavior. **Query budget**: allowed number of interactions with an interface.

# Track G — Preferences, alignment and DPO

## G1. Why demonstrations are not enough
SFT fits a target response y by maximizing log πθ(y|x). It teaches imitation of selected demonstrations, but does not directly encode the *relative* preference between two plausible responses. For a cybersecurity assistant, two responses may be grammatically valid: A says “isolate this host now” from one suspicious DNS event; B says “verify the host, process lineage and change window before containment.” A preference rubric can favor B for caution and evidence quality without claiming B is always correct. Preference is conditional on prompt, rubric, annotator, and deployment consequences. Annotations can disagree; majority preference is not moral truth or factual correctness.

The common RLHF pipeline (Ouyang et al., 2022) is:
~~~text
permissioned prompts -> curated demonstrations -> SFT policy π_SFT
                                             |
same prompt -> sampled response A/B -> human ranking -> reward model rφ
                                             |
π_SFT (reference) + reward model -> KL-regularized RL/PPO -> candidate
                                             |
                              independent held-out evaluation and safety gate
~~~
PPO is a reinforcement-learning optimizer; it is not a synonym for RLHF. Reward-model optimization may exploit reward-model weaknesses (reward hacking). A reward trained on one annotator population may misrepresent another. Hold out prompts, incident groups and annotation sources, not merely randomly selected rows.

## G2. Derive the preference model
The Bradley–Terry model posits P(y+ preferred to y− |x)=σ(r(x,y+)−r(x,y−)), where σ(t)=1/(1+exp(−t)). The negative log likelihood for one pair is L_BT=−log σ(Δr). Differentiating: dL_BT/d(Δr)=σ(Δr)−1. At Δr=0 the loss is ln 2≈0.6931, and the derivative is −0.5: the objective pushes the winner's reward above the loser's. With Δr=ln 3, preference probability=0.75 and loss≈0.28768. These are modeling assumptions, not calibrated probabilities of factual truth.

The KL-regularized idealized policy objective for a fixed x is
J(π)=E_{y~π}[r(x,y)]−β KL(π(.|x)||πref(.|x)).
Here β>0 controls the cost of departing from πref. Enforce Σ_y π(y|x)=1 with Lagrange multiplier λ. Differentiating each π(y):
r(y)−β(log(π(y)/πref(y))+1)+λ=0,
so π*(y|x)=πref(y|x) exp(r(x,y)/β)/Z(x), where Z=Σ_y πref(y|x)exp(r/β).
Rearrange: r(x,y)=β log[π*(y|x)/πref(y|x)]+β log Z(x). In a same-prompt pair the log Z term cancels. Substitute into Bradley–Terry and parameterize π* by πθ:
L_DPO=−log σ(β[(log πθ(y+|x)−log πref(y+|x))−(log πθ(y−|x)−log πref(y−|x))]).
This is the central DPO loss (Rafailov et al., 2023). The derivation assumes an appropriate KL-regularized reward formulation and preference model; practical finite models, imperfect labels, response length and optimization dynamics break idealizations.

Define h=β[(logπθ+−logπref+)−(logπθ−−logπref−)]. Then dL/dh=σ(h)−1. Consequently ∂L/∂logπθ+=β(σ(h)−1), ∂L/∂logπθ−=β(1−σ(h)). Gradient descent increases the relative winner likelihood and decreases the loser likelihood, subject to shared parameters and normalization. It does **not** guarantee the winner's absolute probability increases on every optimizer step.

**Worked numerical example:** πθ+=0.6, πθ−=0.4, πref+=0.5, πref−=0.5, β=1. Then h=ln(1.5)≈0.4055, σ(h)=0.6 and L≈0.5108. If πθ and πref are identical, h=0 and L=ln2. If β→0, the loss approaches ln2 and its direct gradients shrink for fixed log ratios; this is not a universal monotonic quality statement.

## G3. Mechanisms, implementation traps and trade-offs
~~~text
verified preference pairs (x,y+,y−) -> tokenizer and chat template
                    |                              |
             reference logprobs              policy logprobs
                    +----------- log-ratio difference --------+
                                           |
                                   sigmoid / DPO loss
                                           |
                                update only policy (or LoRA)
                                           |
                         test: correctness, refusal, drift, cost
~~~
For autoregressive models, log π(y|x)=Σ_t log p(y_t|x,y_<t), with only response tokens included, not prompt padding. Keep winner/loser under the **same prompt** and consistent tokenizer/template. Mask padding and exclude truncated or unscorable pairs. Length effects matter: a longer answer sums more negative log probabilities; compare length-matched subsets and report response length distributions. Never compute reference logprobs from a moving reference unless deliberately studying a different objective. A small DPO loss is not evidence of robustness, privacy, or correct citations. DPO avoids a separately trained reward model and on-policy RL loop in its basic formulation; it still needs preference data, reference inference, tuning and independent evaluation.

**Counterexamples:** (1) Annotators favor confident fabricated citations, so preference learning can amplify hallucinations. (2) An always-refuse response may score well under a narrow safety rubric but fail legitimate tasks. (3) Two annotators disagree because one optimizes brevity and another evidence completeness; forcing one winner erases meaningful uncertainty. (4) Quantized adapter fine-tuning from yesterday can save memory but changes logprobs and may affect rare safety decisions; compare BF16+LoRA and QLoRA+DPO on identical held-out cases.

## G4. Runnable lab — DPO loss and finite-difference check (Python 3.10+, standard library)
Save as dpo_lab.py; no network, models or GPU required.
~~~python
from math import exp, log, isclose
def sigmoid(z):
    return 1/(1+exp(-z))
def dpo_loss(lp_w, lp_l, lr_w, lr_l, beta):
    h=beta*((lp_w-lr_w)-(lp_l-lr_l))
    return -log(sigmoid(h))
lpw,lpl=log(.6),log(.4)
lrw,lrl=log(.5),log(.5)
loss=dpo_loss(lpw,lpl,lrw,lrl,1.)
assert isclose(loss,-log(.6),abs_tol=1e-12)
h=(lpw-lrw)-(lpl-lrl)
grad_w=sigmoid(h)-1
grad_l=1-sigmoid(h)
eps=1e-6
fd_w=(dpo_loss(lpw+eps,lpl,lrw,lrl,1.)-dpo_loss(lpw-eps,lpl,lrw,lrl,1.))/(2*eps)
fd_l=(dpo_loss(lpw,lpl+eps,lrw,lrl,1.)-dpo_loss(lpw,lpl-eps,lrw,lrl,1.))/(2*eps)
assert isclose(fd_w,grad_w,abs_tol=1e-8)
assert isclose(fd_l,grad_l,abs_tol=1e-8)
assert isclose(dpo_loss(lrw,lrl,lrw,lrl,1.),log(2),abs_tol=1e-12)
print("loss",round(loss,6))
print("winner gradient",round(grad_w,6),"loser gradient",round(grad_l,6))
print("finite differences PASS")
~~~
Expected: loss 0.510826; winner gradient −0.4; loser gradient 0.4; finite differences PASS. **Extensions:** vary β, simulate 10% reversed labels, compare pairwise preference accuracy and held-out factual correctness, and implement token masking with a tiny locally available language model. Never interpret this two-number calculation as a trained language model.

## G5. Critical reading and research question
Ouyang et al. (2022), *Training language models to follow instructions with human feedback*, arXiv:2203.02155, https://arxiv.org/abs/2203.02155: demonstrates a demonstration→ranking→reward→RLHF pipeline and human preference results; limitations include annotator dependence and distribution shift. Rafailov et al. (2023; revised 2024), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290, https://arxiv.org/abs/2305.18290: derives the classification-style loss and evaluates it on selected generation tasks; this does not prove universal alignment or immunity to label bias. **Research design:** on authorized synthetic SOC answers, compare SFT, DPO, and a simple ranking baseline with equal compute and fixed response budgets. Pre-register rubric, annotator disagreement, grounded-citation accuracy, false refusal, unsafe tool recommendation, held-out incident families, and calibration. A preference gain with worse factual support is a meaningful negative finding.

# Track C — Model extraction, membership and attribute inference

## C1. Three distinct attacker questions
**Extraction:** “Can I approximate your model's decision function?” **Membership inference:** “Was this particular row used in training?” **Attribute inference:** “Can I estimate an unobserved property about an individual or record?” They require different target labels, access, success measures and ethical safeguards. Exposure of confidence scores, repeated queries, memorization and training overfitting can change risk, but none guarantees attack success. An API returning labels only may still leak through many adaptive queries; rate limiting reduces opportunities but is not proof of security.

~~~text
authorized test harness -> target API (labels / scores / text)
          |                         |
     strict query ledger       observable outputs
          |                         |
    surrogate fitting        audit statistics
          |                         |
   extraction fidelity   MI member/nonmember AUC; attribute baseline
          +-------------------+
                   |
           privacy and utility report
~~~
Never run probing against an unconsenting third-party service. Use offline synthetic or owned models and data.

## C2. Extraction formalization and counterexamples
Given target f and surrogate g, extraction can minimize empirical disagreement: L_extract=(1/n)Σ_i 1[g(x_i)≠f(x_i)] over queried x_i. **Fidelity** on a separately sampled test distribution D is P_{x~D}(g(x)=f(x)); **task accuracy** is P(g(x)=y_true). High fidelity can copy a flawed model and need not imply high task accuracy. If queries only cover common benign network flows, a surrogate may appear faithful yet fail on rare ransomware-like flows. Query selection, response richness, architecture assumptions and rate limits all affect efficiency. For a logistic target f(x)=σ(wᵀx+b), exact real-valued scores would allow logit(p)=ln[p/(1−p)]=wᵀx+b; sufficiently independent queries could identify parameters in an ideal noiseless setting. Real APIs may round, clip, randomize or return labels only. This is a conceptual risk derivation, not an instruction to probe a real system.

## C3. Membership inference, Bayes and generalization
Let M=1 mean a record is in training, M=0 otherwise. An MI auditor computes score s(x,y), perhaps negative loss or confidence on the correct label, then tests thresholds. For threshold τ, TPR=P(s≥τ|M=1) and FPR=P(s≥τ|M=0). If membership prior is π=P(M=1), then posterior precision P(M=1|s≥τ)=πTPR/[πTPR+(1−π)FPR]. At π=.01, TPR=.8, FPR=.1, precision=.008/(.008+.099)=0.074766, just 7.48%; a high true-positive rate alone does not make positive claims trustworthy. The prior depends on the sampling scenario and must be stated.

Overfitting may increase the train/nontrain loss gap and leakage, but membership can be detectable without a large average gap, and a small gap does not prove privacy. Data duplicates, label noise, query adaptation, distribution mismatch and record rarity confound evaluation. Split by user/source/time before training; hold out an independent population; include attack confidence intervals. ROC-AUC=0.5 is chance for balanced pairwise ranking, not a guarantee of no individual leakage. Attack advantage at a chosen threshold is TPR−FPR; report the operating point.

**Attribute inference** differs: infer a hidden attribute a from public features z and model observations o. Compare P(a|z,o) against baseline P(a|z) to measure incremental leakage; a high raw accuracy could come from public features alone. Protect sensitive demographics, and avoid inferring real people's traits without consent. In a SOC setting, device or user properties can be confidential even if an incident label is public.

## C4. Runnable defensive audit — membership metrics (standard Python)
Save as privacy_audit_lab.py. Scores and labels are fabricated; no attack against a live model.
~~~python
from math import isclose
member_scores=[.95,.85,.75,.65,.55]
nonmember_scores=[.90,.60,.40,.30,.20]
tau=.70
tp=sum(s>=tau for s in member_scores)
fn=len(member_scores)-tp
fp=sum(s>=tau for s in nonmember_scores)
tn=len(nonmember_scores)-fp
tpr=tp/(tp+fn);fpr=fp/(fp+tn)
assert (tp,fn,fp,tn)==(3,2,1,4)
assert isclose(tpr,.6) and isclose(fpr,.2)
prior=.01
ppv=prior*tpr/(prior*tpr+(1-prior)*fpr)
assert isclose(ppv,.029411764705882353)
# Pairwise ranking AUC, including half credit for ties.
pairs=[(a,b) for a in member_scores for b in nonmember_scores]
auc=sum(1 if a>b else .5 if a==b else 0 for a,b in pairs)/len(pairs)
assert isclose(auc,.8)
print("TP FN FP TN",tp,fn,fp,tn)
print("TPR FPR",round(tpr,2),round(fpr,2))
print("PPV at 1% prior",round(ppv,6))
print("AUC",round(auc,2))
~~~
Expected: TP FN FP TN 3 2 1 4; TPR FPR 0.6 0.2; PPV at 1% prior 0.029412; AUC 0.8. **Important:** the 1% prior is hypothetical and differs from the deliberately balanced 5/5 toy evaluation sample. Reweight using Bayes rather than reporting toy-sample precision as deployment precision. **Extension:** vary threshold; compare attack AUC and TPR at 1% FPR; group-bootstrap confidence intervals; test an independently sampled nonmember cohort and a public-feature-only attribute baseline.

## C5. Mitigations, ethics and critique
Train-data minimization, deduplication, regularization, restricted score disclosure, query auditing, throttling and privacy-preserving training can reduce some risks. None is a universal defense. Differential privacy, when correctly implemented with a specified (ε,δ) and accounting, provides a formal bound on how much a single record can affect the distribution of outputs under a neighboring-dataset definition; it is not a blanket guarantee against extraction or all attribute inference. Utility/privacy trade-offs must be measured. A model may leak through retrieval or logs even if training is private; secure the entire pipeline. Avoid exposing raw per-user audit scores.

Shokri et al. (2017), *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P, DOI:10.1109/SP.2017.41, https://doi.org/10.1109/SP.2017.41: studies membership inference and shadow-model attacks; critique its assumed interface and training distributions against modern language models. Tramèr et al. (2016), *Stealing Machine Learning Models via Prediction APIs*, USENIX Security, https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/tramer: studies extraction under prediction API access; distinguish fidelity from theft of proprietary training data. Carlini et al. (2021), *Extracting Training Data from Large Language Models*, USENIX Security, https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting: demonstrates memorized-sequence exposure under studied settings; not a claim that every prompt yields training records. **Research question:** compare calibrated MI risk at low FPR for BF16+LoRA, QLoRA and DPO-adapted SOC assistants, holding training data and access interface fixed. Control record rarity and duplication; preregister a privacy-utility Pareto frontier. Never use private production logs without authorization.

# Cross-track synthesis
Preference-tuned assistants and privacy attacks meet at the **output interface**. A DPO policy optimized for detailed answers may reveal more sensitive facts if its preference labels reward specificity without privacy review. This is a hypothesis, not an established causal effect. Pair alignment evaluation with privacy canaries constructed from synthetic data, held-out member/nonmember tests, output logging and permission-aware refusal tests. Human reviewers must approve any operational SOC action; a preferred response is not automatically authorized.

# Progressive oral Socratic examination (question → answer → follow-up)
1. **What does SFT optimize?** Demonstration log likelihood. *Probe:* what relative judgment is missing?
2. **What is a preference pair?** Two candidate responses for the same prompt, with a chosen winner. *Probe:* why keep prompts identical?
3. **What is the Bradley–Terry probability?** σ(r+−r−). *Probe:* probability at equal rewards? 0.5.
4. **What is L_BT at equal rewards?** ln2. *Probe:* derivative? −0.5 wrt reward difference.
5. **What does a reward model learn?** Proxy for labeled preferences. *Probe:* why not ground truth?
6. **Name RLHF stages.** Demonstrations/SFT, ranking/reward, policy optimization. *Probe:* where can reward hacking occur?
7. **What does KL constrain?** Deviation from a reference policy. *Probe:* why can too much constraint hurt adaptation?
8. **Derive optimal π*.** πref exp(r/β)/Z. *Probe:* why must Z normalize?
9. **Why does Z cancel in DPO?** Same-prompt reward differences. *Probe:* what if prompts differ?
10. **State DPO logit.** β times winner-minus-loser policy/reference log-ratio. *Probe:* explain β.
11. **DPO loss at unchanged reference?** ln2. *Probe:* does that imply poor factuality? No.
12. **Winner gradient sign at h=ln1.5?** −0.4. *Probe:* loser sign? +0.4.
13. **Why mask prompt tokens?** Only response likelihood is being compared. *Probe:* what about padding?
14. **What can preference optimization worsen?** Factuality, minority preferences, overrefusal. *Probe:* name an independent metric.
15. **What is extraction?** Approximate target behavior via observations. *Probe:* fidelity versus accuracy?
16. **What is membership inference?** Infer train-set inclusion of a record. *Probe:* how is it different from memorized text extraction?
17. **What is attribute inference?** Infer hidden property. *Probe:* compare to public-feature baseline.
18. **Why is query access a threat surface?** Outputs convey information. *Probe:* what if only labels are returned?
19. **Calculate MI PPV at π=.01, TPR=.8, FPR=.1.** About 7.48%. *Probe:* why so low?
20. **Does AUC=.5 prove privacy?** No; low-FPR/rare-record leakage can persist. *Probe:* what else report?
21. **Does throttling guarantee protection?** No; reduces some attack budgets. *Probe:* what about side channels?
22. **What would make a dissertation-grade study?** Preregistered privacy-utility experiment with controlled access, held-out groups, low-FPR metrics and ethics review. *Probe:* which negative results matter?

## Recall, misconceptions and failure checks
Without notes: derive DPO from a KL-constrained policy; calculate loss and signs of gradients; distinguish reward preference from correctness; explain extraction fidelity versus task accuracy; derive MI precision with a realistic prior; name three privacy controls and their limitations. Misconceptions to reject: “DPO is just SFT”; “RLHF means PPO”; “high preference score proves truth”; “membership AUC near chance proves no privacy risk”; “an API's labels contain no information”; “a signed model or quantized model cannot leak.” Common failures: prompt/response masking errors, shifting reference models, pair-label bias, test leakage, mismatched member/nonmember populations, overconfident claims from tiny samples, and unconsented probing.

## Research protocol to produce after the walk
Draft one page: objective, attacker knowledge and query budget, datasets and licenses, membership sampling prior, target/surrogate versions, prompt templates, DPO β, seed count, held-out incident groups, pre-registered low-FPR privacy endpoint, preference/factuality endpoint, cluster bootstrap, acceptable privacy-utility margin, ethics review, and a reproducible artifact manifest. A null result must report power and uncertainty.

## Source map
- **Syllabus/continuity:** original Gmail PDFs 19 October 2026, GenAI Unit 3 Session 5 and Cyber Unit 3 Session 4; 13 November morning and evening GitHub lessons.
- **G1–G3:** local Bradley–Terry and KL derivation; Ouyang 2022 (RLHF pipeline); Rafailov 2023 (DPO).
- **G4:** local executable synthetic finite-difference validation, not an LLM training result.
- **C1–C3:** local threat-model definitions and Bayes derivation; Tramèr 2016 (extraction); Shokri 2017 (membership).
- **C4:** local deterministic synthetic privacy audit, not a real-model leak estimate.
- **C5/cross-track:** Carlini 2021 (memorization), local proposed experiment; speculative causal questions explicitly labeled.
- **Paper metadata, access notes and local PDF inventory:** [companion references index](references/index.md). No email, audio or narration is generated.
