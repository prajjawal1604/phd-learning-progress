# Morning PhD teaching package — 13 November 2026 (Asia/Kolkata)

**Continuity.** The committed 12 November morning and evening lessons on `main` covered LoRA factorization, optimization and physically constrained adversarial evaluation. The two original PDFs attached to the 19 October 2026 email, *Advanced Generative AI and Foundation Models Learning Pathway* (Unit 3 Session 4) and *AI for Cybersecurity Learning Pathway* (Unit 3 Session 3), prescribe today's next topics: **quantization and efficient adaptation** and **poisoning, backdoors and model supply-chain security**. This is an accelerated but syllabus-aligned progression. All numerical examples and labs are synthetic; no claims about live systems.

## Voice-tutor contract and routes
**10 minutes:** ask the learner to explain LoRA versus QLoRA, calculate 4-bit storage and one quantization error, then distinguish poisoning from evasion and backdoors; Q1–Q5. **30 minutes:** derive quantization, trace gradients through a frozen quantized backbone, examine a training-data integrity failure and a paired-trigger audit; Q1–Q12. **Extended 90–150 minutes:** run both labs, discuss experimental design and paper critiques, answer Q1–Q20, write one research hypothesis. Ask one question, wait for an answer, probe the learner's assumptions, then reveal the model answer. Spoken notation: “w hat” means reconstructed weight; “delta” means reconstruction error.

## Prerequisites and definitions
From yesterday: `W_eff=W0+(alpha/r)BA` with frozen `W0`, trainable low-rank factors `A,B`; attacker goal, control surface, feasibility and robust-risk evaluation. **Quantization:** represent continuous or high-precision values by a finite codebook. **Bit width b:** bits per stored code. **Scale s:** conversion between integer code and reconstructed real value. **Zero point z:** integer code representing real zero. **Quantization error:** reconstructed minus original weight. **Dequantization:** reconstruct approximate values for arithmetic. **Calibration:** choose ranges or scales using permitted data. **Outlier:** value unusually far from typical distribution, possibly dominating a scale. **QLoRA:** freeze a low-bit stored base and train floating-point low-rank adapters through its dequantized computation. **Poisoning:** adversarial manipulation of training inputs or labels. **Backdoor:** learned conditional behavior activated by an attacker-selected trigger, often while clean performance remains acceptable. **Supply-chain compromise:** unauthorized modification or substitution of data, model, dependency or update artifacts. **Provenance:** verifiable origin and transformation history.

# Track G — Quantization and efficient adaptation

## G1. Intuition, mechanism and exact equations
Imagine transporting a detailed map as a small palette of colors: common values map to nearby representatives; fine distinctions disappear. Uniform affine quantization for real weight `w` uses integer code
`q=clip(round(w/s)+z,qmin,qmax)`, reconstructed `w_hat=s(q-z)`, with `s>0`, integer zero point `z` and integer bounds `qmin,qmax`. Round-to-nearest ties and clipping rules must be specified for reproducibility. Symmetric signed 4-bit uses codes -8 through 7; a common symmetric design instead uses -7 through 7 to balance magnitudes. **These are different conventions**. We use -7…7 in the lab, scale `s=max(abs(weights))/7`, zero point zero. Values in range have nearest-grid absolute error at most `s/2`; clipping and incorrect calibration invalidate that bound.

For a matrix `W` and vector `x`, the output error is `e=(W_hat-W)x=Ex`. By Cauchy–Schwarz, `||e||_2 <= ||E||_F ||x||_2` (Frobenius norm is a valid upper bound on spectral norm). This is an upper bound, not an empirical accuracy prediction; nonlinear networks can amplify or sometimes attenuate errors. **Counterexample:** weights [0.01,0.02,8] with one shared symmetric scale quantize the first two to zero, potentially erasing a rare but important feature. Per-channel scales or nonuniform codebooks may reduce this distortion, at extra metadata/kernel cost.

**Worked calculation:** weights [-1, -0.4, 0, 0.4, 1], scale `1/7`. Codes [-7,-3,0,3,7], reconstructions [-1,-3/7,0,3/7,1]. Each middle error magnitude is `|3/7-0.4|≈0.028571`. Squared reconstruction error summed across five values is `2(0.028571)^2≈0.00163265`. These are synthetic weights, not model-quality measurements.

## G2. Storage arithmetic and why 4-bit is not 4-bit training
For `N` parameters, ideal raw weight storage is `Nb/8` bytes. With `N=1,000,000,000`, 16-bit raw weights use 2.0 billion bytes (decimal GB); 4-bit raw codes use 0.5 billion bytes. But block scales, zero points, code packing, embeddings, temporary dequantized tiles, activations, LoRA factors, gradients, optimizer states, KV cache and runtime allocator overhead remain. For blocks of `B=64` weights with one fp16 scale, scale overhead is `16/B=0.25` bits per weight, so a simplified 4-bit scheme uses 4.25 bits/weight before padding and other metadata. Double quantization can reduce scale metadata; it does not magically remove activations.

QLoRA computation:
```text
trusted base revision -> quantize + block scales -> frozen packed 4-bit weights
                                                | dequantize for matmul
input x -------------------------------> W_hat x + (alpha/r) B A x
                                                          |
                                      loss -> gradients to A,B only
                                       |             |
                             base codes unchanged   optimizer updates A,B
```
For `z=W_hat x + sBAx` and upstream gradient `g=∂L/∂z`, `∂L/∂B=s g(Ax)^T`, `∂L/∂A=s(B^Tg)x^T`, as yesterday. `W_hat` is frozen but participates in `∂L/∂x=W_hat^Tg+sA^TB^Tg`. Quantization noise may change optimal adapter factors and convergence. In standard QLoRA the base is not trained by straight-through estimators: only adapter parameters are updated. NF4 is a nonuniform normal-distribution-informed 4-bit codebook (Dettmers et al., 2023), not the uniform toy scheme used here.

**Why it works:** large pretrained base retained approximately; adapters absorb some task-specific changes; memory saved in base storage and trainable state. **When it fails:** outliers, calibration mismatch, sensitive small weights, unsupported kernels, precision overflow, a narrow rank, sequence-length activation spikes, quality regressions, base/adapter mismatch. Quantization is not compression of the information in the dataset or a security mechanism. Compare BF16+LoRA and 4-bit+LoRA at equal data, rank, optimizer, steps, seeds and target modules; report held-out task accuracy, calibration, rare-class recall, peak allocated/reserved memory and wall time. A 4-bit implementation can be slower because of dequantization overhead.

## G3. Runnable Python lab: exact quantization and error bounds
Python 3.10+, NumPy (`python -m pip install numpy`). Save as `quantization_lab.py`.
```python
import numpy as np
w=np.array([-1.,-.4,0.,.4,1.])
scale=np.max(np.abs(w))/7
q=np.clip(np.rint(w/scale),-7,7).astype(np.int8)
recon=q.astype(float)*scale
err=recon-w
assert q.tolist()==[-7,-3,0,3,7]
assert np.all(np.abs(err)<=scale/2+1e-12)
assert np.isclose(np.sum(err**2),2*(3/7-.4)**2)
W=np.array([[1.,-.4],[.4,-1.]])
x=np.array([2.,-1.])
s=np.max(np.abs(W))/7
Wq=np.clip(np.rint(W/s),-7,7)*s
e=(Wq-W)@x
bound=np.linalg.norm(Wq-W,'fro')*np.linalg.norm(x)
assert np.linalg.norm(e)<=bound+1e-12
print("codes",q.tolist())
print("squared error",round(float(np.sum(err**2)),8))
print("output error",np.round(e,6).tolist(),"bound",round(float(bound),6))
print("ideal bytes for 1B parameters: fp16",2_000_000_000,"4bit",500_000_000)
```
Expected codes `[-7,-3,0,3,7]`; squared error approximately `0.00163265`; output error and bound are deterministic and satisfy the assertion. **Extensions:** implement block sizes 2,4,8; inject a single large outlier; compare per-row versus per-tensor quantization; plot MSE versus bit width and per-class error on a synthetic classifier. Do not call this NF4 or an LLM benchmark.

## G4. Paper critique and research proposal
Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314, https://arxiv.org/abs/2305.14314. Contributions: frozen 4-bit base, NF4, double quantization, paged optimizers; question transferability of chatbot evaluation, model-specific kernels, memory accounting and rare safety failures. Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, https://arxiv.org/abs/2106.09685: establishes factorization baseline; distinguish rank reduction from bit reduction. **Research hypothesis:** quantization calibration errors disproportionately affect rare cybersecurity indicators. Use synthetic and authorized incident data, chronological and host-grouped test sets, fixed rank and training tokens, and confidence intervals clustered by incident. Pre-register outlier frequency, false-negative cost, compute budget and failure thresholds.

# Track C — Training-data poisoning, backdoors and supply-chain security

## C1. Start at the trust boundary
Yesterday's evasion changed a test input after training. **Poisoning changes the training process**: labels, examples, sampling weights or update artifacts. A targeted backdoor attempts to make a trigger `T(x)` produce an attacker-desired label `y*` while retaining benign-looking performance on ordinary examples. Not every label error is malicious poisoning; intent, capability and provenance matter. An adversary who can edit one user query cannot necessarily change the training corpus.

```text
source records -> provenance/hash/rights -> quarantine + schema checks
      |                                     |
  contributor ACLs                      label review
      |                                     |
      +----------> versioned train-only dataset -> training job
                                                |
             signed base + adapter registry <- artifact validation
                                                |
                       frozen clean holdout + authorized trigger audit
                                                |
                               promotion gate -> SOC analyst review
```

**Threat model:** asset is integrity of a phishing classifier; adversary controls at most 5% of *training* records from one lower-trust ingestion source, but not the sealed holdout, model signing keys or production threshold. Goal: a particular synthetic marker causes malicious examples to be classified benign. Budget: at most 5% modified rows; attack success only on still-malicious triggered examples. The defender must compare both clean utility and trigger-conditioned performance, not merely aggregate accuracy. An indicator appearing in text may be ordinary evidence, not an attacker-controlled trigger; distinguish source provenance.

## C2. Mathematical objective and bias from poisoned data
Let clean distribution be `P`, poisoning distribution `Q`, fraction `epsilon∈[0,1]`. The observed training distribution is `P_epsilon=(1-epsilon)P+epsilon Q`. For model `theta`, empirical risk `L_epsilon(theta)=(1-epsilon) E_P[ell(theta;z)]+epsilon E_Q[ell(theta;z)]`. Differentiating, `∇L_epsilon=(1-epsilon)g_P+epsilon g_Q`: a small poisoned fraction can dominate if `g_Q` is large or aligned with a sensitive parameter direction. If clean optimum `theta0` satisfies `∇L_0(theta0)=0` and Hessian `H=∇²L_0(theta0)` is invertible, first-order implicit differentiation yields
`d theta*/d epsilon |0 = -H^{-1}(g_Q(theta0)-g_P(theta0))`.
Derivation: set `∇L_epsilon(theta*(epsilon))=0`; differentiate both sides and solve `H theta' + (g_Q-g_P)=0`. This is a *local smooth optimum approximation*, not a universal poisoning success formula for nonconvex deep nets. Regularization and data validation change `H` and feasible `Q`.

**Worked label-flip toy:** 100 training records, 50 positives and 50 negatives. Flip 5 positive labels to negative: the observed positive fraction falls from 0.50 to 0.45. A constant Bernoulli maximum-likelihood predictor moves from 0.50 to 0.45; the clean evaluation labels have **not** changed. This demonstrates training contamination but not a learned backdoor. To test a backdoor, use a separate trigger-conditioned holdout and preserve ground-truth labels.

## C3. Why clean accuracy can conceal a backdoor
Suppose a synthetic evaluation has 990 ordinary cases and 10 trigger-bearing malicious cases. A classifier correctly handles all 990 ordinary cases but misclassifies all 10 triggered cases: overall accuracy is 99%, yet triggered malicious recall is 0%. Reporting only 99% hides the conditional failure. **Backdoor attack success rate (ASR)** should declare the eligible source class and denominator: number of eligible triggered examples mapped to target class divided by number of eligible triggered examples. Also report clean accuracy/recall, trigger false alarms on benign cases, target-label baseline, and whether the trigger changes semantics. Do not mistake a naturally correlated benign feature for a confirmed malicious trigger.

**Supply-chain controls:** immutable raw-data snapshots, contributor identities, row-level provenance, train/validation/test isolation, review of label changes, signed model and adapter artifacts, hashes bound to base/tokenizer versions, dependency lockfiles and least-privilege CI/CD, independent promotion gates and rollback. A hash verifies identity relative to a trusted digest; it does not prove the underlying model is safe. Dataset filtering may exclude rare legitimate cases, creating fairness and operational harm. Protect personal data and never deploy untested trigger experiments.

## C4. Runnable Python lab: label contamination and conditional audit
Python 3.10+ standard library. This does not construct an operational backdoor or touch a live model.
```python
from statistics import mean
clean_labels=[1]*50+[0]*50
training_labels=clean_labels.copy()
for i in range(5): training_labels[i]=0
assert sum(clean_labels)==50 and sum(training_labels)==45
print("clean prevalence",mean(clean_labels),"contaminated",mean(training_labels))
# Paired, fabricated evaluation: all ordinary examples correct,
# ten malicious trigger-bearing cases missed by a hypothetical detector.
ordinary_truth=[0]*900+[1]*90
ordinary_pred=ordinary_truth.copy()
trigger_truth=[1]*10
trigger_pred=[0]*10
accuracy=sum(a==b for a,b in zip(ordinary_truth+trigger_truth,
                                ordinary_pred+trigger_pred))/1000
trigger_recall=sum(p==1 for p in trigger_pred)/len(trigger_pred)
assert accuracy==.99 and trigger_recall==0.
print("overall accuracy",accuracy,"triggered malicious recall",trigger_recall)
# A signed digest protects identity, not model behavior.
import hashlib
artifact=b"synthetic-model-v1"
digest=hashlib.sha256(artifact).hexdigest()
assert hashlib.sha256(artifact).hexdigest()==digest
assert hashlib.sha256(artifact+b"tampered").hexdigest()!=digest
print("digest identity check PASS")
```
Expected: `clean prevalence 0.5 contaminated 0.45`; `overall accuracy 0.99 triggered malicious recall 0.0`; `digest identity check PASS`. **Advanced authorized extension:** train a regularized logistic model on synthetic data with 0%, 1%, 5% label flips (train only), evaluate unchanged grouped test records, repeat across seeds, and bootstrap at source-group level. Separately design a *defensive* conditional-slice audit for a hypothetical marker without teaching deployment of a malicious trigger.

## C5. Critique and real-world interpretation
Gu, Dolan-Gavitt & Garg (2017/2019), *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain*, arXiv:1708.06733, https://arxiv.org/abs/1708.06733: demonstrates trigger-dependent failures and outsourced-model trust concerns; challenge transfer from image triggers to temporal SOC telemetry and the representativeness of trigger choice. Vassilev et al. (2025), *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025: taxonomy of poisoning, evasion and mitigations; taxonomy alone does not measure a defense's effectiveness. **Research question:** which provenance signals best predict poisoning-related degradation under realistic partial contributor control? Pre-register source groups, poisoned fractions, label audit budgets, rare-class recalls, uncertainty and cost; ensure experiments are offline and authorized.

# Cross-track synthesis: quantized adapters are security artifacts
A QLoRA adapter can be small enough to share casually, but its base revision, tokenizer, quantization configuration, training provenance and cryptographic digest are required for trustworthy reuse. A low-bit model can still be poisoned; compression is not validation. For a SOC assistant, use signed artifact manifests, read-only training snapshots, isolated evaluation sets, human approval for containment and logged rollback. Distinguish **model integrity**, **input validity**, **training-data integrity**, **factual grounding** and **operational safety**: no single metric covers all five.

# Progressive oral Socratic Q&A (ask, pause, reveal, challenge)
1. **What does quantization discard?** Fine distinctions between real values. *Follow-up:* is it lossless? Generally not.
2. **Define scale and zero point.** Parameters mapping codes to real values. *Follow-up:* why is calibration needed?
3. **What is a 4-bit raw code budget?** Half a byte per parameter. *Follow-up:* what metadata is omitted?
4. **Why can an outlier be harmful?** It enlarges a shared scale, collapsing small weights. *Follow-up:* name per-channel scaling.
5. **Calculate 1B fp16 versus 4-bit raw bytes.** 2.0B versus 0.5B. *Follow-up:* is total GPU use 4× lower? Not necessarily.
6. **What is QLoRA's frozen component?** Quantized base weights. *Follow-up:* which tensors learn? LoRA A and B.
7. **How do gradients reach A if W is frozen?** Through differentiable dequantized matmul and adapter path; W need not update. *Follow-up:* give derivative wrt A.
8. **Why is NF4 not the lab's uniform quantizer?** NF4 uses a nonuniform normal-informed codebook. *Follow-up:* how would you test both?
9. **Derive an output-error bound.** `||Ex||≤||E||F||x||`. *Follow-up:* is it a quality guarantee? No.
10. **Could 4-bit run slower?** Yes, dequantization and kernel overhead. *Follow-up:* what timings should be measured?
11. **Evasion versus poisoning?** Inference-time manipulation versus training-time influence. *Follow-up:* who controls which surface?
12. **Define a backdoor.** Trigger-dependent attacker-desired behavior. *Follow-up:* why may clean accuracy remain high?
13. **Explain the contamination mixture.** `(1-epsilon)P+epsilon Q`. *Follow-up:* how does the gradient change?
14. **Why can small epsilon matter?** Poisoned gradient direction/magnitude and sensitivity. *Follow-up:* what does Hessian inverse capture?
15. **What does label flipping demonstrate?** Training contamination, not necessarily a backdoor. *Follow-up:* what extra trigger audit is required?
16. **99% accuracy and 0% triggered recall: contradiction?** No, rare conditional failures hide in aggregate metrics. *Follow-up:* choose denominators.
17. **Does SHA-256 prove model safety?** No, only integrity against a trusted expected digest. *Follow-up:* what verifies origin?
18. **Name three supply-chain gates.** Provenance, signed artifact checks, independent holdout/promotion. *Follow-up:* what about dependencies?
19. **How can provenance controls cause harm?** Overfilter rare legitimate populations and create blind spots. *Follow-up:* audit subgroup recall.
20. **Propose a doctoral study.** Compare quantization precision and training-data contamination under fixed compute and sealed grouped test. *Follow-up:* preregister metrics, attack budget, and uncertainty.

## Recall checkpoint
Without notes, derive the affine quantizer, explain QLoRA gradients, compute storage including one scale per 64 weights, distinguish a poisoned label from a backdoor trigger, and propose a conditional-slice audit. If you cannot defend the validity of the test split and source provenance, do not claim robustness.

## Source map and integrity notes
- Syllabus scope and progression: original 19 Oct Gmail attachments, GenAI Unit 3 Session 4; Cyber Unit 3 Session 3; prior 12 Nov morning/evening lessons on `main`.
- Quantizer equations, norm bound, storage examples: derived in G1–G3; reproducible toy code in G3. QLoRA architecture and NF4: Dettmers et al. (2023); LoRA factors: Hu et al. (2022).
- Poisoning mixture, local sensitivity and toy prevalence: derivations C2 and code C4; attack taxonomy: NIST (2025); trigger-conditioned supply-chain case: Gu et al.
- These are explanatory derivations and toy examples, not measured production outcomes. Bibliographic summaries and download/access notes: [companion index](references/index.md). No audio, email or narration is generated.
