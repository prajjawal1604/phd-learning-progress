# Evening PhD deep dive — 13 November 2026 (Asia/Kolkata)
**Scope lock:** This lesson deepens the committed [morning lesson](../../../../phd-morning/lessons/2026-11-13/lesson.md) and [morning references](../../../../phd-morning/lessons/2026-11-13/references/index.md). GenAI: quantization, QLoRA and efficient adaptation. Cybersecurity: poisoning, backdoors and ML supply-chain integrity. Both 19 October 2026 private syllabus PDFs were read: *Advanced Generative AI and Foundation Models Learning Pathway*, Unit 3 Session 4; *AI for Cybersecurity Learning Pathway*, Unit 3 Session 3. No new syllabus unit. All data below are synthetic, offline and safe.

## Interactive walking route
**10 minutes:** Say what NF4 changes compared with uniform int4; derive one output-error bound; explain why an authentic signed poisoned model can still be unsafe; ask Q1–Q6.
**30 minutes:** Explain G1–G4 and C1–C4 aloud, calculate the 1B-weight budget and poison sensitivity, then Q1–Q14.
**Extended 90–150 minutes:** Execute both labs, critique four papers, design a preregistered experiment, finish Q1–Q22. Tutor protocol: ask one question; wait; demand an assumption or counterexample; then disclose the answer. For voice, pronounce ||E||_F as “Frobenius norm of E”.

## Prerequisites and glossary
Assume basic linear algebra, cross-entropy, SGD, Hessians, LoRA W_eff=W0+(alpha/r)BA, attack surfaces, class imbalance, train/test isolation. Define **codebook** as set of representable weight values; **granularity** as scope sharing a scale (tensor, row, block); **clipping** as saturation beyond code range; **quantization-aware error** as the effect of reconstructed weights on outputs, not just weight MSE; **adapter** as a trained low-rank update to a frozen base; **poison budget** as a bound on attacker-controlled training records; **backdoor** as trigger-conditioned malicious model behavior; **provenance** as authenticated lineage; **artifact integrity** as byte identity relative to a trusted digest. Authentication is not safety validation.

# G — Quantization and QLoRA, advanced

## G1. Mechanism and nontrivial failure
Morning's symmetric toy int4 used q=clip(round(w/s),-7,7), reconstructed w_hat=sq. A block of B weights shares a scale; one extreme value can expand s and zero out many small values. A nonuniform codebook c_0,...,c_15 instead maps a normalized weight to the closest representative; NF4 selects representatives informed by a normal distribution, as studied by Dettmers et al. This is *not* the same as standard float4 or our uniform toy. Calibration distribution, block length, rounding, packing order and kernel accumulation precision matter. Blockwise quantization reduces outlier influence across blocks but introduces scale metadata and may impede fused kernels.

For a matrix W and input x, let E=W_hat-W. Then y_hat-y=Ex. Since ||Ex||_2 <= ||E||_2||x||_2 <= ||E||_F||x||_2, a small Frobenius error controls a single linear layer's worst-case Euclidean output. The bound can be loose: if x lies in E's null space, error is exactly zero even when ||E||_F>0. Conversely, a small output change can flip an argmax if two class logits nearly tie. **Worked example:** clean logits (0.501,0.499) predict class 0; errors (-0.003,+0.003) change them to (0.498,0.502), predicting class 1. Thus weight-MSE-only evaluation can miss rare safety-critical decisions.

For L layers f_L∘...∘f_1, with each clean layer Lipschitz K_i and local perturbation bound δ_i at its input, output deviation is at most sum over i of δ_i times product of downstream K_j (j>i), under compatible domains. Large downstream gains amplify early perturbations; normalization, residuals and nonlinearities complicate tight estimates. **Counterexample:** a low-norm quantization error aligned with a decision boundary can cause more classification failures than a larger error in irrelevant directions.

## G2. Storage and gradients, derived
Raw N weights at b bits require Nb/8 bytes; for N=10^9, 4-bit=0.5 decimal GB versus fp16=2.0 GB. If each block of B=64 weights stores one 16-bit scale, metadata adds 16/64=0.25 bits/weight, giving 4.25 bits/weight or 531.25 MB per billion weights, ignoring padding, codebooks and extra metadata. Double quantization compresses scale metadata; it does not compress gradients or activations to four bits. A 4096×4096 linear layer with rank r=16 adds r(4096+4096)=131072 trainable LoRA parameters versus 16777216 dense weights: 0.78125% trainable relative to that matrix. Its adapter fp16 weights alone take 262144 bytes, excluding gradients/optimizer.

Write z=W_hat x+κBAx, κ=alpha/r, g=∂L/∂z. Chain rule: ∂L/∂B=κg(Ax)^T, ∂L/∂A=κ(B^Tg)x^T and ∂L/∂x=W_hat^Tg+κA^TB^Tg. W_hat stays frozen; no gradient is needed for the packed codes. At B=0, ∂L/∂A=0 on the initial step but ∂L/∂B can be nonzero if A is initialized nonzero; later B becomes nonzero and A begins learning. Initializing both factors to zero would stall both gradients. A 4-bit stored base does not imply 4-bit accumulation or 4-bit optimization.

**Adapter merge trap:** W_merge=W_hat+κBA is a floating-point effective matrix; re-quantizing it can introduce a *second* error. If the runtime instead keeps packed W_hat and a separate adapter, arithmetic, latency and reproducibility differ. Base revision, tokenizer, rank, scaling, target modules, quantizer and kernel version belong in the deployment manifest.

## G3. Advanced lab — output error versus decision error (Python 3.10+, NumPy)
```python
import numpy as np
W=np.array([[.501,.001],[.499,-.001]],dtype=float)
x=np.array([1.,0.])
def q4(M):
    s=np.max(np.abs(M))/7
    return np.rint(np.clip(M/s,-7,7))*s
Wq=q4(W)
E=Wq-W
err=np.linalg.norm(E@x)
bound=np.linalg.norm(E,'fro')*np.linalg.norm(x)
assert err<=bound+1e-12
# An independently constructed near-tie counterexample
clean=np.array([.501,.499]); perturbed=clean+np.array([-.003,.003])
assert np.argmax(clean)==0 and np.argmax(perturbed)==1
print("quantized",np.round(Wq,6).tolist())
print("error <= bound",bool(err<=bound+1e-12))
print("near-tie flipped",bool(np.argmax(clean)!=np.argmax(perturbed)))
# Finite-difference check of LoRA gradients
A=np.array([[.3,.7]]); B=np.array([[.4],[-.2]])
v=np.array([1.,-2.]); g=np.array([.5,-1.]); k=.5
def objective(a,b): return float(g@(k*(b@a)@v))
dA=k*np.outer(B.T@g,v); dB=k*np.outer(g,A@v)
h=1e-6
for j in range(2):
    p=A.copy();m=A.copy();p[0,j]+=h;m[0,j]-=h
    assert np.isclose((objective(p,B)-objective(m,B))/(2*h),dA[0,j],atol=1e-8)
for i in range(2):
    p=B.copy();m=B.copy();p[i,0]+=h;m[i,0]-=h
    assert np.isclose((objective(A,p)-objective(A,m))/(2*h),dB[i,0],atol=1e-8)
print("LoRA finite-difference gradients PASS")
```
Expected final lines: `error <= bound True`, `near-tie flipped True`, `LoRA finite-difference gradients PASS`. The quantized matrix is deterministic but not claimed to produce the separate near-tie example. **Advanced extension:** compare per-row vs global scales on synthetic imbalanced feature matrices; measure worst-subgroup recall, margin flips and runtime, not only mean squared weight error. Use paired records and group bootstrap confidence intervals.

## G4. Paper critique and research design
Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314, https://arxiv.org/abs/2305.14314: frozen quantized backbone, NF4, double quantization and paged optimizers. Strong engineering and ablations; chatbot ratings and selected tasks do not establish rare-event calibration, safety parity or universal speedups. Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685, https://arxiv.org/abs/2106.09685: rank-constrained trainable updates, but not proof that 4-bit rounding preserves all tasks. **Proposed study:** BF16+LoRA versus uniform int4+LoRA versus NF4+LoRA, equal seeds/data/tokens/rank/targets; evaluate on sealed chronological incident groups. Primary endpoint false negatives at a fixed analyst alert budget; secondary endpoints ECE, subgroup recall, memory allocated/reserved, tokens/s and energy where measurable. Pre-register an equivalence margin rather than treating nonsignificant differences as equivalence.

# C — Poisoning, backdoors and ML supply chain, advanced

## C1. Causal chain and trust boundaries
```text
external source -> authenticated contributor -> immutable raw snapshot
      |                     |                       |
  untrusted records     access audit          hash + provenance
      +-----------------------> schema/label review -> train-only data
                                                          |
                                     sealed holdout <- train job
                                                          |
                                  signed base + signed adapter manifest
                                                          |
                         independent conditional audit + human approval
                                                          |
                                  promotion -> monitored SOC service
```
Attackers controlling training rows cannot automatically change the sealed test set or signing key. A compromised signer can produce authentic signatures on malicious artifacts: signature validity is necessary for provenance, not sufficient for safety. A poisoned but reproducible dataset can yield reproducibly bad models. Separate **data lineage**, **label correctness**, **model behavior** and **operational authorization**.

## C2. Poison sensitivity: full derivation and limits
Let P be clean distribution, Q adversarial training distribution, ε poison fraction, ℓ(θ;z) differentiable loss, θ parameters. L(θ,ε)=(1-ε)E_Pℓ+εE_Qℓ. Define θ*(ε)=argmin_θ L(θ,ε). At a differentiable local optimum, ∇_θ L(θ*(ε),ε)=0. Differentiate w.r.t. ε: H θ' +(g_Q-g_P)=0, where H=∇²_θ L(θ*(0),0), g_Q=E_Q∇ℓ and g_P=E_P∇ℓ at θ*(0). Thus θ'(0)=-H^{-1}(g_Q-g_P). This is a *local influence approximation* requiring invertible H and a stable differentiable optimum; neural nets with flat directions, nonconvex jumps, optimizer dependence and data-dependent preprocessing can violate it. Small ε can matter if gradients align with high-sensitivity directions (small positive Hessian eigenvalues).

**Worked scalar example:** clean loss L_P(θ)=θ²/2, poisoned loss L_Q(θ)=(θ-4)²/2. Mixture derivative (1-ε)θ+ε(θ-4)=θ-4ε; optimum θ*=4ε. At ε=.05, θ*=.2. Here H=1, g_P(0)=0, g_Q(0)=-4; influence formula gives derivative 4 exactly. This is a shift of a synthetic optimum, not evidence of a learned trigger or a field attack.

## C3. Conditional auditing and denominator discipline
Suppose 990 ordinary cases all correct and 10 malicious trigger-bearing cases all missed: aggregate accuracy=99%, conditional malicious recall=0%. Define conditional recall TP_T/(TP_T+FN_T) for genuinely malicious eligible trigger-bearing cases. An attack success rate must specify eligible class, target label and whether the trigger changes the ground truth. Never divide by all test samples and call it ASR. A clean-only validation set cannot rule out untested trigger classes. But probing an arbitrary marker and finding low recall does not prove malicious intent; provenance and comparison to paired controls matter.

**Bayesian operational calculation:** with attack prevalence π=.001, detector TPR=.9 and FPR=.01, precision = π·TPR/[π·TPR+(1-π)·FPR] = .0009/(.0009+.00999)≈.08264 (8.26%). Even a detector with 90% recall yields mostly false alerts at this base rate. A poisoning audit should report conditional recall and alert burden together; an automatic quarantine policy needs a separate cost and human-review analysis.

## C4. Advanced lab — influence and audit denominators (standard Python)
```python
from math import isclose
def optimum(epsilon):
    # Minimize (1-e)*theta^2/2 + e*(theta-4)^2/2
    return 4*epsilon
for e in [0.,.01,.05]:
    print(f"epsilon={e:.2f} optimum={optimum(e):.2f}")
assert isclose((optimum(1e-5)-optimum(0))/1e-5,4.)
ordinary=[(0,0)]*900+[(1,1)]*90
trigger=[(1,0)]*10  # fabricated malicious cases missed
all_pairs=ordinary+trigger
acc=sum(y==p for y,p in all_pairs)/len(all_pairs)
recall_T=sum(p==1 for y,p in trigger)/len(trigger)
assert acc==.99 and recall_T==0.
pi,tpr,fpr=.001,.9,.01
precision=pi*tpr/(pi*tpr+(1-pi)*fpr)
assert isclose(precision,.08264462809917356)
print("accuracy",acc,"trigger recall",recall_T)
print("precision at 0.1% prevalence",round(precision,4))
```
Expected: ε=0 -> 0.00, ε=.01 -> .04, ε=.05 -> .20; accuracy 0.99; trigger recall 0; precision 0.0826. **Defensive extension:** build a sealed synthetic dataset with contributor groups, a clean label-review budget, 0/1/5% *training-only* label noise, and paired conditional slices. Repeat across 20 seeds; bootstrap by contributor, not individual correlated rows. This lab does not teach creating or deploying a backdoor.

## C5. Critique, operational response and ethics
Gu, Dolan-Gavitt & Garg (2017 preprint, revised 2019), *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain*, arXiv:1708.06733, https://arxiv.org/abs/1708.06733: demonstrates conditional model failures in outsourced/trusted-model settings; image trigger experiments are not a validated threat rate for phishing or SOC telemetry. Vassilev et al. (2025), *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2025, DOI:10.6028/NIST.AI.100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025: useful definitions, not proof that a particular mitigation works. Defensive workflow: restrict contributor rights, preserve raw evidence and rights metadata, audit relabeling, isolate holdout and secrets, sign base and adapter manifests, review rare conditional failures, gate deployment and monitor rollback. Protect personal telemetry; do not publish private incident examples or operational triggers. Overaggressive filtering can suppress legitimate minority or rare incident classes.

# Cross-track synthesis: quantization meets supply-chain trust
A compact 4-bit backbone plus small LoRA adapter is a composite artifact. A benign-looking adapter can change behavior while leaving base checksum unchanged; quantization changes decision margins and may amplify rare conditional failures. Sign and bind **both** base and adapter to tokenizer, data snapshot, quantization scales, kernel revision, evaluation report and deployment authorization. The scientific hypothesis is *not* “quantization causes backdoors”: test whether low-bit adaptation alters the detectability of preexisting conditional failures under controlled, authorized conditions. Distinguish correlation from mechanism.

# Progressive oral Socratic examination — ask, pause, answer, probe
1. **What is quantization granularity?** Scope sharing codebook/scale. *Probe:* how do outliers affect it?
2. **Why isn't NF4 uniform int4?** Nonuniform normal-informed levels. *Probe:* when might normality fail?
3. **Derive raw 1B-parameter 4-bit storage.** 0.5B bytes. *Probe:* name four omitted costs.
4. **Why is 4.25 bits/weight plausible?** 4 codes +16/64 scale bits. *Probe:* what does double quantization target?
5. **State the matrix error.** Ex where E=W_hat-W. *Probe:* prove the norm bound.
6. **Can zero output error coexist with nonzero weight error?** Yes, x in null space. *Probe:* construct x.
7. **Why can tiny logit error flip a class?** Near-tied margins. *Probe:* show a numerical pair.
8. **Does QLoRA train packed codes?** No, adapters train through frozen reconstructed base. *Probe:* derivative wrt A.
9. **Why initialize one factor nonzero?** Avoid zero gradients to both. *Probe:* what happens first step?
10. **Is adapter merging lossless?** Not if re-quantized. *Probe:* compare separate-adapter inference.
11. **Which metric tests rare security impact?** Conditional false negatives at fixed alert budget. *Probe:* why not MSE?
12. **Define poison fraction ε.** Fraction of training distribution drawn from Q. *Probe:* is test set affected?
13. **Derive θ'=-H^-1(gQ-gP).** Differentiate stationarity. *Probe:* list assumptions.
14. **What does small Hessian eigenvalue imply?** Potential high local sensitivity. *Probe:* what if H singular?
15. **At ε=.05 in scalar toy, optimum?** .2. *Probe:* does that prove backdoor? No.
16. **Why can clean accuracy miss backdoors?** Rare conditional subset. *Probe:* compute 990/1000.
17. **Define triggered recall denominator.** Eligible malicious trigger-bearing examples. *Probe:* can semantic label change?
18. **Does a valid hash prove safe behavior?** No. *Probe:* what if malicious model was signed?
19. **What does provenance prove?** Claimed source/lineage under trusted controls, not label truth. *Probe:* how to review labels?
20. **Calculate precision at π=.001, TPR=.9, FPR=.01.** About 8.26%. *Probe:* how does alert budget change policy?
21. **What paper claim is not transferable directly?** Image BadNets trigger success to SOC streams. *Probe:* design domain-valid tests.
22. **What is the dissertation-ready hypothesis?** Test low-bit adaptation's impact on conditional security recall under fixed compute and sealed groups. *Probe:* preregister negative result and equivalence margin.

## Recall and research deliverable
Without notes: (a) state the quantization error bound and its limits, (b) derive LoRA factor gradients, (c) derive local poisoning influence, (d) distinguish signed artifact integrity from behavioral safety, (e) design a leakage-free rare-event experiment. Produce a one-page protocol with attacker capability, train/validation/test split, quantization conditions, conditional recall, false-alert budget, seed count, confidence interval unit, licensing and ethics review. A negative result is publishable evidence if power and equivalence margins are explicit.

## Source map and reproduction boundary
- Syllabus: 19 October private PDFs, GenAI Unit 3 Session 4 and Cyber Unit 3 Session 3; [morning package](../../../../phd-morning/lessons/2026-11-13/lesson.md).
- G1/G2: local algebra and lab G3; Dettmers 2023 for NF4/double quantization; Hu 2022 for LoRA.
- C1/C2: local causal assumptions and scalar proof; NIST 2025 for taxonomy.
- C3/C5: local fabricated conditional audit and Bayes calculation; Gu et al. for motivating backdoor threat, not toy rates.
- All code uses synthetic data. Local PDF inventory and critical summaries: [companion references index](references/index.md). No audio, email or narration.
