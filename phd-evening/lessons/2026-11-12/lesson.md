# Evening PhD deep dive — 12 November 2026 (Asia/Kolkata)

**Dependency verified first.** Read the committed same-day [morning lesson](../../../phd-morning/lessons/2026-11-12/lesson.md) and [morning references](../../../phd-morning/lessons/2026-11-12/references/index.md) from `main` before writing. The two original private 19 October syllabus PDFs were retrieved from Gmail and read: *Advanced Generative AI and Foundation Models Learning Pathway*, Unit 3 (adaptation, LoRA and quantization), and *AI for Cybersecurity Learning Pathway*, Unit 3 (threat models, evasion and mitigations). **Same topics, greater depth:** LoRA optimization, expressivity, initialization and adapter composition; physically feasible adversarial evaluation for security telemetry. All experiments below use synthetic, authorized data. No audio, narration or email is part of this package.

## Spoken tutor routes
**10 minutes:** ask what rank constrains, derive the LoRA gradients aloud, explain why two rank-one adapters can yield rank two; contrast a box-bound adversary with a protocol-consistent adversary; ask Q1–Q5.
**30 minutes:** work the scalar factorization example and rank bottleneck; solve the coupled-feature linear program; discuss the two research critiques; ask Q1–Q12.
**Extended 90–150 minutes:** run both labs, compare finite differences and exact enumeration, critique papers, design an experimental protocol and defend answers to Q1–Q20. Voice partner: ask one question at a time, wait for reasoning, challenge with a counterexample, then reveal answer. Speak equations as sentences; never read code punctuation.

## Prerequisites and precise definitions
From morning: a frozen projection `W0` with shape `k × d`, trainable `B` of shape `k × r`, `A` of shape `r × d`, scaling `s=alpha/r`, output `z=W0 x+s BA x`, upstream gradient `g=∂L/∂z`; adversarial feasible set `S(x)` and robust risk `(1/n)Σ max_{δ∈S(x_i)} ℓ(f(x_i+δ),y_i)`. Here `r` is factor rank, `alpha` a positive scale, `x` an input vector, `L` the scalar loss, `n` the number of evaluation examples, `f` the frozen classifier, `y` the ground-truth label, `δ` a permitted transformation. **Expressivity** is the class of updates representable by the factors; **identifiability** asks whether different factors represent the same update; **robust recall** is attack recall under a declared manipulation set; **attack validity** requires preserving semantics, conservation laws and actual attacker control. A worst-case score over a relaxed superset is an upper bound on the *score* optimum, not a certified lower bound on deployed robustness.

# Part G — LoRA beyond the introductory equation

## G1. Why the factorization is useful, and what it cannot represent
For `W0∈R^(k×d)`, a full update `ΔW` needs `kd` scalars. LoRA writes `ΔW=sBA`, requiring `r(k+d)` stored scalars; rank(`ΔW`) ≤ `r`. At `k=d=4096,r=8`, the morning count is `65,536/16,777,216=0.00390625`, or `0.390625%`. But this is only the selected projection's trainable weight count; embedding, layernorm, base storage, activations and optimizer accounting must be reported separately.

**Expressivity counterexample.** Let the desired update be the 2×2 identity. Its rank is two. A rank-one factorization cannot reproduce it exactly, however long it is trained. The Eckart–Young–Mirsky theorem says the best rank-`r` approximation in squared Frobenius error retains the top `r` singular values: if `ΔW*=U diag(σ1,…,σm)Vᵀ`, the minimum squared error is `Σ_{j>r}σ_j²`. For identity 2×2 with `r=1`, minimum squared error is 1. This theorem concerns matrix approximation, **not** a guarantee about task loss in a nonlinear Transformer. A network can sometimes compensate in other layers.

**When it works:** a task's useful weight change is well approximated by low-dimensional directions, with sufficient target-module coverage and training data. **When it fails:** intrinsic update rank exceeds allocated capacity; dataset shift, a mislabeled response mask, tokenizer mismatch or a stale base checkpoint invalidate a seemingly good adapter. The effective update may be low rank without the task itself being intrinsically low-dimensional.

## G2. Gradients, zero initialization and non-identifiability
Let `u=Ax`, `z=W0x+sBu`. Applying differentials,
`dz=s(dB)Ax+sB(dA)x` for frozen `W0`. Since `dL=gᵀdz`, rearrange traces to obtain
`∂L/∂B=s g(Ax)ᵀ` and `∂L/∂A=s(Bᵀg)xᵀ`.
For a batch, aggregate per-example gradients according to the declared token/sequence normalization. The input gradient is `W0ᵀg+sAᵀBᵀg`; a frozen weight can still transmit gradients to earlier activations.

If `B=0` initially and `A` is random nonzero, the first gradient to `A` is zero while `∂L/∂B` can be nonzero. If **both** `A=B=0`, both factor gradients are zero: gradient descent cannot escape. This is an important counterexample to the naive claim that “zero adapter initialization always works.”

**Gauge freedom.** For any invertible `r×r` matrix `Q`, `(BQ)(Q⁻¹A)=BA`. Thus factors are not uniquely identified by their product. In the scalar case `ba=c`, pairs `(b,a)=(1,c)` and `(10,c/10)` produce the same prediction but different raw gradients and optimizer dynamics. With scalar `x=1`, target `t=2`, prediction `p=ba`, squared loss `L=½(p-t)²`: `∂L/∂a=(ba-t)b`, `∂L/∂b=(ba-t)a`. At `a=1,b=0`, `∂L/∂b=-2` and `∂L/∂a=0`; after a gradient step `η=0.1`, `b=0.2`, and `a` now receives gradient `(0.2-2)0.2=-0.36`. These are exact toy values, not an LLM benchmark.

**Scale and regularization caveat.** Rescaling factors leaves the product invariant but not `||A||²+||B||²` or optimizer updates. Adam, weight decay and gradient clipping can therefore favor different factorizations. The convention `alpha/r` is a hyperparameterization, not a theorem guaranteeing rank-invariant training. Report `r`, `alpha`, initialization, optimizer and seeds.

## G3. Adapter composition, merging and serving
Two adapters on the same base have `W1=W0+s1 B1A1`, `W2=W0+s2 B2A2`. A weighted arithmetic merge `Wmerge=W0+λ1s1B1A1+λ2s2B2A2` is mathematically defined, but its update rank can reach `r1+r2`; it is not automatically representable by one rank-`r1` adapter. A block factorization gives an exact rank-at-most-`r1+r2` representation: `B'=[λ1s1B1, λ2s2B2]`, `A'=[A1;A2]`. If a serving system insists on rank `r`, truncated SVD compresses the combined delta with approximation error; evaluate behavior after compression.

**Nonlinear counterexample:** even if averaging adapter deltas is algebraically valid, the output of a nonlinear network with averaged weights need not equal the average of the outputs of the two original networks. Task interference and safety regressions require independent tests.

**Serving diagram**
```text
signed base revision + tokenizer hash -----> immutable frozen backbone
                   |                               |
signed adapter A --+--> registry/compatibility ----+--> chosen projection delta
signed adapter B --+       and access checks         |
                                   |                v
                            merge offline OR runtime routing
                                   |
                        regression, retention, safety tests
                                   |
                        canary -> monitored deployment -> rollback
```
Adapter-only checkpoints are not complete models. Merging into a quantized base may require dequantization/requantization, changing numerical error and memory. Multi-tenant routing trades base reuse against adapter isolation, batch compatibility and cold-load latency. A compromised adapter is a supply-chain artifact, not “just a prompt.” Store hashes, licenses, lineage, base revision, dataset manifests, test metrics and signing provenance.

## G4. Lab G — exact gradients, rank bottleneck and composition
**Requirements:** Python 3.10+, NumPy (`python -m pip install numpy`). Run offline. Copy to `evening_lora.py`.
```python
import numpy as np
np.set_printoptions(precision=5, suppress=True)
# scalar zero-B initialization
a,b,t,eta=1.,0.,2.,0.1
ga=(b*a-t)*b
gb=(b*a-t)*a
assert ga==0 and gb==-2
a-=eta*ga; b-=eta*gb
assert abs(b-.2)<1e-12
ga2=(b*a-t)*b
assert abs(ga2+.36)<1e-12
print("first gradients",ga,gb,"next dA",round(ga2,2))
# rank-one approximation of 2x2 identity cannot be exact
target=np.eye(2)
u,s,vt=np.linalg.svd(target)
rank1=(u[:,:1]*s[:1])@vt[:1,:]
err=np.linalg.norm(target-rank1,"fro")**2
assert np.isclose(err,1.)
# two rank-one adapters add to rank two
B1=np.array([[1.],[0.]])
A1=np.array([[1.,0.]])
B2=np.array([[0.],[1.]])
A2=np.array([[0.,1.]])
delta=B1@A1+B2@A2
assert np.allclose(delta,target)
assert np.linalg.matrix_rank(delta)==2
print("rank-one squared error",round(err,2),"combined rank",np.linalg.matrix_rank(delta))
# finite difference on factor gradient for a nontrivial matrix
x=np.array([1.,-2.]); A=np.array([[.3,.7]]); B=np.array([[.4],[-.2]])
g=np.array([.5,-1.]); scale=.5
def objective(aa,bb): return float(g @ (scale*(bb@aa)@x))
dA=scale*np.outer(B.T@g,x)
dB=scale*np.outer(g,A@x)
eps=1e-6
for j in range(2):
    ap=A.copy(); am=A.copy()
    ap[0,j]+=eps; am[0,j]-=eps
    assert abs((objective(ap,B)-objective(am,B))/(2*eps)-dA[0,j])<1e-8
for i in range(2):
    bp=B.copy(); bm=B.copy()
    bp[i,0]+=eps; bm[i,0]-=eps
    assert abs((objective(A,bp)-objective(A,bm))/(2*eps)-dB[i,0])<1e-8
print("factor finite differences PASS")
```
Expected lines: `first gradients -0.0 -2.0 next dA -0.36`; `rank-one squared error 1.0 combined rank 2`; `factor finite differences PASS`. The sign of printed zero may be `0.0` depending on runtime. **Advanced experiment:** train identical synthetic linear targets with rank 1, 2 and full dense updates, holding training examples, initialization seeds, steps and optimizer fixed; report train/test MSE, singular spectra and optimizer-state bytes. Then simulate adapter merge and rank-1 recompression. A negative result is informative.

## G5. Papers and critique
**Hu et al. (2022), LoRA**, arXiv:2106.09685, https://arxiv.org/abs/2106.09685: motivates low-rank updates; interrogate task coverage, rank placement, baselines and whether parameter counts are confused with total training memory. **Dettmers et al. (2023), QLoRA**, arXiv:2305.14314, https://arxiv.org/abs/2305.14314: quantized frozen weights with adapters; distinguish peak-memory claims from speed, convergence and portability. **Houlsby et al. (2019), Parameter-Efficient Transfer Learning for NLP**, arXiv:1902.00751, https://arxiv.org/abs/1902.00751: bottleneck modules are not algebraically identical to low-rank weight deltas. **Eckart & Young (1936), The Approximation of One Matrix by Another of Lower Rank**, DOI:10.1007/BF02288367, https://doi.org/10.1007/BF02288367: low-rank approximation under particular matrix norms, not a theorem about language-model task quality. **Research extension:** under fixed compute, measure whether adapter rank and merge compression change security-triage calibration and forgetting across chronologically separated incidents; publish split manifests and uncertainty intervals.

# Part C — adversarial evaluation with realistic constraints

## C1. A defensible attacker model and causal boundary
Start with the protected **asset** (correct SOC triage), **goal** (a malicious flow is not flagged), **knowledge** (scores only, or full model), **control** (timing of requests but not authenticated server logs), **budget** (maximum delay and queries), **validity** (monotonic event times, conserved byte totals, feasible TCP semantics), **success** (a still-malicious event becomes a false negative), and **response** (review, rate limit or containment). State what the attacker cannot change. A black-box score query does not grant white-box gradients. An attack on a retrieved document is not automatically a training-time poisoning attack.

```text
offline labeled sessions -> chronological/group split -> frozen feature extractor
        |                                               |
        +--> immutable event facts ------+               v
        |                                |        frozen detector
        +--> authorized edit surfaces --> validator --> attacked paired scores
                                         |               |
                                  reject impossible     |
                                  transformations       v
                                                  clean/robust recall,
                                              false alarms, time and cost
```

A detector trained on “packets per second” may react to delaying packets. But modifying a rate while keeping both event duration and packet count fixed violates `rate=count/duration`. If the attack budget allows changing duration, recompute derived rate instead of editing it independently. Protocol validity, semantics and observation-time availability are **part of** the threat model, not post-hoc excuses.

## C2. Derive relaxed bounds versus feasible optima
For a linear score `z=wᵀx+b` and independent box `|δ_j|≤ε_j`, the maximum increase is `Σ|w_j|ε_j`; choose `δ_j=ε_j sign(w_j)`. If coordinates are coupled by `Cδ=0` (conservation constraints) or inequality `Dδ≤h`, the exact optimum solves a constrained linear program:
`max wᵀδ` subject to `-ε≤δ≤ε, Cδ=0, Dδ≤h`.
The box-only maximum is an **upper bound** because it relaxes constraints; it may be unattainable. For a score-decreasing evader, minimize rather than maximize.

**Worked example:** `w=(2,-1)`, `ε=(0.1,0.1)`, `δ1+δ2=0`. With `δ2=-δ1`, the change is `2δ1-δ2=3δ1`, so maximum is `0.3` at `(0.1,-0.1)`. Without the coupling, the box maximum is also `0.3` coincidentally; to see the real gap, impose `δ1-δ2=0` instead: then change is `δ1`, maximum `0.1`. The same box has three times the apparent score sensitivity. With `δ2=0` and `δ1+δ2=0`, only zero edit is possible (the morning example).

**Robust decision margin:** for a binary classifier that flags if `z≥τ`, a malicious example with `z(x)=0.5`, threshold `τ=0.25`, and feasible minimum score shift `-0.1` remains flagged (`0.4≥0.25`); if a relaxed box predicts `-0.3`, it suggests a false negative (`0.2<0.25`) that may not be feasible. This margin reasoning certifies only the *linear score under the exact declared constraints*, not the full deployed detection pipeline.

## C3. Robust risk, estimators and evaluation design
Define `R_rob(θ)=(1/n)Σ max_{δ∈S(x_i)} ℓ(f_θ(x_i+δ),y_i)`. If inner maximization is approximated, observed worst loss is generally a **lower bound on the true maximum loss**, not a security certificate. Conversely, a mathematically justified relaxation can upper-bound the worst score or loss if its assumptions hold. Distinguish the two directions.

For a fixed attack procedure `a`, report paired malicious-event robust recall `Σ1[f(a(x_i))=malicious]/N_mal`, and clean recall on the same malicious sessions. Also report benign false-positive rate, attack validity rate, query cost, latency, rejection/abstention, and confidence intervals clustered by incident or host. Never use an attacker-generated sample that has lost malicious semantics as a successful evasion. Tune thresholds on validation only; freeze them before testing. A single perturbation method or seed is a lower-bound *attack search*, not evidence of universal resistance.

**Counterexample: leakage.** If records from one session appear in both train and test, a classifier can learn device identifiers instead of threat behavior. Attack evaluation then tests a memorized identity shortcut, not deployed robustness. Group by host/session and split chronologically where operationally appropriate. **Counterexample: class prevalence.** Even with unchanged recall and FPR, precision changes as attack prevalence changes. At prevalence `π`, recall `t`, FPR `f`, precision is `πt/[πt+(1-π)f]`. With `π=.01,t=.8,f=.02`, precision is `.008/(.008+.0198)≈.28777`; at `π=.001` it falls to `.0008/(.0008+.01998)≈.03850`. A “robust” detector may still overwhelm analysts.

## C4. Lab C — exact enumeration of physically coupled edits
**Requirements:** Python 3.10+, standard library only; no network traffic. Save as `evening_constraints.py`.
```python
from itertools import product
w=(2.,-1.)
eps=(.1,.1)
def delta_score(d): return sum(wi*di for wi,di in zip(w,d))
grid=(-.1,0.,.1)
box=[d for d in product(grid,repeat=2)]
same=[d for d in box if abs(d[0]-d[1])<1e-10]
conserved=[d for d in box if abs(d[0]+d[1])<1e-10]
def bounds(xs): return (min(map(delta_score,xs)),max(map(delta_score,xs)))
assert bounds(box)==(-.30000000000000004,.30000000000000004) or all(abs(a-b)<1e-10 for a,b in zip(bounds(box),(-.3,.3)))
assert all(abs(a-b)<1e-10 for a,b in zip(bounds(same),(-.1,.1)))
assert all(abs(a-b)<1e-10 for a,b in zip(bounds(conserved),(-.3,.3)))
print("box",tuple(round(v,2) for v in bounds(box)))
print("same-change",tuple(round(v,2) for v in bounds(same)))
print("conserved",tuple(round(v,2) for v in bounds(conserved)))
# A score threshold: relaxed evasion versus coupled feasible non-evasion
base=.5; threshold=.25
assert base+min(map(delta_score,box))<threshold
assert base+min(map(delta_score,same))>=threshold
print("relaxed evades",True,"coupled evades",False)
# Prevalence-dependent alert precision
def precision(pi,tpr,fpr): return pi*tpr/(pi*tpr+(1-pi)*fpr)
assert abs(precision(.01,.8,.02)-.28776978417266186)<1e-12
print("precision at 1% prevalence",round(precision(.01,.8,.02),4))
```
Expected: `box (-0.3, 0.3)`; `same-change (-0.1, 0.1)`; `conserved (-0.3, 0.3)`; `relaxed evades True coupled evades False`; `precision at 1% prevalence 0.2878`. **Advanced extension:** simulate session events with immutable byte count, variable duration and recomputed rates; compare box, conservation and protocol-valid transformations. Bootstrap paired session-level recall differences; add uncertainty intervals and a validity audit. This is defensive offline methodology, not a recipe for attacking an external IDS.

## C5. Primary literature and critical interpretation
**Goodfellow, Shlens & Szegedy (2015), Explaining and Harnessing Adversarial Examples**, arXiv:1412.6572, https://arxiv.org/abs/1412.6572: first-order sensitivity explains some gradient-based perturbations; image-pixel norms do not translate automatically to valid packets, URLs or logs. **Biggio et al. (2013), Evasion Attacks against Machine Learning at Test Time**, DOI:10.1007/978-3-642-40994-3_25, https://doi.org/10.1007/978-3-642-40994-3_25: attacker knowledge and feature-control assumptions matter; inspect operational feasibility and evaluation splits. **Carlini & Wagner (2017), Towards Evaluating the Robustness of Neural Networks**, arXiv:1608.04644, https://arxiv.org/abs/1608.04644: illustrates why a weak attack may overestimate robustness; does not by itself validate structured telemetry edits. **NIST (2025), Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations**, DOI:10.6028/NIST.AI.100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025: useful taxonomy of capabilities and defenses, not a certification of any detector.

**Research extension:** pre-register the hypothesis “robust recall estimated under independent normalized-feature edits differs materially from recall under physically feasible event transformations.” Freeze the detector and test sessions, publish the transformation validator, compare equal search budgets, and use paired clustered bootstrap intervals. Negative results must be retained. Conduct only authorized offline tests, redact personally identifiable telemetry and avoid releasing actionable real-world exploit traces.

# Cross-track synthesis: security of adapted SOC assistants
An organization might use LoRA to adapt an incident summarizer and a separate detector to flag malicious telemetry. The **developer** controls adapter weights; the **attacker** might control a log message or retrieved document; a third actor might compromise the adapter registry. Those are different control surfaces. A prompt-injection string in a log cannot by itself change model weights, but can influence a poorly isolated summarizer; a malicious adapter checkpoint can persistently change behavior without any suspicious input. Defensive architecture: signed artifact registry, frozen base revision, tokenizer and data provenance, typed log ingestion, untrusted-content separation, evidence citations, least-privilege tools, human confirmation for containment and drift monitoring.

# Progressive oral Socratic examination
1. **What exactly does rank constrain?** Rank of each learned weight delta, not the frozen matrix or total network. *Follow-up:* could a rank-one update still change many output coordinates? Yes.
2. **Why is parameter percentage not GPU-memory percentage?** Base weights and activations remain, and optimizer/master states depend on precision. *Follow-up:* what is quantized in QLoRA? Frozen base.
3. **Give both LoRA gradients.** `s g(Ax)ᵀ` for B and `s(Bᵀg)xᵀ` for A. *Follow-up:* verify shapes.
4. **Why does B=0 initialize safely?** Initial delta is zero; A nonzero allows B to learn. *Follow-up:* both zero? Training stalls.
5. **Can rank-one fit identity 2×2?** No; best squared Frobenius error is 1. *Follow-up:* does that prove bad language-model accuracy? No.
6. **What is gauge freedom?** `BQ` and `Q⁻¹A` have same product. *Follow-up:* why do optimizers differ? Regularization/gradient geometry.
7. **Can two rank-one adapters sum to rank two?** Yes, orthogonal coordinate projections do. *Follow-up:* what rank is required for exact merged representation? At most two.
8. **Does averaging weights average predictions?** Not in nonlinear networks. *Follow-up:* what test is needed? Task and safety regression.
9. **When would rank compression be acceptable?** When held-out quality, calibration and safety remain within declared tolerances. *Follow-up:* why inspect drift?
10. **What must an adapter checkpoint identify?** Base revision, tokenizer, training configuration, hashes and licensing. *Follow-up:* supply-chain consequence?
11. **Name the attacker's actual control surface.** Explicitly editable fields and their provenance. *Follow-up:* can a network client rewrite trusted server logs? Generally not.
12. **What is a feasibility constraint?** A rule preserving physical/protocol semantics. *Follow-up:* rate/count/duration dependency?
13. **Why does independent box sensitivity mislead?** It permits coupled edits that cannot coexist. *Follow-up:* derive `δ1=δ2` example.
14. **What is the score bound for w=(2,-1), ε=(.1,.1)?** Box 0.3; equal-change constraint 0.1. *Follow-up:* conservation `δ1+δ2=0`? 0.3.
15. **Which way does approximate inner maximization bias robust loss?** Observed max ≤ true max. *Follow-up:* can weak attacks certify security? No.
16. **When is a relaxed bound useful?** To prove no feasible edit can cross a threshold when the relaxation itself cannot cross it. *Follow-up:* state assumptions.
17. **Why retain paired clean and attacked sessions?** Control baseline differences and permit paired inference. *Follow-up:* why cluster by host?
18. **Why does precision fall with prevalence?** False alarms dominate at low base rate. *Follow-up:* compute at 0.1% prevalence.
19. **Is a poisoned adapter the same as prompt injection?** No; persistent artifact manipulation versus input instruction crossing trust boundary. *Follow-up:* what controls differ?
20. **Defend a doctoral experiment.** Pre-register threat model, immutable splits, validity checker, fixed detector, attack budgets, robust recall/FPR, clustered CIs, negative results and reproducible artifacts. *Follow-up:* what evidence would falsify the hypothesis?

## Recall and next-morning handoff
Without notes: derive the factor gradients; show the both-zero initialization trap; construct two rank-one adapters whose sum is rank two; explain why LoRA factors are non-identifiable; derive constrained linear score bounds; distinguish weak-search loss from certified worst-case loss; calculate precision at changing prevalence; name three trust boundaries in an SOC assistant. The next morning may progress the syllabus toward quantization/preference optimization and adversarial mitigations, but **this evening did not introduce them as new core units**.

## Claim-to-source map and verification limits
- Syllabus and same-day continuity: private 19 October pathway PDFs (GenAI Unit 3, Cyber Unit 3), [morning lesson](../../../phd-morning/lessons/2026-11-12/lesson.md) and [morning index](../../../phd-morning/lessons/2026-11-12/references/index.md).
- LoRA mechanism and parameterization: [G1] Hu et al.; mathematical gradient/zero-init/gauge examples are explicitly derived here in G1–G4.
- QLoRA and alternative adapters: [G2] Dettmers; [G3] Houlsby; compare architectures rather than claiming universal performance.
- Best low-rank approximation: [G4] Eckart & Young; G1 synthetic identity demonstration.
- First-order adversarial methods: [C1] Goodfellow; attacker-aware evasion: [C2] Biggio; evaluation pitfalls: [C3] Carlini & Wagner; terminology: [C4] NIST.
- Constrained score optimization, threshold counterexample and precision: original mathematical derivations and synthetic tests C2–C4; not attributed as experimental findings from papers.
- Governance: cross-track synthesis and NIST taxonomy, design guidance not a certification.
- **Access/validation:** Canonical paper identifiers and stable links are recorded in [companion index](references/index.md); no PDF is claimed downloaded, licensed or redistributed. Code is designed to run locally with stated dependencies and assertions; a remote GitHub CI run is not claimed. External URLs are canonical reference destinations, not individually HTTP-verified in this run. No unverified 12 November news is asserted.
