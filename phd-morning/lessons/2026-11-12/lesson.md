# Morning PhD teaching package — 12 November 2026 (Asia/Kolkata)

**Continuity and syllabus provenance.** Read the committed 11 November morning lesson and references on `main`: supervised fine-tuning (SFT), response-only masks, cross-task cybersecurity benchmarks, leakage and cost-sensitive evaluation. Read both original 19 October 2026 Gmail syllabus PDFs: *Advanced Generative AI and Foundation Models Learning Pathway*, Unit 3 (adaptation, adapters and LoRA), and *AI for Cybersecurity Learning Pathway*, Unit 3 Session 1 (adversarial threat models). Today's accelerated progression is **parameter-efficient fine-tuning (PEFT), especially LoRA**, and **explicit adversarial-ML threat modeling**. All measurements below are synthetic illustrations, not empirical claims about deployed systems.

## Spoken route and learning contract
**10 minutes:** contrast full SFT with LoRA; derive rank-2 parameter count; name the six fields of an attacker model; answer Q1–Q5. **30 minutes:** follow the low-rank gradient, compare adapter placements, design an attacker-constrained IDS evaluation, discuss Q1–Q11. **Extended 90–150 minutes:** execute both labs, critique papers, design a controlled experiment and answer Q1–Q18. Voice tutor: ask one question, wait for the learner's reasoning, then give the answer and a counterexample. Read symbols verbally, not code punctuation.

## Prerequisites and definitions
Recall the previous lesson's conditional likelihood, SFT loss, gradient descent, chronological/grouped evaluation, prevalence and decision thresholds. **Weight matrix** W: learned linear map from d input coordinates to k output coordinates. **Full fine-tuning:** update all selected pretrained weights. **PEFT:** update a small subset or added parameters. **LoRA:** low-rank adaptation, keeping pretrained W frozen and learning factors A and B. **Rank:** dimension of a matrix's independent column/row subspace. **Adapter:** small trainable module added to a frozen backbone. **Threat model:** precise statement of assets, adversary goal, capabilities, knowledge, access, budget and constraints. **Evasion:** inference-time manipulation; **poisoning:** training-time manipulation; **backdoor:** conditional malicious behavior associated with a trigger. **White box:** model internals accessible; **black box:** only an interface/observations accessible. **Targeted attack:** force a particular incorrect output; **untargeted:** cause any incorrect output. **Validity constraint:** a perturbation must still represent a possible input with the intended underlying semantics.

# Track G — Low-rank adaptation after SFT

## G1. Mechanism and intuition
A dense projection computes h=Wx where W has shape k×d, x has shape d, and h has shape k. In full SFT every element of W may change. LoRA writes W_eff=W_0+(α/r)BA, with frozen W_0∈R^{k×d}, A∈R^{r×d}, B∈R^{k×r}, rank r≪min(k,d), scaling α/r. The trainable update BA has rank at most r, even if W_0 is full rank. Intuition: task-specific changes may lie in a lower-dimensional subspace; this is a *hypothesis*, not a guarantee for all tasks. For a 4096×4096 projection, full update has 16,777,216 parameters; r=8 factors have 8(4096+4096)=65,536, or 0.390625% of the projection's parameters. This excludes other trainable layers, biases, optimizer states and activation memory.

**Forward and backward, step by step.** (1) Token embeddings and previous layer produce x. (2) Frozen path computes W_0x. (3) A projects x to r dimensions; B projects back to k; scale by α/r. (4) Sum paths and pass to downstream loss. (5) Backpropagate through A,B and other explicitly trainable modules, not W_0. (6) Save adapter weights and base-model revision separately; the adapter alone is not a runnable full model. Often B is initialized to zero and A nonzero, making the initial update exactly zero. Initially gradient through B can be nonzero while gradient through A is zero, which changes after B moves.

**Architecture:**
```text
prompt -> tokenizer -> frozen transformer backbone
                         | each selected projection |
                    x -> W0 x (frozen) -----------+
                    x -> A (d->r) -> B (r->k) --+ -> sum -> next block
                           trainable factors      |
                 response-only SFT objective -> optimizer(A,B)
evaluation -> task quality + general retention + memory + latency
```

## G2. Derivation, gradients and trade-offs
Let z=W_0x+sBAx with s=α/r, upstream gradient g=∂L/∂z (k-vector), and u=Ax (r-vector). By the chain rule,
```math
∂L/∂B=s g u^T,       ∂L/∂A=s (B^T g)x^T,
∂L/∂x=W_0^Tg+s A^TB^Tg.
```
Explain shapes: g u^T is k×r, (B^Tg)x^T is r×d, and ∂L/∂x has d elements. For a minibatch, sum/average the per-example gradients according to the declared loss normalization. The effective update rank is ≤r, but a sequence of optimization steps may explore different low-rank directions through changing A and B.

**Memory estimate:** for fp16/bf16 trainable adapter parameters P, raw weights occupy about 2P bytes; fp32 Adam moments typically 8P additional bytes, and gradients/master weights depend on optimizer and precision. A full fine-tuning optimizer may need states for all selected pretrained parameters, while LoRA still must retain frozen base weights and activations needed for gradient computation. Thus 0.39% trainable *projection parameters* does not imply 0.39% GPU memory. Quantization of frozen weights (e.g., QLoRA) can further reduce base storage but changes numerical behavior and kernel constraints.

**Worked example:** d=4,k=3,r=1, α=1, x=[1,2,0,-1], A=[1,0,1,0], B=[2,-1,3]^T. Ax=1; BAx=[2,-1,3]. With g=[1,2,-1], ∂L/∂B=[1,2,-1]^T and B^Tg=2-2-3=-3, so ∂L/∂A=[-3,-6,0,3]. If α=2 then both derivatives double for fixed A,B,x,g. This calculation is independent of the unknown W_0 because g is stipulated as upstream gradient.

**When it works:** modest compute, multiple specialized tasks sharing a base, bounded behavior changes, and frozen-weight governance. **When it fails:** rank too small for a complex transformation, mismatched tokenizer/base revision, unstable target-module selection, catastrophic forgetting, noisy instruction labels, stale facts, and distribution shift. LoRA is not retrieval and cannot guarantee citations. **Design choices:** r, α, dropout, q/v versus all-linear placement, learning rate, adapter merge versus separate serving, precision and checkpoint provenance. Compare at *equal training tokens and compute budget*, not just equal epochs. A larger r is not automatically better; validation variance matters.

## G3. Lab G — runnable low-rank algebra and finite-difference gradients
Python 3.10+ and NumPy (`python -m pip install numpy`); synthetic vectors only.
```python
import numpy as np
x=np.array([1.,2.,0.,-1.])
A=np.array([[1.,0.,1.,0.]])
B=np.array([[2.],[-1.],[3.]])
g=np.array([1.,2.,-1.])
s=1.
def loss(a,b):
    return float(g @ (s*(b@a)@x))
dB=s*np.outer(g,A@x)
dA=s*np.outer(B.T@g,x)
assert np.allclose(dB.ravel(),[1,2,-1])
assert np.allclose(dA.ravel(),[-3,-6,0,3])
eps=1e-6
for i,j in [(0,0),(0,1),(0,2),(0,3)]:
    ap=A.copy(); am=A.copy()
    ap[i,j]+=eps; am[i,j]-=eps
    numerical=(loss(ap,B)-loss(am,B))/(2*eps)
    assert abs(numerical-dA[i,j])<1e-8
assert np.linalg.matrix_rank(B@A)<=1
d=k=4096; r=8
full=d*k; lora=r*(d+k)
assert full==16777216 and lora==65536
print("dB",dB.ravel().tolist())
print("dA",dA.ravel().tolist())
print("trainable fraction",round(100*lora/full,6),"%")
```
Expected: `dB [1.0, 2.0, -1.0]`, `dA [-3.0, -6.0, 0.0, 3.0]`, `trainable fraction 0.390625 %`. **Advanced extension:** implement an actual tiny PyTorch linear layer with frozen W_0 and trainable A/B, train on a reproducible synthetic mapping; compare r∈{1,2,4}, validation MSE, full fine-tuning, training time and optimizer-state bytes. Keep random seed, target matrix, split, CPU/GPU device and all library versions. Do not present toy results as LLM results.

## G4. Research and critical reading
Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685: motivates low-rank updates; scrutinize chosen tasks, module placement and parameter-count accounting. Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314: combines quantized frozen weights with adapters; separate storage savings from training quality and runtime. Houlsby et al. (2019), *Parameter-Efficient Transfer Learning for NLP*, arXiv:1902.00751: inserted bottleneck adapters are related but architecturally distinct from LoRA's weight update. **Research question:** under a fixed token and accelerator budget, how does LoRA rank interact with temporal drift in security incident classification? Pre-register rank, datasets, group/time splits, cost-sensitive metrics, forgetting benchmark, latency, energy and uncertainty intervals. Avoid production secrets and copyrighted training material without rights.

# Track C — Adversarial ML threat modeling

## C1. Begin with the attacker, not the attack algorithm
A detector's clean-test accuracy says nothing about a manipulator's access. Write a seven-field threat model: **asset** (e.g. integrity of triage), **attacker objective** (evade a review flag), **knowledge** (scores only versus weights), **control surface** (editable log field, URL, packet timing, training row), **budget** (number of edits/queries), **validity constraints** (protocol and physical consistency), **success measure** (misclassification while maintaining event semantics), plus **defender response** (abstain, analyst escalation). Treat attacker objective and legitimate label as separate; a transformed event that is no longer malicious is not an evasion success. Distinguish benign drift from maliciously optimized changes.

```text
authorized synthetic telemetry -> feature extraction -> frozen detector -> score
             |                                        |
  constrained simulated adversary -> valid edits -----+
             |                     |                  |
         semantics check      budget log          threshold + action
             +------------------ audit -> clean/attacked paired evaluation
```

**Scenario:** an IDS uses request-rate and connection-duration features. A simulator allows at most 2 seconds of timing change but must preserve packet count, total bytes, and chronological event order. Changing packet count without changing bytes and protocol constraints may be physically inconsistent. An arbitrary L-infinity perturbation of standardized columns is a mathematical sensitivity probe, not automatically a feasible network attack. A malicious actor who controls a public retrieved document does not automatically control a system instruction or the model weights. Explicit trust boundaries prevent conflating prompt injection, poisoning and evasion.

## C2. Formal constrained risk and counterexample
Let f_θ(x) be a classifier, y the ground-truth label, ℓ its loss, and S(x) the *valid* transformation set under the adversary's control. Empirical robust risk is
```math
R_rob(θ)=(1/n) Σ_{i=1}^n max_{δ∈S(x_i)} ℓ(f_θ(x_i+δ),y_i).
```
Here n is test examples, δ a permitted perturbation, θ model parameters; the maximum is the strongest attack in the declared feasible set. **This is not a proof of security** if the inner maximum is approximated by a weak search, or S omits plausible attack actions. For scalar linear score z=w^Tx+b and box constraints |δ_j|≤ε_j, the maximum score increase is Σ_j |w_j|ε_j (choose δ_j=ε_j sign(w_j)); if only editable coordinates E are permitted, restrict the sum to E. If edits must conserve a total (Σδ_j=0), this independent-coordinate bound may be unattainable.

**Worked example:** z=2x_1−x_2−1; x=[1,1], initial z=0. Editable coordinate x_1 has |δ_1|≤0.1; x_2 is immutable. Maximum z increase is 0.2, not 0.3 from falsely assuming both coordinates are editable. If the classifier flags when z>0.1, feasible manipulation of x_1 can change its decision; whether that is a *successful evasion* depends on the true label, desired direction and preserved semantics. Report both false negatives and false positives.

**Threat-model failures:** unrealistic independent feature edits, no query-rate limits, changing the ground truth during manipulation, leaked labels, testing only one seed, adversary knowing future intelligence, assuming a gradient attacker can access a black-box production system, and measuring only attack success while ignoring clean-data performance. Ethical rule: conduct experiments on synthetic or explicitly authorized offline data; never probe external targets.

## C3. Lab C — constrained score sensitivity without attack traffic
Python 3.10+ standard library; purely synthetic.
```python
w=[2.,-1.]; x=[1.,1.]; b=-1.
def score(v): return sum(a*t for a,t in zip(w,v))+b
epsilon=[.1,0.]  # second feature immutable
base=score(x)
upper=sum(abs(a)*e for a,e in zip(w,epsilon))
candidate=[x[j]+epsilon[j]*(1 if w[j]>=0 else -1) for j in range(2)]
assert base==0 and abs(upper-.2)<1e-12
assert abs(score(candidate)-.2)<1e-12
assert candidate[1]==x[1]
# Check a conservation rule: δ1+δ2=0 and δ2 immutable -> only zero edit.
valid=[(a,b) for a in [-.1,0.,.1] for b in [0.]
       if abs(a+b)<1e-12]
assert valid==[(0.,0.)]
print("base",base,"box upper",round(upper,2))
print("candidate",candidate,"conserved edits",valid)
```
Expected: `base 0.0 box upper 0.2`; `candidate [1.1, 1.0] conserved edits [(0.0, 0.0)]`. **Advanced extension:** create 100 synthetic labeled flow summaries with group/time split, a frozen logistic classifier and a bounded perturbation simulator. Audit feature dependencies, compute robust recall and clean precision with bootstrap by session, and compare abstention versus automatic blocking. Do not use real malware, unauthorized probing, or invented published benchmark scores.

## C4. Critique, security and research
Goodfellow, Shlens & Szegedy (2015), *Explaining and Harnessing Adversarial Examples*, arXiv:1412.6572: first-order perturbation insight, but image-space assumptions do not establish feasible changes to structured network data. Biggio et al. (2013), *Evasion Attacks against Machine Learning at Test Time*, DOI:10.1007/978-3-642-40994-3_25: attacker-aware evasion framing; inspect what feature edits and knowledge are allowed. NIST (2025), *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*, AI 100-2e2025, https://doi.org/10.6028/NIST.AI.100-2e2025: taxonomy for capabilities, lifecycle and mitigations; terminology does not replace evaluation. **Research question:** how much does estimated robust recall fall when arbitrary numeric perturbations are replaced by protocol-consistent transformations? Specify a feasible-set validator, same clean test events, paired confidence intervals, and independently verified labels. A defense that blocks one toy attack is not a universal defense.

## Cross-track synthesis
LoRA controls *which parameters* an authorized developer may update; an adversarial threat model controls *which inputs or artifacts* an adversary may manipulate. In an SOC assistant, adapter checkpoints, retrieval documents and detector telemetry are different trust surfaces. A malicious document is not a gradient update; a compromised adapter artifact is not ordinary user prompting. Protect the base/adapter artifact provenance, verify hashes, restrict training updates, isolate untrusted retrieved text, enforce tool permissions and retain human approval for destructive actions.

## Progressive oral Socratic examination (ask, pause, then reveal)
1. **Why PEFT?** Reduce trainable state while retaining a reusable backbone. *Follow-up:* does it remove activation memory? No.
2. **What is rank?** Maximum independent column/row dimension. *Follow-up:* rank(BA)≤r.
3. **LoRA equation?** W_eff=W_0+(α/r)BA. *Follow-up:* which tensor stays frozen? W_0.
4. **Count for 4096 square, r=8?** 65,536 vs 16,777,216. *Follow-up:* what memory costs remain? Base, activations, optimizer.
5. **Why initialize B=0?** Initial update is zero. *Follow-up:* which factor gets gradient first? Usually B.
6. **Give derivative wrt B.** s g(Ax)^T. *Follow-up:* shape k×r.
7. **Derivative wrt A?** s(B^Tg)x^T. *Follow-up:* shape r×d.
8. **Can low training loss prove facts?** No; evidence and sealed generation evaluation are separate. *Follow-up:* SFT or RAG for changing manuals? RAG.
9. **LoRA versus bottleneck adapter?** Low-rank weight delta versus inserted activation module. *Follow-up:* can both use a frozen backbone? Yes.
10. **Why can QLoRA save more storage?** Quantized frozen base plus trainable adapters. *Follow-up:* what must be validated? Quality, numerical stability and kernels.
11. **Define threat model.** Asset, goal, knowledge, control, budget, constraints, success and response. *Follow-up:* choose one for a URL detector.
12. **Evasion versus poisoning?** Inference-time inputs versus training-time influence. *Follow-up:* is document injection always poisoning? No.
13. **White versus black box?** Internal access versus interface access. *Follow-up:* can query feedback still leak information? Yes.
14. **Why is L∞ often unrealistic for flows?** Features obey coupled protocol constraints. *Follow-up:* give a conserved quantity.
15. **What does inner max mean?** Worst loss over declared feasible edits. *Follow-up:* does an approximate attack certify robustness? No.
16. **Calculate box bound for w=[2,-1], ε=[.1,0].** 0.2. *Follow-up:* what if conservation forces δ=0? Bound becomes 0.
17. **What does an attack success metric omit?** Clean utility, feasibility, false positives, costs. *Follow-up:* add analyst capacity and latency.
18. **Design one doctoral study.** Compare feasible vs arbitrary edits under fixed detector, split and budget, paired robust recall and CI. *Follow-up:* how will a third party reproduce constraints? Publish validator and manifests.

## Recall and next-session handoff
Without notes: derive LoRA's parameter count and gradients; explain why optimizer memory is not proportional to trainable fraction; specify a valid threat model for a phishing detector; calculate a constrained linear bound; describe why an attack simulation may overstate deployment risk. The evening lesson should deepen **LoRA optimization/adapter composition** and **feasible-set adversarial evaluation**, not introduce unrelated topics.

## Source map
- **Curricular progression:** the two original October 19 pathway PDFs, GenAI Unit 3 and Cybersecurity Unit 3; previous committed 11 November lesson and companion index.
- **Low-rank formulation, trainable-parameter savings:** Hu et al. [G1]; derivation and synthetic arithmetic G1–G3.
- **Quantized frozen weights:** Dettmers et al. [G2]; do not extrapolate performance.
- **Bottleneck adapters:** Houlsby et al. [G3].
- **Adversarial first-order intuition:** Goodfellow et al. [C1].
- **Evasion attacker model:** Biggio et al. [C2].
- **Taxonomy:** NIST [C3].
- **Constraint mathematics, examples and code:** original worked derivations in C2–C3, not empirical literature results.
- **Risk and governance:** threat-boundary analysis in C4 and cross-track synthesis.

**Validation/access note:** Code is self-contained except for the declared NumPy dependency in Lab G; expected outputs are analytically checked, but remote execution in the repository's CI is not claimed. No private syllabus PDFs or research PDFs are redistributed. External reference links are canonical identifiers; current-day news and individual live HTTP resolutions are not asserted without separate verification.
