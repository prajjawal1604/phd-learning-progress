# Evening PhD deep dive — 23 November 2026 (Asia/Kolkata)
**Dependency verified:** [same-day morning lesson](../../../../phd-morning/lessons/2026-11-23/lesson.md) and [morning reference index](../../../../phd-morning/lessons/2026-11-23/references/index.md) were fetched from GitHub main before drafting. Deepens exactly **Generative AI U5/S4: speech, audio and video** and **AI for Cybersecurity U5/S4: code review, cyber ranges and synthetic data**. Syllabus: *Advanced Generative AI and Foundation Models Learning Pathway*, p.12 U5/S4, and *AI for Cybersecurity Learning Pathway*, p.12 U5/S4, attached to the 19 Oct 2026 Gmail message. No production targets, recordings or identities are used.

## Spoken navigation and prerequisites
**10 minutes:** Ask why CTC sums paths, work a two-frame probability, explain why “not” errors dominate safety, then calculate precision under a low vulnerability base rate and critique synthetic test leakage. Use questions 1–4 and 13–16.
**30 minutes:** Add CTC forward recurrence, audio-video timing, selective risk, taint analysis, prior-shift odds and range evaluation; questions 1–10 and 13–22.
**Extended 90–120 minutes:** Work every derivation and counterexample, run both labs only while stationary, critique papers and design research studies. Finish with spaced recall. **Tutor protocol:** pose one question, wait for spoken reasoning, challenge one assumption, then disclose answer and follow-up. Never execute code while walking.

Prerequisites: conditional probability and logarithms; dynamic programming; sequence alignment; sampling; confusion matrices; Bayesian odds; trust boundaries; SQL parameterization; train/test separation. Symbols are defined at first use. A **logit** is an unnormalized model score; **CTC blank** is a special non-character alignment symbol; **calibration** means predicted probability matches observed frequency; **taint** means data may originate from an untrusted source; **covariate shift** changes input distribution; **prior shift** changes class prevalence with class-conditionals held fixed; **cyber range** is an isolated authorized experiment.

# Part I — Speech, audio and video: deeper mechanics

## 1. CTC from path enumeration to dynamic programming
Morning established P(y|x)=Σ_{π:B(π)=y}∏_{t=1}^T p_t(π_t|x), where x is the acoustic input, y the desired transcript, T the number of acoustic frames, π a frame-level token path and B the collapse operation. Conditional frame independence is an assumption **given encoder representations**, not a claim that speech frames are physically independent. A blank between two identical symbols is needed to preserve the repeated output.

**Derive the forward algorithm.** Expand transcript y=(y₁,…,y_U) into z=(blank,y₁,blank,y₂,…,y_U,blank) of length S=2U+1. Let α_t(s) be total probability of valid paths ending at expanded state s at time t. For t>1,
α_t(s)=p_t(z_s)[α_{t−1}(s)+α_{t−1}(s−1)+I(z_s≠blank and z_s≠z_{s−2})α_{t−1}(s−2)].
Out-of-range α terms are zero. Initialize α₁(1)=p₁(blank), α₁(2)=p₁(y₁), all others zero. The final probability is α_T(S)+α_T(S−1). The skip condition prohibits jumping over a blank to repeat the same letter: otherwise “AA” would collapse to “A”. Computational complexity O(TU) and memory O(U) with rolling rows, versus exponentially many naive paths. Train with log-sum-exp to prevent underflow.

**Tiny worked case:** one-character transcript A, two frames, vocabulary {blank,A}, p₁(A)=0.6, p₂(A)=0.7. Valid paths (A,A), (A,blank), (blank,A) have probabilities .42, .18 and .28; total P(A)=.88, negative log likelihood ≈0.127833. The excluded (blank,blank) has probability .12. Greedy argmax outputs (A,A)→A here, but greedy decoding is not guaranteed to maximize transcript probability: sums over paths can reverse rankings.

**Gradient intuition:** for frame t and symbol k, derivative of negative log probability with respect to logit a_t(k) is p_t(k)−P(π_t=k|x,y). This is model probability minus the *posterior occupancy* of the symbol among alignments consistent with y, obtained by forward-backward. The target is a distribution over alignments, not a single hard timestamp. CTC timestamps should not be treated as verified word boundaries without alignment evaluation.

**When it works:** approximately monotonic alignment and sufficient frame resolution. **When it fails:** overlapping speakers, ambiguous short negations, heavy clipping, and language-model hallucination. A lower WER can conceal a more dangerous error: “do not isolate”→“isolate” may be one deletion but reverses the action. Define weighted command risk R=Σ_i c_i·1(error on token i), where c_i is the consequence weight, and report it alongside WER. Costs must be predeclared, not tuned to flattering results.

## 2. Timing and sampling beyond the morning intuition
For sampling frequency f_s and N-point STFT window, nominal bin spacing Δf=f_s/N; hop H gives frame advance Δt=H/f_s. At f_s=16000, N=400, H=160, bin spacing is 40 Hz and hop 10 ms. Zero-padding adds interpolated spectral samples, not new resolving power. Longer windows improve frequency discrimination at the expense of temporal localization; short events can be smeared.

For periodic video samples separated by Δ and a visible event of duration d≤Δ with uniformly random start phase, P(hit)=d/Δ. If the event lasts 100 ms and Δ=1 s, hit probability .1. For n **independent** sampling phases the probability at least one sees it is 1−(1−d/Δ)^n; frames on a fixed periodic grid are **not independent**, so this expression cannot be applied to n consecutive fixed-phase samples. Counterexample: a recurring 100-ms event synchronized between each 1-second frame is always missed despite arbitrarily many frames. Random jitter may break synchronization but creates coverage and timestamp complications.

**Audio-video sync model:** let video-clock time t_v=a·t_a+b, where t_a is audio clock, b offset and a relative clock-rate drift. Estimate a,b from trusted synchronization markers, fit residuals, and report uncertainty. One offset measurement cannot estimate drift. A timestamped frame and a transcript are observations; neither authenticates the speaker, the capture device or the intervening event.

## 3. Safety-aware spoken agent architecture
```text
captured PCM + signed capture metadata -------> immutable evidence store
     |                                                   |
ASR logits -> transcript + alternatives -> semantic risk gate
     |                  |                      |
 diarization      timing uncertainty      verify negation/numbers
     |                  |                      |
video frames -> timestamp calibration -> evidence reconciliation
                                                |
                         low-risk answer / ask replay / human review
                                                |
                                  text response -> consented TTS
```
The evidence store must distinguish raw capture, derived transcript, synthetic audio and edited media. Hashes detect later alteration **only if capture and key custody are trustworthy**; a hash does not prove the event occurred. Voice similarity is not authentication. For high-consequence instructions, preserve alternatives, request repetition and require authorized human approval. Avoid inferring identity, emotion or protected traits from voice.

**Advanced Lab A — exact CTC, gradient finite difference, temporal aliasing (Python standard library).** Save as `ctc_evening.py`; run `python ctc_evening.py`.
```python
import math
def prob_A(logits):
    # each frame has logits (blank, A)
    ps=[]
    for b,a in logits:
        m=max(b,a); eb,ea=math.exp(b-m),math.exp(a-m)
        ps.append((eb/(eb+ea),ea/(eb+ea)))
    return 1-ps[0][0]*ps[1][0] # every path except blank,blank
def loss(logits): return -math.log(prob_A(logits))
p=prob_A([(math.log(.4),math.log(.6)),
          (math.log(.3),math.log(.7))])
assert abs(p-.88)<1e-12
print("P(A)",round(p,6),"NLL",round(-math.log(p),6))
L=[(0.0,0.4),(0.2,0.8)]
h=1e-5
plus=[tuple(x) for x in L]; minus=[tuple(x) for x in L]
plus[0]=(L[0][0],L[0][1]+h)
minus[0]=(L[0][0],L[0][1]-h)
numeric=(loss(plus)-loss(minus))/(2*h)
b,a=L[0]; q=math.exp(b)/(math.exp(b)+math.exp(a))
b2,a2=L[1]; q2=math.exp(b2)/(math.exp(b2)+math.exp(a2))
analytic=-(q*(1-q)*q2)/(1-q*q2)
assert abs(numeric-analytic)<1e-8
print("gradient_check",abs(numeric-analytic)<1e-8)
# Event repeats during [k+.2,k+.3); sample at integer seconds
events=[(k+.2,k+.3) for k in range(10)]
hits=sum(any(lo<=t<hi for lo,hi in events) for t in range(11))
assert hits==0
print("phase_locked_hits",hits)
```
Expected: `P(A) 0.88 NLL 0.127833`, `gradient_check True`, `phase_locked_hits 0`. This is a mathematical toy, not ASR benchmark or real camera experiment. Extension: implement log-domain CTC for “AA” with three or more frames, compare finite-difference gradients, add a negation-weighted error score, simulate drifting clocks. Check repeated-symbol transitions carefully.

# Part II — Cybersecurity code review, cyber ranges and synthetic data

## 4. Code review as an evidence graph, not a label
The morning example concatenated untrusted username into SQL and replaced it with parameter binding. A doctoral-level reviewer should build an evidence chain:
```text
versioned source + dependencies
  -> entry point (request parameter)
  -> transformations (decode/validate/concatenate)
  -> sensitive sink (SQL execution)
  -> trust-boundary and reachability proof
  -> isolated regression test
  -> parameterized patch + non-regression checks
  -> reviewer signoff and residual risk
```
A model-generated CWE-89 label is a hypothesis. A taint-analysis rule asks whether untrusted input can flow into a query **structure** without sanitization appropriate to that sink. The same string can be safe when bound as a parameter and unsafe when interpolated into SQL syntax. Escaping is dialect-dependent and inferior to binding for values. Parameter binding does not authorize database access or safely parameterize SQL identifiers.

**Counterexamples:** (i) a syntactically concatenated query with a constant trusted value may be unreachable from attackers; (ii) a query using bound values but dynamically concatenated user-selected table name remains risky; (iii) a passing test for one quote-containing username cannot prove all code paths safe. Record line-level evidence, reachability assumptions, tests, and unresolved uncertainty. Never run untrusted code outside a sandbox.

## 5. Bayesian base rates and synthetic-to-real calibration
Define π=P(V) as true vulnerability prevalence, s=P(flag|V) sensitivity, f=P(flag|not V) false-positive rate. Bayes gives
P(V|flag)=sπ/[sπ+f(1−π)].
For π=.01, s=.90, f=.05: numerator .009, denominator .0585, precision ≈.153846. A model with 90% recall can produce mostly false alerts at low base rates. An evaluation set balanced 50/50 can conceal this. Confidence intervals should reflect grouping by repository and vulnerability family, not assume independent near-duplicate snippets.

For synthetic prior π_s and real target prior π_t, assume P_s(x|y)=P_t(x|y) for each class y (the strong **label-shift** assumption). By Bayes, posterior odds satisfy
O_t(x)=O_s(x)·[π_t/(1−π_t)]/[π_s/(1−π_s)].
If synthetic posterior p_s=.8, π_s=.5, π_t=.01, O_s=4, factor=1/99, O_t=4/99 and p_t=4/103≈.038835. A high synthetic score is not deployment confidence. This correction is invalid when the synthetic generator changes attack features, correlations, sequence causality or selection mechanisms. If the generator memorizes source templates, group leakage may invalidate even held-out synthetic scores.

**Counterexample:** train on synthetic SQLi samples with the token `SELECT` always present in vulnerable examples; test on same generator. A classifier learns a superficial token. On real code where benign SQL queries also contain `SELECT`, precision collapses. Evaluate on independent projects and incident families with version and time splits; use a real authorized holdout when available. Synthetic data are not automatically anonymous; check memorization, membership inference, consent, retention and data provenance. Differential privacy needs a formally specified mechanism and privacy accounting.

## 6. Rigorous range protocol and failure analysis
A defensible range uses isolated networking, fake secrets, pinned image digests, resource limits, reset snapshots, telemetry, and a pre-registered ground truth. Authoring and evaluation must be separated. Evaluate findings by (a) line localization, (b) source-to-sink explanation, (c) verified weakness under stated assumptions, (d) patch correctness, (e) functional regression, (f) false-positive burden, and (g) reproducibility across independent seeds and projects. Human approval is required for operational changes.

**Advanced Lab B — parameter binding, base rates, and leakage trap.** Save as `cyber_evening.py`; run `python cyber_evening.py`. This is fully local and defensive.
```python
import sqlite3
db=sqlite3.connect(":memory:")
db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
db.executemany("INSERT INTO users VALUES (?,?)",[(1,"alice"),(2,"O'Reilly")])
def lookup(name):
    return db.execute("SELECT id FROM users WHERE name=?",(name,)).fetchall()
assert lookup("O'Reilly")==[(2,)]
assert lookup("nobody")==[]
print("bound_values_pass",True)
pi,s,f=.01,.9,.05
precision=s*pi/(s*pi+f*(1-pi))
assert abs(precision-.009/.0585)<1e-12
print("base_rate_precision",round(precision,6))
ps,pis,pit=.8,.5,.01
odds=(ps/(1-ps))*((pit/(1-pit))/(pis/(1-pis)))
pt=odds/(1+odds)
assert abs(pt-4/103)<1e-12
print("shifted_posterior",round(pt,6))
# Same incident family across splits is a leakage warning.
train={"family-A","family-B"}
test={"family-B","family-C"}
assert train & test == {"family-B"}
print("family_leakage_detected",bool(train & test))
```
Expected: `bound_values_pass True`, `base_rate_precision 0.153846`, `shifted_posterior 0.038835`, `family_leakage_detected True`. This does not execute a real exploit, establish real-world detector quality or demonstrate privacy guarantees. Extension: implement group-disjoint split checks and a bootstrap by repository; measure patch correctness on authorized toy services.

## 7. Critical reading of primary literature
**Graves et al. (2006), CTC, DOI 10.1145/1143844.1143891:** major contribution is exact marginalization over monotonic alignment paths using dynamic programming. Critique: conditional independence and monotonicity do not resolve overlapped speech, speaker identity, or high-cost command semantics. Research extension: risk-weighted token objectives with calibrated abstention.
**Radford et al. (2023), Whisper, arXiv:2212.04356:** large-scale weak supervision improves broad ASR robustness; benchmark WER cannot establish safety for rare negations or noisy security commands. Research extension: adversarially stratified command benchmarks with provenance and error severity.
**Pearce et al. (2022), Asleep at the Keyboard?, arXiv:2108.09293:** demonstrates security evaluation of generated code under selected prompts. Critique: prompt/model/version specificity and generated-code focus limit generalization to current code-review assistants. Research extension: versioned, source-to-sink, human-adjudicated benchmarks.
**NIST AI RMF 1.0 (2023), DOI 10.6028/NIST.AI.100-1:** useful governance framework, not evidence that a particular system is secure. Tie governance controls to measured test artifacts.
**Ho et al. (2020), DDPM, arXiv:2006.11239; Rombach et al. (2022), LDM, arXiv:2112.10752:** prior-day generative context only; neither is evidence that synthetic media are authentic or cyber ranges representative.

## 8. Doctoral research designs
**Study A: consequence-aware ASR for security commands.** Hypothesis: calibration plus negation-sensitive selective prediction lowers cost-weighted errors at a fixed coverage. Data: consented synthetic spoken commands across accents/noise and independently recorded authorized holdout. Baselines: raw ASR, confidence threshold, n-best verification. Outcomes: WER, critical-token error, risk-coverage area, false refusal, latency. Pre-register costs and speaker-disjoint splits. Threat: generated speech does not reproduce real acoustic conditions.
**Study B: phase-aware multimodal evidence.** Hypothesis: jittered video sampling with trusted clock alignment improves detection of short events at fixed bandwidth. Baselines: fixed 1 fps, random jitter, higher-rate bursts. Outcomes: missed-event rate, timestamp residual, false event attribution. Evaluate clock drift and occlusion. Never infer identity from ambiguous video.
**Study C: leakage-resistant AI code review.** Hypothesis: source-to-sink explanations and independent range validation reduce false positives relative to model-only labels. Split by project, family and time before generating synthetic cases; keep held-out real authorized samples isolated. Report per-CWE performance, patch tests, reviewer hours, calibration and privacy audit. Threat: hidden overlap and synthetic artifacts.

## 9. Progressive oral Socratic examination (answer, then follow-up)
1. **Why marginalize CTC paths?** The transcript has many valid frame alignments; sum their probabilities. *Follow-up:* why is greedy path decoding insufficient?
2. **Why insert blanks?** They allow adjacent identical output symbols to remain distinct. *Follow-up:* trace A–blank–A.
3. **What does α_t(s) mean?** Probability mass of valid prefixes ending in expanded state s at frame t. *Follow-up:* explain the skip constraint.
4. **What is the computational improvement?** O(TU) dynamic programming rather than path enumeration. *Follow-up:* where does log-sum-exp enter?
5. **Compute two-frame P(A).** 1−.4×.3=.88. *Follow-up:* calculate −ln(.88).
6. **What is posterior occupancy?** Probability of frame-symbol alignment conditioned on audio and transcript. *Follow-up:* derive logit gradient sign.
7. **Why can WER mislead?** Errors have unequal consequences. *Follow-up:* score a deleted “not”.
8. **Does 40 Hz bin spacing imply 40 Hz true resolution?** No; window and signal assumptions determine resolvability. *Follow-up:* what does zero-padding change?
9. **Why can 100 repeated fixed samples miss every event?** Phase-locked periodic events fall between samples. *Follow-up:* what does jitter buy?
10. **Can one synchronization marker estimate drift?** No, only offset under assumptions. *Follow-up:* what extra observations are needed?
11. **Does hashing video authenticate its origin?** No; only integrity relative to trusted capture and custody. *Follow-up:* define trust root.
12. **When should spoken action abstain?** When uncertainty or consequence exceeds approved thresholds. *Follow-up:* distinguish abstention from silent correction.
13. **What is a taint source?** An origin of untrusted data. *Follow-up:* name a SQL sink.
14. **Why bind SQL values?** It separates values from query syntax. *Follow-up:* what about dynamic table names?
15. **Does CWE-89 prove exploitability?** No; reachability and control must be established. *Follow-up:* provide a false-positive example.
16. **What is the base-rate formula?** sπ/[sπ+f(1−π)]. *Follow-up:* compute for π=.01.
17. **Why mostly false flags despite 90% sensitivity?** Low prevalence and nonzero FPR dominate. *Follow-up:* which parameter most helps review burden?
18. **What assumption enables prior correction?** Stable class-conditional features. *Follow-up:* invent a synthetic counterexample.
19. **Why is p_s=.8 not deployment p=.8?** Different class priors; here adjusted ≈.038835. *Follow-up:* derive odds factor.
20. **Why split before synthesis?** Prevent memorized incidents contaminating evaluation. *Follow-up:* define incident-family leakage.
21. **Does synthetic mean private?** No; memorization and inference remain possible. *Follow-up:* what would formal DP require?
22. **What makes a range lawful?** Authorization, isolation, safe targets and controlled artifacts. *Follow-up:* list evidence for reproducibility.
23. **Can a passing patch test prove universal security?** No; finite tests cover only tested behavior. *Follow-up:* what residual risk remains?
24. **How should findings be adjudicated?** Source-to-sink evidence, independent tests, severity and human review. *Follow-up:* what if reviewers disagree?
25. **What is a valid ASR research baseline?** Raw model with fixed evaluation protocol. *Follow-up:* what data split prevents speaker leakage?
26. **How evaluate a safety gate?** Risk-coverage curve plus false refusal and latency. *Follow-up:* what costs must be preregistered?
27. **How test video aliasing?** Vary event phase and sampling policy at equal bandwidth. *Follow-up:* control occlusion and clock drift.
28. **What invalidates a synthetic-code benchmark?** Template leakage, unrealistic distributions, weak ground truth. *Follow-up:* propose an independent holdout.

## Recall and source map
Recall tomorrow: (1) write CTC skip condition from memory; (2) compute P(A)=.88; (3) explain fixed-phase aliasing; (4) distinguish transcript, original waveform and authenticated capture; (5) calculate base-rate precision; (6) derive prior odds correction; (7) explain source-to-sink proof; (8) describe group-disjoint synthetic evaluation.

|Major claim|Self-contained local teaching|Primary/official source|
|---|---|---|
|CTC path sum, recurrence, gradients|§1 and Lab A|R1|
|ASR limitations and command-specific evaluation|§1, §3, Study A|R2|
|Sampling, temporal aliasing, clock drift|§2 and Lab A|Explicit mathematical assumptions and demonstration|
|AI code review evidence and generalization|§4–6, Lab B, Study C|R3, R4|
|Base-rate and label-shift calculations|§5, Lab B|Derivation here; not an empirical paper claim|
|Synthetic provenance and governance|§6–8|R4|
|Generative outputs not authenticated captures|§3, §7|Logical provenance distinction; R5/R6 context only|

References, stable links, access rights and paper index: [companion evening references](references/index.md). **Validation status:** both labs are standard-library-only and include deterministic assertions/expected outputs; do not claim full execution until independently run. No research PDF redistribution is claimed without verified license. No email, narration, audio generation, real-world security tests or operational changes.
