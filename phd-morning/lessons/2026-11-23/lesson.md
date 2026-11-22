# Morning PhD lesson — Monday 23 November 2026 (Asia/Kolkata)
**Generative AI Unit 5 Session 4: Speech, audio and video. AI for Cybersecurity Unit 5 Session 4: Code review, cyber ranges and synthetic data.**

## Provenance and learning contract
Syllabus sources read directly from attachments to the 19 October 2026 Gmail message *PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027*: `Advanced_Generative_AI_and_Foundation_Models_Learning_Pathway.pdf`, page 12, Unit 5 Session 4; `AI_for_Cybersecurity_Learning_Pathway.pdf`, page 12, Unit 5 Session 4. Continuity: [22 Nov morning](../../2026-11-22/lesson.md) on diffusion/conditioning and controlled SOC agents; [22 Nov evening](../../../../phd-evening/lessons/2026-11-22/lesson.md) on diffusion posteriors and execution-time policy. Today's connection: generated sound/video are *outputs*, not authenticated records; synthetic cyber incidents are *training/test artifacts*, not real attacks or proof of deployment performance. No production tools or live systems are touched.

## Spoken lesson routes
**10-minute walk:** Explain ASR versus TTS; derive WER on one sentence; explain frame aliasing; then trace one vulnerable-code review and synthetic-range provenance. Oral questions 1–4, 13–16.
**30-minute walk:** Include CTC recurrence intuition, timestamps and synchronization, code-review evidence hierarchy, label leakage and domain shift; work examples verbally; questions 1–9 and 13–22.
**Extended 80–100 minutes:** Derivations, both labs when stationary, case discussions, paper critiques, research designs and all oral questions. Never read or run code while walking.

## Prerequisites and vocabulary
Recall conditional probability, sequence models, attention, log likelihood, dynamic programming, precision/recall, confusion matrices, authorization, synthetic-data provenance, train/test leakage. A waveform is a sampled pressure signal `x[n]` at sample rate `f_s` Hz. A spectrogram is a time-frequency representation, usually from the short-time Fourier transform (STFT). **ASR** maps audio to text; **TTS** maps text to speech; **speech-to-speech** may cascade ASR→language model→TTS or use joint audio-token models. **Diarization** estimates who spoke when; it does not establish identity. **WER** counts substitutions, deletions and insertions divided by reference word count. **Frame sampling** extracts images from a temporal stream; **temporal aliasing** occurs when sampling misses or misrepresents rapid changes. **Cyber range** is a controlled, isolated environment for training and evaluation. **Synthetic data** is artificially generated; it can preserve or distort causal relations. **CWE** is a weakness category, **CVE** a disclosed vulnerability identifier; neither alone proves exploitability in a particular deployment. **Data leakage** is contamination of evaluation with information unavailable in the intended setting.

# Part A — Speech, audio and video

## A1. Why audio is not merely text
A spoken request carries lexical content, timing, pauses, speaker turns, noise, accents and potentially paralinguistic cues. ASR transcribes words, often discarding timing and uncertainty. A transcript cannot reliably establish who spoke, whether a command was interrupted or whether the sound was synthetic. A voice tutor must distinguish recognition error from reasoning error: if ASR hears “fifteen” as “fifty,” a flawless downstream calculation can still be wrong. In security, never authenticate a person from a claimed voice alone.

**Architecture:**
```text
microphone -> PCM samples -> framing/STFT or learned audio encoder
                                |
                        ASR token probabilities -> transcript + timestamps
                                |                    |
                       uncertainty checks <---------+
                                |
                        dialogue/reasoning controller
                                |
                      response text + safety checks
                                |
                    TTS acoustic model / vocoder -> audio
Video camera -> timestamped frames -> visual encoder -> temporal fusion --^
Trusted metadata (clock, provenance, consent) ---------------------------^
```
The controller should retain modality provenance: a text transcript is not equivalent to the original waveform; a generated video frame is not a captured camera frame.

## A2. Sampling and STFT derivation
For continuous signal `s(t)`, uniform sampling gives `x[n]=s(n/f_s)`. The Nyquist frequency is `f_s/2`: an ideal band-limited signal below it can be reconstructed under sampling-theorem assumptions. This does *not* guarantee speech intelligibility in noise, nor correct inference of events between frames. For a length-`N` window `w[m]`, hop `H`, frame index `k`, frequency bin `r`:
`X[k,r]=sum_{m=0}^{N-1} x[kH+m]w[m] exp(-j2πrm/N)`.
The magnitude `|X[k,r]|` captures local spectral energy; phase may matter for reconstruction. A 16 kHz signal with 400-sample windows spans 25 ms; a hop of 160 samples advances 10 ms. Time-frequency resolution trades off: longer windows distinguish close frequencies better but blur abrupt timing.

**Worked example:** 3.2 seconds at 16 kHz contains 51,200 samples. A non-padded 400-sample window with hop 160 produces `floor((51200-400)/160)+1=318` frames. With padding or centered windows, the count changes: always specify conventions.

## A3. Sequence alignment and CTC
Speech and transcripts differ in length. Connectionist Temporal Classification (CTC) [R1] introduces a blank token and sums probabilities over framewise paths that collapse to a transcript. For target text `y`, paths `π`, collapse map `B` (remove repeats separated by no blank, then blanks), frame probabilities `p_t`:
`P(y|x)=sum_{π:B(π)=y} product_{t=1}^T p_t(π_t|x)`.
Training minimizes `L_CTC=-log P(y|x)`. Dynamic programming shares prefixes to avoid enumerating exponentially many paths. With three frames, paths `[A,blank,B]` and `[A,A,B]` both collapse to “AB”; `[A,B,B]` does too. By contrast `[A,A]` collapses to “A,” not “AA”; a blank between repeated letters is required for “AA”. CTC makes a conditional-independence assumption across output frames given encoder representations; attention encoder-decoders and transducers use different factorization and latency tradeoffs.

**When it works/fails:** CTC is useful for monotonic alignment with more frames than tokens; it struggles when input is severely clipped, speaker overlap is unresolved, or lexical context requires strong language modeling. Beam search and external language models may improve words but can also hallucinate plausible text unsupported by sound.

## A4. WER and the difference between transcription and truth
Let `R=(r_1,...,r_N)` be reference words and `H=(h_1,...,h_M)` a hypothesis. Edit distance recurrence:
`D(i,j)=min[D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+1_{r_i≠h_j}]`
with `D(i,0)=i`, `D(0,j)=j`. Then `WER=(S+D+I)/N=D(N,M)/N`, where S,D,I are optimal edit operations. WER can exceed 100% due to insertions; normalization choices (case, punctuation, numbers) materially change it.
**Example:** reference “isolate host after review” versus hypothesis “isolate host now” needs one substitution (“after”→“now”) and one deletion (“review”): `WER=2/4=0.5`. A 50% WER is not an instruction to isolate a host; the original waveform and approval must be checked.

## A5. TTS, voice agents and video timing
A TTS system predicts acoustic features or waveform tokens conditioned on text, then synthesizes sound; modern systems may use neural vocoders or generative acoustic models. Naturalness and intelligibility are distinct from factuality, consent, impersonation resistance and speaker verification. A cascaded spoken tutor has separately measurable ASR WER, dialogue correctness, TTS intelligibility, latency and interruption handling. A speech model producing fluent output can still invent a source.

For video, let event duration be `d` seconds and frames be sampled every `Δ` seconds. If event start is uniformly distributed relative to sampling phase and `d≤Δ`, the probability at least one instantaneous sample lands in the event is `d/Δ` under ideal periodic sampling. Thus a 0.1-second event sampled once per second is observed only about 10% of the time under those assumptions. If `d>Δ`, detection is guaranteed only for ideal point samples and uninterrupted capture; real shutter, occlusion and dropped frames invalidate that simplification. Frame order and timestamps matter: two frames “door open” and “door closed” do not prove the intervening action or actor. Audio/video sync needs a common clock and measured offset/drift; aligning by file index is unsafe.

**Real-world-style case:** A security camera sampled at 1 fps and an ASR transcript say “disable the alarm.” An operator actually said “do not disable the alarm,” with the “not” masked by noise. The video misses a short gesture. The correct system flags uncertainty, requests the original high-rate clip, preserves timestamps and avoids automated containment. This is a constructed teaching case, not a reported incident.

## A6. Lab A — WER, STFT framing and missed-event simulation (standard library)
Save as `speech_video_lab.py`, run `python speech_video_lab.py`.
```python
import math
def edits(ref,hyp):
    a,b=ref.split(),hyp.split()
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):
            new.append(min(new[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=new
    return d[-1],len(a)
e,n=edits("isolate host after review","isolate host now")
assert (e,n)==(2,4)
print("WER",round(e/n,3))
fs,seconds,N,H=16000,3.2,400,160
samples=round(fs*seconds)
frames=(samples-N)//H+1
assert (samples,frames)==(51200,318)
print("samples",samples,"frames",frames)
# deterministic phase-grid experiment: event occupies [phase,phase+d)
d,interval=0.1,1.0
phases=[(i+0.5)/10000 for i in range(10000)]
hits=sum(1 for phase in phases if phase+d>=interval)
print("sample_hit_fraction",round(hits/len(phases),3))
assert abs(hits/len(phases)-.1)<.001
```
Expected:
```text
WER 0.5
samples 51200 frames 318
sample_hit_fraction 0.1
```
This is not a real ASR model, camera benchmark or TTS synthesis. **Extensions:** add a “not” deletion cost analysis; compare frame rates 1, 5, 25 fps under a fixed 0.1-second event; test window/hop choices; add timestamps and a transcript confidence policy. For real recordings, secure consent and retain original hashes.

# Part B — Cybersecurity code review, cyber ranges and synthetic data

## B1. What AI code review can and cannot establish
An LLM can propose suspicious data flows, identify likely CWE categories, write unit tests and summarize a patch. But “this code is vulnerable” requires checking actual reachability, trust boundaries, deployed configuration and exploitability. Static analysis often has high coverage and false positives; dynamic tests establish behavior for tested inputs, not universal safety. A secure review pipeline combines parser/linters, dataflow analysis, dependency inventory, threat modeling, tests and human review. Never execute an unknown repository or untrusted payload in a personal environment.

**Architecture:**
```text
version-pinned source -> parser / static checks -> candidate findings
             |                  |                      |
      dependency/SBOM     source locations       model-assisted explanation
             |                  |                      |
             +------------------+----------------------+
                                |
                 deterministic reproduction in isolated range
                                |
                 evidence + severity + false-positive review
                                |
                  patch proposal -> tests -> human approval
                                |
                 tracked remediation / residual-risk record
```
AI-generated comments are hypotheses; tests and traceable source locations are evidence.

## B2. Worked code-review example: SQL injection
Consider:
```python
def find_user(conn, username):
    sql = "SELECT id FROM users WHERE name = '" + username + "'"
    return conn.execute(sql).fetchall()
```
If an untrusted string reaches this concatenation, the SQL parser can interpret input as syntax rather than data: a CWE-89 SQL-injection pattern. The correct mitigation is parameter binding:
```python
def find_user_safe(conn, username):
    return conn.execute("SELECT id FROM users WHERE name = ?", (username,)).fetchall()
```
Parameterization separates SQL structure from value. It does not authorize access, solve unsafe dynamic table identifiers, or prove the surrounding system secure. An LLM that suggests escaping quotes without identifying driver-specific rules may miss encoding and dialect issues. **Counterexample:** if the username is a compile-time constant not controlled by a user, the unsafe pattern may be non-exploitable in that exact call path; code review must identify taint sources and sinks.

## B3. Formal evaluation and base-rate traps
Let TP be real vulnerabilities correctly flagged, FP benign code flagged, FN missed vulnerabilities, TN correctly rejected benign code. `precision=TP/(TP+FP)`, `recall=TP/(TP+FN)`, `FPR=FP/(FP+TN)`. A reviewer seeing 100 findings with 10 true vulnerabilities has 10% precision even if the model sounds confident. On a 1%-vulnerable corpus, sensitivity 90% and specificity 95% give
`P(vuln|flag)=0.9·0.01/[0.9·0.01+0.05·0.99]≈0.15385`.
Thus about 15.4% of flagged examples are truly vulnerable under the stated base rate and assumptions. Different weakness types and project contexts require stratified reporting. Report exact-match CWE, source-line localization, evidence sufficiency, fix correctness and regression behavior separately. A synthetic benchmark is not a production deployment result.

## B4. Cyber-range design and evidence discipline
A range should be isolated, resettable, resource-limited, instrumented and legally authorized. Use harmless toy services, synthetic accounts and fake secrets; no production credentials, public attack targets or real customer data. Freeze source version, dependency hashes, operating-system image and evaluator. Record the expected weakness, trigger conditions, mitigation and observable outcome. Separate authoring and evaluation: if an LLM generated both the vulnerable exercise and the answer key, an evaluation can reward stylistic familiarity rather than generalization. Test unseen projects and time-separated vulnerabilities. Measure whether a proposed patch removes the issue without breaking intended functionality.

**Case:** A toy login API uses string-concatenated SQL. A code assistant flags CWE-89, proposes parameter binding, and tests a benign username plus a quoted username. The range records source line, SQLite query behavior, patch diff and regression tests. The lesson does not run an exploit against a real target.

## B5. Synthetic data, utility and privacy
Synthetic security logs can balance rare event classes and exercise parsers. They can also encode impossible sequences: a “successful login” before account creation, negative byte counts, timestamps out of order, or unrealistic base rates. Distinguish three axes:
1. **Fidelity:** distributions, correlations and causal constraints resemble the intended environment.
2. **Utility:** downstream detector performance on independent *real* held-out data, if lawfully available.
3. **Privacy:** whether records expose or permit inference about original individuals; synthetic does not imply private.
A generator trained on the same real incidents as the held-out test can leak templates and labels. Split by entity, time and incident family *before* training the generator. Membership inference and nearest-neighbor checks are risk assessments, not privacy proofs. Differential privacy offers formal guarantees only for correctly specified algorithms and privacy accounting, not merely because generated records look different.

**Mathematical shift example:** Suppose a detector sees 50% malicious synthetic training examples but 1% malicious real-world examples. Even if class-conditional likelihoods were stable, its raw posterior is miscalibrated under prior shift. Let training prior `π_s`, deployment prior `π_t`, training posterior `p_s(y=1|x)`. Under pure prior shift:
`odds_t(x)=odds_s(x) · [π_t/(1-π_t)]/[π_s/(1-π_s)]`.
For `p_s=.8`, `π_s=.5`, `π_t=.01`: training odds 4, prior factor 1/99, deployment odds 4/99, and `p_t≈0.03883`. This adjustment fails if class-conditional feature distributions differ, as they often do for synthetic logs. Do not “correct” all domain shift with prior adjustment alone.

## B6. Lab B — isolated SQLite safety and synthetic prior-shift calculation
Save as `cyber_range_lab.py`; run `python cyber_range_lab.py`. No network or real accounts.
```python
import sqlite3
db=sqlite3.connect(":memory:")
db.execute("CREATE TABLE users(id INTEGER, name TEXT)")
db.executemany("INSERT INTO users VALUES (?,?)",[(1,"alice"),(2,"O'Reilly")])
def safe_lookup(name):
    return db.execute("SELECT id FROM users WHERE name = ?",(name,)).fetchall()
assert safe_lookup("alice")==[(1,)]
assert safe_lookup("O'Reilly")==[(2,)]
assert safe_lookup("nobody")==[]
print("parameterized_lookup_passed",3)
p_s,pi_s,pi_t=.8,.5,.01
odds_s=p_s/(1-p_s)
factor=(pi_t/(1-pi_t))/(pi_s/(1-pi_s))
odds_t=odds_s*factor
p_t=odds_t/(1+odds_t)
assert abs(p_t-4/103)<1e-12
print("adjusted_probability",round(p_t,6))
tp,fp,fn=9,50,1
print("precision",round(tp/(tp+fp),6),"recall",round(tp/(tp+fn),6))
```
Expected:
```text
parameterized_lookup_passed 3
adjusted_probability 0.038835
precision 0.152542 recall 0.9
```
**Extensions:** add property-based tests for quote and Unicode inputs, but do not introduce unsafe string-built queries into a live service. Simulate timestamp constraints, rare-event prevalence and entity-group splits; compare a classifier trained on synthetic versus held-out real data only with proper permission. Log provenance and deletion policies.

## B7. Misconceptions, failure modes and ethics
“Fluent transcription means accurate recognition”: false; inspect WER and audio. “A high-frame-rate camera proves all events”: false; occlusion and provenance remain. “Synthetic equals anonymous”: false; memorization and membership inference are possible. “AI found CWE-89, so exploitation is proven”: false; establish reachability and trust boundary. “A patched test passed, so system is secure”: false; tests sample behavior. “Generated video is forensic evidence”: false; preserve original sensor capture and cryptographic custody. Avoid unauthorized voice cloning, biometric inference, employee surveillance and offensive cyber testing. Disclose synthetic media and retain lawful provenance.

# Paper critiques and research extensions
**[R1] Graves et al. (2006)** supplies CTC alignment machinery; its assumptions do not resolve multi-speaker overlap or semantic ambiguity.
**[R2] Radford et al. (2023)** shows scalable weakly supervised ASR (Whisper); benchmark accuracy does not prove domain-specific medical/security command accuracy or immunity to hallucinated transcriptions.
**[R3] Ho et al. (2020)** and **[R4] Rombach et al. (2022)** from yesterday ground generative diffusion, but neither provides evidence authenticity guarantees for audio/video or cyber artifacts.
**[R5] Pearce et al. (2022)** examines security of AI code suggestions; findings depend on task prompts and model versions, and do not measure today's deployed assistants.
**[R6] NIST AI RMF (2023)** provides governance vocabulary, not certification of a synthetic dataset or assistant.

Research proposal 1: **Negation-preserving spoken SOC commands.** Collect consented, balanced synthetic utterances with “do/do not” contrasts, accents and noise; measure command-critical word errors, abstention and operator confirmation under matched latency. Baseline: deterministic confirmation of all destructive verbs.
Research proposal 2: **Event-preserving video sampling.** Randomize short-event phase and duration, compare uniform sampling, motion-adaptive sampling and timestamp-aware fusion. Evaluate event recall and false attribution at fixed compute, using held-out camera scenes.
Research proposal 3: **Synthetic-range transfer gap.** Generate toy code and log incidents with explicit causal constraints; split by project, time and generator family; evaluate static tools, LLM review and combined pipelines on independent, licensed real-world examples. Measure precision, localization, patch regression, leakage and calibration, with confidence intervals. Do not claim transfer without external evaluation.

# Progressive oral Socratic exam (answers + follow-up prompts)
1. **ASR versus TTS?** Audio→text versus text→audio. *Follow-up:* Where does a dialogue model sit?
2. **Why retain raw waveform?** Transcript discards uncertainty/timing; forensic provenance. *Follow-up:* What hash/clock metadata?
3. **Nyquist at 16 kHz?** 8 kHz under band-limited assumptions. *Follow-up:* Why is it not a quality guarantee?
4. **Window and hop?** Local analysis length and advance. *Follow-up:* 400/160 at 16 kHz equals 25/10 ms.
5. **CTC blank purpose?** Allows flexible monotonic alignment and repeated characters. *Follow-up:* Why must “AA” have separation?
6. **CTC probability?** Sum of framewise path products collapsing to target. *Follow-up:* Why dynamic programming?
7. **What does WER measure?** Word edit distance divided by reference length. *Follow-up:* Can it exceed one? Yes.
8. **Compute WER for 2 edits/4 words.** 0.5. *Follow-up:* Does that quantify safety of negation? Not directly.
9. **Why separate ASR and reasoning evaluation?** Different failure sources. *Follow-up:* Design an error-attribution matrix.
10. **How does TTS fluency relate to truth?** It does not establish truth. *Follow-up:* How to verify source claims?
11. **Why can 1 fps miss 0.1 s?** Phase-dependent sampling; ideal 10% hit probability. *Follow-up:* When does assumption fail?
12. **How to synchronize modalities?** Common timestamps, offset and drift calibration. *Follow-up:* What if clocks reset?
13. **CWE versus CVE?** Weakness category versus disclosed vulnerability record. *Follow-up:* Does either prove local exploitability?
14. **What's the SQL flaw?** Untrusted data concatenated into query syntax. *Follow-up:* Trace source to sink.
15. **Why parameter binding?** Separates SQL code from data. *Follow-up:* Does it authorize access? No.
16. **Can an LLM be the sole reviewer?** No; need tests, provenance and independent validation. *Follow-up:* What is the false-positive cost?
17. **Precision versus recall?** TP/(TP+FP) versus TP/(TP+FN). *Follow-up:* Which matters for analyst fatigue?
18. **Why low precision at low prevalence?** Base-rate effect. *Follow-up:* Derive Bayes denominator.
19. **Why isolate a cyber range?** Prevent unintended effects, ensure reproducibility. *Follow-up:* How to reset?
20. **Why pin dependencies?** Reproducible behavior and supply-chain review. *Follow-up:* What about image hashes?
21. **Synthetic fidelity versus utility?** Resemblance versus real-task benefit. *Follow-up:* Can high fidelity coexist with low utility?
22. **Synthetic privacy guaranteed?** No. *Follow-up:* How might memorization be tested?
23. **Why split before generation?** Prevent incident/entity leakage. *Follow-up:* What if prompts copy test data?
24. **Prior-shift correction assumption?** Stable class-conditionals, changed class prior. *Follow-up:* What if feature distribution also shifts?
25. **Why does 0.8 become ~0.039?** Prior odds drop from 1 to 1/99. *Follow-up:* What calibration would you measure?
26. **Does patch success prove safety?** No; finite tests and unknown paths. *Follow-up:* Which residual risk remains?
27. **What is a strong baseline?** Static analysis + deterministic tests + human review. *Follow-up:* Compare under fixed budget.
28. **What is your doctoral hypothesis?** State falsifiable effect, baseline, independent split and metrics. *Follow-up:* What result would refute it?

## Recall after 24 hours
Without notes: derive CTC path sum; compute a 2/4 WER; state video event-hit probability assumptions; distinguish ASR, TTS and authentication; explain CWE versus CVE; show why parameter binding works; derive the prior-shift odds correction; name two synthetic-data leakage paths and a held-out evaluation protocol.

## Source map and verification boundaries
- Audio sampling, STFT, CTC and WER: A2–A4; CTC [R1]; ASR architecture/limitations [R2]. Sampling and edit-distance derivations are explicitly worked here.
- Generated media is not authentic evidence: A1/A5/B7, risk reasoning grounded in [R6], not a claim from a diffusion paper.
- Code review, parameterization and metrics: B1–B3, B6; empirical motivation [R5]; SQL parameter binding demonstrated with SQLite.
- Cyber range and synthetic transfer/privacy: B4–B7; original teaching design, governance [R6]. Numerical prior-shift calculation derived in B5.
- Yesterday's generative diffusion references [R3,R4] are continuity, not direct evidence for ASR.
All records, attacks and scenarios here are fabricated and defensive. No real recordings, malware or live targets. Citation links and access notes: [companion references index](references/index.md). No claim of independently reproducing published benchmark results.
