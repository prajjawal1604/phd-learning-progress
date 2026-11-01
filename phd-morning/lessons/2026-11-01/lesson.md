# PhD Morning Learning — 1 November 2026

PhD Morning Learning — Day 1 | 1 November 2026

Advanced Generative AI and Foundation Models + AI for Cybersecurity

This is the first counted lesson; the 31 October edition was a preview. This file is reconstructed from the actual November 1 morning email for a full-content narration test.

A. Generative AI: Discriminative versus generative learning

A discriminative model predicts a label y from an input x, estimating P(y|x). A generative model represents a distribution over data P(x), a joint distribution P(x,y), or a conditional output distribution such as P(reply|message). For the message “My login is failing after a reset,” classification may estimate whether the issue is urgent; generation may draft a support reply. The same pretrained network can support both tasks.

Conditional probability obeys P(y|x)=P(x,y)/P(x) when P(x)>0, and Bayes’ rule gives P(y|x)=P(x|y)P(y)/P(x). These equations make base rates important: a phishing classifier tested on a balanced dataset may behave differently in a real inbox with far fewer phishing messages.

An autoregressive language model factors a sequence of tokens into conditional predictions:

P(t1,…,tN) = product over i of P(ti | t1,…,t(i-1)).

In “the server is healthy,” the model predicts each next token from the preceding tokens. During training the next token already appears in the source text, making this self-supervised learning. Maximum-likelihood training minimizes negative log likelihood:

NLL = -(1/N) sum_i log P_theta(ti | preceding tokens).

If two observed next tokens receive probabilities 0.5 and 0.25, average NLL is approximately 1.040 nats per token; perplexity exp(NLL) is approximately 2.83. Perplexity is a measure of predictive uncertainty on a fixed tokenizer and evaluation distribution. It is not a count of factual errors and should not be compared blindly across tokenizers.

A generative model may output a plausible but unsupported claim. Fluency, likelihood and factual correctness are distinct. Reliable applications add retrieval, tools, source checks, or human verification. A foundation model is broadly pretrained and adapted for multiple downstream tasks; an instruction-tuned model and an application built around it are different layers of the system.

Worked bigram example: A corpus contains “alert resolved”, “alert pending”, “ticket resolved”. A unigram baseline ignores context; a bigram model estimates P(next|previous). With add-one smoothing, P(resolved|alert)=(count(alert,resolved)+1)/(count(alert)+V), where V is vocabulary size. This weak but interpretable baseline tests whether complex methods offer a measurable improvement.

Lab A (25 minutes): Write 20–30 reusable or self-authored short documents. Split by document before forming overlapping token windows. Compare character and word tokenization on English, technical identifiers, numbers and multilingual text. Train unigram and add-one-smoothed bigram models on training documents only. Explicitly handle unknown tokens. Measure held-out NLL and perplexity using the same tokenizer, generate five seeded samples, and describe repetition and unsupported content. Vary training-document count while keeping the test set fixed. Record a corpus card, split manifest, model settings and failures.

Research challenge: Does lower perplexity correlate with grounded factual answers? What data contamination or duplicate text could inflate results? What additional evidence would establish a causal explanation?

B. AI for Cybersecurity: Threats and telemetry

Begin with confidentiality (prevent unauthorized disclosure), integrity (prevent unauthorized alteration) and availability (keep services usable). Phishing, credential abuse, malware, denial of service and insider misuse threaten different assets. Choose the prediction target and available evidence before choosing a model.

Telemetry: Network flows summarize communication metadata such as endpoints, protocol, duration, bytes and packets; encrypted payloads may remain invisible. Authentication logs capture account, login outcome, time, device and source. Process/endpoint events describe process creation, file operations and related host activity. DNS logs show name resolution, not proof of compromise. A model cannot reliably detect a threat for which the necessary evidence is absent.

Suppose an account has 80 failed logins followed by a successful login from a new device. Authentication events may support a credential-abuse investigation; flows alone may not identify the user. Define whether the model predicts abuse at the time of the success, within the next five minutes, or retrospectively. Only features available by the chosen decision time are valid. Using post-confirmation information would leak the answer.

Learning paradigms: Supervised learning trains on attack/benign labels. Unsupervised learning scores unusual activity without proving it malicious. Semi-supervised learning combines limited labels with unlabeled telemetry. A software update can appear anomalous while being harmless. Start with simple baselines before adding deep learning.

Why accuracy is misleading: Consider 10,000 events with 100 attacks and 9,900 benign events. A detector correctly flags 80 attacks, misses 20, and falsely flags 198 benign events: TP=80, FN=20, FP=198, TN=9,702. Precision=80/(80+198)=28.8%; recall=80/(80+20)=80%; false-positive rate=198/9,900=2%; accuracy=(80+9,702)/10,000=97.82%. An all-benign classifier achieves 99% accuracy but finds no attacks. The useful detector produces 278 alerts, which may overwhelm a team able to review only 50.

If prevalence falls to 0.1% while recall and false-positive rate remain constant, then among 100,000 events TP=80 and FP=1,998, so precision drops to approximately 3.85%. A published benchmark’s precision does not automatically transfer to deployment. Report prevalence, recall, precision, false-positive rate, alert volume and analyst capacity. Choose thresholds on validation data, then test once on held-out data.

Avoid leakage: A flow is not the same as a host-hour or an incident. Random splits can put related flows, duplicate records, hosts and capture-day artifacts into both training and testing. Prefer defensible time or entity-grouped splits. A temporal split may include attack types absent from training; describe this honestly rather than rearranging records to inflate scores.

Lab B (30 minutes): Reproduce the confusion-matrix calculations. Draw a threat-to-evidence table for phishing, credential abuse and denial of service. For the extended Unit 1 lab, download an authorized CICIDS2017 flow-feature subset from the University of New Brunswick; record provenance and hashes, audit missing values, infinities, duplicates and capture days. Separate audit identifiers from features. Compare a majority-class baseline, logistic regression and tree ensemble. Fit preprocessing on training only, tune thresholds on validation, and analyze held-out false positives and negatives. Do not run malware or probe live systems.

Operational diagram (text version):

Threat hypothesis → Telemetry → Validated features → Model score → Threshold → Alert → Analyst review → Confirmed outcome → Monitoring.

Every stage can fail. A model score is not a confirmed incident.

C. Integrated doctoral research question

“How can evidence-grounded language models improve security alert triage without increasing false positives or unsupported incident claims?” Compare a rules-only baseline, a classifier and a retrieval-grounded assistant on the same incident cases. Measure classification metrics, citation support, unsupported claims, analyst review time and human override. Keep response actions subject to human approval.

Today’s completion evidence: (1) concept map contrasting P(y|x) and P(x); (2) seeded bigram experiment and held-out metric; (3) threat-to-telemetry matrix; (4) confusion matrix and base-rate critique; (5) research log with hypothesis, data split, result, failure and next test.

D. Verified global intelligence — 1 November 2026

28 October — Google Gemini 4 Carbon: Google reports stronger long-horizon and cybersecurity capabilities, including 88.2% SEC-bench Pro, 54.8% ExploitGym and 71.0% CWE-bench. These are provider-reported benchmark results, not proof of safe autonomous deployment. Research relevance: inspect benchmark definitions and operational safeguards. https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/

29 October — Mistral Large 4 open weights: Mistral released weights for “Le Chonk,” highlighting coding and defensive cybersecurity. Open weights can support controlled local evaluation, but do not guarantee transparent training data or eliminate dual-use risk. https://mistral.ai/news/ ; Reuters: https://www.reuters.com/legal/litigation/mistral-ai-publicly-releases-large-4-targets-cybersecurity-users-2026-10-29/

26 October — GPT-6.1 Terra: OpenAI introduced a faster computer-use model; its system-card addendum assesses cybersecurity capability at Critical under its Preparedness Framework. Research relevance: distinguish model capability from authorization and safety controls. https://openai.com/index/introducing-gpt-6-1-terra/ ; https://deploymentsafety.openai.com/gpt-6-1-terra

29 October — “Black October” incident-response strain: The Wall Street Journal reports pressure on incident responders and shrinking response windows. Research relevance: alert volume and time to containment matter beyond raw accuracy. https://www.wsj.com/pro/cybersecurity/cyber-responders-strained-by-black-october-e259ea25

28 October — Axios Future of Cybersecurity newsletter: Emphasizes rapid detection, containment, recovery, multifactor authentication and tested response plans as AI changes attack speed. https://www.axios.com/newsletters/axios-future-of-cybersecurity-340330d0-cf00-11f1-a1fd-f1b7e30b8d45

1 October — Microsoft Digital Defense Report 2026: Describes compressed attack timelines and the importance of continuous defense. It is a vendor report; account for telemetry and sampling biases. https://www.microsoft.com/en-us/security/security-insider/threat-landscape/2026-digital-defense-report

Ongoing newsletter feeds: The Batch (https://www.deeplearning.ai/the-batch) and SANS Stormcast (https://isc.sans.edu/podcast.html). These are reading recommendations, not claims of new issues today.

E. Study resources

Reading: Foundation Models report https://arxiv.org/abs/2108.07258 ; Dive into Deep Learning language modeling https://d2l.ai/chapter_recurrent-neural-networks/language-model.html ; CICIDS2017 https://www.unb.ca/cic/datasets/ids-2017.html ; NIST AML taxonomy https://csrc.nist.gov/pubs/ai/100/2/e2025/final

Videos: 3Blue1Brown LLM overview https://www.youtube.com/watch?v=LPZh9BOjkQs ; 3Blue1Brown Transformers https://www.youtube.com/watch?v=wjZofJX0v4M ; StatQuest precision/recall https://www.youtube.com/watch?v=PWvfrTgaPBI

Podcasts: Practical AI https://podcasts.apple.com/in/podcast/practical-ai/id1406537385 ; Security Now episode 1092 https://twit.tv/shows/security-now/episodes/1092

F. Quick self-test

- What does P(y|x) estimate? A label conditional on input.

- Why is next-token learning self-supervised? The observed next token supplies its own target.

- Does low perplexity certify factuality? No.

- Why can network flows miss credential misuse? Identity/authentication context may be absent.

- What are precision and recall for TP=80, FP=198, FN=20? 28.8% and 80%.

- Why can random IDS splits inflate performance? Duplicate/related flows and capture artifacts can leak across partitions.

Next syllabus topics: Autoregressive versus latent-variable models; supervised versus unsupervised versus semi-supervised security learning.

Source: November 1 morning email. Narration is generated separately by the verified GitHub Actions workflow.

<!-- Integration test: 2026-11-01 morning lesson committed to main to exercise automatic audio generation. -->
