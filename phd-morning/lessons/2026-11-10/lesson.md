# Morning PhD teaching package — 10 November 2026 (Asia/Kolkata)

**Source syllabus:** October 19, 2026 Gmail message “PhD Coursework Progress, First DAC Review and Ludovika Fellowship 2027”, attachments *Advanced Generative AI and Foundation Models Learning Pathway* (21 pages), Unit 3 Session 1, p.8, and *AI for Cybersecurity Learning Pathway* (20 pages), Unit 2 Session 5, p.6. Both source PDFs were retrieved and their relevant sections read. **Progression:** after November 9's Unit 2 Session 6 on inference scaling, Generative AI advances to **Unit 3 Session 1: data curation and pretraining**; cybersecurity advances after ransomware/botnets/phishing to **Unit 2 Session 5: ensembles, graphs and threat intelligence**. All examples are synthetic unless explicitly sourced. This is a teaching session, not university contact-hour certification.

## Spoken navigation
**10 minutes:** Explain why training-data quality affects a model even with unchanged architecture; compute one contamination example; describe a host–domain graph and why an indicator requires provenance; answer oral Q1–Q5. **30 minutes:** Add deduplication, split leakage, cross-entropy, boosting versus bagging, graph edge semantics, precision and source confidence; answer Q1–Q10. **Extended 90–150 minutes:** run both labs, audit every assumption, critique papers and preregister research experiments; answer all Q1–Q18. Tutor: ask one question at a time, pause for the learner, challenge with the follow-up, then reveal the answer. Never read code aloud line by line; explain the algorithm and its invariants.

## Prerequisites and vocabulary
Prior work: tokenization, next-token likelihood, Transformers, prefill/decode, Python dictionaries, precision/recall, Bayesian base rates, malware/phishing evidence. **Corpus**: documents used for learning. **Provenance**: origin, permissions, collection date, transformations and traceable identifiers. **Deduplication**: removal or grouping of exact/near-repeated examples. **Contamination**: overlap or answer leakage between training and evaluation. **Checkpoint**: serialized model/optimizer/training state. **Bagging**: fitting diverse estimators to bootstrap samples and aggregating predictions. **Boosting**: sequentially fitting models to correct a chosen loss or residual. **Graph**: vertices (hosts/users/domains/files) and typed, timestamped edges (observed relationships). **Threat-intelligence indicator**: a structured clue such as a domain/hash, not a verdict. **Calibration**: agreement between predicted probabilities and observed frequencies under a defined distribution.

# Track G — Data curation and pretraining (Unit 3 Session 1)

## 1. Why the dataset is part of the model
Imagine two apprentices with the same memory and learning algorithm. One studies ten independently written technical manuals; the other studies one manual repeated ten times. Equal token counts do not mean equal evidence. Repetition can overweight an author's errors, distort the gradient, inflate apparent coverage, and leak near-duplicate answers into evaluation. Conversely, removing all repeated technical syntax may erase useful regularities. Data selection is an explicit optimization and governance decision, not merely cleaning.

**Mechanism, end to end:**
```text
authorized sources -> provenance ledger -> parse/normalize -> safety/PII checks
   -> exact + near-duplicate clusters -> document/group/time split
   -> tokenizer/version -> packed sequences + causal target mask
   -> batches -> forward -> cross-entropy -> backprop/optimizer
   -> periodic checkpoints -> held-out loss + contamination audit
   -> model card, data card and reproducibility manifest
```
First establish lawful access and retention. Preserve original hashes and document IDs. Normalize with reversible metadata where possible. Detect exact duplicates with cryptographic hashes, near duplicates with token shingles/MinHash or embeddings (each has false positives). Split by document **and duplicate cluster** before making overlapping token windows; otherwise a paragraph can appear in train and test. Fit token vocabulary or preprocessing on training material when evaluating from-scratch pipelines. Record tokenizer revision, seed, sample weighting, corpus cutoff, filters and exceptions. Never silently use test examples for prompt selection or hyperparameter tuning.

**When filtering helps:** removing corrupted text, leaked secrets, license-incompatible content, repeated spam and evaluation contamination. **When it fails:** dialects, minority languages and rare incident reports can be disproportionately removed; “low quality” proxies may encode demographic bias. Keep distributional audits and allow documented review. Synthetic training text may reproduce biases and can create feedback loops.

## 2. Mathematical mechanism and worked examples
Let documents be d_1,...,d_N; a tokenizer maps them to sequences x_1,...,x_T. An autoregressive model with parameters θ assigns
**p_θ(x_1,...,x_T)=∏_(t=1)^T p_θ(x_t | x_<t)**.
With teacher forcing, the empirical token loss is **L(θ)=−(1/M) Σ_(i,t∈valid_i) w_i log p_θ(x_(i,t)|x_(i,<t))**, where M is the sum of included token weights, w_i a declared document/example weight, and valid_i excludes padding and optionally prompt-only positions. θ are learned weights; t is position; i is sequence; p is a conditional probability. In a basic unweighted corpus, all w_i=1 and M counts valid prediction tokens. Padding must not enter the loss.

**Gradient intuition:** for a vocabulary softmax with logits z_j and one-hot target y_j, the derivative of negative log-likelihood with respect to logit z_j is **softmax(z)_j − y_j**. The correct class is pushed up, other classes down. Repeating an example k times repeats its gradient contribution k times (before normalization effects), which is why deduplication changes the effective objective.

**Worked loss:** correct next-token probabilities 0.8, 0.5, 0.25 yield L=−(ln .8+ln .5+ln .25)/3≈0.7675 nats/token, perplexity exp(L)≈2.154. Perplexity is not factual accuracy, and cross-tokenizer comparisons are invalid without care.

**Leakage calculation:** suppose 100 independent documents each have 3 near-identical copies; each copy is randomly assigned train with probability 0.8 and test with 0.2. For one test copy, probability that at least one of its two siblings lands in train is 1−(0.2)^2=**0.96** (assuming independent assignment). Thus 96% of test copies would have a train sibling under these assumptions. A group split by duplicate cluster makes this particular overlap zero. This does not establish that every model memorizes every duplicate.

**Training resources:** With P parameters, naive FP16 weight storage is 2P bytes, but training also stores gradients, optimizer states, activations and often higher-precision master weights. Adam's two FP32 moments alone use ~8P bytes; this is not total training memory. For global batch size G across D data-parallel workers with microbatch m and gradient accumulation a, **G=Dma** if every worker processes m examples per microstep. Data parallelism replicates model state (unless sharded); tensor parallelism partitions operations within layers and adds communication; pipeline parallelism partitions layers and risks idle bubbles. More GPUs do not eliminate data licensing, contamination or evaluation problems.

## 3. Lab G — corpus split, leakage audit and weighted token loss
**Python 3.10+, standard library; synthetic documents only.** Save as `lab_g.py` and run `python3 lab_g.py`.
```python
import hashlib, math, random
docs = [
 ("A1","Incident response requires evidence review."),
 ("A2","Incident response requires evidence review."),
 ("B1","Tokens predict the next symbol."),
 ("B2","Tokens predict the next symbol."),
 ("C1","Check provenance before training."),
 ("D1","Measure held out loss honestly."),
]
def fingerprint(text):
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()
groups = {}
for name, body in docs:
    groups.setdefault(fingerprint(body), []).append(name)
assert sorted(map(len, groups.values())) == [1,1,2,2]
# Assign whole duplicate groups, never individual documents.
keys = sorted(groups)
train_keys, test_keys = set(keys[:3]), set(keys[3:])
assert train_keys.isdisjoint(test_keys)
train = [n for k in train_keys for n in groups[k]]
test = [n for k in test_keys for n in groups[k]]
assert set(train).isdisjoint(test)
probs = [0.8, 0.5, 0.25]
loss = -sum(math.log(p) for p in probs)/len(probs)
perplexity = math.exp(loss)
leak_probability = 1-(1-.8)**2
assert abs(leak_probability-.96)<1e-12
assert abs(loss-0.7675283643)<1e-8
print("clusters", sorted(map(len,groups.values())))
print("loss", round(loss,4), "ppl", round(perplexity,3))
print("random-copy leakage", round(leak_probability,2))
```
**Expected:** `clusters [1, 1, 2, 2]`; `loss 0.7675 ppl 2.154`; `random-copy leakage 0.96`. This checks exact duplicates, not paraphrases. **Advanced extension:** create 3-word shingles, calculate Jaccard similarity, and manually inspect false merges. Compare train/test loss with copy-level versus cluster-level splitting, holding the model fixed. Record any unbalanced domain representation. Do not mistake a deterministic arithmetic demonstration for a pretrained-model experiment.

## 4. Practical case, trade-offs and paper critique
**Case:** an organization trains a security-answering assistant on public incident writeups and internal response procedures. A near-duplicate public benchmark answer appears in training; benchmark scores rise without improved response on new incidents. A privacy filter deletes hostnames but leaves account identifiers in adjacent logs. Controls: data lineage, duplicate-cluster split, canary/PII scanning, authorization, deletion procedures, sealed holdout, human review, and evidence-grounded evaluation. A false-positive PII filter can also delete critical context: quantify both leakage and retained utility.

**Research reading:** Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*, arXiv:2107.06499, studies memorization and evaluation contamination; critique how their corpora, duplicate definitions and model sizes transfer. Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*, arXiv:2203.15556, investigates model/data compute allocation; empirical scaling is regime-dependent, not a license to ingest unvetted data. Dodge et al. (2021), *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*, arXiv:2104.08758, illustrates corpus documentation challenges. Read original methods before quoting effect sizes.

**Research question G:** At fixed token budget and model size, does duplicate-cluster splitting plus quality filtering improve *truly held-out* security QA compared with random document splits? Preregister exact/near-duplicate thresholds, stratify rare domains, evaluate clean accuracy, memorization indicators and coverage, and report confidence intervals. The confounder is changed domain composition; measure it explicitly.

# Track C — Ensembles, graphs and threat intelligence (Unit 2 Session 5)

## 5. Intuition and architecture
A network-flow model says a host contacted an unusual destination. An endpoint model says a suspicious process ran. A threat feed says the destination was reported last month. None alone proves compromise. An ensemble can combine *complementary* signals, while a graph preserves who connected to what, when, and with what evidence. A feed entry must carry provenance, timestamp, confidence semantics and expiry. Without those fields, a stale shared-IP indicator may harm an innocent service.

```text
flows / endpoint / identity -> typed event schema -> time-bounded entity graph
           |                        |                  |
       rules + models         provenance ledger    host-domain-file paths
           |                        |                  |
           +---- calibrated fusion / ensemble --------+
                               |
                 policy threshold + evidence references
                               |
                   human review / authorized action
```
**Bagging** trains base estimators on resampled training records and aggregates, reducing variance when errors differ. Random forests also vary candidate features. **Boosting** iteratively emphasizes errors/gradients under an objective, often reducing bias but increasing sensitivity to noisy labels and leakage. **Voting** combines discrete labels or probabilities; averaging uncalibrated scores does not magically calibrate them. **Stacking** trains a meta-learner on out-of-fold predictions, never the same base model's in-sample predictions. For temporal security data, folds must respect time and entities.

**Graph semantics:** node IDs are pseudonymous host, user, domain, file hash. An edge `(host, contacted, domain, t)` is an observed network event, not a declaration that either endpoint is malicious. Use time windows, direction, source sensor, evidence ID and tenant boundary. Shared cloud/CDN IPs create high-degree benign hubs; graph propagation can spread false suspicion. Entity resolution and missing sensors are major failure modes.

## 6. Derivations and worked example
For two classifiers with independent error indicators of probability e, majority voting helps only under appropriate diversity and error assumptions. If both make exactly the same mistakes, voting adds **zero** information. For three conditionally independent classifiers each with error e=0.2, majority error = **3e²(1−e)+e³=0.104**. Correlation can erase that gain. This is an illustrative calculation, not a measured detection rate.

**Weighted fusion:** let calibrated model probabilities be p_1,p_2. A simple linear pool q=αp_1+(1−α)p_2 for α∈[0,1] may be useful but is not necessarily calibrated; evaluate on a held-out time block. Under explicit conditional independence given class, likelihood ratios multiply: posterior odds = prior odds × LR_1 × LR_2. If sensors share a common cause (e.g. backup software), naive multiplication double counts evidence. A log-odds model with interaction terms or a calibrated meta-model can capture dependence when sufficient data exist.

**Graph example:** host H contacts domain D at 10:02, user U logs into H at 10:01, and file F runs on H at 10:03. A feed marks D “suspicious” based on a report dated 18 months ago. The three-edge path is a *candidate investigation narrative*, not causal proof. If D is a shared update endpoint, high graph connectivity increases benign explanations. Filter by time and typed edge; require an evidence ID for every asserted relation.

**Operational threshold:** with attack prevalence π=0.001, sensitivity 0.9 and specificity 0.99, precision among alerts is 0.08264. Even a “90% sensitive” ensemble can overwhelm analysts. Optimize incident-level recall at a fixed daily alert budget, not accuracy alone. An indicator's “confidence=high” is not a calibrated P(attack|indicator) unless its publisher defines and validates that interpretation.

## 7. Lab C — safe typed-graph correlation and ensemble counterexample
**Python 3.10+, standard library; no network requests or malicious samples.**
```python
from collections import defaultdict
events = [
 ("h1","contacted","d1",10,"flow-01"),
 ("h1","executed","f1",11,"edr-01"),
 ("h2","contacted","d2",10,"flow-02"),
 ("h2","executed","f2",11,"edr-02"),
]
intel = {"d1":{"seen":2,"confidence":"medium","source":"synthetic-feed"},
         "d2":{"seen":1,"confidence":"low","source":"synthetic-feed"}}
by_host = defaultdict(list)
for host, relation, target, minute, evidence in events:
    by_host[host].append((relation,target,minute,evidence))
def explain(host):
    edges = by_host[host]
    destinations = [t for r,t,_,_ in edges if r=="contacted"]
    executed = [t for r,t,_,_ in edges if r=="executed"]
    return {"host":host, "indicator_matches":[d for d in destinations if d in intel],
            "executed":executed,"evidence":[e for *_,e in edges]}
assert explain("h1")["evidence"] == ["flow-01","edr-01"]
assert explain("h2")["indicator_matches"] == ["d2"]
e=.2
majority_error=3*e*e*(1-e)+e**3
assert abs(majority_error-.104)<1e-12
print("majority error",round(majority_error,3))
print(explain("h1"))
```
**Expected first line:** `majority error 0.104`; second line is a dictionary with host `h1`, indicator `d1`, executed `f1` and evidence IDs `flow-01`, `edr-01`. **Important:** the synthetic feed timestamps are abstract units, not current dates, and no “malicious” conclusion is computed. **Extensions:** reject edges outside a 5-minute window; add duplicate sensor events and stale intel expiry; implement three synthetic classifiers with identical errors to demonstrate that the independence assumption is crucial. Count alert burden and compare a rule, a calibrated classifier and a graph-aware model on the same time-held-out incidents.

## 8. Security cases, misconceptions and research critique
**Real-world class of cases:** phishing infrastructure and botnet campaigns may reuse domains, hosts and files; graph links can help analysts connect evidence across sensors. Yet cloud hosting, NAT, CDNs and shared SaaS create legitimate many-to-many links. Do not block a domain merely because it appears one hop from an indicator. Enrich only with authorized telemetry, retain data minimally, protect tenant boundaries and document analyst override.

**Misconceptions:** “More models always improve accuracy” (correlated errors can hurt); “a graph edge proves attack” (it records observation); “a threat-feed score is a probability” (often an ordinal label); “boosting fixes missing evidence” (it cannot invent absent telemetry); “training on future feed labels is harmless” (it leaks future knowledge). **Failure modes:** indicator poisoning, stale intelligence, feed licensing, identity collisions, timestamp skew, label leakage, rare attack families and analyst feedback loops. **Ethics:** protect user and host identities, avoid mass surveillance, document false-positive containment cost and appeal paths.

**Reading:** Breiman (2001), *Random Forests*, Machine Learning 45, DOI:10.1023/A:1010933404324; Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of Statistics 29(5), DOI:10.1214/aos/1013203451. Both describe methods, not validated modern SOC efficacy. MITRE ATT&CK Enterprise knowledge base (official) provides technique taxonomy but no posterior probabilities. STIX 2.1 specification (OASIS, 2021) provides structured cyber-threat-intelligence objects and relationships; a STIX indicator is still not proof of an active attack. Review temporal validity and feed provenance before using any IOC.

**Research question C:** At fixed 30 alerts per analyst-day, does time-bounded host–domain–file graph evidence improve incident-level recall and lead time over an endpoint+flow ensemble? Predeclare entity resolution, time windows, missing-sensor patterns and delayed intel arrival. Train/validate/test chronologically by incident family; compare no-graph, graph, and stale-feed ablations; bootstrap confidence intervals over incidents rather than individual correlated edges.

# Cross-track integration and oral examination
A security model's training corpus and threat feed share the same fundamental question: **Where did the evidence come from, and was it available at decision time?** The generative model may summarize graph observations, but it must never convert an unverified feed assertion into a confirmed compromise. The safe SOC explanation pipeline retrieves evidence IDs, separates observation/inference, cites timestamps and abstains when evidence is missing. Prompt injection in a retrieved indicator note must be treated as data, not instructions. Keep human authorization for containment.

## Progressive Socratic Q&A (ask, wait, then reveal; follow-up tests transfer)
1. **What is provenance?** Origin, rights, timestamp and transformations of a record. **Follow-up:** Which fields would let you retract a compromised source?
2. **Why split by duplicate cluster?** To prevent copies crossing train/test. **Follow-up:** What about paraphrases?
3. **What does pretraining minimize?** Average negative log conditional token probability over included positions. **Follow-up:** Does low loss prove truth?
4. **Why mask padding?** It is not a target token. **Follow-up:** What happens to gradient if included?
5. **What is a checkpoint?** Saved weights and often optimizer/training state. **Follow-up:** What else is needed for reproducible resume?
6. **What is the 96% leakage calculation conditional on?** One test copy, two independently split siblings, 80% train probability. **Follow-up:** How does group split change it?
7. **Why does repeated text change training?** It repeats gradient contributions. **Follow-up:** Could legitimate repetition help?
8. **Compare data/tensor/pipeline parallelism.** Data replicates workloads, tensor partitions layer computation, pipeline partitions layers. **Follow-up:** Which incurs cross-device communication?
9. **What does bagging address?** Variance via aggregation of diverse learners. **Follow-up:** What if their errors are identical?
10. **Why can boosting overfit?** It can pursue noise or mislabeled residuals. **Follow-up:** Which validation split tests temporal drift?
11. **What does a graph edge mean?** A typed observation with time and source. **Follow-up:** Why is a shared CDN domain dangerous?
12. **Is feed confidence a probability?** Not without calibration and definition. **Follow-up:** What reliability plot would you demand?
13. **Derive majority error for three independent 20%-error classifiers.** Exactly two wrong: 3(.2)^2(.8); three wrong: (.2)^3; total .104. **Follow-up:** Why does correlation invalidate it?
14. **What is stacking leakage?** Meta-model learns on base predictions made in-sample. **Follow-up:** How do out-of-fold predictions fix this?
15. **What does time-of-knowledge mean?** Only features/feeds available at prediction timestamp may enter evaluation. **Follow-up:** What if intel arrives next day?
16. **What does 8.26% alert precision mean?** Most alerts are false under the stated low prevalence/sensitivity/specificity. **Follow-up:** How would you budget analyst time?
17. **How can filtering create harm?** It can remove rare-language or minority-domain examples. **Follow-up:** Which coverage metric detects it?
18. **Design a cross-track falsification test.** Poison a synthetic intel note and check that the summarizer quotes evidence but ignores instructions; compare time-held-out graph detection with/without stale intel. **Follow-up:** What result would reject your hypothesis?

## Recall cards and research deliverables
At the end of a 10-minute walk, recall without notes: the autoregressive objective; the 96% leakage assumption; why near-duplicate grouping matters; the difference between bagging and boosting; the 0.104 independent-majority result; the provenance fields on an IOC. At 30 minutes, derive the softmax gradient and explain why an 18-month-old indicator should not cause automatic containment. For extended study, save both lab outputs, a corpus/split manifest, one graph schema with time-of-knowledge fields, and a preregistered one-page research design with negative controls.

## Source map (self-contained explanations above; canonical links in companion index)
- Data provenance, split discipline and pretraining mechanics: syllabus GenAI Unit 3 Session 1; §§1–3; Lee et al. 2022, Dodge et al. 2021.
- Token loss, gradient and compute/data trade-offs: §2; Hoffmann et al. 2022; basic softmax derivative shown here.
- Bagging, boosting and error diversity: §§5–7; Breiman 2001, Friedman 2001; majority-error arithmetic derived here.
- Graph semantics, IOC provenance, time validity and safety: §§5–8; OASIS STIX 2.1, MITRE ATT&CK; illustrative examples explicitly synthetic.
- Security false positives and deployment: §6; November 9 morning lesson's Bayes derivation; new prevalence example uses same stated assumptions.
- Current developments: no unverified November 10 headline is asserted; foundational primary references take precedence over speculative news.

**Reproducibility and access:** Labs are standard-library and contain assertions; a repository write alone is not proof of execution. No external paper PDF is claimed downloaded. Consult the companion `references/index.md` for stable links, summaries, access notes and paper paths.
