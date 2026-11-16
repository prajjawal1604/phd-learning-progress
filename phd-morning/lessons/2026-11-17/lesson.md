# Morning PhD teaching package — 17 November 2026 (Asia/Kolkata)
**Syllabus progression and source verification.** Yesterday's committed morning and evening lessons covered RAG retrieval architecture and adaptive robustness. Today advances to **Generative AI Unit 4 Session 2: query processing, chunking and reranking** (GenAI pathway p.10) and **AI for Cybersecurity Unit 4 Session 1: deep learning and Transformers for event sequences** (cybersecurity pathway p.10). Both original pathway PDFs were read from the 19 October 2026 Gmail message. The tracks deliberately converge on safe SOC evidence processing without repeating yesterday's derivations. References below were checked against canonical arXiv records where stated; no speculative 2026 research announcements are asserted.

## Spoken routes and interactive tutoring
**10 minutes:** Contrast retrieve-then-rerank with retrieve-only; compute pairwise ranking loss; explain why event order matters; answer Q1–Q6 and Q13–Q16.
**30 minutes:** Add chunking/expansion, cross-encoder complexity, attention masking, token-surprise example and the two failure cases; answer Q1–Q12 and Q13–Q21.
**Extended 90–120 minutes:** Work all derivations and both labs; critique papers, design a leakage-free SOC evaluation and discuss Q1–Q24.
**Voice protocol:** Ask a question, pause for reasoning, request a counterexample, give the worked answer, then ask the follow-up. Speak equations in words before notation; never read Python syntax while walking.

Prerequisites: BM25, RRF, conditional probability, softmax, dot products, gradients, train/test separation, confusion matrices, log-event semantics and yesterday's RAG/robustness lessons.
Definitions: *candidate generation* is a fast high-recall shortlist; *reranking* expensive ordering of candidates; *cross encoder* jointly encodes query and document; *bi encoder* encodes them separately; *chunk* a retrievable document segment; *MRR* mean reciprocal rank; *DCG* discounted cumulative gain; *event template* normalized log message pattern; *causal mask* disallows future-event attention; *concept drift* change in input-label relationship; *token surprisal* negative log predicted probability, not a probability of maliciousness.

# Track G — Query processing, chunking and reranking

## G1. Intuition and full mechanism
Suppose a runbook has a table with 'Condition' in one chunk and 'Action' in the next. Search retrieves a row but loses the header. The answer is fluent yet unsupported. A good system must preserve section headers, table row identities, validity intervals, tenant ACLs and exact evidence offsets. First-stage BM25/dense retrieval should favor recall at modest cost; a cross encoder can compare query and candidate jointly, but cannot resurrect absent candidates.

Architecture:
```text
signed sources -> parse/layout/table rows -> semantic chunks + parent ID + version + ACL
 -> tenant/time/status authorization filter -> BM25 / dense candidate lists
 -> optional query rewrite (retain original) -> merge/deduplicate within ACL
 -> cross-encoder score(query, passage) -> top-k evidence with offsets
 -> claim-by-claim answer and citation verifier -> abstain or release
 -> read-only audit trace (query hash, model version, corpus hash, scores)
```
A cross encoder's joint token interactions often improve ranking but are O(K) model passes per query, with input length costs. A bi encoder can precompute passage vectors and use ANN search; it trades interaction richness for cheap retrieval. Latency, token truncation and training-domain mismatch can make reranking worse. Query expansion may broaden recall but change intent: 'disable account' and 'lock account' are not interchangeable for every incident policy. Keep original query, record rewrites and evaluate against fixed human labels.

## G2. Derive a pairwise reranking objective
For query q, relevant passage d+ and less relevant d−, let scores s+=f(q,d+) and s−=f(q,d−). Define Δ=s+−s−. Model probability that d+ wins as σ(Δ)=1/(1+e^(−Δ)). Negative log likelihood:
L=−log σ(Δ)=log(1+e^(−Δ)).
Differentiate: dL/dΔ=−e^(−Δ)/(1+e^(−Δ))=σ(Δ)−1. For Δ=0, loss=ln2≈0.6931 and derivative −0.5. Gradient descent increases the gap. For Δ=ln3≈1.0986, σ=.75, loss≈.2877, derivative −.25. This ranking probability is a training device, **not calibrated probability that the passage is true or authorized**. If labels are biased by obsolete documents, optimizing L can confidently promote wrong passages.

For a list of candidates with grades rel_i≥0, DCG@k=Σ_i≤k (2^rel_i−1)/log2(i+1); nDCG divides by ideal DCG when positive. Example grades [2,0,1] give DCG=3+0+1/2=3.5; ideal [2,1,0] gives 3+1/log2(3)≈3.63093; nDCG≈0.9639. Unlike MRR, nDCG considers multiple graded relevant items. Grades and cutoff must be predeclared.

## G3. Chunking and retrieval experiments
Choose unit (paragraph, section, table row + headers, page) before measuring. Chunk overlap reduces boundary loss but creates near-duplicates, inflates indexing and can crowd out independent evidence. Parent-child retrieval indexes small chunks but returns larger authorized parent context; check ACL and validity on both. For a 100-token document, 40-token chunks with stride 30 produce starts 0,30,60,90 (last may be tiny); a minimum-tail merge rule avoids a 10-token orphan. Semantic chunking is not guaranteed superior to fixed windows: OCR corruption and noisy headings can defeat it.

**Failure case 1:** Correct 2026 runbook is candidate rank 12; reranker sees only top 10. Even perfect reranking yields recall@10=0. Remedy: increase candidate breadth, use exact identifier constraints, test recall versus latency; do not blame generation.
**Failure case 2:** A malicious retrieved snippet says 'ignore authorization'. Treat it as data. No text instruction can grant access to cross-tenant records or invoke containment tools. Pre-filter before ranking, validate metadata from trusted ingestion, keep tool authorization outside the model.

## G4. Lab 1: deterministic candidate and reranking demonstration
Python 3.10+ standard library; save as `rerank_lab.py`; run `python rerank_lab.py`.
```python
import math
docs=[
 ("A","approved",["login","fail","lockout","audit"]),
 ("B","draft",["login","fail","disable","audit"]),
 ("C","approved",["credential","lockout","review","audit"]),
 ("D","approved",["login","failure","investigate","audit"])
]
q={"login","fail","lockout"}
def overlap(d):return len(q & set(d[2]))
ranked=sorted(docs,key=lambda d:(-overlap(d),d[0]))
assert [d[0] for d in ranked]==["A","B","C","D"]
def rerank(d):
    return overlap(d)+ (2 if d[1]=="approved" else -5)
rescored=sorted(ranked[:3],key=lambda d:(-rerank(d),d[0]))
assert [d[0] for d in rescored]==["A","C","B"]
print("candidate", [d[0] for d in ranked])
print("reranked", [d[0] for d in rescored])
print("candidate recall",int("C" in [d[0] for d in ranked[:3]]))
print("top1 contains C",int(rescored[0][0]=="C"))
# A cross encoder cannot recover a missing candidate:
assert "C" not in [d[0] for d in ranked[:2]]
```
Expected: candidate ['A','B','C','D']; reranked ['A','C','B']; candidate recall 1; top1 contains C 0. This is **a hand-scored ranking simulation**, not a trained BERT reranker or proof of accuracy. It demonstrates that reranking can move C from third to second but cannot make it first under the chosen scoring function. Try top-2 candidate truncation; C vanishes and cannot be recovered. Change the draft penalty and explain why policy approval must be a hard filter, not just a ranking weight. For real experiments compare BM25, dense, RRF, cross encoder with identical ACL and version filters and report recall@20, MRR@10, nDCG@10, unsupported citations, p95 latency and cost.

## G5. Paper reading and critique
Nogueira & Cho, *Passage Re-ranking with BERT* (2019, arXiv:1901.04085, DOI:10.48550/arXiv.1901.04085), https://arxiv.org/abs/1901.04085, investigate BERT for passage reranking on information-retrieval benchmarks. Their result motivates a joint query-passage model but does not show that a reranker repairs missing evidence, authorization or document validity. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020, arXiv:2005.11401), https://arxiv.org/abs/2005.11401, models retrieval-conditioned generation; it does not guarantee citation entailment or access control. Research question: at fixed 250ms p95 and zero unauthorized retrieval, does cross-encoder reranking outperform hybrid BM25/dense for versioned SOC runbooks? Group splits by document family, bootstrap CIs and audit all authorization failures separately.

# Track C — Transformers and deep learning for security event sequences

## C1. Why sequences instead of bags of events?
Consider sequence A: login success → privileged change → outbound connection; sequence B: outbound connection → privileged change → login success. Both have identical token counts but different order and possible investigative meaning. Count models collapse these; a sequence model can condition on order and time. But missing sensor events, template changes and host identity leakage can make a Transformer appear impressive on a random split while failing on a new capture day. Start with a count or Markov baseline.

Architecture:
```text
authorized raw events -> redact sensitive identifiers -> parser/template IDs + timestamps
 -> group by entity and session -> chronological windows + missingness indicators
 -> train-only vocabulary + embeddings + time/position encoding
 -> masked Transformer encoder or causal next-event predictor
 -> anomaly score / calibrated classifier -> threshold + abstention
 -> analyst evidence with event offsets -> authorized human response
```
Event templates are not natural-language words; template IDs encode operational types. A new template may be harmless software drift. Record unseen-token handling and ensure no future logs enter the prediction window.

## C2. Attention derivation and worked example
For input X∈R^(T×d), learned W_Q,W_K∈R^(d×d_k), W_V∈R^(d×d_v), Q=XW_Q, K=XW_K, V=XW_V. Attention(Q,K,V)=softmax(QK^T/√d_k+M)V, where M_ij=−∞ for forbidden keys and 0 otherwise. A causal mask requires j≤i; a padding mask prevents attending to padded events. If q=[1,0], k1=[1,0], k2=[0,1], d_k=2, scores [1/√2,0]; softmax ≈[.6698,.3302]. Values v1=[2,0], v2=[0,4] produce ≈[1.3396,1.3208]. If second key is forbidden, weights [1,0] and output [2,0]. Attention is learned context aggregation, **not a causal attribution or proof of compromise**.

Complexity: full attention score matrix has T² entries per head; memory and compute increase with sequence length, although kernels and sparse architectures alter actual resource usage. For T=1000, one head's dense score matrix has one million entries, ~4MB in float32 before other activations. An event model trained on 20-event windows cannot reliably infer a month-long campaign without additional memory or aggregation.

## C3. Losses and false alarms
For next-event prediction, pθ(e_t | e_1...e_(t−1)), loss L=−Σ_t log pθ(e_t|history). Average surprisal S=L/n; higher S suggests rarity under the model, not malicious intent. For masked-event training, hide selected events and predict them from allowed surrounding context; this is useful for representation learning but **bidirectional masking cannot be used to claim online forecasting without leakage**. In a binary classifier with logits z, p=σ(z), cross entropy −y log p−(1−y)log(1−p); threshold selection must use validation data and operational false-positive budget.

Counterexample: an uncommon but legitimate patch rollout introduces a new event template. Surprisal spikes; an anomaly alert without deployment metadata may overwhelm analysts. Conversely an attacker mimicking common event patterns can evade a novelty detector. Compare by host/time-grouped test splits, use clean baseline and label delays, and report prevalence and analyst queue size.

## C4. Lab 2: token surprise, ordering failure and masked-event check
Python 3.10+ standard library; save as `event_lab.py`; run `python event_lab.py`.
```python
import math
from collections import Counter
train=[("normal",["login","ok","logout"]),("normal",["login","ok","logout"]),("alert",["login","fail","lockout"])]
counts=Counter(t for _,seq in train for t in seq)
vocab=sorted(counts)
N=sum(counts.values()); V=len(vocab)
def surprisal(seq):
    return sum(-math.log((counts.get(t,0)+1)/(N+V)) for t in seq)/len(seq)
clean=["login","ok","logout"]
attack=["login","fail","lockout"]
unknown=["login","newtemplate","logout"]
assert surprisal(clean)<surprisal(attack)<surprisal(unknown)
for name,seq in [("clean",clean),("attack",attack),("unknown",unknown)]:
    print(name,round(surprisal(seq),3))
# Independent token model cannot distinguish reversed order:
assert surprisal(clean)==surprisal(list(reversed(clean)))
# Masked-event model target should not use its true label in input:
masked=["login","[MASK]","logout"]
assert masked[1]=="[MASK]"
```
Expected: three lines clean, attack, unknown with strictly increasing scores; exact values are deterministic under this toy Laplace-smoothed unigram model. Reversing clean yields the same score, exposing the bag-of-events limitation. Extension: implement transition probabilities with start/end markers and Laplace smoothing, then verify reversed order changes score. This toy is **not a trained Transformer**, not a measured IDS, and 'attack' is only a synthetic label.

## C5. Real-world case and safety
A SOC receives authentication, process and network telemetry from multiple tenants. A model flags unusual authentication sequences. Ensure data minimization, tenant-isolated storage, timestamp normalization, and missing-log flags. Correlate anomalies with independent evidence (MFA failures, device posture, signed deployment changes); never disable accounts solely from an anomaly score. An attacker may exploit log suppression, injected benign-looking events, or training drift. Keep experiments offline and authorized. Evaluate adaptive perturbations with valid event semantics, not arbitrary changes to event IDs.

## C6. Primary-paper critique and doctoral question
Vaswani et al., *Attention Is All You Need* (2017, arXiv:1706.03762, DOI:10.48550/arXiv.1706.03762), https://arxiv.org/abs/1706.03762, establishes the Transformer on translation; its results do not validate security telemetry. Du et al., *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning* (2017, arXiv:1709.09338), https://arxiv.org/abs/1709.09338, studies log sequence modeling; critique assumptions about stable templates and training normality. Research hypothesis: a small causal event model with missingness flags will improve rare-attack recall at fixed 0.5% FPR over a count/Markov baseline on held-out hosts and later capture periods. Pre-register groups, drift intervals, thresholds and an abstention budget; report negative results.

# Cross-track synthesis, ethics and research agenda
A SOC RAG assistant can explain a detector's alerts by retrieving authorized incident notes, but a fluent explanation is not causal proof. Keep detector scores, retrieved evidence, model-generated hypotheses and analyst-confirmed facts separate. A high retrieval MRR cannot compensate for a high detector false-positive rate. Separate offline synthetic tests from operational claims; never deploy destructive autonomous actions without approvals. Potential PhD experiments: (1) version-aware reranking under latency budgets, (2) cross-tenant leakage probes in retrieval caches, (3) temporal event-model drift under patch rollouts, (4) robustness of anomaly scores to missing sensors, (5) joint retrieval-plus-detector abstention under fixed analyst capacity.

# Progressive oral Socratic dialogue
**Q1. Why rerank?** Answer: Initial search optimizes cheap recall; expensive joint query-passage scoring improves ordering within the shortlist. Follow-up: What if the right passage never enters top-k?

**Q2. What is a cross encoder?** Answer: A model jointly processes query and candidate, allowing token-level interactions before scoring. Follow-up: Why is it costlier than precomputed dense embeddings?

**Q3. Does reranking grant access?** Answer: No; authorization must filter candidates before any model sees their contents. Follow-up: What about caches and logs?

**Q4. Why does chunk overlap help?** Answer: It preserves boundary context, at cost of duplicated tokens and candidate crowding. Follow-up: When does overlap harm retrieval?

**Q5. What is recall@k?** Answer: Fraction of queries with relevant evidence in the first k candidates under a declared relevance definition. Follow-up: What if two passages are both required?

**Q6. What is MRR?** Answer: Mean reciprocal rank of first relevant result, with zero when missing. Follow-up: Why might it conceal missing second evidence?

**Q7. What is a pointwise reranker?** Answer: One score per query-document pair. Follow-up: How does a pairwise loss differ?

**Q8. Derive pairwise logistic loss.** Answer: With score gap Δ=s+−s−, probability preferred=σ(Δ); loss=log(1+exp(−Δ)). Follow-up: What is derivative with respect to Δ?

**Q9. What is its derivative?** Answer: σ(Δ)−1, so gradient descent raises the positive-versus-negative gap. Follow-up: What happens for already large positive gaps?

**Q10. What is nDCG?** Answer: Discounted cumulative relevance gain normalized by ideal ordering; handles graded relevance. Follow-up: Why must relevance grades be fixed before testing?

**Q11. Can expansion change intent?** Answer: Yes; adding synonyms may introduce a different vulnerability or procedure. Follow-up: How test expansion safely?

**Q12. Why version metadata?** Answer: A semantically relevant superseded procedure may be wrong for the requested date. Follow-up: What if no valid version exists?

**Q13. What is event tokenization?** Answer: Mapping typed security events to discrete IDs or structured embeddings. Follow-up: Why might an IP address be a shortcut?

**Q14. Why is order important?** Answer: Login→failure→lockout is not the same temporal pattern as reversed order. Follow-up: What does a bag-of-events lose?

**Q15. What is self attention?** Answer: Each event forms a weighted combination of value vectors using query-key similarities. Follow-up: Why use a mask for future events?

**Q16. Why scale dot products?** Answer: Division by sqrt(key dimension) limits logit growth under common variance assumptions. Follow-up: Does it calibrate anomaly scores?

**Q17. Why compare with counts?** Answer: A simple baseline may be more robust and cheaper, especially with small data. Follow-up: What is a fair split?

**Q18. What is a missing log event?** Answer: Unobserved event from sensor loss, not necessarily absence of activity. Follow-up: How model missingness?

**Q19. What is a false alarm from a new template?** Answer: Novel format may score anomalous even if behavior is benign. Follow-up: How separate template drift from attacks?

**Q20. Why not random window split?** Answer: Overlapping windows from one incident can leak into both train and test. Follow-up: How group by host and time?

**Q21. What is token surprise?** Answer: Negative log probability of observed event under a model. Follow-up: Why is it not attack probability?

**Q22. Why does attention not prove causality?** Answer: Weights are computational intermediates and may not identify causal evidence. Follow-up: What intervention would test causality?

**Q23. How measure operational value?** Answer: Recall, FPR, PR-AUC, calibration, p95 latency, analyst queue and review burden at fixed prevalence. Follow-up: Why report threshold?

**Q24. What is the ethical boundary?** Answer: Use authorized synthetic telemetry, protect identities and prohibit automated destructive response. Follow-up: What permissions are required for containment?

## Recall and self-test
Without notes: derive pairwise ranking loss and its derivative; calculate attention weights for two keys; describe the exact ACL enforcement boundary; explain why reranking cannot repair candidate recall; design a group/time holdout; distinguish surprisal from attack probability. Explain one false positive and one false negative in both tracks. Record your uncertainty and return to the relevant worked section.

## Claim-to-source map
- Syllabus placement and required labs: October 19 Gmail PDFs, GenAI p.10 and Cybersecurity p.10; prior context: local 2026-11-16 morning/evening lessons.
- Cross-encoder reranking and benchmark scope: Nogueira & Cho (2019), arXiv:1901.04085.
- Retrieval-conditioned generation and candidate limitation: Lewis et al. (2020), arXiv:2005.11401; previous local retrieval lessons.
- Transformer scaled attention and masking: Vaswani et al. (2017), arXiv:1706.03762.
- Log anomaly sequence modeling and limitations: Du et al. (2017), arXiv:1709.09338.
- Toy ranking, nDCG example and Laplace unigram results: mathematical derivations and executable examples in this lesson; these are not claimed as empirical paper findings.
- Security constraints and tenant/approval examples: explicitly designed local threat-model requirements, not attributed to an experimental paper.

**Access and validation:** Paper links and access notes are in [companion index](references/index.md). No PDF is claimed downloaded. No narration, email or audio is generated.
