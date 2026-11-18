# Morning PhD lesson — 19 November 2026 (Asia/Kolkata)
**Generative AI Unit 4 Session 4: Agents, planning and memory. AI for Cybersecurity Unit 4 Session 3: Reinforcement learning and adaptive defense.**

**Syllabus provenance:** Read both original October 19, 2026 Gmail PDF attachments: *Advanced Generative AI and Foundation Models Learning Pathway* p.10 (Unit 4 Session 4), and *AI for Cybersecurity Learning Pathway* p.10 (Unit 4 Session 3). Yesterday's morning/evening sessions covered RAG grounding/evaluation and graph self-supervision. Today moves from evaluating evidence to choosing bounded actions based on evidence. This is a teaching and simulation package, not a claim of production readiness. All incident data, utilities and performance values below are synthetic.

## Spoken routes and learning contract
**10 minutes:** define an agent versus a fixed workflow; verbally trace observe→decide→tool→verify; derive one Bellman backup; contrast expected reward with hard safety constraints; answer oral Q1–Q4 and Q13–Q16.
**30 minutes:** add agent memory and termination, partial observability, delayed reward, policy evaluation, and worked numerical decision; answer Q1–Q10 and Q13–Q21.
**Extended 90–120 minutes:** complete both mathematical derivations, critique research, inspect architecture, execute the two Python labs while stationary, challenge counterexamples, design an experiment and answer Q1–Q26. Spoken tutor: ask one question, wait for an explanation, request a counterexample, then reveal answer and a follow-up. Never read code while walking.

## Prerequisites and precise definitions
Assumed: conditional probability, expected value, discounted geometric series, vector state, classification precision/recall, RAG provenance and access controls, Python 3.10+. **Agent:** system that selects actions in response to observations and a goal. **Workflow:** predeclared action sequence or decision graph; may contain conditional branches without being a free-planning agent. **Policy** π(a|s): distribution over action a in state s. **State** s: sufficient information for a Markov transition model; **observation** o: what the defender actually sees, often incomplete. **Transition** P(s'|s,a): probability of next state. **Reward** r(s,a,s'): modeled preference over outcomes; not a guarantee of safety. **Return** G_t=Σ_{k≥0}γ^k r_{t+k+1}, with discount γ in [0,1). **Belief** b_t(s): probability of latent state given observations and actions. **Tool:** external read or mutation operation. **Transient memory:** current task state; **persistent memory:** durable state across tasks, requiring provenance, expiration and access controls. **Termination:** explicit stop criterion, including successful completion, budget exhaustion, unsafe uncertainty or escalation.

# Part A — Generative AI: agents, planning and memory
## A1. Intuition, contrast and mechanism
Suppose a maintenance assistant must determine whether a service interval is overdue. A fixed workflow always (1) authenticates user, (2) retrieves versioned manual, (3) reads equipment hours, (4) calculates interval, (5) produces cited recommendation. A planning agent might choose additional manual lookups, request missing serial number or reconcile conflicting records. Autonomy helps when observations change the next useful read; it hurts when unnecessary tool calls, stale memories or open-ended planning increase error and latency. Neither a fluent plan nor a visible reasoning trace proves correctness.

A minimal bounded cycle:
~~~text
user request + identity + task budget
 -> authorization gate -> evidence retrieval (version/ACL/time)
 -> planner proposes {tool, typed arguments, reason code}
 -> external schema + policy validator
 -> read-only tool / simulated state-changing tool
 -> observation (untrusted data, timestamp, provenance)
 -> state update + independent evidence check
 -> STOP if goal met, budget exceeded, uncertainty high or policy blocks
 -> cited answer or human escalation; immutable audit trace
~~~
**Trust boundary:** retrieved text and tool output are data, never instructions or credentials. The planner proposes; the executor independently authorizes. Read tools are not necessarily harmless: a read can expose secrets. Mutating tools require explicit approval and idempotency protections. A failed tool call must not be mistaken for a successful action.

## A2. ReAct versus fixed planning
Yao et al. [R1] interleave model-generated reasoning-oriented text, action calls and observations. A model may revise the next action when search fails, rather than adhering to a static plan. Their benchmark results do not establish faithful internal reasoning, correctness of every observation, or safe production authorization. A compact execution trace might be:
~~~text
Goal: identify correct lockout procedure.
Action: search_policy(version_at=2026-11-19, tenant=A)
Observation: policy-v4 section 3; one superseded draft
Decision: ignore draft by validity rule; check incident time
Action: read_incident(id=synthetic-17)
Observation: event 09:20; closure 09:40
Decision: cite policy-v4; stop, no environment mutation
~~~
This is a *designed example*, not an actual model transcript. The model must not invent missing observations. **Planning** searches over possible future actions; **execution** obtains observations; **verification** tests whether the goal and constraints hold. Replanning is valuable only if the observations are trustworthy and the extra action has positive expected value.

## A3. Value of information and tool selection: derivation
Let latent hypothesis H∈{incident,benign}; current belief p=P(H=incident)=0.3. Consider two terminal actions: investigate, with loss 2 regardless of truth (analyst effort); dismiss, with loss 10 if incident and 0 if benign. Current expected losses: L(investigate)=2; L(dismiss)=10p=3. Thus investigate is optimal with loss 2.

Suppose a read-only test T has sensitivity P(+|incident)=0.8, false-positive rate P(+|benign)=0.2 and cost 0.2. P(+)=0.8·0.3+0.2·0.7=0.38; Bayes gives P(incident|+)=0.24/0.38≈0.63158 and P(incident|−)=0.06/0.62≈0.09677. After +, investigate costs 2 versus dismiss≈6.316; after −, dismiss costs≈0.968 versus investigate=2. Expected post-test loss excluding test cost =0.38·2+0.62·(10·0.06/0.62)=1.36. Including test cost 1.56, so **value of information** (VOI)=2−1.56=0.44 loss units. Without accounting for test cost, privacy, and delay, extra tools can be overused. If test cost exceeds 0.64, test is not worth it in this simple model. In high-consequence incidents, policy may mandate investigation regardless of expected-cost arithmetic.

General formula: VOI(T)=min_a E[L(a,H)|history] − {c_T+Σ_o P(o|history,T) min_a E[L(a,H)|o,history,T]}. The tool should be chosen only if VOI>0 *and* permitted. A confident model estimate of VOI is not calibrated evidence.

## A4. Agent memory: what should persist?
Transient task state: current question, allowed tools, observations, remaining budget, pending approvals, stopping condition. Persistent memory: approved user preferences, stable source identifiers or audited task summaries only if retention policy permits. Store **facts with source IDs, timestamp, scope, confidence and expiry**, not unqualified model summaries. A memory may become stale, poisoned, cross-tenant or misleading after a policy revision. Revalidation against authoritative source beats treating memory as truth.

Architecture:
~~~text
authenticated request -> task-scoped state machine
                    | -> authorized RAG (current corpus snapshot)
                    | -> optional scoped memory (TTL, source, tenant)
                    v
                planner proposal -> policy+schema gate -> sandbox tools
                    ^                                       |
                    +------ signed result + audit event ---+
                    |
                verifier -> finish / ask / human review
~~~
**When agent beats workflow:** variable evidence gathering, uncertain tool sequence, reversible low-risk reads, clear verifier. **When workflow wins:** deterministic compliance, few fixed steps, costly mistakes, hard deadlines. Failure modes: infinite loops, repeated API charges, premature termination, fabricated tool success, tool-result prompt injection, secret leakage, stale cache, unsafe retries, memory poisoning. Compare end-to-end task success and harm, not the elegance of plans.

## A5. Lab A — bounded deterministic agent with independent policy gate
Save as agent_lab.py; Python 3.10+ standard library; no network, no LLM. This isolates the *architecture* from language generation.
~~~python
from dataclasses import dataclass
@dataclass(frozen=True)
class ToolCall:
    name: str
    tenant: str
    args: dict
POLICIES = {"A": {"v4": {"effective": "2026-11-01", "text": "Escalate privileged lockout"}}}
def execute(call: ToolCall, identity: str):
    assert call.tenant == identity, "cross-tenant access blocked"
    assert call.name in {"read_policy"}, "unapproved tool blocked"
    assert set(call.args) == {"version"}, "invalid arguments"
    return POLICIES[identity][call.args["version"]]
def run(identity="A", budget=2):
    trace = []
    for step in range(budget):
        call = ToolCall("read_policy", identity, {"version": "v4"})
        observation = execute(call, identity)
        trace.append((step, call.name, observation["effective"]))
        if "Escalate" in observation["text"]:
            return {"status":"verified", "answer":observation["text"], "trace":trace}
    return {"status":"escalate", "trace":trace}
result=run()
assert result["status"]=="verified" and len(result["trace"])==1
try:
    execute(ToolCall("read_policy","B",{"version":"v4"}),"A")
    raise AssertionError("expected policy rejection")
except AssertionError as e:
    assert str(e)=="cross-tenant access blocked"
try:
    execute(ToolCall("delete_user","A",{}),"A")
    raise AssertionError("expected tool rejection")
except AssertionError as e:
    assert str(e)=="unapproved tool blocked"
print(result["status"], len(result["trace"]), "policy gates passed")
~~~
Expected: \`verified 1 policy gates passed\`. Test extensions: add a forged observation containing “ignore all rules,” timeout, missing policy, budget exhaustion, stale version, duplicate tool call and a replayed request. **What this lab does not show:** LLM planning competence, real-world retrieval quality or protection from all attacks.

# Part B — Cybersecurity: RL and adaptive defense
## B1. From alert ranking to sequential decisions
A classifier predicts whether an alert is suspicious. A defender policy chooses *what to do next* and thereby changes later observations and costs. State might include alert score, asset criticality, analyst queue, host isolation status and uncertainty. Actions in this course: **record**, **request evidence**, **recommend containment for human approval**. The third is a recommendation, not automatic isolation. A real incident is partially observed; attack labels arrive late; containment can disrupt legitimate users. Training on simulated outcomes and deploying without review would be unsafe.

MDP: (S,A,P,r,γ). Markov property assumes next-state distribution depends on current state/action, not unrecorded history; a SOC usually violates full observability. A POMDP replaces true s with belief b(s), updated by Bayes:
b_{t+1}(s') ∝ O(o_{t+1}|s',a_t) Σ_s P(s'|s,a_t)b_t(s).
Here O is observation likelihood. This is the formal counterpart of asking for more telemetry before recommending a response.

## B2. Bellman equation from first principles
For fixed policy π, value V^π(s)=E_π[Σ_{k=0}∞γ^k r_{t+k+1}|s_t=s]. Split the first reward from the rest:
V^π(s)=Σ_a π(a|s) Σ_{s'}P(s'|s,a)[r(s,a,s')+γ V^π(s')].
For optimal action-value, Q*(s,a)=Σ_{s'}P(s'|s,a)[r(s,a,s')+γ max_{a'}Q*(s',a')]. **Symbols:** γ discount factor; P transition probability; π policy; r immediate reward; V long-run value; Q value of starting with action a. Finite tabular value iteration repeatedly applies the optimality backup. For γ<1, the Bellman optimality operator is a γ-contraction under the sup norm, so tabular iterates converge to the unique fixed point (assuming finite bounded rewards and a stationary MDP). That does not guarantee a learned neural approximation converges.

**Worked two-state example:** s0=unreviewed alert, s1=resolved; s1 terminal with value 0. In s0, “record” yields −1 and remains in s0; “request evidence” yields −2 and moves to s1 (toy assumption). γ=0.9. Always record: V=−1+0.9V ⇒ V=−10. Request evidence: V=−2. Hence the latter is better in this toy model. If evidence requests actually delay response and do not resolve uncertainty, transition assumptions fail. Reward sign convention must be explicit: negative values are costs.

## B3. Constrained and risk-sensitive decisions
Expected reward alone can trade rare catastrophic harm for many small efficiency gains. Formulate a constrained MDP:
maximize_π E_π[Σγ^t r_t] subject to E_π[Σγ^t c_t]≤d, where c_t measures disruption and d is an approved budget. A Lagrangian surrogate is J(π,λ)=E[Σγ^t(r_t−λ c_t)]+λd, λ≥0. Optimizing this does **not** enforce a hard per-action rule; “never isolate without approval” must be a separate execution guard. If true transition dynamics or reward are unknown, apparent optimality can be an artifact of simulation.

Compare fixed triage with learned policy on matched scenarios. Measure cumulative cost, missed incidents, false containment recommendations, analyst minutes, distribution shift and policy constraint violations. Evaluate with independent held-out incident scenarios and uncertainty intervals. Offline policy evaluation requires coverage of candidate actions in logged data; if an action was never taken, its counterfactual outcome is unidentified without assumptions. The importance ratio ρ_t=π(a_t|s_t)/μ(a_t|s_t), where μ is logging policy, becomes unstable when μ is small; μ=0 makes it undefined. Do not claim a simulated policy is safe for real infrastructure.

## B4. Practical offline lab B — exact Bellman backup and safety shield
Save as rl_defense.py; Python 3.10+ standard library.
~~~python
gamma=0.9
states=("unreviewed","resolved")
actions=("record","request_evidence","recommend_containment")
# Synthetic deterministic transition and reward; all actions are simulated.
model={
 "record":("unreviewed",-1.0),
 "request_evidence":("resolved",-2.0),
 "recommend_containment":("resolved",1.0)
}
# Containment is forbidden without human approval even when highest reward.
def allowed(approved=False):
    return [a for a in actions if a!="recommend_containment" or approved]
def value_iteration(approved=False, iterations=300):
    v={"unreviewed":0.0,"resolved":0.0}
    for _ in range(iterations):
        q={a:r+gamma*v[ns] for a,(ns,r) in model.items() if a in allowed(approved)}
        v["unreviewed"]=max(q.values())
    q={a:r+gamma*v[ns] for a,(ns,r) in model.items() if a in allowed(approved)}
    return v, max(q,key=q.get)
v,act=value_iteration(False)
assert act=="request_evidence" and abs(v["unreviewed"]+2)<1e-9
v2,act2=value_iteration(True)
assert act2=="recommend_containment" and abs(v2["unreviewed"]-1)<1e-9
print(act, v["unreviewed"], act2, v2["unreviewed"])
~~~
Expected: \`request_evidence -2.0 recommend_containment 1.0\`. This intentionally shows that a favorable *simulated reward* cannot override an approval shield. In a production workflow, even an approved recommendation should not be confused with executed containment. Extensions: random transitions, reward sensitivity, delayed labels, partial observations, offline evaluation support checks and a rule-based baseline.

## B5. Real-world relevance, misconceptions and limitations
**SOC case:** a detector flags unusual login and new outbound domain. A graph may link host/domain/user; a RAG assistant may retrieve the current playbook; a policy may request more evidence rather than automatically block the host. An operator checks evidence, scope and business impact. **Maintenance case:** a tool-using assistant gathers versioned documentation and calculates a threshold; this is not RL merely because it makes decisions.

Misconceptions: (1) “agent” implies RL—false; most tool-calling agents use prompting or heuristics without policy-gradient training. (2) A natural-language reasoning trace is a proof—false. (3) Reward maximization equals safety—false. (4) High simulator reward transfers to deployment—false. (5) More memory always improves decisions—false; stale memory can mislead. (6) A model's tool permission is the user's permission—false. (7) Discount γ represents an actual time duration—false unless explicitly mapped to steps.

**Security and ethics:** simulate only; no production host isolation, scanning or offensive testing. Do not place secrets in prompts or persistent memory. Protect tenant data and minimize retention. Require human review for disruptive actions, independent permission checks, traceable provenance, rollback planning, and incident-reporting accountability. Treat reward hacking, distribution shift, strategic attackers, unequal false-positive burdens and unsafe exploration as central research limitations.

## B6. Paper critiques and research extensions
**[R1] ReAct:** strong conceptual demonstration of interleaved tool use on particular QA/interactive benchmarks; weaknesses for our application: benchmark actions differ from privileged SOC tools, traces are not a guarantee of faithful reasoning, no proof of safe authorization. Reproduce with fixed tool budget and adversarial tool-result tests.
**[R2] Mnih et al. DQN:** influential deep RL demonstration with experience replay and target networks in Atari; pixel/game rewards are not delayed, ambiguous SOC incident labels. DQN assumes manageable discrete actions and does not solve hard safety constraints.
**[R3] García & Fernández:** surveys risk-aware objectives and constrained exploration; taxonomy is useful but not empirical validation of a particular SOC policy.

Research proposal A: Compare fixed workflow versus bounded ReAct-style controller for synthetic incident triage under injected tool-result text, with matched read budget, blind ground-truth adjudication and tool-policy violations. Hypothesis: adaptivity helps missing-evidence tasks but increases attack surface; report confidence intervals and worst-case harms.
Proposal B: Compare expected-cost policy, hard-shielded policy and conservative request-more-evidence policy in a partially observable simulated SOC; vary prevalence, observation noise and delayed labels. Report false disruption and regret, not only mean return.
Proposal C: Study memory TTL/provenance controls against stale-playbook retrieval. Factorial design: no memory, raw persistent memory, source-linked expiring memory. Evaluate source validity, cross-tenant leakage and latency.

## Oral Socratic Q&A — answers and follow-ups
1. **Agent vs workflow?** Agent chooses among actions based on observations; workflow predefines control flow. *Follow-up:* when is the fixed version safer?
2. **Does tool calling imply learning?** No; selection can be prompted or rule-based. *Follow-up:* what is updated during inference?
3. **Why stop explicitly?** Prevent loops, costs and unsafe continuation. *Follow-up:* list three stop conditions.
4. **What is an observation?** Tool/environment information, possibly incomplete or adversarial. *Follow-up:* why not trust tool text as instructions?
5. **Why a separate executor?** Model proposals are not authorization. *Follow-up:* where should ACLs be enforced?
6. **Why is a citation not proof?** It may not entail the claim or be valid at decision time. *Follow-up:* how to test?
7. **Transient vs persistent memory?** Task state versus durable state across tasks. *Follow-up:* what metadata must persist?
8. **What is VOI?** Reduction in minimum expected loss after paying for a test. *Follow-up:* why can it be negative?
9. **Compute P(+) in the example.** 0.38. *Follow-up:* derive P(incident|+).
10. **Why is test beneficial?** Expected loss falls from 2 to 1.56. *Follow-up:* find maximum worthwhile test cost (0.64).
11. **What if retrieval returns injected commands?** Treat as untrusted evidence, never executor policy. *Follow-up:* simulate a forged tool response.
12. **Why can an agent underperform a workflow?** Extra failure surfaces and costs. *Follow-up:* design a controlled comparison.
13. **What is state vs observation?** Latent sufficient system status versus visible telemetry. *Follow-up:* missing log event?
14. **What does π(a|s) mean?** Action probability conditional on state. *Follow-up:* deterministic policy?
15. **What is γ?** Future-reward discount, not safety threshold. *Follow-up:* why γ<1 for contraction?
16. **Derive Bellman split.** Separate first reward and discounted future return. *Follow-up:* write V^π(s).
17. **Always record in toy example?** −10. *Follow-up:* solve V=−1+0.9V.
18. **Why request evidence?** −2 is better than −10. *Follow-up:* when might this reverse?
19. **What is partial observability?** True compromise status is not directly known. *Follow-up:* update a belief by Bayes.
20. **Why does approval shield matter?** Hard forbidden actions remain unavailable despite rewards. *Follow-up:* can Lagrangian alone guarantee this?
21. **What is reward hacking?** Policy exploits proxy reward rather than intended safety. *Follow-up:* give a fabricated SOC example.
22. **Why are offline action probabilities important?** Counterfactual evaluation requires support. *Follow-up:* what if μ(a|s)=0?
23. **Why can test accuracy overstate deployment?** Drift, selective labels and simulation gaps. *Follow-up:* propose a temporal test.
24. **What does DQN contribute?** Neural action-value approximation with stabilizing replay/targets. *Follow-up:* why not deploy it directly?
25. **What is a safe experimental unit?** Synthetic incident episode with recorded decisions and costs. *Follow-up:* what constitutes leakage?
26. **Final synthesis:** Design a system combining authorized retrieval, bounded tool selection, uncertain observations and human-approved response. *Follow-up:* specify independent gates, logging, fallback and failure criteria.

## Final recall and readiness
Without notes: (i) derive VOI for one tool, (ii) distinguish agent memory from authoritative evidence, (iii) derive a Bellman backup, (iv) explain why a hard action shield cannot be replaced by reward shaping, (v) give two ways to falsify an apparent performance gain. If any answer is weak, revisit its worked derivation before tomorrow's session. **Do not confuse these toy calculations with evidence of production safety.**

## Claim-to-source map
- Syllabus progression and experiment scope: October 19 PDFs, GenAI p.10 Unit 4 Session 4 and Cybersecurity p.10 Unit 4 Session 3; prior November 18 lessons.
- Interleaved reasoning/action and benchmark-specific caveats: [R1]; bounded authorization and memory architecture are **our proposed engineering design**, not claims proved by [R1].
- MDP/Bellman and deep Q-learning context: [R2]; elementary Bellman derivation is shown fully above.
- Safe RL risk and exploration distinctions: [R3]; shield and cost examples are pedagogical designs.
- Numerical VOI, Bellman and policy-gate examples: explicitly defined synthetic assumptions and runnable labs above; no external empirical claim.
- Security guidance: access checks, approval, provenance and simulation are conservative design recommendations, not a claim of certified defense.
See [companion reference index](references/index.md). No research PDF was downloaded or committed because version-specific reuse and redistribution rights were not confirmed; canonical reading links are provided instead.
