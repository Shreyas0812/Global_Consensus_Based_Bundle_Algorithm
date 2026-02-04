# GCBBA Warehouse Point Task - Expected Outcomes and Validation Guide

**Document Version:** 2.0  
**Author:** Shreyas  
**Date:** February 2026  
**Algorithm:** Global Consensus-Based Bundle Algorithm (GCBBA) with RPT Metric

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [RPT Metric Deep Dive](#3-rpt-metric-deep-dive)
4. [Algorithm Walkthrough](#4-algorithm-walkthrough)
5. [Test Scenarios with Expected Outputs](#5-test-scenarios-with-expected-outputs)
6. [Validation Checklist](#6-validation-checklist)
7. [Common Issues and Debugging](#7-common-issues-and-debugging)

---

## 1. Introduction

This document provides a comprehensive guide for validating the GCBBA implementation for warehouse task allocation. The focus is exclusively on the **RPT (Repeated Path Times)** metric, which is a **Diminishing Marginal Gain (DMG)** function that guarantees algorithm convergence and 50% optimality.

### 1.1 System Components

| Component | File | Purpose |
|-----------|------|---------|
| **Orchestrator** | `GCBBA_Orchestrator.py` | Coordinates agents, manages iterations |
| **Agent** | `GCBBA_Agent.py` | Bundle building, conflict resolution |
| **Task** | `GCBBA_Task.py` | Task representation |
| **Entry Point** | `GCBBA_Warehouse.py` | Configuration and execution |

### 1.2 Key Parameters

```python
na      # Number of agents
nt      # Number of tasks
Lt      # Maximum tasks per agent (capacity)
D       # Communication graph diameter
Nmin    # min(nt, Lt × na) - total iterations
nb_cons # 2 × D - consensus rounds per iteration
```

---

## 2. Mathematical Foundation

### 2.1 Core Notation

| Symbol | Description |
|--------|-------------|
| $A$ | Set of agents, $\|A\| = N_a$ |
| $T$ | Set of tasks, $\|T\| = N_t$ |
| $p_i$ | Path (ordered task sequence) for agent $i$ |
| $b_i$ | Bundle (task assignment order) for agent $i$ |
| $S_i(p_i)$ | Utility/score function for path $p_i$ |
| $c_{ij}(p_i)$ | Marginal gain (bid) of agent $i$ for task $j$ |
| $y_j$ | Winning bid for task $j$ |
| $z_j$ | Winning agent for task $j$ |
| $\tau_{ij}(p_i)$ | Completion time for task $j$ by agent $i$ |

### 2.2 Path Insertion Operator

The operator $\oplus_{opt}$ inserts a task at the optimal position:

$$\ell^* = \arg\max_{1 \leq \ell \leq |p_i|+1} \left[ S_i(p_i \oplus_\ell j) - S_i(p_i) \right]$$

where $p_i \oplus_\ell j$ is the path with task $j$ inserted at position $\ell$.

### 2.3 Marginal Gain (Bid) Definition

$$c_{ij}(p_i) = \begin{cases} 0 & \text{if } j \in p_i \\ S_i(p_i \oplus_{opt} j) - S_i(p_i) & \text{otherwise} \end{cases}$$

For **RPT** specifically:
$$c_{ij}^{RPT}(p_i) = S_i(p_i \oplus_{opt} j)$$

Note: In the implementation, since we're maximizing a negative score, we use the absolute path score rather than the marginal gain difference for RPT.

---

## 3. RPT Metric Deep Dive

### 3.1 RPT Formula

The RPT (Repeated Path Times) function is defined as:

$$S_i^{RPT}(p_i) = -\sum_{j \in p_i} L_i(p_i^{:j} \oplus_{opt} j)^\alpha \cdot W_i(p_i^{:j} \oplus_{opt} j)^\beta$$

With standard parameters $\alpha = 1, \beta = 0$:

$$S_i^{RPT}(p_i) = -\sum_{j \in p_i} L_i(p_i^{:j} \oplus_{opt} j)$$

where $L_i(p)$ is the completion time (makespan) of path $p$.

### 3.2 Completion Time Calculation

For a path $p_i = [j_1, j_2, ..., j_n]$:

$$L_i(p_i) = \tau_{i,j_n}(p_i)$$

where the completion time $\tau$ for each task is:

$$\tau_{ij}(p_i) = \sum_{k=1}^{\text{idx}(j)} \left( \frac{d(pos_{k-1}, pos_k)}{v_i} + t_{dur}(j_k) \right)$$

### 3.3 RPT Implementation in Code

```python
def evaluate_path(self, path):
    """
    Evaluate path using RPT metric.
    
    Returns: Negative sum of completion times for each task
    """
    cur_pos = self.pos
    score = 0
    time = 0
    
    for task_id in path:
        task = self.tasks[task_id]
        
        # Travel time
        travel_time = np.linalg.norm(cur_pos - task.pos) / self.speed
        time += travel_time
        
        # Task duration
        time += task.duration
        
        # RPT: subtract completion time
        score -= time
        
        # CRITICAL: Reset time for next task (this is the "Repeated" in RPT)
        time = 0
        
        cur_pos = task.pos
    
    return score
```

### 3.4 Why RPT is DMG

**Theorem (from paper):** $c_{ij}^{RPT}(p_i)$ is Diminishing Marginal Gain.

**Proof Intuition:**
- Adding task $k$ to path $p_i$ can only increase the path length
- This means inserting task $j$ after adding $k$ will have a worse (more negative) score
- Therefore: $c_{ij}(p_i \oplus_{opt} k) \leq c_{ij}(p_i)$

**Practical Implication:** Bids decrease as agents acquire more tasks, ensuring convergence.

---

## 4. Algorithm Walkthrough

### 4.1 GCBBA Main Loop

```
for iteration in range(Nmin):
    # Phase 1: Bundle Building (each agent adds at most 1 task)
    for agent in agents:
        if not agent.converged:
            agent.create_bundle()
    
    # Phase 2: Consensus (2×D rounds)
    for consensus_round in range(2 * D):
        agents_copy = deepcopy(agents)
        for agent in agents:
            if not agent.converged:
                agent.resolve_conflicts(agents_copy, ...)
```

### 4.2 Bundle Building Phase

**Step 1: Compute bids for all unassigned tasks**
```python
for task_id in unassigned_tasks:
    c[task_id], placement[task_id] = compute_c(task_id)
```

**Step 2: Determine which tasks can be bid on**
```python
for j in range(nt):
    if c[j] > y[j]:  # My bid beats current winner
        bids[j] = c[j]
    elif c[j] == y[j] and z[j] > self.id:  # Tie-breaker
        bids[j] = c[j]
    else:
        bids[j] = min_val
```

**Step 3: Select best task and add to bundle**
```python
best_idx = argmax(bids)
b.append(tasks[best_idx].id)
p.insert(placement[best_idx], tasks[best_idx].id)
y[best_idx] = c[best_idx]
z[best_idx] = self.id
```

### 4.3 Conflict Resolution Phase

For each neighbor $k$, for each task $j$:

| Neighbor thinks | I think | Action |
|-----------------|---------|--------|
| $z_{kj} = k$ (k won) | $z_j = i$ (I won) | Compare bids, **update** if k's is better |
| $z_{kj} = k$ | $z_j = k$ | **update** (accept k's info) |
| $z_{kj} = k$ | $z_j = None$ | **update** (accept k's claim) |
| $z_{kj} = k$ | $z_j = m \neq i, k$ | Compare timestamps, **update** or **reset** |
| $z_{kj} = i$ (k says I won) | $z_j = i$ | **leave** (agree) |
| $z_{kj} = i$ | $z_j = k$ | **reset** (inconsistency) |
| ... | ... | (see full conflict table in paper) |

### 4.4 Convergence Detection

The decentralized convergence detection uses the $E^{cvg}$ vector:

$$E_i^{cvg}[0] = (z_i = z_i^{before})$$

$$E_i^{cvg}[d] = E_i^{cvg}[d-1] \land \bigwedge_{k \in \text{neighbors}} E_k^{cvg}[d-1]$$

Agent converges when: $E_i^{cvg}[D-1] = \text{True}$

---

## 5. Test Scenarios with Expected Outputs

### 5.1 Scenario A: Single Agent, Single Task

**Configuration:**
```python
Agent 0: pos=(0, 0), speed=1.0
Task 0:  pos=(3, 0), duration=2.0
Lt = 1
```

**Step-by-Step Calculation:**

1. **Path Evaluation for [0]:**
   - Travel time: $\frac{\sqrt{(3-0)^2 + (0-0)^2}}{1.0} = \frac{3}{1} = 3.0$s
   - Task duration: $2.0$s
   - Completion time: $\tau_0 = 3.0 + 2.0 = 5.0$s
   - RPT score: $S = -5.0$

2. **Bid Computation:**
   - Only position 0 available
   - $c[0] = -5.0$
   - Optimal position: 0

3. **Bundle Building:**
   - $c[0] = -5.0 > y[0] = -\infty$
   - Task 0 added to bundle
   - $y[0] = -5.0, z[0] = 0$

**Expected Output:**
```
Assignment: {Agent 0: [Task 0]}
Total Score (min-sum): 5.0
Makespan: 5.0
```

---

### 5.2 Scenario B: Single Agent, Two Tasks (Sequential)

**Configuration:**
```python
Agent 0: pos=(0, 0), speed=1.0
Task 0:  pos=(2, 0), duration=1.0
Task 1:  pos=(5, 0), duration=1.0
Lt = 2
```

**Iteration 1 - Bundle Building:**

| Task | Score if added | Position |
|------|----------------|----------|
| 0    | -(2+1) = -3.0  | 0 |
| 1    | -(5+1) = -6.0  | 0 |

Best: Task 0 with bid -3.0
Bundle after: [0], Path: [0]

**Iteration 2 - Bundle Building:**

With path [0], evaluate adding Task 1:

| Insert Position | Path After | Score Calculation |
|-----------------|------------|-------------------|
| 0 (before task 0) | [1, 0] | τ₁=(5+1)=-6, τ₀=(3+1)=-4, Total=-10 |
| 1 (after task 0)  | [0, 1] | τ₀=(2+1)=-3, τ₁=(3+1)=-4, Total=-7 |

Best: Position 1 with bid -7.0
Bundle after: [0, 1], Path: [0, 1]

**Expected Output:**
```
Assignment: {Agent 0: [Task 0, Task 1]}
Final Path: [0, 1]
Total Score: 7.0
Path Breakdown:
  - Task 0: travel=2.0, duration=1.0, completion=3.0
  - Task 1: travel=3.0, duration=1.0, completion=4.0
  - Sum of completions: 7.0
```

---

### 5.3 Scenario C: Two Agents, Two Tasks (Conflict Resolution)

**Configuration:**
```python
Agent 0: pos=(0, 0), speed=1.0
Agent 1: pos=(6, 0), speed=1.0
Task 0:  pos=(2, 0), duration=1.0
Task 1:  pos=(4, 0), duration=1.0
G = [[1, 1], [1, 1]]  # Fully connected
D = 1, Lt = 1
```

**Iteration 1 - Bundle Building:**

Agent 0 bids:
- Task 0: -(2+1) = -3.0 ✓
- Task 1: -(4+1) = -5.0

Agent 1 bids:
- Task 0: -(4+1) = -5.0
- Task 1: -(2+1) = -3.0 ✓

Both agents select their closest task.

After bundle building:
- Agent 0: b=[0], y[0]=-3.0, z[0]=0
- Agent 1: b=[1], y[1]=-3.0, z[1]=1

**Consensus Phase:**

No conflicts! Each agent claimed different tasks.

**Expected Output:**
```
Assignment: 
  Agent 0: [Task 0]
  Agent 1: [Task 1]
Total Score: 6.0 (3.0 + 3.0)
Makespan: 3.0
```

---

### 5.4 Scenario D: Two Agents, One Task (Conflict)

**Configuration:**
```python
Agent 0: pos=(0, 0), speed=1.0
Agent 1: pos=(1, 0), speed=1.0  # Closer to task!
Task 0:  pos=(3, 0), duration=1.0
G = [[1, 1], [1, 1]]
D = 1, Lt = 1
```

**Iteration 1 - Bundle Building:**

Agent 0 bids on Task 0: -(3+1) = -4.0
Agent 1 bids on Task 0: -(2+1) = -3.0 (BETTER!)

Both claim Task 0:
- Agent 0: y[0]=-4.0, z[0]=0
- Agent 1: y[0]=-3.0, z[0]=1

**Consensus Phase:**

Agent 0 receives from Agent 1:
- neigh.z[0] = 1 (Agent 1 claims task 0)
- self.z[0] = 0 (I claim task 0)
- neigh.y[0] = -3.0 > self.y[0] = -4.0

**Conflict Resolution Rule:**
> If neighbor claims task AND my bid is worse → UPDATE

Agent 0 updates:
- y[0] = -3.0 (neighbor's bid)
- z[0] = 1 (neighbor wins)
- Remove task 0 from bundle

**Expected Output:**
```
Assignment:
  Agent 0: []
  Agent 1: [Task 0]
Total Score: 3.0
Makespan: 3.0
```

---

### 5.5 Scenario E: Three Agents, Five Tasks (Complex)

**Configuration:**
```python
Agent 0: pos=(0, 0), speed=1.0
Agent 1: pos=(5, 0), speed=1.0
Agent 2: pos=(10, 0), speed=1.0

Task 0: pos=(1, 0), duration=1.0
Task 1: pos=(4, 0), duration=1.0
Task 2: pos=(6, 0), duration=1.0
Task 3: pos=(9, 0), duration=1.0
Task 4: pos=(12, 0), duration=1.0

G = [[1,1,0], [1,1,1], [0,1,1]]  # Linear connectivity
D = 2, Lt = 2
```

**Expected Behavior:**
- Agent 0 claims Task 0 (closest), then Task 1
- Agent 1 claims Task 2 (closest)
- Agent 2 claims Task 3 (closest), then Task 4

**Note:** With limited connectivity (D=2), convergence requires 2×D=4 consensus rounds per iteration to propagate information.

**Expected Output:**
```
Assignment:
  Agent 0: [0, 1]
  Agent 1: [2]
  Agent 2: [3, 4]
```

---

### 5.6 Scenario F: Path Ordering Matters

**Configuration:**
```python
Agent 0: pos=(0, 0), speed=1.0
Task 0: pos=(1, 1), duration=1.0  # Close to origin
Task 1: pos=(5, 5), duration=1.0  # Far from origin
Lt = 2
```

**Comparison of orderings:**

**Path [0, 1]:**
- τ₀ = √2/1 + 1 ≈ 2.414
- τ₁ = √32/1 + 1 ≈ 6.657 (travel from (1,1) to (5,5))
- Total: ≈ -9.071

**Path [1, 0]:**
- τ₁ = √50/1 + 1 ≈ 8.071 (travel from (0,0) to (5,5))
- τ₀ = √32/1 + 1 ≈ 6.657 (travel from (5,5) to (1,1))
- Total: ≈ -14.728

**Conclusion:** Path [0, 1] is significantly better. The algorithm should choose this ordering.

---

## 6. Validation Checklist

### 6.1 Unit Test Checklist

- [ ] **Task Creation**: Verify task ID, position, duration storage
- [ ] **Agent Initialization**: Verify all data structures initialized correctly
- [ ] **Empty Path Score**: Should return 0
- [ ] **Single Task Score**: Verify travel + duration calculation
- [ ] **Multi-Task Score**: Verify RPT time reset behavior
- [ ] **Bid Computation**: Verify all insertion positions evaluated
- [ ] **Optimal Position Selection**: Verify best position chosen

### 6.2 Integration Test Checklist

- [ ] **Bundle Building**: Tasks added to bundle correctly
- [ ] **Capacity Limit**: Lt enforced
- [ ] **Winning Bid Update**: y[j] and z[j] updated correctly
- [ ] **Conflict Detection**: Conflicts identified between agents
- [ ] **Update Action**: Neighbor's bid accepted, tasks removed
- [ ] **Reset Action**: Bid cleared, cascading removal works
- [ ] **Convergence**: Algorithm terminates correctly

### 6.3 Performance Checklist

- [ ] **Min-Sum**: Total path completion times correct
- [ ] **Makespan**: Maximum agent time correct
- [ ] **Convergence Iteration**: Recorded correctly
- [ ] **No Duplicate Assignments**: Each task assigned once

---

## 7. Common Issues and Debugging

### 7.1 Issue: Tasks Not Being Assigned

**Symptoms:** Empty assignments despite available tasks

**Possible Causes:**
1. Bids not exceeding y[j] (check initialization of y to min_val)
2. Capacity Lt already reached
3. Task already in path (filtering not working)

**Debug Steps:**
```python
print(f"Agent {self.id} bids: {self.c}")
print(f"Current y: {self.y}")
print(f"Bundle size: {len(self.b)} / {self.Lt}")
```

### 7.2 Issue: Incorrect Path Scores

**Symptoms:** Scores don't match hand calculations

**Possible Causes:**
1. Time not resetting after each task (RPT specific!)
2. Using task index instead of task ID
3. Wrong distance calculation

**Verification:**
```python
# Manual calculation
travel = np.linalg.norm(agent.pos - task.pos) / agent.speed
completion = travel + task.duration
expected_score = -completion
```

### 7.3 Issue: Convergence Not Detected

**Symptoms:** Algorithm runs full Nmin iterations

**Possible Causes:**
1. z_before not being updated correctly
2. their_net_cvg propagation broken
3. D value incorrect

**Debug:**
```python
print(f"Agent {self.id} z: {self.z}")
print(f"Agent {self.id} z_before: {self.z_before}")
print(f"Agent {self.id} cvg vector: {self.their_net_cvg}")
```

### 7.4 Issue: Cascading Removal Not Working

**Symptoms:** Tasks remain in path after conflict resolution

**Expected Behavior:** When task at index k is removed, all tasks from k onwards should be removed.

**Code Check:**
```python
def update(self, neighbor, task_id):
    if task_id in self.b:
        bundle_index = self.b.index(task_id)
        tasks_to_remove = self.b[bundle_index:]  # All from index onwards
        # Clear all these tasks
        for t in tasks_to_remove:
            idx = self._get_task_index(t)
            self.y[idx] = self.min_val
            self.z[idx] = None
        self.b = self.b[:bundle_index]
        # Remove from path too!
        for t in tasks_to_remove:
            if t in self.p:
                self.p.remove(t)
```

---

## Appendix A: Quick Reference Formulas

### RPT Score
$$S_i^{RPT}(p_i) = -\sum_{j \in p_i} \tau_{ij}(p_i^{:j})$$

### Completion Time
$$\tau_{ij} = \frac{d(\text{prev\_pos}, \text{task\_pos})}{v_i} + t_{dur}(j)$$

### Bid (RPT)
$$c_{ij}^{RPT}(p_i) = S_i(p_i \oplus_{opt} j)$$

### Optimal Insertion
$$\ell^* = \arg\max_\ell S_i(p_i \oplus_\ell j)$$

### DMG Property
$$c_{ij}(p_i \oplus_{opt} k) \leq c_{ij}(p_i), \quad \forall j, k \notin p_i$$

### Convergence Bound
$$\text{max iterations} = N_{min} \cdot D, \quad N_{min} = \min(N_t, L_t \cdot N_a)$$

---

## Appendix B: Sample Test Output

```
================================================================================
GCBBA Warehouse Test Suite - Expected Output
================================================================================

Test: Single Agent, Single Task
  Agent 0 at (0,0), Task 0 at (3,0) dur=2
  Expected score: -5.0
  ✓ PASSED

Test: Path Order [0,1] vs [1,0]
  [0,1] score: -6.0
  [1,0] score: -8.0
  ✓ [0,1] correctly identified as better

Test: Two Agents Conflict Resolution
  Agent 0 bid: -4.0
  Agent 1 bid: -3.0 (better)
  After consensus: Agent 1 wins
  ✓ Conflict resolved correctly

Test: Convergence Detection
  Converged at iteration 3 (Nmin=5)
  ✓ Early convergence detected

================================================================================
All tests passed!
================================================================================
```

---

*End of Document*
