# GCBBA Rerun Strategy

## Question

> GCBBA - total score. = 849.530178; max score = 225.09040704210582; time = 6041.0 ms; assignment = [[3, 9, 65, 5, 41, 68, 48, 35, 77, 28, 58, 32], [14, 17, 12, 11, 30, 29, 10, 16, 4, 44, 78, 25, 62, 55], [26, 34, 33, 74, 70, 75, 73, 39, 60, 59, 69, 38], [47, 49, 57, 52, 2, 45, 15, 8, 50, 64, 24, 67, 20, 0], [72, 79, 71, 63, 31, 18, 43, 61, 21, 7, 66, 22, 42, 36], [53, 40, 46, 6, 54, 13, 27, 19, 56, 76, 51, 23, 1, 37]]
>
> This is my result. It is not feasible to run it every iteration right?
>
> What is the apt time to rerun this one?

---

## Answer

You're absolutely right — **6 seconds per run is too slow for every timestep**. Here's the practical reallocation strategy:

## When to Rerun GCBBA

### 🎯 Event-Driven Triggers (Recommended)

| Trigger | When to Fire | Rationale |
|---------|--------------|-----------|
| **Task Completion Batch** | When 20-30% of tasks complete | Rebalance workload as tasks finish |
| **New Tasks Arrive** | When new task batch is injected | Allocate new work to agents |
| **Agent Completes All Tasks** | An agent's path becomes empty | Idle agent needs new work |
| **Agent Failure** | Agent goes offline | Reassign orphaned tasks |
| **Significant Delay** | Task taking 2x expected time | Something went wrong, replan |

### ⏱️ Time-Based Fallback

As a safety net, rerun every **30-60 seconds** even if no events trigger, to handle edge cases.

---

## Practical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SYSTEM LOOP                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────┐     Run once initially,              │
│   │   GCBBA     │     then on triggers                 │
│   │  (6 sec)    │◄─── • 25% tasks complete             │
│   └──────┬──────┘     • New tasks arrive               │
│          │            • Agent idle                      │
│          ▼            • Every 30-60s fallback          │
│   ┌─────────────┐                                      │
│   │    RHCR     │     Run frequently                   │
│   │ (per step)  │◄─── • Every timestep                 │
│   └──────┬──────┘     • Handles collisions             │
│          │                                              │
│          ▼                                              │
│   ┌─────────────┐                                      │
│   │   Execute   │     Agents move                      │
│   │   Movement  │                                      │
│   └─────────────┘                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Sketch

```python
class WarehouseController:
    def __init__(self):
        self.last_gcbba_time = 0
        self.tasks_completed_since_replan = 0
        self.gcbba_replan_interval = 30  # seconds
        self.replan_threshold = 0.25     # 25% of tasks
        
    def should_rerun_gcbba(self, current_time, total_tasks):
        """Determine if GCBBA should be rerun"""
        
        # Trigger 1: Enough tasks completed
        if self.tasks_completed_since_replan >= self.replan_threshold * total_tasks:
            return True, "task_completion_batch"
        
        # Trigger 2: New tasks arrived
        if self.new_tasks_pending:
            return True, "new_tasks"
        
        # Trigger 3: Any agent is idle (empty path)
        if any(len(agent.p) == 0 for agent in self.agents):
            return True, "agent_idle"
        
        # Trigger 4: Time-based fallback
        if current_time - self.last_gcbba_time > self.gcbba_replan_interval:
            return True, "periodic"
        
        return False, None
    
    def main_loop(self):
        while not done:
            current_time = time.time()
            
            # Check if GCBBA replan needed
            should_replan, reason = self.should_rerun_gcbba(current_time, self.total_tasks)
            
            if should_replan:
                print(f"Replanning GCBBA: {reason}")
                self.run_gcbba()
                self.last_gcbba_time = current_time
                self.tasks_completed_since_replan = 0
            
            # RHCR runs every timestep (fast)
            self.run_rhcr_step()
            
            # Execute movement
            self.move_agents()
            
            # Track completions
            completed = self.check_task_completions()
            self.tasks_completed_since_replan += completed
```

---

## Performance Considerations

| Approach | Pros | Cons |
|----------|------|------|
| **Event-driven only** | Minimal computation | Might miss optimization opportunities |
| **Periodic only** | Simple to implement | Wastes computation, slow response |
| **Hybrid (recommended)** | Best of both | Slightly more complex |

---

## For Your Thesis

Given your 80 tasks / 6 agents scenario:

1. **Initial run**: 6 seconds is fine (one-time cost)
2. **Replan triggers**: 
   - After ~20 tasks complete (~25%)
   - When new tasks injected
   - Fallback every 30 seconds
3. **Between replans**: RHCR handles collision-free paths

This means GCBBA runs maybe **3-5 times** during the full execution rather than hundreds of times.
