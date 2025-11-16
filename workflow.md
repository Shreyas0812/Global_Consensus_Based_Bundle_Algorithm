# GCBBA Real-Time Algorithm Workflow

## Overview
This document describes the workflow and architecture of the Global Consensus-Based Bundle Algorithm (GCBBA) for real-time multi-agent task allocation with communication constraints.

---

## System Architecture

### Core Components

1. **Orchestrator_GCBBA Class**: Main orchestration class managing the entire GCBBA execution
2. **Agents**: Mobile entities that can perform tasks
3. **Tasks**: Jobs to be allocated and executed by agents
4. **Communication Graph**: Dynamic network representing agent connectivity

---

## Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                     START MAIN PROGRAM                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─ Set Random Seed (reproducibility)
                  │
                  ├─ Define Parameters:
                  │  • Number of agents (na)
                  │  • Number of tasks (nt)
                  │  • Max tasks per agent (Lt)
                  │  • Environment limits (xlim, ylim)
                  │  • Speed limits (sp_lim)
                  │  • Duration limits (dur_lim)
                  │  • Metric type ("RPT" or "TDR")
                  │  • Communication range
                  │
                  ├─ Initialize Agents:
                  │  └─> random_agent_init()
                  │      • Generate agent IDs (0 to na-1)
                  │      • Generate random positions (x, y)
                  │      • Generate random speeds
                  │      • Return: List of [agent_id, x_pos, y_pos, speed]
                  │
                  ├─ Initialize Tasks:
                  │  └─> random_task_init()
                  │      • Generate random positions (x, y)
                  │      • Generate random durations
                  │      • Generate lambda values (discount factor)
                  │      • Generate weights
                  │      • Return: List of [x_pos, y_pos, duration, lambda, weight]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│         CREATE Orchestrator_GCBBA INSTANCE                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─ Store Configuration:
                  │  • Number of agents (na)
                  │  • Agent characteristics
                  │  • Number of tasks (nt)
                  │  • Task characteristics
                  │  • Max tasks per agent (Lt)
                  │  • Performance metric
                  │  • Communication range
                  │
                  ├─ Start Performance Clock:
                  │  └─> self.start_time = time.perf_counter()
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│          UPDATE COMMUNICATION GRAPH                          │
│          update_comm_graph()                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─ Initialize Communication Matrix:
                  │  • Create G matrix (na x na) of zeros
                  │  • Set initial diameter D = 1
                  │
                  ├─ For each agent pair (i, j):
                  │  │
                  │  ├─ Calculate Euclidean Distance:
                  │  │  └─> dist_ij = ||agent[i].pos - agent[j].pos||
                  │  │
                  │  ├─ Check Communication Range:
                  │  │  └─> if dist_ij <= comm_range:
                  │  │      • Set G[i][j] = 1.0
                  │  │      • Set G[j][i] = 1.0 (undirected)
                  │  │
                  │  └─ Set Self-Connection:
                  │     └─> G[i][i] = 1.0
                  │
                  ├─ Build Network Graph:
                  │  └─> raw_graph = nx.from_numpy_array(G)
                  │
                  ├─ Check Connectivity:
                  │  ├─> if nx.is_connected(raw_graph):
                  │  │   └─ Calculate Diameter: D = nx.diameter(raw_graph)
                  │  └─> else:
                  │      └─ Set D = na - 1 (max diameter)
                  │      └─ Print Warning: "Graph not connected!"
                  │
                  ├─ Return: (G, D)
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│         DISPLAY INITIAL COMMUNICATION DIAMETER               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              LAUNCH GCBBA AGENTS                             │
│              launch_agents()                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─ Initialize Data Structures:
                  │  • task_assignments = {} (empty dict)
                  │  • tot_score = inf (min-sum objective)
                  │  • makespan = inf (max time objective)
                  │
                  ├─ For each agent:
                  │  ├─> Extract agent_id from agent[0]
                  │  └─> Initialize empty task list: 
                  │      task_assignments[agent_id] = []
                  │
                  ├─ Return:
                  │  └─> (task_assignments, tot_score, makespan)
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              DISPLAY RESULTS                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─ Print Task Assignments per Agent
                  ├─ Print Total Score (min-sum objective)
                  ├─ Print Makespan (max time objective)
                  └─ Print Execution Time
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              END PROGRAM                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Agent Representation
```
Agent = [agent_id, x_position, y_position, speed]
```
- **agent_id**: Unique identifier for the agent (0 to na-1)
- **x_position**: X-coordinate in the environment
- **y_position**: Y-coordinate in the environment
- **speed**: Movement speed of the agent

### Task Representation
```
Task = [x_position, y_position, duration, lambda, weight]
```
- **x_position**: X-coordinate of task location
- **y_position**: Y-coordinate of task location
- **duration**: Time required to complete the task
- **lambda**: Discount factor for Time-Discounted Reward (TDR) metric
- **weight**: Task weight/priority

### Communication Graph (G)
- **Type**: Adjacency matrix (na × na)
- **Values**: 
  - 1.0: Agents within communication range (connected)
  - 0.0: Agents out of communication range (not connected)
- **Properties**: 
  - Symmetric (undirected graph)
  - Self-loops (G[i][i] = 1.0)

### Task Assignments
```
task_assignments = {agent_id: [list of task_ids]}
```
- **Type**: Dictionary mapping agent IDs to assigned task lists
- **Key**: agent_id (integer from 0 to na-1)
- **Value**: List of task IDs assigned to that agent
- **Example**: `{0: [5, 12, 23], 1: [8, 15], 2: []}`

### Performance Objectives
- **tot_score**: Total score for min-sum optimization (lower is better)
  - Sum of travel times + task execution times across all agents
  - Currently initialized to infinity (awaiting implementation)
  
- **makespan**: Maximum time taken by any single agent (lower is better)
  - Represents the bottleneck agent's total time
  - Currently initialized to infinity (awaiting implementation)

---

## Key Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `na` | Number of agents | 10 |
| `nt` | Number of tasks | 100 |
| `Lt` | Maximum tasks per agent | ceil(nt/na) |
| `xlim`, `ylim` | Environment boundaries | [-5, 5] |
| `sp_lim` | Agent speed range | [1, 5] |
| `dur_lim` | Task duration range | [1, 5] |
| `metric` | Performance metric | "RPT" or "TDR" |
| `comm_range` | Communication radius | 30.0 |

---

## Performance Metrics

### RPT (Reward Per Time)
- Measures efficiency of task completion
- Focuses on maximizing reward rate

### TDR (Time-Discounted Reward)
- Incorporates temporal discounting
- Formula uses lambda parameter for time decay
- Prioritizes earlier task completion

---

## Algorithm Flow Details

### Initialization Phase
1. **Setup Environment**: Define spatial boundaries and operational constraints
2. **Generate Agents**: Create agents with random positions and speeds
3. **Generate Tasks**: Create tasks with random locations and characteristics
4. **Create Orchestrator**: Initialize GCBBA orchestration system

### Communication Graph Update
1. **Distance Calculation**: Compute pairwise distances between all agents (using positions at indices 1:3)
2. **Connectivity Check**: Determine which agents can communicate
3. **Graph Construction**: Build adjacency matrix representing network
4. **Topology Analysis**: Calculate diameter to understand communication delays

### GCBBA Execution Phase
1. **Initialize Tracking**: Create empty task assignment dictionary for each agent
2. **Set Objectives**: Initialize tot_score and makespan to infinity
3. **Prepare for Allocation**: Framework ready for bundle algorithm implementation
4. **Return Results**: Task assignments and performance metrics

### Future Extensions (Placeholders in Code)
- Loading agents from YAML configuration files
- Loading tasks from YAML files (e.g., injection stations)
- Core task allocation algorithm in launch_agents()
- Dynamic graph updates as agents move
- Consensus mechanism implementation
- Score and makespan calculation logic

---

## Code Organization

```
GCBBA_real_time.py
│
├── Imports
│   ├── networkx (graph operations)
│   ├── numpy (numerical operations)
│   ├── matplotlib (visualization)
│   └── tools_real_time (helper functions)
│
├── Orchestrator_GCBBA Class
│   ├── __init__(): Initialize system
│   ├── update_comm_graph(): Update network topology
│   └── launch_agents(): Execute GCBBA algorithm
│
└── Main Execution Block
    ├── Parameter Configuration
    ├── Agent Initialization
    ├── Task Initialization
    ├── Orchestrator Creation
    ├── GCBBA Execution
    └── Results Display
```

---

## Design Patterns

### Object-Oriented Design
- **Encapsulation**: All GCBBA logic contained in Orchestrator class
- **Modularity**: Separate helper functions for initialization
- **Extensibility**: TODO markers for future enhancements

### Performance Monitoring
- Built-in timing mechanism using `time.perf_counter()`
- Enables benchmarking and optimization

### Graph Theory Integration
- Uses NetworkX for sophisticated graph operations
- Enables efficient connectivity and diameter calculations

---

## Current Implementation Status

✅ **Completed:**
- Basic orchestrator structure
- Agent and task initialization with unique IDs
- Communication graph construction
- Diameter calculation
- Random scenario generation
- Basic launch_agents() framework
- Task assignment data structure
- Performance metrics tracking (tot_score, makespan)
- Execution time measurement

⏳ **TODO (as noted in code):**
- YAML-based configuration loading
- Task allocation algorithm implementation
- Consensus mechanism
- Bundle building process
- Conflict resolution
- Dynamic updates during execution
- Actual assignment logic in launch_agents()

---

## Dependencies

```
- numpy: Array operations and random generation
- networkx: Graph theory operations
- matplotlib: Visualization (imported but not yet used)
- copy: Deep copying (imported but not yet used)
- math: Mathematical functions
- time: Performance timing
- tqdm: Progress bars (imported but not yet used)
- tools_real_time: Custom helper functions
```

---

## Usage Example

```python
# Set seed for reproducibility
np.random.seed(5)

# Define system parameters
na = 10              # 10 agents
nt = 100             # 100 tasks
Lt = ceil(nt/na)     # Max 10 tasks per agent
comm_range = 30.0    # 30 units communication range

# Create environment
agents = random_agent_init(na=10, pos_lim=[-5,5], sp_lim=[1,5])
tasks = random_task_init(nt=100, pos_lim=[-5,5], dur_lim=[1,5])

# Initialize orchestrator
orch = Orchestrator_GCBBA(agents, tasks, Lt, "RPT", comm_range)

# Execute GCBBA
t_start = time.time()
task_assignments, tot_score, makespan = orch.launch_agents()
t_end = time.time()

# Display results
print("GCBBA Task Assignments:")
for agent_id, tasks in task_assignments.items():
    print(f"Agent {agent_id}: {tasks}")
print(f"Total Score (min-sum): {tot_score}")
print(f"Makespan: {makespan}")
print(f"GCBBA execution time: {np.round((t_end - t_start), 3)} seconds")
```

---

## Next Steps for Development

1. **Implement Task Allocation Logic**
   - Bundle building algorithm
   - Bid calculation based on metric
   - Task selection and assignment

2. **Add Consensus Mechanism**
   - Information sharing protocol
   - Conflict resolution
   - Convergence criteria

3. **Dynamic Updates**
   - Agent movement simulation
   - Periodic graph updates
   - Real-time re-planning

4. **Visualization**
   - Agent positions and trajectories
   - Task locations and status
   - Communication graph overlay
   - Performance metrics display

5. **Configuration Management**
   - YAML file parsers
   - Scenario loading
   - Parameter validation

---

*Document Version: 1.0*  
*Last Updated: November 15, 2025*  
*Algorithm: Global Consensus-Based Bundle Algorithm (GCBBA)*
