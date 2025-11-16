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
                  │      • Generate random positions (x, y)
                  │      • Generate random speeds
                  │      • Return: List of [x_pos, y_pos, speed]
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
│              READY FOR GCBBA EXECUTION                       │
│              (To be implemented)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Agent Representation
```
Agent = [x_position, y_position, speed]
```
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
1. **Distance Calculation**: Compute pairwise distances between all agents
2. **Connectivity Check**: Determine which agents can communicate
3. **Graph Construction**: Build adjacency matrix representing network
4. **Topology Analysis**: Calculate diameter to understand communication delays

### Future Extensions (Placeholders in Code)
- Loading agents from YAML configuration files
- Loading tasks from YAML files (e.g., injection stations)
- Real-time task allocation algorithm
- Dynamic graph updates as agents move
- Consensus mechanism implementation

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
│   └── update_comm_graph(): Update network topology
│
└── Main Execution Block
    ├── Parameter Configuration
    ├── Agent Initialization
    ├── Task Initialization
    └── Orchestrator Creation
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
- Agent and task initialization
- Communication graph construction
- Diameter calculation
- Random scenario generation

⏳ **TODO (as noted in code):**
- YAML-based configuration loading
- Real-time task allocation algorithm
- Consensus mechanism
- Bundle building process
- Conflict resolution
- Dynamic updates during execution

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
