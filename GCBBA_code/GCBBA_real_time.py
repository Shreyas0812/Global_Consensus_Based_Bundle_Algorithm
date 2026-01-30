"""
GCBBA Real-Time Implementation


Notes for self

Think about in real time robotics and where each of these modules will go.

But even before that, implement GCBBA in real time with moving agents and tasks being added dynamically.

First implement GCBBA as is, but with agents moving in real time.

"""


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from math import *
import time
from tqdm import tqdm
from tools_real_time import random_agent_init, random_task_init

class GCBBA_AGENT:
    """
    GCBBA Agent class, defined by an id, a position (x,y), and a speed
    """
    def __init__(self, id, agent_info):
        self.id = id
        # agent_info expected as [x_pos, y_pos, speed]
        self.pos = np.array([agent_info[0], agent_info[1]])
        self.speed = float(agent_info[2])

        # Communication parameters
        self.comm_range = 30.0  # communication range
        self.G = None           # communication matrix
        self.D = None           # diameter of communication graph

        self.makespan = 0
        self.updated_makespan = 0

        # Tasks not in bundle 
        # TODO: Later, tasks will be added dynamically, for now, initialize with all tasks
        self.nt = 10
        self.filtered_tasks = list(range(self.nt))  # all tasks initially

        self.bundle = []        # current bundle of tasks
        self.path = []          # current path of tasks

        self.flag_won = True    # flag indicating if agent won last consensus
        self.placement = []      # Array that stores optimal placement of tasks in bundle before adding to path
        #TODO: A node which will detect and update communication graph periodically, For now, done in orchestrator when all agents have been initialized

    def create_bundle(self):
        """
        Create bundle of tasks for the agent
        """
        self.updated_makespan = self.makespan
        print(f"Agent {self.id} creating bundle...")

        if len(self.bundle) < self.Lt: # maximum number of tasks not reached
            self.filtered_tasks = [task_id for task_id in self.filtered_tasks if task_id not in self.path]
            if self.flag_won == True: # last consensus won, recalculate gains and optimal placement for all tasks
                self.placement = np.zeros(self.nt)
                for task_idx in self.filtered_tasks:
                    c, opt_placement = self.compute_c(task_idx)
                    self.c[task_idx] = c
                    self.placement[task_idx] = opt_placement

                    bids = []
                    for j in range(self.nt):
                        can_bid = False

                        if j in self.filtered_tasks:
                            if self.c[j] > self.y[j]:
                                can_bid = True
                            elif self.c[j] == self.y[j] and self.id < self.z[j]:
                                can_bid = True

                        if can_bid:
                            bids.append(self.c[j])
                        else:
                            bids.append(self.min_value)
                    
                    J = np.argmax(bids)
            else: # last consensus lost, reuse previous calculations
                bids = []
                for j in range(self.nt):
                    can_bid = False

                    if j in self.filtered_tasks:
                        if self.c[j] > self.y[j]:
                            can_bid = True
                        elif self.c[j] == self.y[j] and self.id < self.z[j]:
                            can_bid = True

                    if can_bid:
                        bids.append(self.c[j])
                    else:
                        bids.append(self.min_value)

                J = np.argmax(bids)
            
            if J in self.path or bids[J] <= self.min_value:
                print(f"Agent {self.id} could not add any task to bundle.")
                return  # No task can be added

            best_task = J
            self.bundle.append(best_task)
            self.path.insert(int(self.placement[J]), best_task)
            self.S.append(self.evaluate_path(self.path))
            self.y[J]   = self.c[J]
            self.z[J]   = self.id
            self.updated_makespan = max(self.updated_makespan, -self.evaluate_path(self.path))


class Task:
    """
    Task class, defined by an id, a position (x,y), duration, and lambda (only for TDR)
    """
    def __init__(self, id, task_info):
        self.id = id
        self.pos = np.array([task_info[0], task_info[1]])
        self.duration = task_info[2]
        self.lamb = task_info[3]

class Orchestrator_GCBBA:
    """
    Orchestrated GCBBA class
    """
    def __init__(self, agents, tasks, Lt, metric = "RPT", comm_range = 30.0):
        """
        Initialization of orchestrated GCBBA
        :param agents: list of agents characteristics
        :param tasks: list of tasks characteristics
        :param Lt: maximum number of tasks per agent
        :param metric: "RPT" or "TDR"
        """
        self.na = len(agents)   # number of agents
        self.agents = []        # list of Agent objects
        self.Lt = Lt            # maximum number of tasks per agent

        self.nt = len(tasks)    # number of tasks
        self.tasks = []         # list of Task objects

        self.metric = metric

        self.comm_range = comm_range

        # Launch Clock
        self.start_time = time.perf_counter()

        # initialize
        # Populates self.tasks with Task objects
        self.initialize_tasks(tasks)        
        # Populates self.agents with Agent objects
        self.initialize_agents(agents)             
    
    def initialize_tasks(self, tasks_list):
        """
        Initialize tasks for GCBBA
        tasks characteristics - [x_pos, y_pos, duration, lambda, weight]
        """
        self.tasks = []
        for i in range(self.nt):
            task_info = tasks_list[i]
            self.tasks.append(Task(id=i, task_info=task_info))

    def initialize_agents(self, agents_list):
        """
        Initialize agents for GCBBA
        agents characteristics - [agent_id, x_pos, y_pos, speed]
        """
        # TODO: Define GCBBA_Agent class
        self.agents = []
        for i in range(self.na):
            agent_info = agents_list[i]
            gcbba_agent_object = GCBBA_AGENT(id=i, agent_info=agent_info)
            self.agents.append(gcbba_agent_object)

        # TODO: Some module in ROS which detects and updates graph will be added to GCBBA_Agent, done here with other locations for now

        # TODO: Orchestrator should not be updating agent comm graph, each agent should do it themselves, done here for now
        self.G, self.D = self.update_comm_graph()
        for agent in self.agents:
            agent.G = self.G
            agent.D = self.D


    def launch_agents(self):
        """
        Launch GCBBA for all agents
        :return:
        """
        task_assignments = {}  # dictionary of task assignments {agent_id: [task_ids]}
        tot_score = np.inf          # min-sum objective (total travel + task duration time), Goal is to minimize this
        makespan = np.inf           # makespan objective (maximum time taken by any agent), Goal is to minimize this

        for agent in self.agents:
            agent_id = agent.id
            task_assignments[agent_id] = []  # initialize empty assignment for each agent

        # TODO: Implement GCBBA algorithm
        # - Bundle building phase
        # - Consensus phase
        # - Task allocation and assignment
        # - Calculate tot_score and makespan

        D = self.D
        Nmin = int(min(self.nt, self.Lt * self.na)) # total number of iterations

        build_bundle = "ADD"
        nb_consensus = 2 * D                        # number of consensus rounds per iteration
        nb_iter = Nmin                              # number of iterations


        for iter in tqdm(range(nb_iter)):
            # Bundle Building Phase
            for agent in self.agents:
                agent.create_bundle()

            # Consensus Phase
            for _ in range(nb_consensus):
                for agent in self.agents:
                    pass  # TODO: Implement consensus for each agent

            # Task Assignment Phase
            for agent in self.agents:
                pass  # TODO: Implement task assignment for each agent

        return task_assignments, tot_score, makespan
    
    # Function which updates based on agents current position, will be a node in ROS later
    def update_comm_graph(self):
        """
        Update communication graph based on current positions of agents
        :return: communication matrix G, distance matrix D
        """

        G = np.zeros((self.na, self.na))
        D = 1                                       # initial diameter - fully connected (placeholder)

        comm_range = self.comm_range

        for i in range(self.na):
            for j in range(self.na):
                if i != j:
                    # use agent positions stored on agent objects
                    dist_ij = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                    if dist_ij <= comm_range:
                        G[i][j] = 1.0
                        G[j][i] = 1.0 # undirected graph 

            G[i][i] = 1.0

        raw_graph = nx.from_numpy_array(G)
        if nx.is_connected(raw_graph):
            D = nx.diameter(raw_graph)
        else:
            D = self.na - 1  # set to max diameter if not connected
            print("Warning: Communication graph is not connected!")

        return G, D

if __name__ == "__main__":
    """ 
    To see GCBBA execution 
    """

    seed = 5
    np.random.seed(seed)

    # Numbrt of agents
    na = 10

    # Number of tasks
    nt = 100

    # Maximum number of tasks per agent
    Lt = ceil(nt/na)

    #Creating environment
    xlim = [-5, 5]      # x limits
    ylim = [-5, 5]      # y limits
    sp_lim = [1, 5]     # speed limits
    dur_lim = [1, 5]    # task duration limits

    metric = "RPT"

    # Communication range
    comm_range = 30.0

    # Creating agents randomly
    agents = random_agent_init(na=na, pos_lim=xlim, sp_lim=sp_lim)

    # TODO: Creating agents from yaml file

    # Creating tasks randomly
    tasks = random_task_init(nt=nt, pos_lim=xlim, dur_lim=dur_lim)

    # TODO: Creating tasks from yaml file -- inject stations

    orch_GCBBA = Orchestrator_GCBBA(agents, tasks, Lt, metric, comm_range)

    # GCBBA execution
    t_start = time.time()

    # task_assignments = {}  # dictionary of task assignments {agent_id: [task_ids]}
    task_assignments, tot_score, makespan = orch_GCBBA.launch_agents()

    t_end = time.time()

    print("GCBBA Task Assignments:")
    for agent_id, tasks in task_assignments.items():
        print(f"Agent {agent_id}: {tasks}")
    print(f"Total Score (min-sum): {tot_score}")
    print(f"Makespan: {makespan}")
    print(f"GCBBA execution time: {np.round((t_end - t_start), 3)} seconds")
