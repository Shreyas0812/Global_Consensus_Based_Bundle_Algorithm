import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
from math import *
import time
from tqdm import tqdm
from tools_real_time import random_agent_init, random_task_init

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
        self.na = len(agents) # number of agents
        self.agents = agents  # agents characteristics - [x_pos, y_pos, speed]
        self.Lt = Lt          # maximum number of tasks per agent   

        self.nt = len(tasks) # number of tasks
        self.tasks = tasks  # tasks characteristics - [x_pos, y_pos, duration, lambda, weight]

        self.metric = metric

        self.comm_range = comm_range

        # Launch Clock
        self.start_time = time.perf_counter()

        # Establish Communication Graph based on current positions
        self.G, self.D = self.update_comm_graph() # communication matrix G, diameter D

        print(f"Initial Communication Graph Diameter: {self.D}")
    
    def update_comm_graph(self):
        """
        Update communication graph based on current positions of agents
        :return: communication matrix G, distance matrix D
        """
        G = np.zeros((self.na, self.na))
        D = 1                               # initial diameter - Fully connected graph
        comm_range = self.comm_range

        for i in range(self.na):
            for j in range(self.na):
                if i != j:
                    dist_ij = np.linalg.norm(self.agents[i][0:2] - self.agents[j][0:2])
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
