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
    def __init__(self, agents, tasks, Lt, metric = "RPT"):
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

        # Launch Clock
        self.start_time = time.perf_counter()

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

    # Creating agents randomly
    agents = random_agent_init(na=na, pos_lim=xlim, sp_lim=sp_lim)

    # TODO: Creating agents from yaml file

    # Creating tasks randomly
    tasks = random_task_init(nt=nt, pos_lim=xlim, dur_lim=dur_lim)

    # TODO: Creating tasks from yaml file -- inject stations

    orch_GCBBA = Orchestrator_GCBBA(agents, tasks, Lt, metric)
