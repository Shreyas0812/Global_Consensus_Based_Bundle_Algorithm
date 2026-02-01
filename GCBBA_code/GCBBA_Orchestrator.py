import networkx as nx
import numpy as np
import time
from tqdm import tqdm
from GCBBA_Task import GCBBA_Task
from GCBBA_Agent import GCBBA_Agent


class GCBBA_Orchestrator:
    """
    GCBBA Orchestrator for warehouse task allocation
    """
    def __init__(self, G, D, char_t, char_a, Lt=1, metric="RPT"):
        self.G = G
        # int, number of agents
        self.na = G.shape[0]
        # int, number of tasks
        self.nt = len(char_t)
        # capacity per agent
        self.Lt = Lt
        # task characteristics
        self.char_t = char_t
        # agent characteristics
        self.char_a = char_a
        # list of all agents
        self.agents = []
        # list of all tasks
        self.tasks = []
        
        # clock launch
        self.start_time = time.perf_counter()
        
        self.metric = metric
        self.D = D
        
        # initialize tasks and agents
        self.initialize_all()
        print("Orchestrator initialized with {} agents and {} tasks.".format(self.na, self.nt))
        self.bid_history = []
        self.assig_history = []
        self.max_times = []
        self.all_times = [0 for _ in range(self.na)]
        
        self.cvg_iter = self.nt
    
    def initialize_all(self):
        self.initialize_tasks()
        self.initialize_agents()
    
    def initialize_tasks(self):
        self.tasks = []
        for j in range(self.nt):
            char = self.char_t[j]
            self.tasks.append(GCBBA_Task(id=j, char=char))
    
    def initialize_agents(self):
        self.agents = []
        for i in range(self.na):
            char = self.char_a[i]
            self.agents.append(
                GCBBA_Agent(id=i, G=self.G, char=char, tasks=self.tasks, Lt=self.Lt, 
                           start_time=self.start_time, metric=self.metric, D=self.D))
    
    def launch_agents(self, method="global", detector="decentralized"):
        """
        Launch GCBBA allocation algorithm
        :param method: "baseline" or "global"
        :param detector: "none", "centralized", or "decentralized"
        :return: allocation, minsum, makespan
        """
        D = self.D
        Nmin = int(min(self.nt, self.Lt * self.na)) # minimum number of assigned tasks

        nb_iter = Nmin # number of main iterations
        nb_cons = 2*D  # number of consensus rounds per iteration

        for iter in tqdm(range(nb_iter)):
            for i in range(self.na):
                # Bundle creation phase
                self.agents[i]
                if self.agents[i].converged == False:
                    self.agents[i].create_bundle(iter)

        return None, None, None