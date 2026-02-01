"""
GCBBA Agent class for warehouse task allocation
"""

import numpy as np
from math import inf


class GCBBA_Agent:
    """
    GCBBA Agent for warehouse operations
    """
    def __init__(self, id, G, char, tasks, Lt=2, start_time=0, metric="RPT", D=1):
        # int, id of agent
        self.id = id
        # communication matrix G (symmetrical), size Na * Na
        self.G = G
        # int, nb of agents
        self.na = G.shape[0]
        # int, number of neighbors according to G
        self.nb_neigh = np.sum(self.G[id, :])
        # tuple, position in cartesian plane
        self.pos = np.array([char[0], char[1]])
        self.speed = char[2]
        self.agent_id = char[3]  # Unique agent identifier from warehouse config
        # list of tasks
        self.tasks = tasks
        # int, nb of tasks
        self.nt = len(tasks)
        # capacity
        self.Lt = Lt
        self.D = D
        
        self.metric = metric
        if self.metric == "RPT":
            self.min_val = -1e20
        elif metric == "TDR":
            self.min_val = 0
        
        # list of winning bids for each task (size Nt)
        self.y = [self.min_val for _ in range(self.nt)]
        # list of winners for each task (size Nt)
        self.z = [None for _ in range(self.nt)]
        self.z_before = [None for _ in range(self.nt)]
        # list of bids on each task (size Nt)
        self.c = [self.min_val for _ in range(self.nt)]
        
        # bundle
        self.b = []
        # path /ordered bundle
        self.p = []
        
        # timestamps
        self.s = [-inf for _ in range(self.na)]
        self.s[self.id] = 0
        # clock start time
        self.start_time = start_time
        self.converged = False
        
        # convergence observation list
        self.their_net_cvg = [False for _ in range(self.D)]
        self.cvg_counter = 0

        # # Marginal gain list
        # self.Wa = []
        # # List to maintain tasks before insertion
        # self.placement = []
        # Has agent won previous bid? (used for reusing previous path)
        self.flag_won = True
        # # size of path at previous iteration
        # self.len_p_before = 0

    def create_bundle(self, iter):
        
        if len(self.p) < self.Lt: # Check if bundle is not full already
            filtered_task_ids = [t.id for t in self.tasks if t.id not in self.p]
            if self.flag_won == True:
                placement = np.zeros(self.nt)
                for j in filtered_task_ids:
                    c, opt_place = self.compute_c(j) # c_ij(p_i) = S_i(p_i ⊕_opt j) - S_i(p_i)

        pass  # Placeholder for bundle creation method

    def compute_c(self, task_id):
        """
        Compute the bid for a given task based on current path
        :param task_id: id of the task to compute bid for
        :return: bid value, optimal placement index
        """
        # Placeholder for bid computation logic
        return 0, 0  # Return dummy values for now