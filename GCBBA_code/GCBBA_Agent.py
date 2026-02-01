"""
GCBBA Agent class for warehouse task allocation
"""

import numpy as np
from math import inf
import copy


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

        c^RPT_ij(p_i) = X - min_{1≤l≤|p_i|+1} [L_i(p_i ⊕_l j)^α · W_i(p_i ⊕_l j)^β]
        X is implicit
        alpha = 1, beta = 0 for RPT
        L_i(p): Completion time of entire path p (Equation 13)
        """
        path_bids = []
        P = self.p # List of currrent task ids in path
        
        for pos in range(len(self.p) + 1):
            P1 = copy.deepcopy(P)
            P1.insert(pos, task_id)
            path_score = self.evaluate_path(P1)

            path_bids.append(path_score)
        
        max_bid = np.max(path_bids)
        optimal_pos = np.argwhere(path_bids == max_bid)[-1][0]
        
        return max_bid, optimal_pos

    def evaluate_path(self, path):
        """
        Evaluate the score of a given path based on the selected metric
        :param path: list of tasks in the path
        :return: score of the path -- total path time or makespan
        Later TODO: When the tasks are updated to start and end positions, update this function to
        account for travel time between tasks.

        L_i([j^i_1, ..., j^i_n]) = τ_{i,j^i_n}([j^i_1, ..., j^i_n])
        S_i(p_i) = Σ_{j∈p_i} c_ij(p^:j_i)
        """
        cur_pos = self.pos
        score = 0
        time = 0

        if len(path) > 0:
            for j in range(len(path)):
                task = self.tasks[path[j]]
                time += np.linalg.norm(cur_pos - task.pos) / self.speed
                time += task.duration
                score -= time

                time = 0  # Reset time for next task

                cur_pos = task.pos
        return score

