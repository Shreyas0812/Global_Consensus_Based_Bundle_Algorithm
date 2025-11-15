"""
File of auxiliary functions
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import cycle


def random_agent_init(na = 10, pos_lim = [-5, 5], sp_lim = [1, 5]):
    """
    Create agents characteristics
    :param na: number of agents
    :param pos_lim: (x,y) limits
    :param sp_lim: agent speed limits
    :return:
    """
    # Agents: [x_pos, y_pos, speed]
    agents = [np.concatenate(
                (np.random.uniform(pos_lim[0], pos_lim[1], 2), 
                 np.random.uniform(sp_lim[0], sp_lim[1], 1))
                 ) for _ in range(na)]          
    
    return agents

def random_task_init(nt =100, pos_lim = [-5, 5], dur_lim = [1,5], lamb_lim = [0.95, 0.95], clim = [1, 1]):
    """
    Create tasks characteristics
    :param nt: number of tasks
    :param pos_lim: (x,y) limits
    :param sp_lim: agent speed limits
    :param dur_lim: task duration limits
    :param lamb_lim: lambda (TDR) limits
    :param clim: weights (TDR) limits (useless in this implementation)
    :return:
    """
    # Tasks: [x_pos, y_pos, duration, lambda, weight]
    tasks = [np.concatenate(
                (np.random.uniform(pos_lim[0], pos_lim[1], 2), 
                 np.random.uniform(dur_lim[0], dur_lim[1], 1), 
                 np.random.uniform(lamb_lim[0], lamb_lim[1], 1),
                np.random.uniform(clim[0], clim[1], 1))
                ) for _ in range(nt)]
    
    return tasks