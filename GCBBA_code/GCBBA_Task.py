"""
Task class for warehouse GCBBA
"""

import numpy as np


class GCBBA_Task:
    """
    Task class, defined by an id, a position (x,y), duration, and lambda (only for TDR)
    """
    def __init__(self, id, char):
        self.id = id
        self.pos = np.array([char[0], char[1]])
        self.duration = char[2]
        self.lamb = char[3]
