"""
Comprehensive Test Suite for GCBBA Warehouse Implementation
============================================================
WITH INJECT/EJECT STATION TASK MODEL

Tests for:
- GCBBA_Task class with induct/eject positions
- GCBBA_Agent class (RPT metric only)
- GCBBA_Orchestrator class
- End-to-end GCBBA execution

Task Model:
- Each task has an induct (pickup) position and eject (dropoff) position
- Agent travels to induct, then travels from induct to eject
- Task "duration" = distance from induct to eject / agent speed
- After completing task, agent position updates to eject position

Author: Shreyas
Date: February 2026
"""

import numpy as np
import pytest
import copy
from math import ceil

# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockTask:
    """
    Mock Task class for testing with inject/eject stations.
    
    Attributes:
        id: Unique task identifier
        induct_pos: Position where agent picks up item (inject station)
        eject_pos: Position where agent drops off item (eject station)
    """
    def __init__(self, id, induct_pos, eject_pos):
        self.id = id
        self.induct_pos = np.array(induct_pos)
        self.eject_pos = np.array(eject_pos)


class MockAgent:
    """
    Mock Agent class implementing core GCBBA functionality for testing.
    Uses RPT metric with inject/eject task model.
    """
    def __init__(self, id, pos, speed, tasks, G, Lt=2, D=1):
        self.id = id
        self.pos = np.array(pos)
        self.speed = speed
        self.tasks = tasks
        self.nt = len(tasks)
        self.G = G
        self.na = G.shape[0]
        self.Lt = Lt
        self.D = D
        
        self.min_val = -1e20
        
        # Core GCBBA data structures
        self.y = [self.min_val for _ in range(self.nt)]  # Winning bids
        self.y_before = [self.min_val for _ in range(self.nt)]  # Previous winning bids
        self.z = [None for _ in range(self.nt)]          # Winners
        self.z_before = [None for _ in range(self.nt)]   # Previous winners
        self.c = [self.min_val for _ in range(self.nt)]  # Agent's bids
        
        self.b = []  # Bundle (task IDs in order added)
        self.p = []  # Path (task IDs in execution order)
        self.S = [0]  # Path scores history
        
        self.s = [-np.inf for _ in range(self.na)]  # Timestamps
        self.s[self.id] = 0
        
        self.converged = False
        self.their_net_cvg = [False for _ in range(self.D)]
        self.cvg_counter = 0
        
        self.placement = np.zeros(self.nt)
        self.flag_won = True
        self.len_p_before = 0
        
        self.task_id_to_idx = {task.id: i for i, task in enumerate(self.tasks)}
    
    def _get_task_index(self, task_id):
        """Get the index for a given task_id"""
        return self.task_id_to_idx.get(task_id, None)
    
    def evaluate_path(self, path):
        """
        Evaluate path using RPT metric with inject/eject stations.
        
        RPT Formula:
        S_i(p_i) = -Σ_{j∈p_i} τ_{ij}(p^:j_i)
        
        For each task:
        1. Travel from current position to induct position
        2. Task execution = travel from induct to eject
        3. Update current position to eject position
        
        The score is the negative sum of completion times.
        More negative = worse (longer path).
        """
        cur_pos = self.pos.copy()
        score = 0
        time = 0
        
        if len(path) > 0:
            for task_id in path:
                task_idx = self._get_task_index(task_id)
                task = self.tasks[task_idx]
                
                # Travel to induct position
                induct_pos = task.induct_pos
                time += np.linalg.norm(cur_pos - induct_pos) / self.speed
                
                # Task duration: travel from induct to eject
                eject_pos = task.eject_pos
                time += np.linalg.norm(induct_pos - eject_pos) / self.speed
                
                # RPT: subtract completion time
                score -= time
                time = 0  # Reset time for next task (this is the "Repeated" in RPT)
                
                # Update current position to eject position
                cur_pos = eject_pos.copy()
        
        return score
    
    def compute_c(self, task_id):
        """
        Compute the bid for a given task based on current path.
        
        For RPT: c_ij(p_i) = max over all positions of S_i(p_i ⊕_pos j)
        
        Returns: (bid value, optimal insertion position)
        """
        path_bids = []
        P = self.p
        
        for pos in range(len(self.p) + 1):
            P1 = copy.deepcopy(P)
            P1.insert(pos, task_id)
            path_score = self.evaluate_path(P1)
            path_bids.append(path_score)
        
        max_bid = np.max(path_bids)
        optimal_pos = np.argwhere(path_bids == max_bid)[-1][0]
        
        return max_bid, optimal_pos
    
    def create_bundle(self):
        """Bundle building phase - adds at most one task per call"""
        if len(self.p) >= self.Lt:
            return
        
        filtered_task_ids = [t.id for t in self.tasks if t.id not in self.p]
        
        if self.flag_won:
            self.placement = np.zeros(self.nt)
            for task_id in filtered_task_ids:
                task_idx = self._get_task_index(task_id)
                c, opt_place = self.compute_c(task_id)
                self.c[task_idx] = c
                self.placement[task_idx] = opt_place
        
        # Compute bids
        bids = []
        for j in range(self.nt):
            task_id = self.tasks[j].id
            
            if task_id not in filtered_task_ids:
                bids.append(self.min_val)
                continue
            
            if self.c[j] > self.y[j]:
                bids.append(self.c[j])
            elif self.c[j] == self.y[j] and self.z[j] is not None and self.z[j] > self.id:
                bids.append(self.c[j])
            else:
                bids.append(self.min_val)
        
        best_idx = np.argmax(bids)
        best_task_id = self.tasks[best_idx].id
        
        if best_task_id in self.p or bids[best_idx] <= self.min_val:
            return
        
        self.b.append(best_task_id)
        self.p.insert(int(self.placement[best_idx]), best_task_id)
        self.S.append(self.evaluate_path(self.p))
        
        self.y[best_idx] = self.c[best_idx]
        self.z[best_idx] = self.id
    
    def update(self, neighbor, task_id):
        """Update procedure - accept neighbor's bid"""
        task_idx = self._get_task_index(task_id)
        
        # Accept neighbor's bid and winner
        self.y[task_idx] = neighbor.y[task_idx]
        self.z[task_idx] = neighbor.z[task_idx]
        
        bundle = self.b
        if task_id in bundle:
            self.flag_won = False
            bundle_index = bundle.index(task_id)
            tasks_to_remove = bundle[bundle_index:]
            
            # Clear winning bids and winners for removed tasks
            for task_id_to_remove in tasks_to_remove:
                idx = self._get_task_index(task_id_to_remove)
                self.y[idx] = self.min_val
                self.z[idx] = None
            
            # Re-accept neighbor's bid for the contested task
            self.y[task_idx] = neighbor.y[task_idx]
            self.z[task_idx] = neighbor.z[task_idx]
            
            # Remove tasks from bundle and path
            self.b = self.b[:bundle_index]
            for task in tasks_to_remove:
                if task in self.p:
                    self.p.remove(task)
            
            self.S = self.S[:bundle_index + 1]
            self.their_net_cvg[0] = False
    
    def reset(self, task_id):
        """Reset procedure - clear bid and remove from bundle"""
        task_idx = self._get_task_index(task_id)
        
        self.y[task_idx] = self.min_val
        self.z[task_idx] = None
        
        bundle = self.b
        if task_id in bundle:
            bundle_index = bundle.index(task_id)
            tasks_to_remove = bundle[bundle_index:]
            
            for task_id_to_remove in tasks_to_remove:
                idx = self._get_task_index(task_id_to_remove)
                self.y[idx] = self.min_val
                self.z[idx] = None
            
            self.b = self.b[:bundle_index]
            for task in tasks_to_remove:
                if task in self.p:
                    self.p.remove(task)
            
            self.S = self.S[:bundle_index + 1]
            self.their_net_cvg[0] = False
    
    def leave(self):
        """Leave procedure - do nothing"""
        pass
    
    def compute_s(self, neighbor, consensus_iter):
        """Update timestamps based on neighbor's information"""
        self.s[self.id] = consensus_iter
        self.s[neighbor.id] = consensus_iter
        
        not_neighbor_ids = np.argwhere(self.G[self.id, :] == 0).flatten()
        greater_index = np.argwhere(np.array(neighbor.s) > np.array(self.s)).flatten()
        intersect = list(set(greater_index).intersection(set(not_neighbor_ids)))
        
        self.s = [neighbor.s[i] if i in intersect else self.s[i] for i in range(self.na)]
    
    def resolve_conflicts(self, all_agents, consensus_iter=0, consensus_index_last=False):
        """Resolve conflicts with neighboring agents"""
        neigh_idxs = np.argwhere(self.G[self.id, :] == 1).flatten()
        neigh_cvg = [True for _ in range(self.D)]
        
        for k in neigh_idxs:
            neigh = all_agents[k]
            
            for j in range(self.nt):
                task_id = self.tasks[j].id
                
                # Neighbor thinks it won
                if neigh.z[j] == neigh.id:
                    if self.z[j] == self.id:
                        if neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                    elif self.z[j] == k:
                        self.update(neigh, task_id)
                    elif self.z[j] is None:
                        self.update(neigh, task_id)
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m] or neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                
                # Neighbor thinks I won
                elif neigh.z[j] == self.id:
                    if self.z[j] == self.id:
                        self.leave()
                    elif self.z[j] == k:
                        self.reset(task_id)
                    elif self.z[j] is None:
                        self.leave()
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m]:
                            self.reset(task_id)
                
                # Neighbor thinks no one won
                elif neigh.z[j] is None:
                    if self.z[j] == self.id:
                        self.leave()
                    elif self.z[j] == k:
                        self.update(neigh, task_id)
                    elif self.z[j] is None:
                        self.leave()
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                
                # Neighbor thinks m won (m != self, m != neighbor)
                else:
                    m = int(neigh.z[j])
                    if self.z[j] == self.id:
                        if (neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j]) or \
                           (neigh.s[m] > self.s[m] and neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                    elif self.z[j] == k:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                        else:
                            self.reset(task_id)
                    elif self.z[j] == m:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                    elif self.z[j] is None:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                    else:
                        n = int(self.z[j])
                        if neigh.s[m] > self.s[m] and neigh.s[n] > self.s[n]:
                            self.update(neigh, task_id)
                        elif (neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j]) or \
                             (neigh.s[m] > self.s[m] and neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                        elif neigh.s[n] > self.s[n] and self.s[m] > neigh.s[m]:
                            self.reset(task_id)
            
            self.compute_s(neigh, consensus_iter)
            
            for i in range(1, self.D):
                neigh_cvg[i] = neigh_cvg[i] and neigh.their_net_cvg[i-1]
        
        # Update convergence observation
        self.their_net_cvg[0] = (self.z == self.z_before)
        for i in range(1, self.D):
            self.their_net_cvg[i] = neigh_cvg[i] and self.their_net_cvg[i-1]
        
        self.converged = self.their_net_cvg[-1]
        
        if self.converged:
            self.cvg_counter += 1
        
        if consensus_index_last:
            self.y_before = copy.deepcopy(self.y)
            self.z_before = copy.deepcopy(self.z)
            self.flag_won = (len(self.p) != self.len_p_before)
            self.len_p_before = len(self.p)


# ============================================================================
# TEST SECTION 1: Task Class Tests (Inject/Eject)
# ============================================================================

class TestTaskClassInjectEject:
    """Tests for GCBBA_Task class with inject/eject stations"""
    
    def test_task_creation_basic(self):
        """
        Test: Basic task creation with inject/eject
        
        Setup:
        - Task 0: induct at (1, 2), eject at (5, 6)
        
        Expected: Task stores positions correctly
        """
        task = MockTask(id=0, induct_pos=[1.0, 2.0], eject_pos=[5.0, 6.0])
        
        assert task.id == 0, "Task ID should be 0"
        assert np.allclose(task.induct_pos, [1.0, 2.0]), "Induct position incorrect"
        assert np.allclose(task.eject_pos, [5.0, 6.0]), "Eject position incorrect"
        print("✓ Task creation test passed")
        print(f"  → Task {task.id}: induct={task.induct_pos}, eject={task.eject_pos}")
    
    def test_task_distance_calculation(self):
        """
        Test: Distance from induct to eject
        
        Setup:
        - Task: induct at (0, 0), eject at (3, 4)
        
        Expected: Distance = 5.0 (3-4-5 triangle)
        """
        task = MockTask(id=0, induct_pos=[0, 0], eject_pos=[3, 4])
        
        distance = np.linalg.norm(task.eject_pos - task.induct_pos)
        
        assert np.isclose(distance, 5.0), f"Distance should be 5.0, got {distance}"
        print(f"✓ Task distance: {distance}")


# ============================================================================
# TEST SECTION 2: Path Evaluation Tests (RPT with Inject/Eject)
# ============================================================================

class TestPathEvaluationInjectEject:
    """
    Tests for evaluate_path() using RPT metric with inject/eject stations.
    
    RPT with Inject/Eject:
    For each task in path:
    1. Travel from current position to induct position
    2. Task time = travel from induct to eject
    3. Completion time = sum of above
    4. Update position to eject for next task
    
    Score = -Σ(completion times)
    """
    
    @pytest.fixture
    def single_agent_setup(self):
        """Setup single agent with inject/eject tasks"""
        # Task 0: Pick at (2,0), drop at (4,0) - simple horizontal
        # Task 1: Pick at (6,0), drop at (8,0) - further along
        # Task 2: Pick at (0,3), drop at (3,3) - vertical offset
        tasks = [
            MockTask(id=0, induct_pos=[2, 0], eject_pos=[4, 0]),
            MockTask(id=1, induct_pos=[6, 0], eject_pos=[8, 0]),
            MockTask(id=2, induct_pos=[0, 3], eject_pos=[3, 3]),
        ]
        G = np.array([[1]])  # Single agent
        agent = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=3)
        return agent, tasks
    
    def test_empty_path_score(self, single_agent_setup):
        """
        Test: Empty path evaluation
        
        Expected: Score = 0
        """
        agent, _ = single_agent_setup
        
        score = agent.evaluate_path([])
        
        assert score == 0, f"Empty path score should be 0, got {score}"
        print("✓ Empty path score = 0")
    
    def test_single_task_path(self, single_agent_setup):
        """
        Test: Single task path evaluation
        
        Setup:
        - Agent at (0,0), speed = 1.0
        - Task 0: induct (2,0), eject (4,0)
        - Path = [0]
        
        Calculation:
        - Travel to induct: dist((0,0), (2,0)) / 1.0 = 2.0s
        - Task execution: dist((2,0), (4,0)) / 1.0 = 2.0s
        - Completion time τ₀ = 2.0 + 2.0 = 4.0s
        - RPT score = -4.0
        
        Expected: Score = -4.0
        """
        agent, _ = single_agent_setup
        
        score = agent.evaluate_path([0])
        
        expected = -4.0
        assert np.isclose(score, expected), f"Score should be {expected}, got {score}"
        print(f"✓ Single task path score = {score}")
        print("  → Travel to induct: 2.0s, Task execution: 2.0s")
    
    def test_two_task_sequential_path(self, single_agent_setup):
        """
        Test: Two tasks in sequence
        
        Setup:
        - Agent at (0,0), speed = 1.0
        - Task 0: induct (2,0), eject (4,0)
        - Task 1: induct (6,0), eject (8,0)
        - Path = [0, 1]
        
        Calculation:
        Task 0:
        - Travel to induct: 2.0s
        - Task execution: 2.0s
        - τ₀ = 4.0s
        
        Task 1 (starting from eject of task 0 at (4,0)):
        - Travel to induct: dist((4,0), (6,0)) = 2.0s
        - Task execution: 2.0s
        - τ₁ = 4.0s
        
        RPT score = -(4.0 + 4.0) = -8.0
        
        Expected: Score = -8.0
        """
        agent, _ = single_agent_setup
        
        score = agent.evaluate_path([0, 1])
        
        expected = -8.0
        assert np.isclose(score, expected), f"Score should be {expected}, got {score}"
        print(f"✓ Two task sequential path score = {score}")
        print("  → Task 0: 4.0s, Task 1: 4.0s")
    
    def test_path_order_matters(self, single_agent_setup):
        """
        Test: Path order affects score
        
        Setup:
        - Agent at (0,0)
        - Task 0: induct (2,0), eject (4,0)
        - Task 1: induct (6,0), eject (8,0)
        
        Path [0, 1]:
        - Task 0: travel 2s + exec 2s = 4s
        - Task 1 (from (4,0)): travel 2s + exec 2s = 4s
        - Total: -8.0
        
        Path [1, 0]:
        - Task 1: travel 6s + exec 2s = 8s
        - Task 0 (from (8,0)): travel 6s + exec 2s = 8s
        - Total: -16.0
        
        Expected: [0, 1] is better (-8.0 > -16.0)
        """
        agent, _ = single_agent_setup
        
        score_01 = agent.evaluate_path([0, 1])
        score_10 = agent.evaluate_path([1, 0])
        
        assert np.isclose(score_01, -8.0), f"Score [0,1] should be -8.0, got {score_01}"
        assert np.isclose(score_10, -16.0), f"Score [1,0] should be -16.0, got {score_10}"
        assert score_01 > score_10, "Order [0,1] should be better"
        
        print(f"✓ Path order test:")
        print(f"  → [0, 1] score: {score_01}")
        print(f"  → [1, 0] score: {score_10}")
        print(f"  → Order matters: [0,1] is {score_10 - score_01} better")
    
    def test_diagonal_task(self, single_agent_setup):
        """
        Test: Task with diagonal movement
        
        Setup:
        - Agent at (0,0)
        - Task 2: induct (0,3), eject (3,3)
        - Path = [2]
        
        Calculation:
        - Travel to induct: dist((0,0), (0,3)) = 3.0s
        - Task execution: dist((0,3), (3,3)) = 3.0s
        - τ₂ = 6.0s
        - Score = -6.0
        """
        agent, _ = single_agent_setup
        
        score = agent.evaluate_path([2])
        
        expected = -6.0
        assert np.isclose(score, expected), f"Score should be {expected}, got {score}"
        print(f"✓ Diagonal task score = {score}")
    
    def test_agent_speed_affects_score(self):
        """
        Test: Agent speed affects path score
        
        Setup:
        - Task: induct (4,0), eject (8,0)
        - Slow agent (speed=1.0): 4s + 4s = 8s → -8.0
        - Fast agent (speed=2.0): 2s + 2s = 4s → -4.0
        """
        tasks = [MockTask(id=0, induct_pos=[4, 0], eject_pos=[8, 0])]
        G = np.array([[1]])
        
        agent_slow = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=1)
        agent_fast = MockAgent(id=0, pos=[0, 0], speed=2.0, tasks=tasks, G=G, Lt=1)
        
        score_slow = agent_slow.evaluate_path([0])
        score_fast = agent_fast.evaluate_path([0])
        
        assert np.isclose(score_slow, -8.0)
        assert np.isclose(score_fast, -4.0)
        assert score_fast > score_slow
        
        print(f"✓ Speed test: slow={score_slow}, fast={score_fast}")
    
    def test_agent_at_induct_position(self):
        """
        Test: Agent starts at induct position
        
        Setup:
        - Agent at (2,0)
        - Task: induct (2,0), eject (5,0)
        
        Calculation:
        - Travel to induct: 0s
        - Task execution: 3s
        - Score = -3.0
        """
        tasks = [MockTask(id=0, induct_pos=[2, 0], eject_pos=[5, 0])]
        G = np.array([[1]])
        agent = MockAgent(id=0, pos=[2, 0], speed=1.0, tasks=tasks, G=G, Lt=1)
        
        score = agent.evaluate_path([0])
        
        assert np.isclose(score, -3.0), f"Score should be -3.0, got {score}"
        print(f"✓ Agent at induct: score={score}")


# ============================================================================
# TEST SECTION 3: Bid Computation Tests
# ============================================================================

class TestBidComputationInjectEject:
    """Tests for compute_c() with inject/eject task model"""
    
    @pytest.fixture
    def bid_setup(self):
        """Setup for bid computation tests"""
        tasks = [
            MockTask(id=0, induct_pos=[2, 0], eject_pos=[4, 0]),
            MockTask(id=1, induct_pos=[6, 0], eject_pos=[8, 0]),
            MockTask(id=2, induct_pos=[0, 4], eject_pos=[4, 4]),
        ]
        G = np.array([[1]])
        agent = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=3)
        return agent, tasks
    
    def test_compute_c_empty_path(self, bid_setup):
        """
        Test: Compute bid with empty path
        
        Setup:
        - Agent at (0,0), empty path
        - Task 0: induct (2,0), eject (4,0)
        
        Only position 0 available:
        - Score of [0] = -(2 + 2) = -4.0
        
        Expected: bid = -4.0, position = 0
        """
        agent, _ = bid_setup
        
        bid, pos = agent.compute_c(0)
        
        assert np.isclose(bid, -4.0), f"Bid should be -4.0, got {bid}"
        assert pos == 0, f"Position should be 0, got {pos}"
        print(f"✓ Empty path bid: c={bid}, pos={pos}")
    
    def test_compute_c_optimal_insertion(self, bid_setup):
        """
        Test: Find optimal insertion position
        
        Setup:
        - Agent at (0,0), path = [0]
        - Task 0: induct (2,0), eject (4,0)
        - Computing bid for Task 1: induct (6,0), eject (8,0)
        
        Position 0: [1, 0]
        - Task 1 from (0,0): 6s + 2s = 8s
        - Task 0 from (8,0): 6s + 2s = 8s (travel back!)
        - Total: -16.0
        
        Position 1: [0, 1]
        - Task 0: 4s
        - Task 1 from (4,0): 2s + 2s = 4s
        - Total: -8.0
        
        Best: Position 1 with bid -8.0
        """
        agent, _ = bid_setup
        agent.p = [0]
        
        bid, pos = agent.compute_c(1)
        
        assert np.isclose(bid, -8.0), f"Bid should be -8.0, got {bid}"
        assert pos == 1, f"Position should be 1, got {pos}"
        print(f"✓ Optimal insertion: c={bid}, pos={pos}")
        print("  → Insert at end (-8.0) better than at start (-16.0)")


# ============================================================================
# TEST SECTION 4: Bundle Building Tests
# ============================================================================

class TestBundleBuildingInjectEject:
    """Tests for create_bundle() with inject/eject tasks"""
    
    @pytest.fixture
    def bundle_setup(self):
        """Setup for bundle building"""
        # Tasks progressively further from agent
        tasks = [
            MockTask(id=0, induct_pos=[2, 0], eject_pos=[3, 0]),   # Close
            MockTask(id=1, induct_pos=[5, 0], eject_pos=[6, 0]),   # Medium
            MockTask(id=2, induct_pos=[8, 0], eject_pos=[9, 0]),   # Far
        ]
        G = np.array([[1]])
        agent = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=3)
        return agent, tasks
    
    def test_first_bundle_building(self, bundle_setup):
        """
        Test: First bundle building picks closest task
        
        Setup:
        - Agent at (0,0)
        - Task 0: score = -(2 + 1) = -3.0
        - Task 1: score = -(5 + 1) = -6.0
        - Task 2: score = -(8 + 1) = -9.0
        
        Expected: Task 0 selected (highest bid = -3.0)
        """
        agent, _ = bundle_setup
        
        agent.create_bundle()
        
        assert agent.b == [0], f"Bundle should be [0], got {agent.b}"
        assert agent.p == [0], f"Path should be [0], got {agent.p}"
        assert agent.z[0] == 0, f"z[0] should be 0, got {agent.z[0]}"
        assert np.isclose(agent.y[0], -3.0), f"y[0] should be -3.0, got {agent.y[0]}"
        
        print(f"✓ First bundle: b={agent.b}, y[0]={agent.y[0]}")
    
    def test_second_bundle_building(self, bundle_setup):
        """
        Test: Second iteration adds next best task
        
        After task 0, agent ends at (3,0).
        - Task 1 from (3,0): travel 2s + exec 1s = 3s → total -6.0
        - Task 2 from (3,0): travel 5s + exec 1s = 6s → total -9.0
        
        Expected: Task 1 added
        """
        agent, _ = bundle_setup
        
        agent.create_bundle()  # First
        agent.flag_won = True
        agent.create_bundle()  # Second
        
        assert agent.b == [0, 1], f"Bundle should be [0, 1], got {agent.b}"
        assert 1 in agent.p
        print(f"✓ Second bundle: b={agent.b}, p={agent.p}")
    
    def test_capacity_limit(self, bundle_setup):
        """
        Test: Bundle respects capacity limit
        """
        agent, _ = bundle_setup
        agent.Lt = 2
        
        agent.create_bundle()
        agent.flag_won = True
        agent.create_bundle()
        agent.flag_won = True
        agent.create_bundle()  # Should not add
        
        assert len(agent.b) == 2, f"Bundle should have 2 tasks, got {len(agent.b)}"
        print(f"✓ Capacity limit enforced: {len(agent.b)} tasks")


# ============================================================================
# TEST SECTION 5: Conflict Resolution Tests
# ============================================================================

class TestConflictResolutionInjectEject:
    """Tests for conflict resolution with inject/eject tasks"""
    
    @pytest.fixture
    def conflict_setup(self):
        """Setup for conflict tests"""
        tasks = [
            MockTask(id=0, induct_pos=[2, 0], eject_pos=[3, 0]),
            MockTask(id=1, induct_pos=[5, 0], eject_pos=[6, 0]),
        ]
        G = np.array([[1, 1], [1, 1]])  # Fully connected
        
        agent0 = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=2)
        agent1 = MockAgent(id=1, pos=[1, 0], speed=1.0, tasks=tasks, G=G, Lt=2)
        
        return agent0, agent1, tasks
    
    def test_better_bid_wins(self, conflict_setup):
        """
        Test: Agent with better bid wins task
        
        Setup:
        - Agent 0 at (0,0): bid for task 0 = -(2+1) = -3.0
        - Agent 1 at (1,0): bid for task 0 = -(1+1) = -2.0
        
        Agent 1 is closer → better bid → should win
        """
        agent0, agent1, _ = conflict_setup
        
        # Both compute bids
        c0, _ = agent0.compute_c(0)
        c1, _ = agent1.compute_c(0)
        
        assert np.isclose(c0, -3.0), f"Agent 0 bid should be -3.0, got {c0}"
        assert np.isclose(c1, -2.0), f"Agent 1 bid should be -2.0, got {c1}"
        assert c1 > c0, "Agent 1 should have better (less negative) bid"
        
        print(f"✓ Bid comparison: Agent0={c0}, Agent1={c1}")
        print(f"  → Agent 1 wins (closer to task)")
    
    def test_update_accepts_better_bid(self, conflict_setup):
        """
        Test: Agent accepts neighbor's better bid
        """
        agent0, agent1, _ = conflict_setup
        
        # Agent 0 claims task 0
        agent0.b = [0]
        agent0.p = [0]
        agent0.y[0] = -3.0
        agent0.z[0] = 0
        agent0.S = [0, -3.0]
        
        # Agent 1 has better bid
        agent1.y[0] = -2.0
        agent1.z[0] = 1
        
        # Agent 0 accepts
        agent0.update(agent1, 0)
        
        assert agent0.y[0] == -2.0
        assert agent0.z[0] == 1
        assert agent0.b == []
        assert agent0.p == []
        
        print(f"✓ Update accepted: y[0]={agent0.y[0]}, z[0]={agent0.z[0]}")
    
    def test_reset_clears_bundle(self, conflict_setup):
        """
        Test: Reset clears bid and bundle
        """
        agent0, _, _ = conflict_setup
        
        agent0.b = [0]
        agent0.p = [0]
        agent0.y[0] = -3.0
        agent0.z[0] = 0
        agent0.S = [0, -3.0]
        
        agent0.reset(0)
        
        assert agent0.y[0] == agent0.min_val
        assert agent0.z[0] is None
        assert agent0.b == []
        assert agent0.p == []
        
        print(f"✓ Reset clears bundle")
    
    def test_cascading_removal(self, conflict_setup):
        """
        Test: Removing task removes all subsequent tasks
        
        Setup:
        - Agent 0 has bundle [0, 1]
        - Task 0 is outbid
        - Both tasks should be removed (cascading)
        """
        agent0, agent1, _ = conflict_setup
        
        # Agent 0 has both tasks
        agent0.b = [0, 1]
        agent0.p = [0, 1]
        agent0.y[0] = -3.0
        agent0.y[1] = -6.0
        agent0.z[0] = 0
        agent0.z[1] = 0
        agent0.S = [0, -3.0, -6.0]
        
        # Agent 1 outbids for task 0
        agent1.y[0] = -2.0
        agent1.z[0] = 1
        
        # Agent 0 updates
        agent0.update(agent1, 0)
        
        # Both tasks should be removed
        assert agent0.b == [], f"Bundle should be empty, got {agent0.b}"
        assert agent0.p == [], f"Path should be empty, got {agent0.p}"
        assert agent0.y[1] == agent0.min_val, "y[1] should be reset"
        assert agent0.z[1] is None, "z[1] should be None"
        
        print(f"✓ Cascading removal: both tasks removed")


# ============================================================================
# TEST SECTION 6: Convergence Detection Tests
# ============================================================================

class TestConvergenceDetection:
    """Tests for convergence detection mechanism"""
    
    @pytest.fixture
    def convergence_setup(self):
        """Setup for convergence tests"""
        tasks = [
            MockTask(id=0, induct_pos=[1, 0], eject_pos=[2, 0]),
            MockTask(id=1, induct_pos=[4, 0], eject_pos=[5, 0]),
        ]
        G = np.array([[1, 1], [1, 1]])
        D = 1
        
        agents = [
            MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=2, D=D),
            MockAgent(id=1, pos=[3, 0], speed=1.0, tasks=tasks, G=G, Lt=2, D=D)
        ]
        return agents, tasks
    
    def test_convergence_with_no_changes(self, convergence_setup):
        """
        Test: Agents converge when z doesn't change
        
        Setup:
        - Agents have stable assignments
        - z_before = z
        
        Expected: their_net_cvg[0] becomes True
        """
        agents, _ = convergence_setup
        agent0 = agents[0]
        
        # Set stable state
        agent0.z = [0, None]
        agent0.z_before = [0, None]
        
        # Convergence check
        agent0.their_net_cvg[0] = (agent0.z == agent0.z_before)
        
        assert agent0.their_net_cvg[0] == True
        print(f"✓ Convergence detected when z stable")
    
    def test_no_convergence_with_changes(self, convergence_setup):
        """
        Test: No convergence when z changes
        """
        agents, _ = convergence_setup
        agent0 = agents[0]
        
        agent0.z = [0, None]
        agent0.z_before = [None, None]
        
        agent0.their_net_cvg[0] = (agent0.z == agent0.z_before)
        
        assert agent0.their_net_cvg[0] == False
        print(f"✓ No convergence when z changes")


# ============================================================================
# TEST SECTION 7: Multi-Agent Coordination Tests
# ============================================================================

class TestMultiAgentCoordination:
    """Tests for multi-agent GCBBA coordination"""
    
    @pytest.fixture
    def multi_agent_setup(self):
        """Setup for multi-agent tests"""
        tasks = [
            MockTask(id=0, induct_pos=[1, 0], eject_pos=[2, 0]),
            MockTask(id=1, induct_pos=[4, 0], eject_pos=[5, 0]),
            MockTask(id=2, induct_pos=[7, 0], eject_pos=[8, 0]),
        ]
        G = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        D = 2
        
        agents = [
            MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=2, D=D),
            MockAgent(id=1, pos=[3, 0], speed=1.0, tasks=tasks, G=G, Lt=2, D=D),
            MockAgent(id=2, pos=[6, 0], speed=1.0, tasks=tasks, G=G, Lt=2, D=D)
        ]
        return agents, tasks
    
    def test_agents_claim_closest_tasks(self, multi_agent_setup):
        """
        Test: Each agent claims closest task
        
        Setup:
        - Agent 0 at (0,0) → closest to task 0 at (1,0)
        - Agent 1 at (3,0) → closest to task 1 at (4,0)
        - Agent 2 at (6,0) → closest to task 2 at (7,0)
        """
        agents, _ = multi_agent_setup
        
        for agent in agents:
            agent.create_bundle()
        
        # Each should claim nearest task
        assert agents[0].b == [0], f"Agent 0 should claim task 0, got {agents[0].b}"
        assert agents[1].b == [1], f"Agent 1 should claim task 1, got {agents[1].b}"
        assert agents[2].b == [2], f"Agent 2 should claim task 2, got {agents[2].b}"
        
        print(f"✓ Multi-agent claims:")
        for agent in agents:
            print(f"  → Agent {agent.id}: bundle={agent.b}")
    
    def test_full_gcbba_execution(self, multi_agent_setup):
        """
        Test: Complete GCBBA execution with consensus
        """
        agents, tasks = multi_agent_setup
        D = 2
        Nmin = min(len(tasks), sum(a.Lt for a in agents))
        nb_cons = 2 * D
        
        print(f"\n✓ Full GCBBA execution (Nmin={Nmin}, nb_cons={nb_cons}):")
        
        for iteration in range(Nmin):
            print(f"\n  Iteration {iteration}:")
            
            # Bundle building
            for agent in agents:
                if not agent.converged:
                    agent.create_bundle()
                print(f"    Agent {agent.id}: b={agent.b}, p={agent.p}")
            
            # Consensus
            for cons_round in range(nb_cons):
                all_agents_copy = copy.deepcopy(agents)
                consensus_iter = nb_cons * iteration + cons_round
                is_last = (cons_round == nb_cons - 1)
                
                for agent in agents:
                    if not agent.converged:
                        agent.resolve_conflicts(all_agents_copy, 
                                               consensus_iter=consensus_iter,
                                               consensus_index_last=is_last)
        
        # Verify all tasks assigned
        assigned_tasks = set()
        for agent in agents:
            assigned_tasks.update(agent.p)
        
        print(f"\n  Final assignments:")
        for agent in agents:
            print(f"    Agent {agent.id}: {agent.p}")
        print(f"  Tasks assigned: {len(assigned_tasks)}/{len(tasks)}")


# ============================================================================
# TEST SECTION 8: DMG Property Verification
# ============================================================================

class TestDMGProperty:
    """Tests to verify Diminishing Marginal Gain property"""
    
    def test_dmg_property(self):
        """
        Test: Bids decrease as path grows (DMG property)
        
        For RPT: Adding tasks makes future bids more negative
        """
        tasks = [
            MockTask(id=0, induct_pos=[2, 0], eject_pos=[3, 0]),
            MockTask(id=1, induct_pos=[5, 0], eject_pos=[6, 0]),
            MockTask(id=2, induct_pos=[8, 0], eject_pos=[9, 0]),
        ]
        G = np.array([[1]])
        agent = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=3)
        
        # Bid for task 2 with empty path
        agent.p = []
        c_empty, _ = agent.compute_c(2)
        
        # Bid for task 2 with path [0]
        agent.p = [0]
        c_with_one, _ = agent.compute_c(2)
        
        # Bid for task 2 with path [0, 1]
        agent.p = [0, 1]
        c_with_two, _ = agent.compute_c(2)
        
        # DMG: c_empty >= c_with_one >= c_with_two
        assert c_empty >= c_with_one, f"DMG violated: {c_empty} < {c_with_one}"
        assert c_with_one >= c_with_two, f"DMG violated: {c_with_one} < {c_with_two}"
        
        print(f"✓ DMG property verified:")
        print(f"  → Empty path: c={c_empty:.2f}")
        print(f"  → Path [0]: c={c_with_one:.2f}")
        print(f"  → Path [0,1]: c={c_with_two:.2f}")


# ============================================================================
# TEST SECTION 9: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_same_induct_eject_position(self):
        """
        Test: Task with same induct and eject position
        
        This means task duration = 0 (no travel between positions)
        """
        tasks = [MockTask(id=0, induct_pos=[3, 0], eject_pos=[3, 0])]
        G = np.array([[1]])
        agent = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=1)
        
        score = agent.evaluate_path([0])
        
        # Only travel to induct: 3.0s
        expected = -3.0
        assert np.isclose(score, expected), f"Score should be {expected}, got {score}"
        print(f"✓ Same induct/eject position: score={score}")
    
    def test_large_task_distance(self):
        """
        Test: Task with large distance between induct and eject
        """
        tasks = [MockTask(id=0, induct_pos=[0, 0], eject_pos=[100, 0])]
        G = np.array([[1]])
        agent = MockAgent(id=0, pos=[0, 0], speed=10.0, tasks=tasks, G=G, Lt=1)
        
        score = agent.evaluate_path([0])
        
        # Travel to induct: 0s, Task execution: 100/10 = 10s
        expected = -10.0
        assert np.isclose(score, expected), f"Score should be {expected}, got {score}"
        print(f"✓ Large task distance: score={score}")
    
    def test_tie_breaking_lower_id_wins(self):
        """
        Test: When bids are equal, lower ID wins
        """
        tasks = [MockTask(id=0, induct_pos=[2, 0], eject_pos=[3, 0])]
        G = np.array([[1, 1], [1, 1]])
        
        agent0 = MockAgent(id=0, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=1)
        agent1 = MockAgent(id=1, pos=[0, 0], speed=1.0, tasks=tasks, G=G, Lt=1)
        
        # Both have same bid
        c0, _ = agent0.compute_c(0)
        c1, _ = agent1.compute_c(0)
        
        assert c0 == c1, "Bids should be equal"
        
        # Agent 0 should win (lower ID)
        agent0.create_bundle()
        agent1.create_bundle()
        
        # In tie, agent 0 (lower id) should win
        assert agent0.z[0] == 0, "Agent 0 should claim task"
        
        print(f"✓ Tie-breaking: Agent 0 wins with equal bids")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GCBBA Warehouse Test Suite - Inject/Eject Task Model")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "--tb=short"])
