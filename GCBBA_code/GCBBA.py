import networkx as nx
import numpy as np
# To have the matplotlib without pausing
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from math import *
import time
from tqdm import tqdm
from tools import *



class Orchestrator_CBBA:
    """
    CBBA Orchestrator controlling CBBA agents 
    """
    def __init__(self, G, D, char_t, char_a, Lt=1, metric = "RPT"):
        self.G = G
        # aint, number of agents
        self.na = G.shape[0]
        # int, number of agents
        self.nt = len(char_t)
        # capacity per agent
        self.Lt = Lt
        # list of size 2, x limits of problem
        self.char_t = char_t
        # list of size 2, y limits of problem
        self.char_a = char_a
        # list of all agents
        self.agents = []
        # list of all tasks
        self.tasks = []

        # clock launch
        self.start_time = time.perf_counter()

        self.metric = metric

        self.D = D

        # initialize tasks
        self.initialize_all()
        self.bid_history = []
        self.assig_history = []
        self.max_times = []
        self.all_times = [0 for _ in range(self.na)]

        self.cvg_iter = self.nt

    #merged
    def launch_agents(self, method = "baseline", detector = "none", dynamic_communication_graph = True, update_freq = 1, visualize_graph = False, viz_freq = 10):
        """
        CBBA iterations to determine assignment, minsum and makespan
        :param method: "baseline" => gives baseline CBBA allocation. "global" => gives GCBBA (ours) allocation
        :param detector: "none" => no convergence detector. 
        "centralized" => centralized detector suited to CBBA. 
        "decentralized" => our decentralized detector suited to GCBBA.
        :return: allocation, minsum, makespan
        """
        if dynamic_communication_graph:
            self.track_connectivity_history()
        # track progress
        D = self.D
        Nmin = int(min(self.nt, self.Lt * self.na))
        if method == "baseline":
            nb_iter = Nmin*D
            nb_cons = 1
        elif method == "global":
            nb_iter = Nmin
            nb_cons = 2*D
        else:
            nb_iter = Nmin
            nb_cons = 2*D
        build_bundle = "FULLBUNDLE" if method == "baseline" else "ADD"
        for iter in tqdm(range(nb_iter)):

            # Update communication graph if dynamic
            if dynamic_communication_graph and (iter % update_freq == 0):
                G_new, D_new = self.update_communication_graph()

                # log connectivity metrics
                self.log_connectivity(iter)

                # Update Concensus if Diameter changed
                if D_new != self.D and method == "global":
                    D = D_new
                    nb_cons = 2 * D
            
            # visualize dynamic graph
            if visualize_graph and (iter % viz_freq == 0 or iter == nb_iter - 1):
                self.visualize_dynamic_graph(iter, save_fig=True)

            # Update Agents positions in dynamic communication graph scenario
            if dynamic_communication_graph:
                for agent in self.agents:
                    agent.update_position()

            I = list(range(self.na))
            for i in I:
                if detector == "decentralized" and self.agents[i].converged==False:
                    self.agents[i].create_bundle(iter, build_bundle = build_bundle)
                elif detector != "decentralized":
                    self.agents[i].create_bundle(iter, build_bundle=build_bundle)
            for _ in range(nb_cons):
                all_agents = copy.deepcopy(self.agents)
                # last consensus iteration ?
                if _ == nb_cons-1:
                    index = "last"
                else:
                    index = "else"
                for i in I:
                    if detector == "decentralized" and self.agents[i].converged ==False:
                        self.agents[i].resolve_conflict(all_agents, iter = nb_cons * iter + _, index = index)
                    elif detector != "decentralized":
                        self.agents[i].resolve_conflict(all_agents, iter = nb_cons * iter + _, index = index)
            assignment, bid, max_time = self.gather_info()
            self.assig_history.append(assignment)
            self.bid_history.append(bid)
            self.max_times.append(max_time)

            if detector == "centralized":
                if check_conflict(self.assig_history[-1], self.nt):
                    pass
                else:
                    print("EARLY CONVERGENCE AT {}/{}".format(iter+1, nb_iter))
                    self.cvg_iter = iter
                    break

            if detector == "decentralized":
                if np.prod([self.agents[i].converged for i in range(1)]) == False:
                    pass
                else:
                    print("EARLY CONVERGENCE AT {}/{}".format(iter+1, nb_iter))
                    self.cvg_iter = iter
                    break
        
        if dynamic_communication_graph:
            self.plot_connectivity_evolution()
        
        return self.assig_history[-1], self.bid_history[-1], self.max_times[-1]

    def get_cvg_iter(self):
        return self.cvg_iter

    def gather_info(self):
        """
        At a given time (iteration) compute assignment, minsum and makespan
        :return: assignment list, minsum and makespan
        """
        bid_sum = 0
        assignment = []
        max_time = 0
        for i in range(self.na):
            agent = self.agents[i]
            a_time = agent.evaluate_path(agent.p, metric="result")
            if a_time > max_time:
                max_time = a_time
            bid_sum += a_time
            assignment.append(copy.deepcopy(agent.p))

        return assignment, np.round(bid_sum, 6), max_time

    def initialize_all(self):
        self.initialize_tasks()
        self.initialize_agents()

    def initialize_tasks(self):
        self.tasks = []
        for j in range(self.nt):
            char = self.char_t[j]
            self.tasks.append(Task(id=j, char=char))

    def initialize_agents(self):
        self.agents = []
        for i in range(self.na):
            char = self.char_a[i]
            self.agents.append(
                CBBA_Agent(id=i, G=self.G, char=char, tasks=self.tasks, Lt=self.Lt, start_time=self.start_time,
                           metric=self.metric, D = self.D))

    def print_perf(self):
        """
        Performance printing function
        :return:
        """
        for i in range(self.na):
            path = self.assig_history[-1][i]
            score = self.agents[i].evaluate_path(path, metric="RPT")
            cvg_time = np.round(1000 * self.all_times[i])
            print("Agent {} converged: Path = {}, Score = {}, cvg time = {} ms".format(i, path, score, cvg_time))
            
    def compute_TDR_obj(self):
        """
        compute TDR sum value of final allocation (THE HIGHER THE BETTER)
        :return: TDR sum value
        """
        S = 0
        lamb = 0.95
        for i in range(self.na):
            agent = self.agents[i]
            P = agent.p
            for j in range(len(agent.p)):
                S += self.tasks[P[j]].cbar * lamb**(agent.evaluate_path(P[:(j+1)], metric = "result"))
        return S

    def compute_RPT_obj(self):
        """
        compute the effective RPT sum value of final allocation (THE LOWER THE BETTER)
        :return: effective RPT sum value
        """
        S = 0
        for i in range(self.na):
            agent = self.agents[i]
            P = agent.p
            B = agent.b
            J = []
            for j in range(len(B)):
                J.append(B[j])
                Pb = copy.deepcopy(P)
                for k in P:
                    if k not in J:
                        Pb.remove(k)
                S += abs(agent.evaluate_path(Pb, metric="RPT"))
        return S
    
    def update_communication_graph(self):
        """
        Update communication graph G according to current agents positions and communication range
        :return: updated G and diameter D
        """

        G_new = np.zeros((self.na, self.na))

        # Each agent can always communicate with itself
        for i in range(self.na):
            G_new[i, i] = 1.0

        # Checking distances between all agent pairs
        for i in range(self.na):
            for k in range(i+1, self.na):
                if i != k:
                    dist = np.linalg.norm(self.agents[i].pos - self.agents[k].pos)
                    if dist <= self.agents[i].comm_range:
                        G_new[i, k] = 1.0
                        G_new[k, i] = 1.0 # Symmetric graph
                    else:
                        G_new[i, k] = 0.0
                        G_new[k, i] = 0.0

        # Check if full connectivity is maintained
        if not verify_connection(G_new):
            pass

        # Update Graph
        self.G = G_new

        # Update each agent's communication matrix
        for agent in self.agents:
            agent.G = G_new
            agent.nb_neigh = np.sum(agent.G[agent.id, :]) - 1
        
        # Update diameter D
        try:
            raw_graph = nx.from_numpy_array(G_new)
            if nx.is_connected(raw_graph):
                D_new = nx.diameter(raw_graph)
            else:
                D_new = self.D  # Keep previous diameter if not connected
                # D_new = float('inf')  # Infinite diameter if not connected
        except:
            D_new = self.D  # Keep previous diameter if error occurs
        
        return G_new, D_new

    def visualize_dynamic_graph(self, iteration, save_fig=True):
        """
        Visualize agent positions, tasks, and communication links at a given iteration
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Agent positions, tasks, and communication ranges
        ax1.set_title(f'Agent Positions & Tasks (Iteration {iteration})', fontsize=14, fontweight='bold')
        
        # Plot tasks
        task_x = [task.pos[0] for task in self.tasks]
        task_y = [task.pos[1] for task in self.tasks]
        ax1.scatter(task_x, task_y, c='red', marker='x', s=150, linewidths=3, label='Tasks', zorder=5)
        for j, task in enumerate(self.tasks):
            ax1.text(task.pos[0]+0.15, task.pos[1]+0.15, f'T{j}', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot agents with communication ranges
        colors = plt.cm.tab10(np.linspace(0, 1, self.na))
        for i, agent in enumerate(self.agents):
            # Agent position
            ax1.scatter(agent.pos[0], agent.pos[1], c=[colors[i]], marker='o', 
                    s=200, edgecolors='black', linewidths=2, label=f'Agent {i}', zorder=5)
            ax1.text(agent.pos[0]+0.15, agent.pos[1]-0.25, f'A{i}', fontsize=10, 
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='lightblue', alpha=0.8))
            
            # Communication range circle
            circle = plt.Circle(agent.pos, agent.comm_range, color=colors[i], 
                            fill=False, linestyle='--', linewidth=2, alpha=0.4)
            ax1.add_patch(circle)
            
            # Draw current path
            if len(agent.p) > 0:
                path_positions = [agent.pos] + [self.tasks[task_id].pos for task_id in agent.p]
                path_x = [pos[0] for pos in path_positions]
                path_y = [pos[1] for pos in path_positions]
                ax1.plot(path_x, path_y, c=colors[i], linestyle='-', linewidth=2, 
                        alpha=0.6, marker='>', markersize=8)
        
        ax1.set_xlabel('X Position', fontsize=12)
        ax1.set_ylabel('Y Position', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.axis('equal')
        
        # Plot 2: Communication graph
        ax2.set_title(f'Communication Graph (Iteration {iteration})', fontsize=14, fontweight='bold')
        raw_graph = nx.from_numpy_array(self.G)
        
        # Position nodes at agent locations
        pos_dict = {i: self.agents[i].pos for i in range(self.na)}
        
        # Draw edges
        nx.draw_networkx_edges(raw_graph, pos=pos_dict, ax=ax2, width=2, 
                            alpha=0.5, edge_color='gray')
        
        # Draw nodes
        node_colors = [colors[i] for i in range(self.na)]
        nx.draw_networkx_nodes(raw_graph, pos=pos_dict, ax=ax2, 
                            node_color=node_colors, node_size=500, 
                            edgecolors='black', linewidths=2)
        
        # Draw labels
        labels = {i: f'A{i}' for i in range(self.na)}
        nx.draw_networkx_labels(raw_graph, pos=pos_dict, labels=labels, ax=ax2, 
                            font_size=10, font_weight='bold')
        
        # Add statistics
        num_edges = raw_graph.number_of_edges()
        is_connected = nx.is_connected(raw_graph)
        try:
            diameter = nx.diameter(raw_graph) if is_connected else 'inf'
        except:
            diameter = 'N/A'
        
        stats_text = f'Edges: {num_edges}\nConnected: {is_connected}\nDiameter: {diameter}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_xlabel('X Position', fontsize=12)
        ax2.set_ylabel('Y Position', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'dynamic_graph_iter_{iteration:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def track_connectivity_history(self):
        """
        Track how connectivity changes over iterations
        """
        self.connectivity_history = []
        self.diameter_history = []
        
    def log_connectivity(self, iteration):
        """
        Log connectivity metrics at each iteration
        """
        raw_graph = nx.from_numpy_array(self.G)
        is_connected = nx.is_connected(raw_graph)
        num_edges = raw_graph.number_of_edges()
        
        if is_connected:
            diameter = nx.diameter(raw_graph)
        else:
            diameter = float('inf')
        
        self.connectivity_history.append({
            'iteration': iteration,
            'connected': is_connected,
            'edges': num_edges,
            'diameter': diameter
        })

    def plot_connectivity_evolution(self):
        """
        Plot how connectivity metrics evolve over time
        """
        if not hasattr(self, 'connectivity_history') or len(self.connectivity_history) == 0:
            print("No connectivity history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = [h['iteration'] for h in self.connectivity_history]
        connected = [h['connected'] for h in self.connectivity_history]
        edges = [h['edges'] for h in self.connectivity_history]
        diameters = [h['diameter'] if h['diameter'] != float('inf') else None 
                    for h in self.connectivity_history]
        
        # Plot 1: Connectivity status
        axes[0, 0].plot(iterations, connected, marker='o', linewidth=2)
        axes[0, 0].set_title('Network Connectivity Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Connected (1=Yes, 0=No)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([-0.1, 1.1])
        
        # Plot 2: Number of edges
        axes[0, 1].plot(iterations, edges, marker='s', color='green', linewidth=2)
        axes[0, 1].set_title('Number of Communication Links', fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Number of Edges')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Network diameter
        valid_iters = [it for it, d in zip(iterations, diameters) if d is not None]
        valid_diams = [d for d in diameters if d is not None]
        axes[1, 0].plot(valid_iters, valid_diams, marker='^', color='orange', linewidth=2)
        axes[1, 0].set_title('Network Diameter Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Diameter')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Agent positions heatmap (density)
        all_positions = np.array([agent.pos for agent in self.agents])
        axes[1, 1].scatter(all_positions[:, 0], all_positions[:, 1], 
                        s=200, alpha=0.6, c=range(self.na), cmap='viridis')
        axes[1, 1].set_title('Final Agent Positions', fontweight='bold')
        axes[1, 1].set_xlabel('X Position')
        axes[1, 1].set_ylabel('Y Position')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plt.savefig('connectivity_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()

    def create_animation(self, output_filename='dynamic_cbba.gif', fps=2):
        """
        Create an animation from saved iteration images
        Requires: pip install pillow
        """
        import glob
        from PIL import Image
        
        # Get all saved images
        image_files = sorted(glob.glob('dynamic_graph_iter_*.png'))
        
        if len(image_files) == 0:
            print("No images found to create animation")
            return
        
        # Load images
        images = [Image.open(img) for img in image_files]
        
        # Save as GIF
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=int(1000/fps),  # milliseconds per frame
            loop=0
        )
        
        print(f"Animation saved as {output_filename}")
        
        # Clean up individual images (optional)
        # for img_file in image_files:
        #     os.remove(img_file)

class CBBA_Agent:

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
        self.comm_range = char[3] if len(char) > 3 else 3.0
        # list of tasks (should be of size =nb agents)
        self.tasks = tasks
        # int, nb of tasks
        self.nt = len(tasks)
        # capacity
        self.Lt = Lt
        self.D = D
        self.makespan = 0
        self.updated_makespan = 0
        self.length = 0

        self.metric = metric
        if self.metric == "RPT":
            self.min_val = -1e20
        elif metric == "TDR":
            self.min_val = 0

        # list of winning bids for each task (size Nt)
        self.y = [self.min_val for _ in range(self.nt)]
        # list of winners for each task (size Nt)
        # self.z = np.array([None for _ in range(self.nt)]).reshape((1,-1))
        self.z = [None for _ in range(self.nt)]
        self.z_before = [None for _ in range(self.nt)]
        self.z_pre_consensus = [None for _ in range(self.nt)]
        # list of potentially winning bids (size Nt)
        self.h = [self.min_val for _ in range(self.nt)]
        # list of bids on each task (size Nt)
        self.c = [self.min_val for _ in range(self.nt)]

        self.j_outbid = False
        self.outbidder = None
        self.outbid_pos = None
        # bundle
        self.b = []
        # path /ordered bundle
        self.p = []

        # timestamps
        self.s = [-inf for _ in range(self.na)]
        self.s[self.id] = 0
        # tasks not in path / bundle
        self.filtered_index = list(range(self.nt))
        # clock start time
        self.start_time = start_time
        self.S = [0]
        self.converged = False
        #observation list of other agents convergence
        self.their_net_cvg = [False for _ in range(self.D)]
        self.cvg_counter = 0

        #marginal gain list
        self.Wa = []
        #task insert position list
        self.placement = []
        #agent has won at previous iteration ? (optional)
        self.flag_won = True
        #size of path at previous iteration
        self.len_p_before = 0



    def evaluate_path(self, P, metric="RPT"):
        """
        Evaluate path P according to either the RPT metric or the TDR one. Result is the minsum metric (used only for final rendering)
        :param P: path = list of task indices
        :return: evaluation of path
        """
        current_pos = self.pos
        score = 0
        time = 0
        if len(P) > 0:
            for j in range(len(P)):
                task = self.tasks[P[j]]
                time += (np.linalg.norm(np.array(current_pos) - np.array(task.pos))) / self.speed
                time += task.duration
                if metric == "RPT" or metric == "result":
                    score -= time
                    time = 0
                elif metric == "TDR":
                    score += (task.lamb) ** time
                current_pos = task.pos
        if metric == "result":
            return -score
        else:
            return score


    def compute_c(self, j, metric="RPT"):
        """
        Compute marginal gain c[j] of j and optimal position in path
        :param j: task j, assumed to not be in bundle
        :return: c[j], optimal position (index) of j in the path
        """
        # self.filtered_index = [k for k,task in enumerate(self.tasks) if task.id not in self.p]
        path_bids = []
        placements = []
        P = self.p
        cur_task = self.tasks[j].id
        for pos in range(len(self.p) + 1):
            P1 = copy.deepcopy(P)
            P1.insert(pos, cur_task)
            if metric == "TDR":
                mg = self.evaluate_path(P1, metric) - self.S[-1]
            elif metric == "RPT":
                mg = self.evaluate_path(P1, metric)
            path_bids.append(mg)
            placements.append(pos)
        c = np.max(path_bids)
        optimal_pos = np.argwhere(path_bids == c)[-1][0]
        return c, optimal_pos

    def create_bundle(self, iter, build_bundle = "ADD"):
        """
        Path building phase for each agent
        :param iter:
        :param build_bundle: "ADD" is the method of GCBBA (adds only one task). "FULLBUNDLE" is baseline CBBA's method (builds full path)
        :return:
        """
        self.updated_makespan = self.makespan

        if build_bundle == "FULLBUNDLE":
            while len(self.b) < self.Lt:
                optimal_placement = np.zeros(self.nt)
                self.filtered_index = [k for k, task in enumerate(self.tasks) if task.id not in self.p]
                for j in self.filtered_index:
                    c, opt_place = self.compute_c(j, metric = self.metric)
                    self.c[j] = c
                    optimal_placement[j] = opt_place
                if self.metric == "TDR":
                    cbar = copy.deepcopy(self.c)
                    if len(self.b)>0:
                        min_c = np.min([self.c[j] for j in self.b])
                        for j in self.filtered_index:
                            cbar[j] = min(self.c[j],min_c)
                    bids = [self.c[j] if (cbar[j] > self.y[j] and j in self.filtered_index or cbar[j] == self.y[
                        j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                            range(self.nt)]
                    J = np.argmax(bids)
                    self.c = cbar
                else:
                    bids = [self.c[j] if (self.c[j] > self.y[j] and j in self.filtered_index or self.c[j] == self.y[j] and j in self.filtered_index and self.z[j]>self.id) else self.min_val for j in
                            range(self.nt)]
                    J = np.argmax(bids)
                if J in self.p or (bids[J] <= self.min_val):
                    return

                best_task = self.tasks[J].id
                self.b.append(best_task)
                self.p.insert(int(optimal_placement[J]), best_task)
                self.S.append(self.evaluate_path(self.p, self.metric))
                self.y[J] = self.c[J]
                self.z[J] = self.id
        #build_bundle = add
        else:
            if len(self.b) < self.Lt:
                self.filtered_index = [k for k, task in enumerate(self.tasks) if task.id not in self.p]
                if self.flag_won==True:
                    self.placement = np.zeros(self.nt)
                    for j in self.filtered_index:
                        c, opt_place = self.compute_c(j, metric=self.metric)
                        self.c[j] = c
                        self.placement[j] = opt_place
                    if self.metric == "TDR":
                        cbar = copy.deepcopy(self.c)
                        if len(self.b)>0:
                            min_c = np.min([self.c[j] for j in self.b])
                            for j in self.filtered_index:
                                #cbar[j] = min(self.c[j],min_c)
                                cbar[j] = self.c[j]
                        bids = [self.c[j] if (cbar[j] > self.y[j] and j in self.filtered_index or cbar[j] == self.y[
                            j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                                range(self.nt)]
                        J = np.argmax(bids)
                        self.c = cbar
                    else:
                        bids = [self.c[j] if (self.c[j] > self.y[j] and j in self.filtered_index or self.c[j] == self.y[
                            j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                                range(self.nt)]
                        J = np.argmax(bids)
                else:
                    bids = [self.c[j] if (self.c[j] > self.y[j] and j in self.filtered_index or self.c[j] == self.y[
                        j] and j in self.filtered_index and self.z[j] > self.id) else self.min_val for j in
                            range(self.nt)]
                    J = np.argmax(bids)

                if J in self.p or (bids[J] <= self.min_val):
                    return

                best_task = self.tasks[J].id
                self.b.append(best_task)
                self.p.insert(int(self.placement[J]), best_task)
                self.S.append(self.evaluate_path(self.p, self.metric))
                self.y[J] = self.c[J]
                self.z[J] = self.id
                self.updated_makespan = max(self.updated_makespan, -self.evaluate_path(self.p, "RPT"))

    def compute_s(self, neigh, iter):
        """
        compute the new timestamps based on communication (reception) with agent neigh
        :param neigh: agent object, assumed to be neighbor of self
        :param iter: the timestamp updated alue is the current iteration of algo (can also be based on clock time -> current time - start time)
        :return:
        """
        cur_time = time.perf_counter()
        diff = cur_time - self.start_time
        self.s[self.id] = iter
        self.s[neigh.id] = iter
        #self.s[neigh.id] = diff
        not_neigh_index = np.argwhere(self.G[self.id, :] == 0).flatten()
        greater_index = np.argwhere(np.array(neigh.s) > np.array(self.s)).flatten()
        intersect = list(set(greater_index).intersection(set(not_neigh_index)))
        self.s = [neigh.s[i] if i in intersect else self.s[i] for i in range(self.na)]


    def update(self, neigh, j):
        """
        update procedure
        :param neigh: agent object, assumed to be neighbor of self
        :param j: task j
        :return:
        """
        self.y[j] = neigh.y[j]
        self.z[j] = neigh.z[j]

        bundle = self.b
        if j in bundle:
            self.flag_won = False
            bundle_index = bundle.index(j)
            tasks_to_remove = self.b[bundle_index:]

            self.y = [self.y[i] if i not in tasks_to_remove else self.min_val for i in range(self.nt)]
            self.z = [self.z[i] if i not in tasks_to_remove else None for i in range(self.nt)]
            self.y[j] = neigh.y[j]
            self.z[j] = neigh.z[j]

            self.b = self.b[:bundle_index]
            for j in tasks_to_remove:
                self.p.remove(j)
            self.S = self.S[:bundle_index+1]
            self.j_outbid = j
            self.outbidder = neigh.id
            self.outbid_pos = bundle_index
            self.their_net_cvg[0]=False
            self.length = self.evaluate_path(self.p, "RPT")
            if self.length<self.makespan:
                self.updated_makespan = self.makespan
            else:
                self.updated_makespan = self.length


    def reset(self, j):
        """
        reset procedure
        :param j: task j
        :return:
        """
        self.y[j] = self.min_val
        self.z[j] = None
        bundle = self.b
        if j in bundle:
            bundle_index = bundle.index(j)
            tasks_to_remove = self.b[bundle_index:]
            index_to_remove = [t for (i, t) in enumerate(bundle) if t in tasks_to_remove]

            self.y[j] = self.min_val
            self.z[j] = None

            self.y = [self.y[i] if i not in index_to_remove else self.min_val for i in range(self.nt)]
            self.z = [self.z[i] if i not in index_to_remove else None for i in range(self.nt)]

            self.b = self.b[:bundle_index]
            for j in tasks_to_remove:
                self.p.remove(j)
            if self.length < self.makespan:
                self.updated_makespan = self.makespan
            else:
                self.updated_makespan = self.length
            self.S = self.S[:bundle_index+1]
            self.their_net_cvg[0]= False

    def leave(self):
        """
        leave procedure, does nothing
        :return:
        """
        pass



    def resolve_conflict(self, all_agents, iter=0, index = "else"):
        """
        Resolution conflict phase (consensus)
        :param all_agents: list of all agents in the problem
        :param iter: current interation (for printing purposes only)
        :return:
        """
        neigh_index = np.argwhere(self.G[self.id, :] == 1).flatten()
        neigh_cvg = [True for _ in range(self.D)]
        for k in neigh_index:
            neigh = all_agents[k]
            #self.converged = neigh.converged
            for j in range(self.nt):
                # agent k (sender) thinks zkj is k
                if neigh.z[j] == neigh.id:
                    if self.z[j] == self.id:
                        if neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id <self.id):
                            update = True
                            if update == True:
                                self.update(neigh, j)
                    elif self.z[j] == k:
                        update = True
                        if update == True:
                            self.update(neigh, j)
                    # (unassigned/none)
                    elif self.z[j] == None:
                        update = True
                        if update == True:
                            self.update(neigh, j)
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m] or neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id <self.id):
                            self.update(neigh, j)
                # agent k (sender) thinks zkj is i
                elif neigh.z[j] == self.id:
                    if self.z[j] == self.id:
                        self.leave()
                    elif self.z[j] == k:
                        self.reset(j)
                    elif self.z[j] == None:
                        self.leave()
                    else:
                        m = self.z[j]
                        if neigh.s[m] > self.s[m]:
                            self.reset(j)
                elif neigh.z[j] == None:
                    if self.z[j] == self.id:
                        self.leave()
                    elif self.z[j] == k:
                        self.update(neigh, j)
                    elif self.z[j] == None:
                        self.leave()
                    else:
                        m = self.z[j]
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                # agent k (sender) thinks zkj is m not in  {i,k (, none)}
                else:
                    m = neigh.z[j]
                    if self.z[j] == self.id:
                        if neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j] or (  neigh.s[m] > self.s[m] and (neigh.y[j] == self.y[j] and neigh.id <self.id)):
                            self.update(neigh, j)
                    elif self.z[j] == k:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                        else:
                            self.reset(j)
                    elif self.z[j] == m:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                    elif self.z[j] == None:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, j)
                    else:
                        n = self.z[j]
                        #if n not in [self.id, neigh.id, m, None]:
                        if neigh.s[m] > self.s[m] and neigh.s[n] > self.s[n]:
                            self.update(neigh, j)
                        elif neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j] or (neigh.s[m] > self.s[m] and (neigh.y[j] == self.y[j] and neigh.id <self.id)):
                            self.update(neigh, j)
                        elif neigh.s[n] > self.s[n] and self.s[m] > neigh.s[m]:
                            self.reset(j)
            # update stamp if some updates/resets/leaves were made with a neighbor
            self.updated_makespan = max(self.updated_makespan, neigh.updated_makespan)
            self.compute_s(neigh, iter)
            for i in range(1, self.D):
                neigh_cvg[i] = neigh_cvg[i] and neigh.their_net_cvg[i-1]
        self.makespan = self.updated_makespan
        self.their_net_cvg[0] = (self.z == self.z_before)
        for i in range(1, self.D):
            self.their_net_cvg[i] = neigh_cvg[i] and self.their_net_cvg[i - 1]

        self.converged = self.their_net_cvg[-1]

        if self.converged==True:
            self.cvg_counter += 1

        if index == "last":
            self.y_before = copy.deepcopy(self.y)
            self.z_before = copy.deepcopy(self.z)
            self.flag_won = (len(self.p) != self.len_p_before)
            self.len_p_before = len(self.p)

    def update_position(self):
        """
        Update agent position based on current task assignment
        :param new_pos: new position (x,y)
        :return:
        """
        dt = 0.1  # time step
        if len(self.p) > 0:
            # Move towards the first task in the path
            next_task = self.tasks[self.p[0]]
            direction = next_task.pos - self.pos
            distance_to_next_task = np.linalg.norm(direction)

            if distance_to_next_task > 0:
                # Move towards the task at the agent's speed
                step_size = min(self.speed * dt, distance_to_next_task)
                self.pos += (direction / distance_to_next_task) * step_size

            if distance_to_next_task < 0.1:
                # Arrived at the task, remove it from the path
                self.p.pop(0)
                if len(self.b) > 0:
                    self.b.pop(0)

if __name__ == "__main__":
    """
    To have GCBBA allocation: method = "global", detector = "none" or "decentralized" in launch_agents
    To switch to CBBA allocation: method = "baseline", detector = "none" or "centralized" in launch_agents
    """
    # On seed 5, CBBA and GCBBA (SGA) allocations are different for RPT
    seed = 5
    np.random.seed(seed)
    na = 10
    nt = 100
    Lt = ceil(nt / na)
    xlim = [-5, 5]
    ylim = xlim
    sp_lim = [1, 5]
    dur_lim = [1, 5]
    metric = "RPT"
    comm_range = 3.0

    # communication graph initialization
    raw_graph, G = create_graph(na, p=0.5, graph_type="random", seed=seed)
    D = nx.diameter(raw_graph)
    agents, tasks = task_agent_init(na=na, nt=nt, pos_lim=xlim, sp_lim=sp_lim, dur_lim=dur_lim, lamb_lim=[0.95, 0.95],
                                    clim=[1, 1])
    
    # Add communication range to agents
    for i in range(na):
        agents[i] = np.concatenate((agents[i], np.array([comm_range])))
    
    orch_cbba= Orchestrator_CBBA(G, D, tasks, agents, Lt, metric=metric)

    # allocation launching
    t0 = time.time()
    assig, tot_score, makespan = orch_cbba.launch_agents(method="global", detector = "decentralized", dynamic_communication_graph=True, update_freq=5, visualize_graph = True, viz_freq=10)
    tf0 = np.round(1000 * (time.time() - t0))

    print("GCBBA-{} total score. = {}; max score = {}; time = {} ms; assignment = {}".format(metric,tot_score, makespan, tf0,
                                                                                              assig))

    orch_cbba.create_animation('cbba_dynamic.gif', fps=2)

    draw_paths(tasks, agents, assig, tot_score, title="GCBBA-{}".format(metric))








