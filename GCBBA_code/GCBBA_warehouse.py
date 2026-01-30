import yaml
import os
from math import ceil

from tools import *
from tools_warehouse import create_graph_with_range

def read_warehouse_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    seed = 5 # Random seed for reproducibility

    # Get the path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config', 'gridworld_warehouse_small.yaml')
    
    # Read the configuration
    config = read_warehouse_config(config_path)
    
    # Access the parameters
    params = config['create_gridworld_node']['ros__parameters']
    
    # GCBBA specific parameters
    na = len(params['agent_positions']) // 3  # Number of agents
    nt = 100 
    Lt = ceil(nt / na)  # Tasks per agent
    xlim = [0, int(params['grid_width']) * params['grid_resolution']]
    ylim = [0, int(params['grid_height']) * params['grid_resolution']]
    sp_lim = [1, 5] # Speed limits in units/sec
    dur_lim = [1, 10] # Task duration limits in seconds
    comm_range = 30  # Communication range in units
    
    # Extract agent positions from config (reshape flattened array)
    agent_pos_flat = params['agent_positions']
    # LATER TODO: add agent IDs from config
    # Add agent_id to agent positions and then use the commented line instead
    agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], i//3 + 1) 
                       for i in range(0, len(agent_pos_flat), 3)]
    # agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2]) 
    #                    for i in range(0, len(agent_pos_flat), 4)]
    

    # Extract induct station positions from config
    induct_pos_flat = params['induct_stations']
    induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3]) 
                        for i in range(0, len(induct_pos_flat), 4)]
    
    # Create communication graph based on distance
    raw_graph, G = create_graph_with_range(agent_positions, induct_positions, comm_range)
    D = nx.diameter(raw_graph)


    
