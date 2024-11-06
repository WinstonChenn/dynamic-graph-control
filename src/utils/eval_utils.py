from tqdm import tqdm
import numpy as np
import torch

def evaluate_policy(policy, environment, num_int=3, num_times=100, verbose=True):
    model_rewards = []
    for i in tqdm(range(num_times), disable=not verbose):
        int_idx = policy(environment.G, num_int)
        environment.update()
        environment.intervene(int_idx)
        reward = environment.get_num_nodes_in_state("S") / len(environment.G.nodes)
        model_rewards.append(reward)
    return np.array(model_rewards)