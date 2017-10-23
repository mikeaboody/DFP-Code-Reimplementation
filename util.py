import numpy as np
import random
import math

def get_at_indeces(lst, indeces):
    return [lst[i] for i in indeces]

def biased_coin(prob):
    return random.random() < prob

def create_label(obs, temp_offsets, recent_obs_act_pairs):
    pairs_at_offsets = recent_obs_act_pairs.index_from_back(temp_offsets)
    meas_at_offsets = [pair[0].meas for pair in pairs_at_offsets]
    temp_diffs = [meas - obs.meas for meas in meas_at_offsets]
    to_vector = np.array(temp_diffs).flatten()
    return to_vector
    
def enumerate_action_one_hots(num_isolated_actions):
	return list(np.identity(2 ** num_isolated_actions))

def action_to_one_hot(action):
	res = np.zeros(2 ** len(action))
	index = 0
	i = 0
	for ele in action:
		index += ele * (2 ** i)
		i += 1
	res[index] = 1
	return res

def action_from_one_hot(action_one_hot):
	index = 0
	for ele in action_one_hot:
		if ele == 1:
			break
		index += 1
	res = [0] * int(math.log(len(action_one_hot), 2))
	i = 0
	while (index != 0):
		if index % 2 == 1:
			res[i] = 1
		index = index // 2
		i += 1
	return res
