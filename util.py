import numpy as np
import random
import math
from abstraction import *

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

def action_number(action_one_hot):
	index = 0
	for ele in action_one_hot:
		if ele == 1:
			break
		index += 1
	return index

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
	index = action_number(action_one_hot)
	res = [0] * int(math.log(len(action_one_hot), 2))
	i = 0
	while (index != 0):
		if index % 2 == 1:
			res[i] = 1
		index = index // 2
		i += 1
	return res

def serialize_experiences(exp_lst, filename):
	serialized_arr = np.array([exp.to_array() for exp in exp_lst])
	np.save(filename, serialized_arr)

def deserialize_experiences(filename):
	serialized_arr = np.load(filename)
	exp_lst = [Experience.from_array(serialized_arr[i]) for i in range(len(serialized_arr))]
	return exp_lst

def create_experience():
	sens = np.random.random_sample(size=(84,84,1))
	meas = np.random.random_sample(size=(1,))
	goal = np.random.random_sample(size=(6,))
	# last value indicate index of action
	label = np.random.random_sample(size=(6,))
	obs = Observation(sens, meas)
	exp = Experience(obs, action_to_one_hot([0,1,0]), goal, label)
	return exp
	