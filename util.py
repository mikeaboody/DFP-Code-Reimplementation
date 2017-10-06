import numpy as np
import random

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
