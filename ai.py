import numpy as np 
import json
import random
from util import *
from abstraction import *
from network import Network
from bounded_cache import BoundedCache
from experience_creator import ExperienceCreator

class Agent(object):
    """
    M: # of points in memory
    N: # of points subsampled during training
    k: number of new experiences to update prediction parameters
    temp_offsets: temporal offsets
    g_train: fixed goal vector (TODO variable goals) for training

    all of these are set by a config
    """

    def __init__(self, agent_params, possible_actions, network_builder):
        self.load_agent_params(agent_params)
        self.dict_simulator_to_experience_creators = {}
        self.experience_memory = BoundedCache(self.M)
        self.num_exp_added = 0
        self.eps = 1
        self.possible_actions = possible_actions
        self.num_times_weights_updated = 0
        self.network = network_builder()

    def load_agent_params(self, agent_params):
        self.M = agent_params["M"]
        self.N = agent_params["N"]
        self.k = agent_params["k"]
        self.temp_offsets = agent_params["temp_offsets"]
        self.g_train = np.array(agent_params["g_train"])
        self.network_backing_file = agent_params["network_backing_file"]
        self.network_save_period = agent_params["network_save_period"]
        self.eps_decay_func = agent_params["eps_decay_func"]

    def observe(self, obs, action, simul_i=0):
        if simul_i not in self.dict_simulator_to_experience_creators:
            self.dict_simulator_to_experience_creators[simul_i] = ExperienceCreator(self.g_train, self.temp_offsets)
        self.num_exp_added += 1
        experience_creator = self.dict_simulator_to_experience_creators.get(simul_i)

        new_experience = experience_creator.add_and_get_experience(obs, action)
        if new_experience is None:
            return
        self.experience_memory.add(new_experience)

        if self.num_exp_added % self.k == 0:
            sampled_exp = self.experience_memory.sample(self.N)
            self.network.update_weights(sampled_exp)
            self.num_times_weights_updated += 1

            if self.network_save_period > 0 and self.num_times_weights_updated % self.network_save_period  == 0:
                self.network.save_network()

            self.eps = max(self.eps_decay_func(self.num_exp_added), 0)

    def act(self, obs=None, training=False, goal=None):
        assert training or goal is not None

        if obs == None:
            return random.choice(self.possible_actions)

        if training:
            if biased_coin(self.eps): #random action
                return random.choice(self.possible_actions)
            goal = self.g_train

        num_actions = len(self.possible_actions)
        #get prediction from network for all actions
        prediction = self.network.predict(obs, goal)

        assert len(prediction) == num_actions * len(goal)
        
        action_predictions = np.split(prediction, num_actions)
        action_indeces = list(range(num_actions))
        max_index = max(action_indeces, key=lambda i: action_predictions[i].dot(goal))
        return self.possible_actions[max_index]

    def signal_episode_end(self, simul_i):
        self.dict_simulator_to_experience_creators[simul_i].reset()
