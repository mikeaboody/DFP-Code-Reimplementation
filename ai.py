import numpy as np 
import json
import random
from util import *
from abstraction import *
from network import Network
from bounded_cache import BoundedCache

class Agent(object):
    """
    M: # of points in memory
    N: # of points subsampled during training
    k: number of new experiences to update prediction parameters
    temp_offsets: temporal offsets
    g_train: fixed goal vector (TODO variable goals) for training
    eps_decay: fixed schedule for which eps decays

    all of these are set by a config
    """

    def __init__(self, agent_config, possible_actions, network_builder):
        self.load_agent_config(agent_config)
        self.recent_obs_act_pairs = BoundedCache(max(self.temp_offsets) + 1)
        self.experience_memory = BoundedCache(self.M)
        self.num_exp_added = 0
        self.eps = 1
        self.network = network_builder()
        self.possible_actions = possible_actions

    def load_agent_config(self, json_filename):
        with open(json_filename) as json_data_file:
            data = json.load(json_data_file)
            assert set([u'M', u'N', u'k', u'g_train', u'eps_decay', u'temp_offsets']) == set(data.keys())
            for key in data:
                if isinstance(data[key], list):
                    setattr(self, key, np.array(data[key]))
                else:
                    setattr(self, key, data[key])

    def observe(self, obs, action):
        self.recent_obs_act_pairs.add((obs, action))
        self.num_exp_added += 1
        if not self.recent_obs_act_pairs.at_capacity():
            return
        #create an experience
        exp_obs, exp_act = self.recent_obs_act_pairs.index_from_back(0)[0]
        exp_label = create_label(exp_obs, self.temp_offsets, self.recent_obs_act_pairs)
        self.experience_memory.add(Experience(exp_obs, exp_act, self.g_train, exp_label))

        if self.num_exp_added % self.k == 0:
            sampled_exp = self.experience_memory.sample(self.N)
            self.network.update_weights(sampled_exp)

        #TODO do we do this only when we update weights or for every experience we get?
        self.eps = max(self.eps - self.eps_decay, 0)

    def act(self, obs=None, training=False, goal=None):
        assert training or goal != None

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

    def signal_episode_end(self):
        self.recent_obs_act_pairs = BoundedCache(max(self.temp_offsets) + 1)
