import numpy as np 
import json

class Observation(object):
    """
    An observation at tim et is pair <s_t, m_t> where s_t is the sensory input
    s_t and measurement stream m_t
    """
    def __init__(self, s_t, m_t, t):
        self.s_t = s_t
        self.m_t = m_t
        self.t = t

    def as_vector(self):
        return np.concat((self.s_t, self.m_t))


class Agent(object):
    """
    M: # of points in memory
    N: # of points subsampled during training
    k: number of new experiences to update prediction parameters
    temp_offsets: temporal offsets
    g_train: fixed goal vector (TODO variable goals) for training
    eps_init: initial probability of random action 
    eps_decay: fixed schedule for which eps decays

    all of these are set by a config
    """

    def __init__(self, agent_config):
        self.load_agent_config(agent_config)

    def load_agent_config(self, json_filename):
        with open(json_filename) as json_data_file:
            data = json.load(json_data_file)
            for key in data:
                if isinstance(data[key], list):
                    setattr(self, key, np.array(data[key]))
                else:
                    setattr(self, key, data[key])

    def add_experience(obs):
        pass

    def act():
        pass

class PerceptionModule(object):
    pass

class MeasurementModule(object):
    pass

class GoalModule(object):
    pass

class ExpectationStream(object):
    pass

class ActionStream(object):
    pass
