import numpy as np 

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
    g_train: fixed goal vector (TODO variable goals) for training (MUST MATCH len(temporal_offsets) * # of measurements)
    eps_init: initial probability of random action 
    eps_decay: fixed schedule for which eps decays
    """
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