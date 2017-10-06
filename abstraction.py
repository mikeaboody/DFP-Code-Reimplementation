import numpy as np

class Observation(object):
    """
    An observation <sens, meas> where sens is the sensory input and 
    measurement stream.
    """
    def __init__(self, sens, meas):
        self.sens = sens
        self.meas = meas

class Experience(object):
    """
    An experience is just a training point. It has an observation,
    an action, an associated goal (the training goal), and an associated
    label which is just the temporal differences between the current
    measurements and future measurements.
    """
    def __init__(self, obs, a, g, label):
        self.obs = obs
        self.a = a
        self.g = g
        self.label = label

    def sens(self):
        return np.copy(self.obs.sens)
    def meas(self):
        return np.copy(self.obs.meas)
    def goal(self):
        return np.copy(self.g)
    def label(self):
        return np.copy(self.label)
        