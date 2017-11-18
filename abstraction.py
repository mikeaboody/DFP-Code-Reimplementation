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
    def __init__(self, obs, a, g, l):
        self.obs = obs
        self.a = a
        self.g = g
        self.l = l

    def __eq__(self, other):
        sens_equal = np.array_equal(self.sens(), other.sens())
        meas_equal = np.array_equal(self.meas(), other.meas())
        action_equal = np.array_equal(self.action(), other.action())
        goal_equal = np.array_equal(self.goal(), other.goal())
        label_equal = np.array_equal(self.label(), other.label())
        return sens_equal and meas_equal and action_equal and goal_equal and label_equal

    def sens(self):
        return np.copy(self.obs.sens)
    def meas(self):
        return np.copy(self.obs.meas)
    def action(self):
        return np.copy(self.a)
    def goal(self):
        return np.copy(self.g)
    def label(self):
        return np.copy(self.l)
    def to_array(self):
        return np.array([self.sens(), self.meas(), self.action(), self.goal(), self.label()])

    @staticmethod
    def from_array(arr):
        copy_arr = [np.copy(ele) for ele in arr]
        sens, meas, a, g, l = copy_arr
        return Experience(Observation(sens, meas), a, g, l)
