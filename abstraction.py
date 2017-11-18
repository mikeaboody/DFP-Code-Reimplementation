import numpy as np
from util import action_to_one_hot

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

def create_experience():
    sens = np.random.random_sample(size=(84,84,1))
    meas = np.random.random_sample(size=(1,))
    goal = np.random.random_sample(size=(6,))
    # last value indicate index of action
    label = np.random.random_sample(size=(6,))
    obs = Observation(sens, meas)
    exp = Experience(obs, action_to_one_hot([0,1,0]), goal, label)
    return exp

lst = [create_experience() for _ in range(1000)]
for exp in lst:
    copy = Experience.from_array(exp.to_array())
    same = np.array_equal(exp.sens(), copy.sens()) and np.array_equal(exp.meas(), copy.meas()) and np.array_equal(exp.action(), copy.action()) and np.array_equal(exp.goal(), copy.goal()) and np.array_equal(exp.label(), copy.label())
    if not same:
        print("BAD")
