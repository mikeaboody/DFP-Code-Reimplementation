import numpy as np

class ExperienceMemory(object):
    """Efficient tracker of last 'capacity' points. Makes use of clock algorithm but
       with no notion of reference bits
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.clock_hand = 0
        self.np_arrs = np.array([])

    def add(self, np_arr):
        if self.size < self.capacity:
            if self.size == 0:
                self.np_arrs = np.array([np_arr])
            else:
                self.np_arrs = np.append(self.np_arrs, np.array([np_arr]), axis=0)
            self.size += 1
        else:
            self.np_arrs[self.clock_hand] = np_arr
            self.clock_hand = (self.clock_hand + 1) % self.capacity
        

    def sample(self, k):
        indeces = np.random.choice(self.size, k, replace=True)
        return self.np_arrs[indeces]
