import numpy as np

class BoundedCache(object):
    """Efficient cache of last 'capacity' nump arrays. Makes use of clock algorithm 
       but with no notion of reference bits.
       
       Can be used for experience memory and temporal offsets.
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

    def temporal_offsets(self, offsets):
        #not allowed to ask for offset greater than number of points in memory
        assert max(offsets) <= self.size
        indeces = (self.clock_hand - np.array(offsets)) % self.size
        return self.np_arrs[indeces]
