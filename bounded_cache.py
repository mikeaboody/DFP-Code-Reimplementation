import numpy as np
from util import get_at_indeces

class BoundedCache(object):
    """Efficient cache of last 'capacity' items. Makes use of clock algorithm 
       but with no notion of reference bits.

       Can be used for experience memory and for calculating temporal offsets.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.clock_hand = 0
        self.items = []

    def add(self, item):
        if self.size < self.capacity:
            self.items.append(item)
            self.size += 1
        else:
            ret = self.items[self.clock_hand]
            self.items[self.clock_hand] = item
            self.clock_hand = (self.clock_hand + 1) % self.capacity

    def sample(self, k):
        indeces = np.random.choice(self.size, k, replace=True)
        return get_at_indeces(self.items, indeces)

    def index_from_back(self, indeces):
        if isinstance(indeces, int):
            indeces = [indeces]
        indeces_ = (self.clock_hand + np.array(indeces)) % self.size
        return get_at_indeces(self.items, indeces_)

    def get_all(self):
        return self.items[:]

    def at_capacity(self):
        return self.size == self.capacity
