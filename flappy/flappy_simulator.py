'''
ViZDoom wrapper
'''
from __future__ import print_function
import sys
import os
from game import wrapped_flappy_bird as flappy
import random
import time
import numpy as np
import re

class FlappySimulator:
    
    def __init__(self):        
        self._game = flappy.GameState()

    def close_game(self):
        self._game.quit()

    def step(self, action=0):
        if action == [0]:
            action = [1, 0]
        elif action == [1]:
            action = [0, 1]
        # action[0] == 1: do nothing
        # action[1] == 1: flap the bird
        img, rwd, term = self._game.frame_step(action)
        meas = self._game.score
        return img, meas, rwd, term
    
    def get_random_action(self):
        if random.random() > 0.5:
            return [1, 0]
        else:
            return [0, 1]


