from __future__ import print_function
import sys
sys.path = ['../..'] + sys.path
from flappy_simulator import FlappySimulator
import numpy as np
import time

def create_basic_simulator():

    fs = FlappySimulator()
    return fs
