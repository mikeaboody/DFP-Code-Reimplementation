from __future__ import print_function
import sys
sys.path = ['../..'] + sys.path
from doom_simulator import DoomSimulator
from multi_doom_simulator import MultiDoomSimulator
import numpy as np
import time

#NOTE: most of code taken from https://github.com/IntelVCL/DirectFuturePrediction

def create_basic_simulator(num_simulators=1):
	
	## Simulator
	simulator_args = {}
	simulator_args['config'] = 'D1_basic.cfg'
	simulator_args['resolution'] = (84,84)
	simulator_args['frame_skip'] = 4
	simulator_args['color_mode'] = 'GRAY'	
	simulator_args['maps'] = ['MAP01']
	simulator_args['switch_maps'] = False
	simulator_args['game_args'] = ""
	#train
	simulator_args['num_simulators'] = 8
	
	# Create and return
	if num_simulators != 1:
		ds = MultiDoomSimulator([simulator_args] * num_simulators)
	else:
		ds = DoomSimulator(simulator_args)
	return ds
