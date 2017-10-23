import sys
sys.path.append("..")
from ai import Agent
from basic_doom_simulator import create_basic_simulator
from basicnetwork import basicNetwork
from abstraction import *

config_path = "../sample_config.json"
possible_actions = [[1,0,0], [0,1,0], [0,0,1]]

def run():
	doom_simulator = create_basic_simulator()
	agent = Agent(config_path, possible_actions, lambda: basicNetwork(3))
	img = None
	meas = None
	terminated = None

	i = 0
	while not terminated:
		if i == 0:
			action_taken = agent.act(training=True)
		else:
			action_taken = agent.act(Observation(img, meas), training=True)
		print(i, action_taken)
		img, meas, _, terminated = doom_simulator.step(action_taken)
		if (terminated):
			agent.signal_episode_end()
		i += 1
		print(meas)




run()