import sys
sys.path.append("..")
from ai import Agent
from basic_doom_simulator import create_basic_simulator
from network import blank_network_builder
from abstraction import *

config_path = "../sample_config.json"
possible_actions = [[1,0,0], [0,1,0], [0,0,1]]

def run():
	doom_simulator = create_basic_simulator()
	agent = Agent(config_path, possible_actions, blank_network_builder(len(possible_actions)))
	agent.eps = 0
	img = None
	meas = None
	terminated = None

	i = 0
	while i < 1000:
		if i == 0:
			action_taken = agent.act(training=True)
		else:
			action_taken = agent.act(Observation(img, meas), training=True)
		print(i, action_taken)
		img, meas, _, terminated = doom_simulator.step(action_taken)
		
		if (terminated):
			agent.signal_episode_end()
		else:
			agent.observe(Observation(img, meas), action_taken)
		i += 1

		print(meas)
	doom_simulator.close_game()




run()