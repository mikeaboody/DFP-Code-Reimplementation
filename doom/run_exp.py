import sys
sys.path.append("..")
from ai import Agent
from basic_doom_simulator import create_basic_simulator
from basicnetwork import basicNetwork_builder
from abstraction import *
from util import *

config_path = "../sample_config.json"
eps_func = lambda step: (0.02 + 145000. / (float(step) + 150000.))

def run_basic():
	possible_actions = [[1,0,0], [0,1,0], [0,0,1]]
	doom_simulator = create_basic_simulator()
	agent = Agent(config_path, possible_actions, blank_network_builder(len(possible_actions)), eps_func)
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


def train(num_iterations):
	doom_simulator = create_basic_simulator()
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(config_path, possible_actions, basicNetwork_builder(len(possible_actions)), eps_func)

	img = None
	meas = None
	terminated = None

	i = 0
	while i < num_iterations:
		if i == 0:
			action_taken_one_hot = agent.act(training=True)
		else:
			action_taken_one_hot = agent.act(Observation(img, meas), training=True)
		img, meas, _, terminated = doom_simulator.step(action_from_one_hot(action_taken_one_hot))
		
		if (terminated):
			agent.signal_episode_end()
		else:
			agent.observe(Observation(img, meas), action_taken_one_hot)
		i += 1

	doom_simulator.close_game()

def test(num_iterations):
	doom_simulator = create_basic_simulator()
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(config_path, possible_actions, basicNetwork_builder(len(possible_actions)), eps_func)

	img = None
	meas = None
	terminated = None

	i = 0
	while i < num_iterations:
		if i == 0:
			action_taken_one_hot = agent.act(goal=goal)
		else:
			action_taken_one_hot = agent.act(Observation(img, meas), goal=goal)
		action_taken = action_from_one_hot(action_taken_one_hot)
		print(action_taken)
		img, meas, _, terminated = doom_simulator.step(action_taken)
		i += 1

	doom_simulator.close_game()


test(20000)
