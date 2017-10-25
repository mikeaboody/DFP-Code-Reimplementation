import sys
sys.path.append("..")
from ai import Agent
from basic_doom_simulator import create_basic_simulator
from basicnetwork import basicNetwork_builder
from network import blank_network_builder
from abstraction import *
from util import *
#agent_params for the agent
from doom_config import agent_params
from doom_config import network_params
from log_config import log_agent_param
import logging

logging.basicConfig(filename=log_agent_param['log_path'],
                    filemode='a',
                    level=logging.DEBUG)

def run_basic():
	possible_actions = [[1,0,0], [0,1,0], [0,0,1]]
	network_params["num_actions"] = 3
	doom_simulator = create_basic_simulator()
	agent = Agent(agent_params, possible_actions, blank_network_builder(network_params))
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

def log_measurements(episode_count, episode_healths, terminated, meas, i, train):
	# log some shit
	# assume there's only 1 action for now
	if train:
		freq = log_agent_param['train_eval_freq']
	else:
		freq = log_agent_param['test_eval_freq']
	if i % freq == 0 and not terminated:
		logging.debug("Episode {0} iter {1} Health: {3}".format(episode_count, i / episode_count, meas[0]))
	if terminated:
		logging.debug("*****Episode {0} is ending*****".format(episode_count))
		logging.debug("*****Episode {0} has mean health {1} *****".format(episode_count, np.mean(episode_healths)))
		episode_healths = []
		episode_count += 1
	episode_healths.append(meas[0])
	return episode_count, episode_healths


def train(num_iterations):
	if log_agent_param['to_log']:
		logging.debug("**********TRAINING BEGANS**********")
	doom_simulator = create_basic_simulator()
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
	img = None
	meas = None
	terminated = None
	i = 0
	episode_healths = []
	episode_count = 1
	while i < num_iterations:
		if i == 0:
			action_taken_one_hot = agent.act(training=True)
		else:
			action_taken_one_hot = agent.act(Observation(img, meas), training=True)
		img, meas, _, terminated = doom_simulator.step(action_from_one_hot(action_taken_one_hot))
		if log_agent_param['to_log']:
			episode_count, episode_healths = log_measurements(episode_count, episode_healths, terminated, meas, i, True)
		if (terminated):
			agent.signal_episode_end()
		else:
			agent.observe(Observation(img, meas), action_taken_one_hot)
		i += 1

	doom_simulator.close_game()

def test(num_iterations):
	if log_agent_param['to_log']:
		logging.debug("**********TESTING BEGANS**********")
	doom_simulator = create_basic_simulator()
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))

	img = None
	meas = None
	terminated = None
	i = 0
	episode_healths = []
	episode_count = 1
	while i < num_iterations:
		if i == 0:
			action_taken_one_hot = agent.act(goal=goal)
		else:
			action_taken_one_hot = agent.act(Observation(img, meas), goal=goal)
		action_taken = action_from_one_hot(action_taken_one_hot)
		img, meas, _, terminated = doom_simulator.step(action_taken)
		if log_agent_param['to_log']:
			if i % log_agent_param['test_eval_freq'] == 0:
				logging.debug("Episode {0} iter {1} action taken: {0}".format(episode_count, i, action_taken))
		episode_count, episode_healths = log_measurements(episode_count, episode_healths, False, meas, i, False)
		i += 1
	doom_simulator.close_game()

def train_and_test():
	for i in range(0, log_agent_param['num_times_trained'])
		if log_agent_param['to_log']:
			logging.debug("**********Bot Training/Test Iter {0} BEGANS**********".format(i))
		train(log_agent_param['training_num_iter'])
		test(log_agent_param['testing_num_iter'])


test(100)
train_and_test()
