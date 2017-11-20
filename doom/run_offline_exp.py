import sys
sys.path.append("..")
import csv
import os
import shutil
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
from doom_config import num_simulators

def run_test(num_episodes, goal, curr_training_iter, agent):
	"""Returns a size 3 list of the metrics we want: [current # of training iterations for agent,
		avg of avg health over episodes, avg terminal health over episodes].
	"""
	testing_doom_simulator = create_basic_simulator()
	img = None
	meas = None
	terminated = False
	curr_episode = 0
	curr_episode_step = 0
	episode_healths = []
	#metrics
	avg_healths = []
	terminal_healths = []


	while curr_episode < num_episodes:
		if curr_episode_step == 0:
			action_taken_one_hot = agent.act(goal=goal)
		else:
			action_taken_one_hot = agent.act(Observation(img, meas), goal=goal)
		action_taken = action_from_one_hot(action_taken_one_hot)
		img, meas, _, terminated = testing_doom_simulator.step(action_taken)

		curr_episode_step += 1
		if terminated:
			#collect metrics
			avg_health = np.mean(np.array(episode_healths))
			terminal_health = episode_healths[-1]
			avg_healths.append(avg_health)
			terminal_healths.append(terminal_health)

			curr_episode += 1
			curr_episode_step = 0
			episode_healths = []
		else:
			episode_healths.append(meas[0])
	testing_doom_simulator.close_game()

	return [curr_training_iter, np.mean(np.array(avg_healths)), np.mean(np.array(terminal_healths))]

def train_and_test_offline(folder, samples_per_epoch):
	num_episode_test = log_agent_param['testing_num_episodes']

	doom_simulator = create_basic_simulator(num_simulators)
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
	agent.offline_training(folder, samples_per_epoch, 1)
	run_test(num_episode_test, goal, samples_per_epoch, agent)
	doom_simulator.close_game()

train_and_test_offline("dfp_exp", 100)
