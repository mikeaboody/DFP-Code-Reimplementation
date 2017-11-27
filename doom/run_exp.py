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

def train(num_iterations):
	doom_simulator = create_basic_simulator()
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
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
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
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
		img, meas, _, terminated = doom_simulator.step(action_taken)
		i += 1
	doom_simulator.close_game()

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



def train_and_test():
	try:
		shutil.rmtree(os.path.dirname(log_agent_param['test_data_file']))
	except OSError:
		pass
	os.makedirs(os.path.dirname(log_agent_param['test_data_file']))
	num_episode_test = log_agent_param['testing_num_episodes']
	num_training_steps = log_agent_param['training_num_steps']
	freq = log_agent_param['test_eval_freq']

	doom_simulator = create_basic_simulator(num_simulators)
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
	imgs = [None] * num_simulators
	meas = [None] * num_simulators
	terminated = [None] * num_simulators
	i = 0
	while i < num_training_steps:
		if i == 0:
			actions_taken_one_hot = [agent.act(training=True) for _ in range(num_simulators)]
		else:
			observations = [Observation(imgs[simul_i], meas[simul_i]) for simul_i in range(num_simulators)]
			actions_taken_one_hot = [agent.act(obs, training=True) for obs in observations]
		actions = [action_from_one_hot(action_one_hot) for action_one_hot in actions_taken_one_hot]
		imgs, meas, _, terminated = doom_simulator.step(actions)

		for simul_i in range(num_simulators):
			if i % freq == 0:
				#time to test the agent on real episodes
				test_data = run_test(num_episode_test, goal, i, agent)
				with open(log_agent_param['test_data_file'],'a') as ep_f:
					test_writer = csv.writer(ep_f)
					test_writer.writerow(test_data)
			if (terminated[simul_i]):
				agent.signal_episode_end(simul_i)
			else:
				agent.observe(Observation(imgs[simul_i], meas[simul_i]), actions_taken_one_hot[simul_i], simul_i)
			i += 1

	doom_simulator.close_game()

train_and_test()