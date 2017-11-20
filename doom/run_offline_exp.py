import sys
sys.path.append("..")
import csv
import os
import shutil
from ai import Agent
from basic_doom_simulator import create_basic_simulator
from offlinebasicnetwork import offlineBasicNetwork_builder
from network import blank_network_builder
from abstraction import *
from util import *
#agent_params for the agent
from doom_config import agent_params
from doom_config import network_params
from log_config import log_agent_param
from doom_config import num_simulators
from experience_serialization import ExperienceDeserializer

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

def train_and_test_offline(exp_folder):
	try:
		shutil.rmtree(os.path.dirname(log_agent_param['test_offline_data_file']))
	except OSError:
		pass
	os.makedirs(os.path.dirname(log_agent_param['test_offline_data_file']))
	num_episode_test = log_agent_param['testing_num_episodes']
	num_training_steps = log_agent_param['training_num_steps']
	freq = log_agent_param['test_eval_freq']
	exp_des = ExperienceDeserializer(exp_folder).deserialized_generator()
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, offlineBasicNetwork_builder(network_params))
	for i in range(num_training_steps):
		if i % freq == 0:
			test_data = run_test(num_episode_test, goal, i, agent))
			with open(log_agent_param['test_offline_data_file'],'a') as ep_f:
					test_writer = csv.writer(ep_f)
					test_writer.writerow(test_data)
		agent.offline_training(exp_des)

train_and_test_offline("dfp_exp")
