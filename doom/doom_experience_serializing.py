import sys
sys.path.append("..")
import csv
import os
import shutil
import random
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
from experience_creator import ExperienceCreator
from experience_serialization import ExperienceSerializer

def serialize_experiences_from_dfp_agent(folder, chunk_size, num_steps):
	log_freq = 10000
	doom_simulator = create_basic_simulator(num_simulators)
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
	imgs = [None] * num_simulators
	meas = [None] * num_simulators
	terminated = [None] * num_simulators
	exp_creators = [ExperienceCreator(goal, agent_params["temp_offsets"]) for simul_i in range(num_simulators)]
	exp_serializer = ExperienceSerializer(folder, chunk_size)
	i = 0
	while i < num_steps:
		if i == 0:
			actions_taken_one_hot = [agent.act(training=False, goal=goal) for _ in range(num_simulators)]
		else:
			observations = [Observation(imgs[simul_i], meas[simul_i]) for simul_i in range(num_simulators)]
			actions_taken_one_hot = [agent.act(obs, training=False, goal=goal) for obs in observations]
		actions = [action_from_one_hot(action_one_hot) for action_one_hot in actions_taken_one_hot]
		imgs, meas, _, terminated = doom_simulator.step(actions)

		for simul_i in range(num_simulators):
			exp_creator = exp_creators[simul_i]
			if i % log_freq == 0:
				#log it
				with open("dfp_serialization_progress.txt", "a") as myfile:
					myfile.write(str(i) + "\n")
			if (terminated[simul_i]):
				exp_creator.reset()
			else:
				exp = exp_creator.add_and_get_experience(Observation(imgs[simul_i], meas[simul_i]), actions_taken_one_hot[simul_i])
				if exp is not None:
					exp_serializer.serialize_experience(exp)
			i += 1
	exp_serializer.flush()
	doom_simulator.close_game()

def serialize_experiences_from_random_agent(folder, chunk_size, num_steps):
	log_freq = 10000
	doom_simulator = create_basic_simulator(num_simulators)
	goal = np.array([0,0,0,0.5,.5,1])
	possible_actions = enumerate_action_one_hots(3)
	imgs = [None] * num_simulators
	meas = [None] * num_simulators
	terminated = [None] * num_simulators
	exp_creators = [ExperienceCreator(goal, agent_params["temp_offsets"]) for simul_i in range(num_simulators)]
	exp_serializer = ExperienceSerializer(folder, chunk_size)
	i = 0
	while i < num_steps:
		actions_taken_one_hot = [random.choice(possible_actions) for _ in range(num_simulators)]
		actions = [action_from_one_hot(action_one_hot) for action_one_hot in actions_taken_one_hot]
		imgs, meas, _, terminated = doom_simulator.step(actions)

		for simul_i in range(num_simulators):
			exp_creator = exp_creators[simul_i]
			if i % log_freq == 0:
				#log it
				with open("random_serialization_progress.txt", "a") as myfile:
					myfile.write(str(i) + "\n")
			if (terminated[simul_i]):
				exp_creator.reset()
			else:
				exp = exp_creator.add_and_get_experience(Observation(imgs[simul_i], meas[simul_i]), actions_taken_one_hot[simul_i])
				if exp is not None:
					exp_serializer.serialize_experience(exp)
			i += 1
	exp_serializer.flush()
	doom_simulator.close_game()
serialize_experiences_from_dfp_agent("dfp_exp", 250, 5000)
# serialize_experiences_from_dfp_agent("dfp_exp", 25000, 5000000)
# serialize_experiences_from_random_agent("random_exp", 25000, 5000000)