from abstraction import *
import numpy as np
from util import create_label
from bounded_cache import BoundedCache

class ExperienceCreator(object):
	"""Creates experiences"""
	def __init__(self, experience_goal, temp_offsets):
		self.goal = experience_goal
		self.temp_offsets = temp_offsets
		self.recent_obs_act_pairs = BoundedCache(max(self.temp_offsets) + 1)
	def add_and_get_experience(self, obs, action):
		"""Adds the current observation/action pair and returns a created experience if 
			there is one"""
		self.recent_obs_act_pairs.add((obs, action))
		
		if not self.recent_obs_act_pairs.at_capacity():
			return None

		#create an experience
		exp_obs, exp_act = self.recent_obs_act_pairs.index_from_back(0)[0]
		exp_label = create_label(exp_obs, self.temp_offsets, self.recent_obs_act_pairs)
		return Experience(exp_obs, exp_act, self.goal, exp_label)
	def reset(self):
		"""Resets to when initially created"""
		self.recent_obs_act_pairs = BoundedCache(max(self.temp_offsets) + 1)
