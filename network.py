import numpy as np

class Network(object):
	"""
	A wrapper class for our neural network. This will make it easier
	to implement the network in different frameworks. All a network needs
	to do is implement these methods.
	"""
	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.model = None

	def build_network(self):
		"""
			Model should be defined here
		"""
		pass

	def update_weights(self, exps):
		pass

	def loss_func(self, exps):
		pass

	def predict(self, obs, goal):
		pass
	
class BlankNetwork(object):
	"""
	Completely blank network for testing purposes
	"""
	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.model = None
		self.build_network()

	def build_network(self):
		"""
			Model should be defined here
		"""
		print("Building Network")

	def update_weights(self, exps):
		print("Updating weights for {} experiences.".format(len(exps)))
		print("Update stats: sens shape: {}, meas shape {}, goal shape: {}, label shape: {}" \
			.format(exps[0].sens().shape, exps[0].meas().shape, exps[0].goal().shape, exps[0].label().shape))

	def loss_func(self, exps):
		pass

	def predict(self, obs, goal):
		print("Predicting. sens shape: {}, meas shape: {}, goal shape: {}".format(obs.sens.shape, obs.meas.shape, goal.shape))
		return np.random.random_sample(self.num_actions * len(goal))

def blank_network_builder(num_actions):
	return lambda: BlankNetwork(num_actions)
