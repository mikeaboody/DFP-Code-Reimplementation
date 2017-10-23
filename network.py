import numpy as np

class Network(object):
	"""
	A wrapper class for our neural network. This will make it easier
	to implement the network in different frameworks. All a network needs
	to do is implement these methods.
	"""
	def __init__(self, num_actions, backing_file_name, load_from_backing_file):
		self.num_actions = num_actions
		self.model = None
		self.backing_file_name = backing_file_name
		self.load_from_backing_file = load_from_backing_file

	def build_network(self, backing_file_name, load_from_backing_file):
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

	def save_network(self):
		pass
	
class BlankNetwork(object):
	"""
	Completely blank network for testing purposes
	"""
	def __init__(self, num_actions, backing_file_name, load_from_backing_file):
		self.num_actions = num_actions
		self.model = None
		self.backing_file_name = backing_file_name
		self.build_network(backing_file_name)

	def build_network(self, backing_file_name):
		"""
			Model should be defined here
		"""
		print("Building Network")

	def update_weights(self, exps):
		print("Updating weights for {} experiences.".format(len(exps)))
		print("Update stats: sens shape: {}, meas shape {}, goal shape: {}, label shape: {}" \
			.format(exps[0].sens().shape, exps[0].meas().shape, exps[0].goal().shape, exps[0].label().shape))
		print("Action vector length: {}".format(len(exps[0].a)))

	def loss_func(self, exps):
		pass

	def predict(self, obs, goal):
		print("Predicting. sens shape: {}, meas shape: {}, goal shape: {}".format(obs.sens.shape, obs.meas.shape, goal.shape))
		return np.random.random_sample(self.num_actions * len(goal))

	def save_network(self):
		print("Saving network")

def blank_network_builder(num_actions):
	return lambda: BlankNetwork(num_actions, "", False)
