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
	