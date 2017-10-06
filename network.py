class Network(object):
	"""
	A wrapper class for our neural network. This will make it easier
	to implement the network in different frameworks. All a network needs
	to do is implement these methods.
	"""
	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.network = self.build_network()

	def build_network():
		pass

	def update_weights(exps):
		pass

	def loss_func(exps):
		pass

	def predict(obs, goal):
		pass
	