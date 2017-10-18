class Network(object):
	"""
	A wrapper class for our neural network. This will make it easier
	to implement the network in different frameworks. All a network needs
	to do is implement these methods.
	"""
	def __init__(self, num_actions):
		self.num_actions = num_actions

	def build_network(self):
		pass

	def update_weights(self, exps):
		pass

	def loss_func(self, exps):
		pass

	def predict(self, obs, goal):
		pass

        def choose_action(self, actions):
                pass
	
