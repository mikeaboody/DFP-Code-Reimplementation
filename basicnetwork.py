from network import Network
import numpy as np
import pandas as pd
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, concatenate, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


class basicNetwork(Network):
	"""
	A wrapper class for our neural network. This will make it easier
	to implement the network in different frameworks. All a network needs
	to do is implement these methods.
	"""
	def __init__(self, num_actions, optimizer="Adam"):
		super(basicNetwork, self).__init__(num_actions)
		# put arguments here
		self.optimizer = optimizer


	def is_network_defined(self):
		return self.model != None

	# override
	def build_network(self, perception_shape, measurements_shape, goals_shape):
		# perception layer
		perception_input = Input(shape=perception_shape)
		perception_conv = Convolution2D(32, 8, strides=4, input_shape=perception_shape)(perception_input)
		perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
		perception_conv = Convolution2D(64, 4, strides=2)(perception_input)
		perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
		perception_conv = Convolution2D(64, 3, strides=1)(perception_input)
		perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
		perception_flattened = Flatten()(perception_conv)
		perception_fc = Dense(512, activation="linear")(perception_flattened)
		# measurement layer
		measurement_input = Input(shape=measurements_shape)
		measurement_flatten = Flatten()(measurement_input)
		measurements_fc = Dense(128)(measurement_flatten)
		measurements_fc = LeakyReLU(alpha=0.2)(measurements_fc)
		measurements_fc = Dense(128)(measurements_fc)
		measurements_fc = LeakyReLU(alpha=0.2)(measurements_fc)
		measurements_fc = Dense(128, activation="linear")(measurements_fc)
		# goals layer
		goal_input = Input(shape=goals_shape)
		goals_flatten = Flatten()(goal_input)
		goals_fc = Dense(128)(goals_flatten)
		goals_fc = LeakyReLU(alpha=0.2)(goals_fc)
		goals_fc = Dense(128)(goals_fc)
		goals_fc = LeakyReLU(alpha=0.2)(goals_fc)
		goals_fc = Dense(128, activation="linear")(goals_fc)
		# merge together
		mrg_j= concatenate([perception_fc,measurements_fc,goals_fc])
		#expectations
		expectation = Dense(512)(mrg_j)
		expectation = LeakyReLU(alpha=0.2)(expectation)
		expectation = Dense(3*6, activation="linear")(expectation)
		#action
		action = Dense(512)(mrg_j)
		action = LeakyReLU(alpha=0.2)(action)
		action = Dense(3*6*self.num_actions, activation="linear")(action)
		#concat expectations for number of actions there are
		expectation = concatenate([expectation]*self.num_actions)
		# sum expectations with action
		expectation_action_sum = Add()([action, expectation])
		self.model = Model(inputs=[perception_input, measurement_input, goal_input], outputs=expectation_action_sum)
		opt = None
		if self.optimizer == "Adam":
			opt = Adam(lr=1e-04, beta_1=0.95, beta_2=0.999, epsilon=1e-04, decay=0.3)
		self.model.compile(loss='mean_squared_error', optimizer=opt)

	def update_weights(self, exps):
		# please make sure that exps is a batch of the proper size (in basic it's 64)
		# note must change to have it NOT reset learning rate as it currently resets it
		# also need to double check how exps is structured not sure rn
		x_train = exps[:len(exps)-1]
		y_train = exps[len(exps)-1:]
		self.model.train_on_batch(x_train, y_train)

	def loss_func(self, exps):
		pass

	def predict(self, obs, goal):
		pass

bn = basicNetwork(3)
bn.build_network((84,84,1), (3,1), (18,1))