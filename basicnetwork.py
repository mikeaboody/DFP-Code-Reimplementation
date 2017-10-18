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
	def __init__(self, num_actions):
		super(basicNetwork, self).__init__(num_actions)
		# put arguments here
                self.model = build_network((84, 84, 1), (3, 1), (18, 1))

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
		adam = Adam(lr=1e-04, beta_1=0.95, beta_2=0.999, epsilon=1e-04, decay=0.3)
		self.model.compile(loss='mean_squared_error', optimizer=adam)
                return self.model

	def update_weights(self, exps):
		# note must change to have it NOT reset learning rate as it currently resets it
		# also need to double check how exps is structured not sure rn
		x_train = exps[:len(exps)-1]
		y_train = exps[len(exps)-1:]
		# either just put x_train in and let batch be taken care of here or modify this a bit
		self.model.fit(x_train, y_train, batch_size=64)

	def loss_func(self, exps):
		pass

	def predict(self, obs, goal):
                """
                obs is of the form <s_t, m_t>, where s_t is raw sensory input and
                m_t is a set of measurements
                """
                # assuming that obs is a vector of vectors that looks like
                # [[sensory_input_0, sensory_input_1, ...], [measurement_0, measurement_1, ...]]
                # and goal is a vector that looks like
                # [goal_component_0, goal_component_1, ...]
                prediction_t_a = self.model.predict(obs + goal)
                return self.model.predict

        def choose_action(self, prediction_t_a, actions):
                # need to implement actions... a one-hot vector?
                action_and_action_values = [(action, np.dot(goal, prediction_t_a)) 
                                                for action in actions]
                return max(action_and_action_values, key=lambda x:x[1])


bn = basicNetwork(3)
bn.build_network((84,84,1), (3,1), (18,1))




































