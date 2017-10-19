from network import Network
import numpy as np
import pandas as pd
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, concatenate, Add
from keras.layers.normalization import BatchNormalization
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
        self.batch_size = 64
        self.perception_shape = (84, 84, 1)
        self.measurements_shape = (3, 1)
        self.goals_shape = (18, 1)

    def is_network_defined(self):
        return self.model != None

    def set_perception_shape(shape):
        self.perception_shape = shape

    def set_measurement_shape(shape):
        self.measurements_shape = shape

    def set_goals_shape(shape):
        self.goals_shape = shape

    def set_batch_size(size):
        self.batch_size = 64

    # override
    def build_network(self):
        # perception layer
        perception_input = Input(shape=self.perception_shape)
        perception_conv = Convolution2D(32, 8, strides=4, input_shape=self.perception_shape)(perception_input)
        perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
        perception_conv = Convolution2D(64, 4, strides=2)(perception_input)
        perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
        perception_conv = Convolution2D(64, 3, strides=1)(perception_input)
        perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
        perception_flattened = Flatten()(perception_conv)
        perception_fc = Dense(512, activation="linear")(perception_flattened)
        # measurement layer
        measurement_input = Input(shape=self.measurements_shape)
        measurement_flatten = Flatten()(measurement_input)
        measurements_fc = Dense(128)(measurement_flatten)
        measurements_fc = LeakyReLU(alpha=0.2)(measurements_fc)
        measurements_fc = Dense(128)(measurements_fc)
        measurements_fc = LeakyReLU(alpha=0.2)(measurements_fc)
        measurements_fc = Dense(128, activation="linear")(measurements_fc)
        # goals layer
        goal_input = Input(shape=self.goals_shape)
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
        action = BatchNormalization()(action)
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
        # also need to double check how exps is structured not sure rn
        assert self.batch_size == exps and self.model != None
        x_train = [[], [], []]
        y_train = []
        for i in range(0, self.batch_size):
            experience = exps[i]
            s = experience.sens()
            m = experience.meas()
            g = experience.goal()
            label = experience.label()
            x_train[0].append(s)
            x_train[1].append(m)
            x_train[2].append(g)
            y_train.append(label)
        x_train[0] = np.array(x_train[0]).reshape((self.batch_size,
                                                self.perception_shape[0],
                                                self.perception_shape[1],
                                                self.perception_shape[2]))
        x_train[1] = np.array(x_train[1]).reshape((self.batch_size,
                                                self.measurements_shape[0],
                                                self.measurements_shape[1]))
        x_train[2] = np.array(x_train[2]).reshape((self.batch_size,
                                                self.goals_shape[0],
                                                self.goals_shape[1]))
        # y train is tensor of batch size over samples (which are actions*goals length vectors)
        y_train = np.array(y_train).reshape((self.batch_size,
                                            self.goals_shape[0]*self.num_actions,
                                            self.goals_shape[1]))
        self.model.train_on_batch(x_train, y_train)

    def predict(self, obs, goal):
        """
        obs is of the form <s_t, m_t>, where s_t is raw sensory input and
        m_t is a set of measurements
        """
        # assuming that obs is a vector of vectors that looks like
        # [[sensory_input_0, sensory_input_1, ...], [measurement_0, measurement_1, ...]]
        # and goal is a vector that looks like
        # [goal_component_0, goal_component_1, ...]
        prediction_t_a = self.model.predict(obs, verbose=1)
        return prediction_t_a

    def choose_action(self, prediction_t_a, actions):
        # need to implement actions... a one-hot vector?
        action_and_action_values = [(action, np.dot(goal, prediction_t_a)) 
                                        for action in actions]
        return max(action_and_action_values, key=lambda x:x[1])


bn = basicNetwork(3)
bn.build_network()




































