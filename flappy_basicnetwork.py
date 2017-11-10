from network import Network
from abstraction import *
from util import *
import numpy as np
import pandas as pd
import math
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, concatenate, Add, Multiply
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal, Constant
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import os.path

#     def custom_objective(y_true, y_pred):
#     print(y_pred.eval(session=tf.Session()))
#     return -K.dot(y_true,K.transpose(y_pred))


def pad_label(label, action, num_actions):
    # make label [0, 0, 0, label, 0, 0, 0] (0 for all other actions)
    # assume action is 0-indexed
    # this way dot product  0 out everything that's irrelevant
    goal = label[:len(label)].flatten()
    an = action_number(action)
    action_index = int(an*len(goal))
    new_label = np.zeros(len(goal)*num_actions)
    mask = np.zeros(len(goal)*num_actions)
    new_label[action_index:action_index+len(goal)] = goal
    mask[action_index:action_index+len(goal)] = [1]*len(goal)
    return new_label, mask

def msra_stddev(x, k_h, k_w): 
    return 1/math.sqrt(0.5*k_w*k_h*x._keras_shape[-1])

class basicNetwork(Network):
    """
    A wrapper class for our neural network. This will make it easier
    to implement the network in different frameworks. All a network needs
    to do is implement these methods.
    """
    def __init__(self, network_params, backing_file=None, load_from_backing_file=False, optimizer="Adam", k_h=[8, 4, 3], k_w=[8, 4, 3], decay_steps=250000):
        super(basicNetwork, self).__init__(network_params, backing_file, load_from_backing_file)
        self.num_actions = network_params["num_actions"]
        self.preprocess_img = network_params["preprocess_img"]
        self.preprocess_meas = network_params["preprocess_meas"]
        self.preprocess_label = network_params["preprocess_label"]
        self.postprocess_label = network_params["postprocess_label"]
        self.optimizer = optimizer
        self.batch_size = 64
        self.perception_shape = (288, 512, 3)
        self.measurements_shape = (1,)
        self.goals_shape = (6, 1)
        self.learning_rate = 1e-04
        self.decay_rate = 0.3
        self.msra_coef = 0.9
        self.k_h = k_h
        self.k_w = k_w
        self.action_mask_shape = (1*6*self.num_actions,)
        self.backing_file = backing_file
        self.load_from_backing_file = load_from_backing_file
        self.num_updates = 0
        self.decay_steps = decay_steps
        self.build_network()


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
        if self.load_from_backing_file and os.path.isfile(self.backing_file) :
            self.model = load_model(self.backing_file)
            return
        # perception layer
        perception_input = Input(shape=self.perception_shape)
        perception_conv = Convolution2D(32, 8, strides=4,
                                        input_shape=self.perception_shape,
                                        padding="same",
                                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(perception_input, self.k_h[0], self.k_w[0])),
                                        bias_initializer=Constant(value=0))(perception_input)
        perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
        perception_conv = Convolution2D(64, 4, strides=2,
                                        padding="same",
                                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(perception_conv, self.k_h[1], self.k_w[1])),
                                        bias_initializer=Constant(value=0))(perception_conv)
        perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
        perception_conv = Convolution2D(64, 3, strides=1,
                                        padding="same",
                                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(perception_conv, self.k_h[2], self.k_w[2])),
                                        bias_initializer=Constant(value=0))(perception_conv)
        perception_conv = LeakyReLU(alpha=0.2)(perception_conv)
        perception_flattened = Flatten()(perception_conv)
        perception_fc = Dense(512, activation="linear",
                                   kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(perception_flattened, 1, 1)),
                                   bias_initializer=Constant(value=0))(perception_flattened)
        # measurement layer
        measurement_input = Input(shape=self.measurements_shape)
#        measurement_flatten = Flatten()(measurement_input)
        measurement_flatten = measurement_input
        measurements_fc = Dense(128,
                                kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(measurement_flatten, 1, 1)),
                                bias_initializer=Constant(value=0))(measurement_flatten)
        measurements_fc = LeakyReLU(alpha=0.2)(measurements_fc)
        measurements_fc = Dense(128,
                                kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(measurements_fc, 1, 1)),
                                bias_initializer=Constant(value=0))(measurements_fc)
        measurements_fc = LeakyReLU(alpha=0.2)(measurements_fc)
        measurements_fc = Dense(128, activation="linear",
                                kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(measurements_fc, 1, 1)),
                                bias_initializer=Constant(value=0))(measurements_fc)
        # goals layer
        goal_input = Input(shape=self.goals_shape)
        goals_flatten = Flatten()(goal_input)
        goals_fc = Dense(128,
                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(goals_flatten, 1, 1)),
                        bias_initializer=Constant(value=0))(goals_flatten)
        goals_fc = LeakyReLU(alpha=0.2)(goals_fc)
        goals_fc = Dense(128,
                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(goals_fc, 1, 1)),
                        bias_initializer=Constant(value=0))(goals_fc)
        goals_fc = LeakyReLU(alpha=0.2)(goals_fc)
        goals_fc = Dense(128, activation="linear",
                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(goals_fc, 1, 1)),
                        bias_initializer=Constant(value=0))(goals_fc)
        # merge together
        mrg_j= concatenate([perception_fc,measurements_fc,goals_fc])
        #expectations
        expectation = Dense(512,
                            kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(mrg_j, 1, 1)),
                            bias_initializer=Constant(value=0))(mrg_j)
        expectation = LeakyReLU(alpha=0.2)(expectation)
        expectation = Dense(1*6, activation="linear",
                            kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(expectation, 1, 1)),
                            bias_initializer=Constant(value=0))(expectation)
        #action
        action = Dense(512,
                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(mrg_j, 1, 1)),
                        bias_initializer=Constant(value=0))(mrg_j)
        action = LeakyReLU(alpha=0.2)(action)
        action = Dense(1*6*self.num_actions, activation="linear",
                        kernel_initializer=TruncatedNormal(stddev=self.msra_coef*msra_stddev(action, 1, 1)),
                        bias_initializer=Constant(value=0))(action)
        action = BatchNormalization()(action)
        #concat expectations for number of actions there are
        expectation = concatenate([expectation]*self.num_actions)
        # sum expectations with action
        expectation_action_sum = Add()([action, expectation])
        action_mask_layer = Input(shape=self.action_mask_shape)
        expectation_action_sum = Multiply()([action_mask_layer, expectation_action_sum])
        self.model = Model(inputs=[perception_input, measurement_input, goal_input, action_mask_layer], outputs=expectation_action_sum)
        opt = None
        # worst case we can do learning rate step size of 250000
        if self.optimizer == "Adam":
            opt = Adam(lr=self.learning_rate, beta_1=0.95, beta_2=0.999, epsilon=1e-04)
        self.model.compile(loss='mean_squared_error', optimizer=opt)

    def exponentially_decay(self, global_step):
        decayed_learning_rate = self.learning_rate * (math.pow(self.decay_rate, global_step / self.decay_steps))
        return decayed_learning_rate


    def update_weights(self, exps):
        # please make sure that exps is a batch of the proper size (in basic it's 64)
        # also need to double check how exps is structured not sure rn
        global_step = self.num_updates*self.batch_size
        new_lr = self.exponentially_decay(global_step)
        self.model.optimizer.lr.assign(new_lr)
        p100irint(new_lr)
        assert self.batch_size == len(exps) and self.model != None
        x_train = [[], [], [], []]
        y_train = []
        for i in range(0, self.batch_size):
            experience = exps[i]
            s = self.preprocess_img(experience.sens())
            m = self.preprocess_meas(experience.meas())
            g = experience.goal()
            label, mask = pad_label(experience.label(), experience.a, self.num_actions)
            label = self.preprocess_label(label)
            x_train[0].append(s)
            x_train[1].append(m)
            x_train[2].append(g)
            x_train[3].append(mask)
            y_train.append(label)
        x_train[0] = np.array(x_train[0]).reshape((self.batch_size,
                                                self.perception_shape[0],
                                                self.perception_shape[1],
                                                self.perception_shape[2]))
        x_train[1] = np.array(x_train[1]).reshape((self.batch_size,
                                                self.measurements_shape[0]))
        x_train[2] = np.array(x_train[2]).reshape((self.batch_size,
                                                self.goals_shape[0],
                                                self.goals_shape[1]))
        x_train[3] = np.array(x_train[3]).reshape((self.batch_size,
                                                self.action_mask_shape[0]))
        # y train is tensor of batch size over samples (which are actions*goals length vectors)
        y_train = np.array(y_train).reshape((self.batch_size,
                                            self.goals_shape[0]*self.num_actions))
        res = self.model.train_on_batch(x_train, y_train)
        if self.num_updates % 100 == 0:
            with open("results.txt", "a") as myfile:
                myfile.write(str(res) + "\n")
        self.num_updates += 1

    def predict(self, obs, goal):
        """
        obs is of the form <s_t, m_t>, where s_t is raw sensory input and
        m_t is a set of measurements
        """
        # assuming that obs is a vector of vectors that looks like
        # [[sensory_input_0, sensory_input_1, ...], [measurement_0, measurement_1, ...]]
        # and goal is a vector that looks like
        # [goal_component_0, goal_component_1, ...]
        num_s = len(obs.sens[0])
        #need to put these into np arrays to make shape (1, original shape)
        sens = np.array([self.preprocess_img(obs.sens)])
        meas = np.array([np.expand_dims(self.preprocess_meas(obs.meas), 1)])
        goal = np.array([np.expand_dims(goal, 1)])
        prediction_t_a = self.model.predict([sens, meas, goal, np.ones((num_s, 1*6*self.num_actions))])[0]
        prediction_t_a = self.postprocess_label(prediction_t_a)
        return prediction_t_a

    def save_network(self):
        self.model.save(self.backing_file)

    def choose_action(self, prediction_t_a, actions):
        # need to implement actions... a one-hot vector?
        action_and_action_values = [(action, np.dot(goal, prediction_t_a)) 
                                        for action in actions]
        return max(custom_objective, key=lambda x:x[1])


def basicNetwork_builder(network_params):
    return lambda: basicNetwork(network_params, backing_file="bn.h5", load_from_backing_file= True)

# bn = basicNetwork_builder({"num_actions": 8,
#                             "preprocess_img": np.array([]),
#                             "preprocess_meas": np.array([]),
#                             "preprocess_label": np.array([]),
#                             "postprocess_label": np.array([])})()

def create_obs_goal_pair(bn):
    sens = np.random.random_sample(size=(84,84,1))
    meas = np.random.random_sample(size=(1,))
    goal = np.random.random_sample(size=(6,))
    obs = Observation(sens, meas)
    return obs, goal

def create_experience(bn):
    sens = np.random.random_sample(size=(84,84,1))
    meas = np.random.random_sample(size=(1,))
    goal = np.random.random_sample(size=(6,))
    # last value indicate index of action
    label = np.random.random_sample(size=(6,))
    obs = Observation(sens, meas)
    exp = Experience(obs, action_to_one_hot([0,1,0]), goal, label)
    return exp

# experiences = [create_experience(bn) for _ in range(64)]
# # # import pdb; pdb.set_trace()
# bn.update_weights(experiences)
# bn.update_weights(experiences)
# bn.update_weights(experiences)
# bn.update_weights(experiences)
# bn.update_weights(experiences)
# bn.update_weights(experiences)


# a = create_obs_goal_pair(bn)
# bn.predict(a[0], a[1])


