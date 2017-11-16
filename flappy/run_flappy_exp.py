import sys
sys.path.append("..")
import csv
import os
import shutil
from ai import Agent
from basic_flappy_simulator import create_basic_simulator
from flappy_basicnetwork import basicNetwork_builder
from network import blank_network_builder
from abstraction import *
from util import *
#agent_params for the agent
from doom.log_config import log_agent_param
from flappy.flappy_config import agent_params
from flappy.flappy_config import network_params
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

def run_basic():
    possible_actions = [[1,0], [0,1]]
    network_params["num_actions"] = 2
    flappy_simulator = create_basic_simulator()
    agent = Agent(agent_params, possible_actions, blank_network_builder(network_params))
    agent.eps = 0
    img = None
    meas = None
    terminated = None

    i = 0
    while i < 1000:
        if i == 0:
            action_taken = agent.act(training=True)
        else:
            action_taken = agent.act(Observation(img, meas), training=True)
        print(i, action_taken)
        img, meas, _, terminated = flappy_simulator.step(action_taken)
        
        if (terminated):
            agent.signal_episode_end()
        else:
            agent.observe(Observation(img, meas), action_taken)
        i += 1

        print(meas)
    flappy_simulator.close_game()

def log_measurements(episode_count, episode_scores, terminated, meas, i, train):
    # assume there's only 1 action for now
    if train:
        freq = log_agent_param['train_eval_freq']
    else:
        freq = log_agent_param['test_eval_freq']
    if i % freq == 0 and not terminated:
        logging.debug("Episode {0} iter {1} Score: {3}".format(episode_count, i / episode_count, meas))
    if terminated:
        logging.debug("*****Episode {0} is ending*****".format(episode_count))
        logging.debug("*****Episode {0} has mean score {1} *****".format(episode_count, np.mean(episode_scores)))
        episode_scores = []
        episode_count += 1
    episode_scores.append(meas)
    return episode_count, episode_scores


def train(num_iterations):
    flappy_simulator = create_basic_simulator()
    possible_actions = enumerate_action_one_hots(1)
    agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
    img = None
    meas = None
    terminated = None
    i = 0
    action_taken_one_hot = agent.act(training=True)
    action = action_from_one_hot(action_taken_one_hot)
    img, meas, _, terminated = flappy_simulator.step(action)
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img,(80,80))
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))

    while i < num_iterations:

        action_taken_one_hot = agent.act(Observation(img, meas), training=True)
        action = action_from_one_hot(action_taken_one_hot)

        img, meas, _, terminated = flappy_simulator.step(action)

        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img,(80,80))
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))

        if (terminated):
            agent.signal_episode_end()
        else:
            agent.observe(Observation(img, meas), action_taken_one_hot)
        i += 1

    flappy_simulator.close_game()

def test(num_iterations):
    flappy_simulator = create_basic_simulator()
    goal = np.array([0,0,0,0.5,.5,1])
    possible_actions = enumerate_action_one_hots(1)
    agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
    img = None
    meas = None
    terminated = None
    i = 0
    while i < num_iterations:
        if i == 0:
            action_taken_one_hot = agent.act(goal=goal)
        else:
            action_taken_one_hot = agent.act(Observation(img, meas), goal=goal)
        action_taken = action_from_one_hot(action_taken_one_hot)
        img, meas, _, terminated = flappy_simulator.step(action_taken)
        i += 1
    flappy_simulator.close_game()

def run_test(num_episodes, goal, curr_training_iter, agent):
    """Returns a size 3 list of the metrics we want: [current # of training iterations for agent,
        avg of avg score over episodes, avg terminal score over episodes].
    """
    print("Testing...")
    testing_flappy_simulator = create_basic_simulator()
    img = None
    meas = None
    terminated = False
    curr_episode = 0
    curr_episode_step = 0
    episode_scores = []
    #metrics
    avg_scores = []
    terminal_scores = []

    action_taken_one_hot = agent.act(goal=goal)
    action = action_from_one_hot(action_taken_one_hot)
    img, meas, _, terminated = testing_flappy_simulator.step(action)
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img,(80,80,1))
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))

    while curr_episode < num_episodes:

        action_taken_one_hot = agent.act(Observation(img, meas), goal=goal)
        action = action_from_one_hot(action_taken_one_hot)

        img, meas, _, terminated = testing_flappy_simulator.step(action)

        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img,(80,80,1))
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))

        curr_episode_step += 1
        if terminated:
            #collect metrics
            avg_score = np.mean(np.array(episode_scores))
            terminal_score = episode_scores[-1]
            avg_scores.append(avg_score)
            terminal_scores.append(terminal_score)

            curr_episode += 1
            curr_episode_step = 0
            episode_scores = []
            print("Average score: {}".format(avg_score))
        else:
            episode_scores.append(meas)
    testing_flappy_simulator.close_game()

    print("Testing finished")
    return [curr_training_iter, np.mean(np.array(avg_scores)), np.mean(np.array(terminal_scores))]



def train_and_test():
    try:
        shutil.rmtree(os.path.dirname(log_agent_param['test_data_file']))
    except OSError:
        pass
    os.makedirs(os.path.dirname(log_agent_param['test_data_file']))
    num_episode_test = log_agent_param['testing_num_episodes']
    num_training_steps = log_agent_param['training_num_steps']
    freq = log_agent_param['test_eval_freq']
    #freq = 100

    flappy_simulator = create_basic_simulator()
    goal = np.array([1,1,1,1,1,1])
    possible_actions = enumerate_action_one_hots(1)
    agent = Agent(agent_params, possible_actions, basicNetwork_builder(network_params))
    img = None
    meas = None
    terminated = None
    i = 0
    action_taken_one_hot = agent.act(training=True)
    action = action_from_one_hot(action_taken_one_hot)
    img, meas, _, terminated = flappy_simulator.step(action)
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img,(80,80,1))
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))

    while i < num_training_steps:

        action_taken_one_hot = agent.act(Observation(img, meas), training=True)
        action = action_from_one_hot(action_taken_one_hot)

        img, meas, _, terminated = flappy_simulator.step(action)

        if meas > 0:
            print("Meas is {} at {}".format(meas, i))
        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img,(80,80,1))
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))

        if i % freq == 0 and i != 0:
            #time to test the agent on real episodes
            test_data = run_test(num_episode_test, goal, i, agent)
            with open(log_agent_param['test_data_file'],'a') as ep_f:
                test_writer = csv.writer(ep_f)
                test_writer.writerow(test_data)

        if (terminated):
            agent.signal_episode_end()
        else:
            agent.observe(Observation(img, meas), action_taken_one_hot)
        i += 1

    flappy_simulator.close_game()

train_and_test()
