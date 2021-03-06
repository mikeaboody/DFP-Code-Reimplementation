from basicnetwork import basicNetwork_builder

#number of simulators
num_simulators = 8

#stuff the agent will use
agent_params = {}
agent_params["M"] = 20000
agent_params["N"] = 64
agent_params["k"] = 64
agent_params["temp_offsets"] = [1,2,4,8,16,32]
agent_params["g_train"] = [0,0,0,0.5,0.5,1]
agent_params["network_backing_file"] = None
agent_params["network_save_period"] = 10
agent_params["eps_decay_func"] = lambda k: (0.02 + 145000. / (float(k) + 150000.))

#stuff that network will use
network_params = {}
network_params["num_actions"] = 8
network_params["preprocess_img"] = lambda x: x / 255. - 0.5
network_params["preprocess_meas"] = lambda x: x / 100. - 0.5
network_params["preprocess_label"] = lambda x: x / 30.
network_params["postprocess_label"] = lambda x: x * 30.
