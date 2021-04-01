
"""
Author:         Anonymous
Description:
                Controller policy architectures:
                - nn_policy: fully connected, tanh for all activations
                - nn_policy_ppo: architecture based on PPO implementation 
                - nn_policy_relu: no tanh non-linearities
                
                - nn_policy: CNN based for image inputs
"""

import logging
import tensorflow as tf 
# import keras
# from keras import layers as ly
layer = tf.keras.layers


logger = logging.getLogger(__name__)



def nn_policy(state_input, policy_arch, dim_action, **kwargs):
    """ 
        Fully-connected agent policy network 
    """
    with tf.variable_scope('policy_net', reuse=tf.AUTO_REUSE):
        for i, h in enumerate(policy_arch):
            state_input = layer.Dense(h, activation='tanh', # dtype='float64', 
                                      name="fc_{}".format(i))(state_input)
        action_out = layer.Dense(dim_action, 
                                 activation='tanh', # dtype='float64',
                                 name="fc_action_out")(state_input)
    return action_out, state_input


def nn_policy_ppo(state_input, policy_arch, dim_action, **kwargs):
    """ 
        Policy definition taken from: 
        https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/core.py
    """
    with tf.variable_scope('policy_net', reuse=tf.AUTO_REUSE):
        for i, h in enumerate(policy_arch):
            state_input = layer.Dense(h, activation='tanh', # dtype='float64', 
                                      name="fc_{}".format(i))(state_input)
        action_out = layer.Dense(dim_action, 
                                 activation='linear', # dtype='float64',
                                 name="fc_action_out")(state_input)
    return action_out, state_input


def nn_policy_relu(state_input, policy_arch, dim_action, **kwargs):
    """ 
        Fully-connected agent policy network 
    """
    with tf.variable_scope('policy_net', reuse=tf.AUTO_REUSE):
        for i, h in enumerate(policy_arch):
            state_input = layer.Dense(h, activation='relu', # dtype='float64', 
                                      name="fc_{}".format(i))(state_input)
        action_out = layer.Dense(dim_action, 
                                 activation='linear', # dtype='float64',
                                 name="fc_action_out")(state_input)
    return action_out, state_input
