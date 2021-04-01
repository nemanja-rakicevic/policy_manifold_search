
"""
Author:         Anonymous
Description:
                AE architecture building functions, fully-connected 
                encoder/decoder definitions
                
"""

import os
import logging
import csv
import numpy as np 
import tensorflow as tf 
# import keras
# from keras import layers as ly
layer = tf.keras.layers


logger = logging.getLogger(__name__)




def fc_param_encoder(x_input, encoder_arch, latent_dim, parameter_arch, **kwargs):
    """ Fully-connected encoder from the AE architecture blueprint """

    with tf.variable_scope('ae_encoder', reuse=tf.AUTO_REUSE):
        # Make sure the input is flat and NON NANS   
        if parameter_arch is not None:
            split_list = tf.split(x_input, 
                                  num_or_size_splits=len(parameter_arch), 
                                  axis=-1)
            in_data = []
            for st, pa in zip(split_list, parameter_arch):
                tmp = st[:,:pa[0],:pa[1],:]
                tmp = tf.reshape(tmp, [-1]+[np.prod(pa)])
                in_data.append(tmp)
            x_input = tf.concat(in_data, axis=1)


        # Construct the network
        for i, h in enumerate(encoder_arch):
            if h[0] == 'fc':
                x_input = layer.Dense(h[1], activation='elu', 
                                  name="encoder_fc_{}".format(i))(x_input)
                # x_input = layer.BatchNormalization()(x_input, training=True)
            elif h[0] == 'cct':
                orig = x_input
                x_input = layer.Dense(h[1], activation='elu', 
                                  name="encoder_cct_{}".format(i))(x_input)
                x_input = tf.concat([orig, x_input], 1)

            elif h[0] == 'res':
                x_input = layer.Dense(h[1], activation='elu', 
                                  name="encoder_res_{}a".format(i))(x_input)
                res = x_input
                x_input = layer.Dense(h[1], activation='elu', 
                                  name="encoder_res_{}b".format(i))(x_input)
                # x_input = layer.Activation(activation='relu')(x_input)
                x_input = layer.add([res, x_input])

        with tf.variable_scope('latent_representation', reuse=tf.AUTO_REUSE):
            z_mean = layer.Dense(latent_dim, activation='linear', 
            # z_mean = layer.Dense(latentdim_param_ae, activation='tanh', 
                                 # name="encoder_mean"
                                 )(x_input)
            z_log_std = layer.Dense(latent_dim, activation='linear', 
                                    # name="encoder_std"
                                    )(x_input)
            z_std = tf.exp(z_log_std) + 1e-20
            
            # z_var = layer.Dense(latent_dim, activation='relu')(x_input)

        eps = tf.random_normal(tf.shape(z_std), mean=0., stddev=1.0,
                               dtype=tf.float32, name='epsilon')
        z = tf.add(z_mean, tf.multiply(tf.sqrt(z_std), eps), 
                   name="latent_var")
    return z, z_mean, z_std
        

def fc_param_decoder(z_input, decoder_arch, input_dim, 
                     parameter_dims, parameter_arch, scale_type=None, **kwargs):
    """ Fully-connected decoder from the AE architecture blueprint """
    with tf.variable_scope('ae_decoder', reuse=tf.AUTO_REUSE):
    # with tf.variable_scope('fc_decoder'):
        for i,h in enumerate(decoder_arch[::-1]):
            if h[0] == 'fc':
                z_input = layer.Dense(h[1], activation='elu',
                                            name="decoder_fc_{}".format(i))(z_input)
                # z_input = layer.BatchNormalization()(z_input, training=True)
            elif h[0] == 'cct':
                orig = z_input
                z_input = layer.Dense(h[1], activation='elu', 
                                            name="decoder_cct_{}".format(i))(z_input)
                z_input = tf.concat([orig, z_input], 1)

            elif h[0] == 'res':
                z_input = layer.Dense(h[1], activation='elu',
                                            name="decoder_res_{}a".format(i))(z_input)
                res = z_input
                z_input = layer.Dense(h[1], activation='elu',
                                            name="decoder_res_{}b".format(i))(z_input)
                z_input = layer.add([res, z_input])
  
        # Reconstruct the original shape
        if parameter_arch is not None:
            recn_dim = sum([np.prod(pa) for pa in parameter_arch])
            
            if scale_type is not None:      
                if scale_type == 'tanh':         
                    x_hat = layer.Dense(recn_dim, activation='tanh',
                                              name="X_reconstructed")(z_input)           
                elif scale_type == 'clip':  
                    x_hat = layer.Dense(recn_dim, activation='linear',
                                              name="X_reconstructed")(z_input)                                 
                    x_hat = tf.clip_by_value(x_hat, -1, 1)
                else:
                    raise ValueError("Incorrect scale_type!")
            else:
                x_hat = layer.Dense(recn_dim, activation='linear',
                                          name="X_reconstructed")(z_input)
            cum_dim = 0
            out_data = []
            for i, pa in enumerate(parameter_arch):
                tmp_dim = np.prod(pa)
                tmp = x_hat[:, cum_dim:cum_dim+tmp_dim]
                tmp = tf.reshape(tmp, [-1]+list(pa)+[1])
                _padd = tf.constant([[0, 0], 
                           [0, parameter_dims[0]-pa[0]], 
                           [0, parameter_dims[1]-pa[1]], 
                           [0, 0]])
                tmp = tf.pad(tmp, paddings=_padd, constant_values=np.inf) 
                out_data.append(tmp)
                cum_dim += tmp_dim
            x_hat = tf.concat(out_data, axis=-1)
        else:
            recn_dim = np.prod(parameter_dims)

            if scale_type is not None:      
                if scale_type == 'tanh':         
                    x_hat = layer.Dense(recn_dim, activation='tanh',
                                              name="X_reconstructed")(z_input)           
                elif scale_type == 'clip':  
                    x_hat = layer.Dense(recn_dim, activation='linear',
                                              name="X_reconstructed")(z_input)                                 
                    x_hat = tf.clip_by_value(x_hat, -1, 1)
                else:
                    raise ValueError("Incorrect scale_type!")
            else:
                x_hat = layer.Dense(recn_dim, activation='linear',
                                          name="X_reconstructed")(z_input)

    return x_hat  

