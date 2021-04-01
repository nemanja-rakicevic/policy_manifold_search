
"""
Author:         Anonymous
Description:
                AE loss functions
"""

import numpy as np
import tensorflow as tf

from itertools import combinations


_EMB_NORM = 2  # 0.5
outcome_vals = np.array([-1, 0, 1, 2])


def _sample_distances(embeddings, squared=False):
    """
    Get distances through the covariance matrix
    """
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + \
                    tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)
    if not squared:
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances


def _combinations(bs):
    a = tf.range(bs)
    tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(a)[0]])  
    tile_a = tf.expand_dims(tile_a, 2) 
    tile_b = tf.tile(tf.expand_dims(a, 0), [tf.shape(a)[0], 1]) 
    tile_b = tf.expand_dims(tile_b, 2) 
    cart = tf.concat([tile_a, tile_b], axis=2) 
    cart = tf.reshape(cart,[-1,2])
    return tf.boolean_mask(cart, tf.less(cart[:,0], cart[:,1]))


def test_loss(input_batch, recn_batch, **kwargs):
    with tf.name_scope("TEST_loss"):
        # MSE
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(input_batch - recn_batch), 
            # sum over other axes except the batch axis
            axis=list(range(1, len(recn_batch.get_shape()) ))) )

        # # Cross-Entropy (continuous)
        # reconstruction_loss = \
        #     tf.reduce_mean(-tf.reduce_sum(x_input*tf.log(1e-20 + x_hat), 1))
    return reconstruction_loss, []
    
    
###############################################################################


""" REGULARISATION LOSSES """


def l2_loss(**kwargs):
    """ Weight decay loss on weights only """
    with tf.name_scope("L2_loss"):
        all_vars = tf.trainable_variables() 
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars])
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars \
        #                                      if 'bias' not in v.name ])
        l2_loss = tf.add_n([tf.norm(v, ord=2) for v in all_vars \
                                             if 'bias' not in v.name ])
    return l2_loss, []


def l1_loss(**kwargs):
    """ Weight decay loss on weights only """
    with tf.name_scope("L1_loss"):
        all_vars = tf.trainable_variables() 
        l1_loss = tf.add_n([tf.norm(v, ord=1) for v in all_vars \
                                             if 'bias' not in v.name ])
    return l1_loss, []


"""  BASIC AE LOSSES """


def reconstruction_loss(**kwargs):
    """ If it is high-dimensional apply slicing to get proper values """
    if tf.shape(kwargs['input_batch']).get_shape()[0] <= 3:
        return reconstruction_flat_loss(**kwargs)
    else:
        return reconstruction_tensor_loss(**kwargs)



def reconstruction_flat_loss(input_batch, recn_batch, **kwargs):
    with tf.name_scope("RECN_loss"):
        # MSE
        # sum over other axes except the batch axis
        sum_list = list(range(1, len(input_batch.get_shape()) ))
        reconstruction_loss = \
          tf.reduce_mean(tf.reduce_sum(tf.square(input_batch - recn_batch), 
                                       axis=sum_list))
    return reconstruction_loss, []



def reconstruction_tensor_loss(input_batch, recn_batch, parameter_arch, **kwargs):
    with tf.name_scope("RECN_CNN_loss"):
        sum_list = list(range(1, len(input_batch.get_shape()) ))
        n_lay = len(parameter_arch)
            
        split_x_in = tf.split(input_batch, num_or_size_splits=n_lay, axis=-1)
        split_x_hat = tf.split(recn_batch, num_or_size_splits=n_lay, axis=-1)
        recnloss_list = []
        
        for xin, xhat, pa in zip(split_x_in, split_x_hat, parameter_arch):
            tmp_xin = xin[:,:pa[0],:pa[1],:]  # tf.slice(noise_spec, [0,0],[rows, cols])
            tmp_xhat = xhat[:,:pa[0],:pa[1],:]
            tmp = tf.reduce_sum(tf.square(tmp_xin - tmp_xhat), axis=sum_list)
            recnloss_list.append(tmp)
        
        # concatenate layers and sum
        reconstruction_loss = tf.stack(recnloss_list, axis=1)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    return reconstruction_loss, []



def elbo_loss(embedding_batch, z_var, **kwargs):
    # z_var = kwargs['z_var']
    with tf.name_scope("ELBO_loss"):
        KL_loss = 1 + tf.log(z_var) - tf.square(embedding_batch) - z_var
        KL_loss = -0.5 * tf.reduce_sum(KL_loss, 1)
        KL_loss = tf.reduce_mean(KL_loss)
    return KL_loss, []

