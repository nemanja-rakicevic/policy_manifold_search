
"""
Author:         Anonymous
Description:
                Training classes for different model versions:
                - TrainBase: basic training initialisation
                - TrainAE: Parameter AE
"""

import os
import time
import logging
import csv
import joblib
import numpy as np 
import tensorflow as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
# import keras
# from keras import layers as ly
layer = tf.keras.layers

from functools import partial
from scipy.stats import linregress
from sklearn.decomposition import PCA

import behaviour_representations.training.ae_losses as uloss
from behaviour_representations.utils.utils import timing, _TAB, _SEED


logger = logging.getLogger(__name__)




class NNutil(object):
    """ Helper methods for interfacing NN """

    def _tf_remove_padding(self, data_arch):
        if self.parameter_arch is not None:
            split_list = tf.split(data_arch, 
                                  num_or_size_splits=len(self.parameter_arch), 
                                  axis=-1)
            in_data = []
            for st, pa in zip(split_list, self.parameter_arch):
                tmp = st[:,:pa[0],:pa[1],:]
                tmp = tf.reshape(tmp, [-1]+[np.prod(pa)])
                in_data.append(tmp)
            data_arch = tf.concat(in_data, axis=1)
        return data_arch


    def _remove_padding(self, data_arch):
        if self.parameter_arch is not None:
            split_list = np.split(data_arch, 
                                  indices_or_sections=len(self.parameter_arch), 
                                  axis=-1)
            in_data = []
            for st, pa in zip(split_list, self.parameter_arch):
                tmp = st[:,:pa[0],:pa[1],:]
                tmp = np.reshape(tmp, [-1]+[np.prod(pa)])
                in_data.append(tmp)
            data_arch = np.concatenate(in_data, axis=1)
        return data_arch


    def _add_padding(self, data_flat):
        if self.parameter_arch is not None:
            cum_dim = 0
            n_smp = data_flat.shape[0]
            out_data = np.inf * np.ones([n_smp]+list(self.parameter_dims))
            for i, pa in enumerate(self.parameter_arch):
                tmp_dim = np.prod(pa)
                tmp = data_flat[:, cum_dim:cum_dim+tmp_dim]
                tmp = np.reshape(tmp, [-1]+list(pa))
                out_data[:, :pa[0], :pa[1], i] = tmp
                cum_dim += tmp_dim
            data_flat = out_data.copy()
        return data_flat


###############################################################################
###############################################################################
###############################################################################



class TrainBase(NNutil):
    """
        Base class to create AE networks
    """
    def __init__(self, seed_model, recn_init, testset_ratio, **kwargs):
        # Data
        self.recn_init = recn_init
        self.testset_ratio = testset_ratio

        self.dirname = self.data_object.dirname
        self.load_model_path = self.data_object.load_model_path

        self.metric_name = self.data_object.datagen_object.metric_name
        self.metric_dim = np.prod(self.data_object.datagen_object.metric_dim)

        # TF graph and session
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.reset_default_graph()
        tf.set_random_seed(seed_model)
        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.sess = session
        if self.load_model_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.restore_model()        
        self.sess.graph.finalize()
        
        # TF tensorboard
        # uniq_id = "/tmp/tensorboard-layers-api/" + uuid.uuid1().__str__()[:6]
        # summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
        # RUN_PATH = './logs/run_wtf2/'
        # summary_writer = tf.summary.FileWriter(RUN_PATH+'summary', graph=self.sess.graph)


    """ TRAINING """

    def _build_graph(self, graph_name="AE_net"):
        raise NotImplementedError


    def _total_loss(self):
        raise NotImplementedError


    def _train_step_batch(self, nloop, it):
        raise NotImplementedError


    def _test_step(self, nloop):
        raise NotImplementedError


    def _logger_inital(self, nloop, num_epochs, num_iter, 
                             ae_name='param', ae_type=None):
        # if nloop == 0:
        #     logger.info("Initial batching: '{}'; after: '{}'".format(
        #                   self.data_object.batch_object.batchmode_init, 
        #                   self.data_object.batch_object.batchmode))
        if self.recn_init and nloop==0: 
              logger.info("[RECONSTRUCTION loss only]")
        in_sz = self.parameter_arch if self.parameter_arch is not None \
                                    else self.inputdim_param_ae
        logger.info("{}_AE {}FITTING: {}-D -> {}-D;\n{}"
                    "- Datapoints: {:4};\n{}"
                    "- Epochs:     {:4};\n{}"
                    "- Iterations: {:4};\n".format(
                    ae_name.upper(), 
                    "'{}' ".format(ae_type) if ae_type is not None else '',
                    in_sz if ae_name=='param' else self.inputdim_traj_ae, 
                    self.latentdim_param_ae if ae_name=='param' \
                        else self.latentdim_traj_ae, _TAB,
                    self.data_object.num_datapoints, _TAB,
                    num_epochs, _TAB, num_iter))


    def _logger_loss(self, avg_loss, num_iter=None, ae_name='param', 
                     test_loss=None, nloop=None, ep=None, num_epochs=None):
        """ Log the current epoch average loss over minibatches """
        _loss_fn_list = getattr(self, 'loss_fn_{}_ae'.format(ae_name))
        # log training message
        train_list = ["\n{}\t\t{}: {:8.4e}".format(_TAB, fn, fl) for fn, fl \
                        in zip(_loss_fn_list[::-1], avg_loss[::-1])]
        train_loss = '\tTraining loss: {:8.4e};{}'.format(
                          np.sum(avg_loss), ', '.join(train_list))
        if ep is not None:
            if  np.array(test_loss).all() != None:
                test_list = ["\n{}\t\t{}: {:8.4e}".format(_TAB, fn, fl) for \
                    fn, fl in zip(_loss_fn_list[::-1], test_loss[::-1])]
                test_loss = \
                    '\n{}\tTest loss ({}%): {:8.4e};{}'.format(
                    _TAB, int(self.testset_ratio * 100), 
                    np.sum(test_loss), ', '.join(test_list))
            else:
                test_loss = ''
            logger.info('> loop {} > epoch {:4} > n_iters {}:\n{}{}{}\n'.format(
                        nloop, ep+1, num_iter, _TAB, train_loss, test_loss))
        elif num_epochs is not None:
            logger.info("{}_AE DONE; {} epochs;\n{}{}\n".format(
                        ae_name.upper(), num_epochs, _TAB, train_loss))


    def _loss_slope(self, loop_losses, ep, buff_sz, name):
        # buff_sz = len(loop_losses)
        _buff = np.vstack(loop_losses[-buff_sz:])
        _tmp_test = np.stack(_buff[:,1])
        if name == 'test' and self.testset_ratio and (_tmp_test!=None).all():
            _tmp_test = np.mean(_tmp_test, axis=1)
            _slope = linregress(np.arange(buff_sz),_tmp_test).slope
        elif name == 'train':
            _tmp_train = np.mean(np.stack(_buff[:,2]), axis=1)
            _slope = linregress(np.arange(buff_sz),_tmp_train).slope
        else:
            return False

        # Check if slope is rising
        if _slope >= 1e-5:
            logger.info(">>> EARLY STOPPING (epoch: {}; {} slope: "
                        "{:8.4e})\n".format(ep, name , _slope))
            return True
        else:
            return False


    @timing
    def _fit_autoencoder(self, nloop, ae_name, embedding_fn, recn_fn,
                         train_init_fn, train_step_fn, test_step_fn, 
                         ae_type=None, save_dataset=False, 
                         save_training_model=True, save_training_info=True,
                         verbose=True, **kwargs):
        """ Run AE training """
        # Setup number of training iterations and epochs to run
        _buff_sz = 100
        num_iter, num_epochs = train_init_fn(nloop, ae_name, verbose=True)
        verbose_freq = max(100, num_epochs/10)
        # Log info
        self._logger_inital(nloop, num_epochs, num_iter,
                            ae_name=ae_name, ae_type=ae_type)
        # Main training loop
        _loop_losses = []
        for ep_ in range(num_epochs):
            # Go through the dataset
            num_iter, _ = train_init_fn(nloop, ae_name)
            _epoch_losses = []
            for iter_ in range(num_iter):
                _, mb_loss_list = train_step_fn(nloop, iter_)
                _epoch_losses.append(mb_loss_list)
            # Evaluate on testset
            _test_losses = test_step_fn(nloop)
            _epoch_losses = list(np.mean(_epoch_losses, axis=0))
            _loop_losses.append([nloop] + [_test_losses] + [_epoch_losses])
            # Log progress
            if not (ep_ + 1) % verbose_freq and verbose:
                self._logger_loss(nloop=nloop, ep=ep_, num_iter=num_iter,
                                  ae_name=ae_name,
                                  test_loss=_test_losses,
                                  avg_loss=_epoch_losses)

             # Early stopping based on test set
            if self.early_stop and ep_>_buff_sz: # and self.testset_ratio:
                if self._loss_slope(_loop_losses, ep_, _buff_sz, name='test'):
                    break

        # Log final data
        self._logger_loss(_epoch_losses, ae_name=ae_name, num_epochs=ep_+1)
# loss info for plot ### TODO: make this nicer
        self.data_object.training_loss_dict[ae_name] = \
            {(fn.__name__[:-5] if type(fn)!=str else fn): fl for \
             fn, fl in zip(getattr(self, 'loss_fn_{}_ae'.format(ae_name)), 
                              _epoch_losses)}

        # Update all the data with the latest model
        self.data_object.update_representations(ae_name=ae_name,
                                                embedding_fn=embedding_fn, 
                                                recn_fn=recn_fn, 
                                                verbose=verbose, 
                                                save_dataset=save_dataset)
        if save_training_model:
            self.save_model(nloop)
        if save_training_info:
            self.save_training_data(ae_name, _loop_losses)


    # @timing
    def run_training(self):
        raise NotImplementedError


    """ SAVE/LOAD """

    # @timing
    def save_training_data(self, ae_name, loop_losses):
        """ Appends epoch losses after each loop """
        savepath = self.dirname+"/saved_models/"
        if not os.path.isdir(savepath): os.makedirs(savepath)
        filepath = savepath + "training_losses_{}_ae.csv".format(ae_name)
        with open(filepath, 'a') as outfile: 
            writer = csv.writer(outfile) 
            writer.writerows(loop_losses)

    # @timing
    def save_model(self, num_loop):
        """ Save tf model data """
        savepath = self.dirname+"/saved_models/"
        modelpath = savepath + "loop_{num:05d}".format(num=num_loop)
        if not os.path.isdir(modelpath): os.makedirs(modelpath)
        self.saver.save(self.sess, modelpath+'/model_info', 
                        global_step=num_loop)

    # @timing
    def restore_model(self):
        model_dir = os.path.join(self.load_model_path, 'saved_models')
        # Load last saved model
        model_last = sorted([mn for mn in os.listdir(model_dir) \
                                if 'loop' in mn])[-1]
        # path = self.load_model_path \
        #     + "saved_models/loop_{num:05d}/model_info-{num}".format(num=nloop)
        model_num = int(model_last.split('loop_')[1])
        path = os.path.join(model_dir, model_last,
                            'model_info-{num}'.format(num=model_num))
        self.saver.restore(self.sess, path)
        logger.info("RESTORED AE MODEL; loop {}"
                    "\n{}\tLocation: {}".format(model_num, _TAB, path))



###############################################################################
###############################################################################
###############################################################################


class TrainAE(TrainBase):
    """ Training AE for parameter reconstruction, with regularisation """

    def __init__(self, data_object, dim_latent,
                       ae_param,
                       param_encoder_fn, param_decoder_fn, 
                       **kwargs):

        self.data_object = data_object
        self.inputdim_param_ae = self.data_object.inputdim_param_ae
        self.latentdim_param_ae = dim_latent
        # Losses
        self.loss_fn_param_ae = ae_param['loss_fn']
        self.loss_coef_param_ae = ae_param['loss_coeff']
        self.loss_kwargs = ae_param['loss_kwargs']
        # Training hyperparams
        self.lr_param_ae = ae_param['lr']
        self.num_epochs = ae_param['num_epochs']
        self.early_stop = ae_param['early_stop']

        if 'outcome_branch' in self.loss_fn_param_ae:
            self.get_out_prediction = self._get_out_prediction
        else:
            self.get_out_prediction = None


        # Extract parameter AE model building functions
        self.parameter_arch = self.data_object.datagen_object.parameter_arch
        scale_type = self.data_object.datagen_object.stype
        paramae_kwargs = dict(latent_dim=self.latentdim_param_ae,
                              input_dim=self.inputdim_param_ae,
                              encoder_arch=ae_param['architecture'], 
                              decoder_arch=ae_param['architecture'],
                              branch_arch=None, 
                              parameter_dims=self.inputdim_param_ae, 
                              parameter_arch=self.parameter_arch,
                              scale_type=scale_type)
        self._encoder = partial(param_encoder_fn, **paramae_kwargs)
        self._decoder = partial(param_decoder_fn, **paramae_kwargs)

        super().__init__(**kwargs)


    """ GRAPH CONSTRUCTION """

    def _build_graph(self, graph_name="AE_net"):

        # Placeholders: Main AE input 
        self.X_PARAM = layer.Input(shape=(*self.inputdim_param_ae,), 
                                   name="PARAM_input")
        self.Z_PARAM = layer.Input(shape=(self.latentdim_param_ae,), 
                                   name="Z_MAIN_input")

        # Placeholders: Outcome
        self.TO = layer.Input(shape=(1,), name="Target_outcome")
        # Placeholders: Auxilliary Regularization
        self.METRIC_BD = layer.Input(shape=(self.metric_dim,), 
                                        name="MetricBD")

        # Single value placeholders
        _METRIC_TRAJ_DM = tf.placeholder(tf.float32, shape=[None, 1], 
                                        name="MetricTraj_ph")
        self._METRIC_TRAJ_DM = layer.Input(tensor=_METRIC_TRAJ_DM, 
                                           name="MetricTraj")
        
        _N_CLST = tf.placeholder(tf.int32, shape=(), name="NumClusters_ph")
        self._N_CLST = layer.Input(tensor=_N_CLST, name="NumClusters")
        
        _BS = tf.placeholder(tf.int32, shape=(), name="BatchSize_ph")
        self._BS = layer.Input(tensor=_BS, name="BatchSize")

        self.JIDX = tf.placeholder("int32")

        # Additional ops
        with tf.name_scope("get_embedding"):
            _, self.encoder_op, _ = self._encoder(self.X_PARAM)
        with tf.name_scope("get_reconstruction"):
            self.decoder_op = self._decoder(self.Z_PARAM)

        if 'outcome_branch' in self.loss_fn_param_ae:
            with tf.name_scope("get_out_branch"):
                self.branch_op = self._branch_out(self.Z_PARAM)

        # Jacobian ops
        with tf.name_scope("decoder_jacobian_grad"):
            dec_out = self._decoder(self.Z_PARAM)
            # dec_out = self.data_object.apply_denormalisation(dec_out)
            dec_out = self._tf_remove_padding(dec_out)
            out_slice = tf.slice(dec_out,[0, self.JIDX],[-1, 1])
            self.dec_jac_grad_op = tf.gradients(out_slice, self.Z_PARAM)

        with tf.name_scope("encoder_jacobian_grad"):
            # enc_in = self.data_object.apply_normalisation(self.X_PARAM)
            # _, enc_out, _ = self._encoder(enc_in)
            _, enc_out, _ = self._encoder(self.X_PARAM)
            out_slice = tf.slice(enc_out,[0, self.JIDX],[-1, 1])
            tmp_grad = tf.gradients(out_slice, self.X_PARAM)
            self.enc_jac_grad_op = self._tf_remove_padding(tmp_grad)

        # Define losses
        with tf.name_scope("param_ae_loss"):
            self.op_loss, \
            self.op_losses_list, \
            self.op_other, \
            self.op_loss_init = self._loss_param_ae()

        # Define optimizer
        self.ae_optimizer = tf.train.AdamOptimizer(
                                        learning_rate=self.lr_param_ae,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08,)

        with tf.name_scope("train_init"):
            self.op_optimizer_init = \
                self.ae_optimizer.minimize(self.op_loss_init)

        with tf.name_scope("train"):
            self.op_optimizer = self.ae_optimizer.minimize(self.op_loss)

        self.reset_op = tf.variables_initializer(self.ae_optimizer.variables())

        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ae_decoder')[0]
        self.var_grad = tf.gradients(self.op_loss, [var])



    def _reset_optimizer(self):
        """ 
            Resets the list of variables which encode the current state of the
            Optimizer. Includes slot variables and additional global variables
            created by the optimizer in the current default graph.
        """
        self.sess.run(self.reset_op)
        # self.sess.run(tf.global_variables_initializer())


    def _loss_param_ae(self):
        """ Caluclate loss using a weighted combination of loss functions. """
        z, z_mean, z_var = self._encoder(self.X_PARAM)
        # x_hat = self._decoder(z)
        x_hat_exact = self._decoder(z_mean)
        out_hat = self._branch_out(z_mean) if 'outcome_branch' \
                    in self.loss_fn_param_ae else None

        # Inputs to loss functions
        default_kwargs = {
            'input_batch': self.X_PARAM,
            # 'recn_batch': x_hat,
            'recn_batch': x_hat_exact,
            'recn_exact_batch': x_hat_exact,
            'embedding_batch': z_mean,
            'z_var': z_var,
            'outcome_batch': self.TO, # tf.reshape(self.TO, [tf.shape(self.TO)[0]]),
            'out_hat_batch': out_hat,

            'metric_vect_batch': self.METRIC_BD,
            'metric_traj_dm': self._METRIC_TRAJ_DM,

            'parameter_arch': self.parameter_arch,

            'n_samples': self.data_object.batch_object._n_samples, 
            'n_clusters': self._N_CLST, 
            'batch_size': self._BS, 
            'latent_dim': self.latentdim_param_ae}

        losses_list = []
        other_list = []
        # Graph for all losses
        for lossfn, losscoeff, losskw in zip(self.loss_fn_param_ae,
                                             self.loss_coef_param_ae,
                                             self.loss_kwargs):
            loss_factor, other = uloss.__dict__['{}_loss'.format(lossfn)](
                                    **default_kwargs, **losskw)
            losses_list.append(tf.multiply(loss_factor, losscoeff))
            other_list.append(other)
        # Graph for only reconstruction 
        init_loss, _ = uloss.__dict__['reconstruction_loss'](**default_kwargs)
        # return tf.reduce_sum(losses_list), losses_list, other_list, init_loss
        return tf.reduce_sum(losses_list), losses_list, other_list, init_loss


    """ TRAINING """

    def run_training(self, **kwargs):
        self._reset_optimizer()
        self._fit_autoencoder(ae_name='param', 
                              embedding_fn=self.get_param_embedding, 
                              recn_fn=self.get_param_reconstruction,
                              train_init_fn=self._train_init, 
                              train_step_fn=self._train_step, 
                              test_step_fn=self._test_step,
                              save_dataset=True, **kwargs)


    def _train_init(self, nloop, *args, **kwargs):
        num_iter = self.data_object.init_batching(
                    nloop=nloop, testset_ratio=self.testset_ratio, **kwargs)
        num_epochs = self.recn_init * self.num_epochs \
                        if self.recn_init and nloop==0 else self.num_epochs
        return num_iter, num_epochs


    def _train_step(self, nloop, it):
        """ Execute one AE training batch """
        # Get batch
        batch_dict = self.data_object.generate_batch(
                          iteration=it, recn_fn=self.get_param_reconstruction)
        # Normalize batch data using dataset normalisation statistics
        batch_params = \
            self.data_object.apply_normalisation(batch_dict['batch_params'])
        feed_dict = {self.X_PARAM:  batch_params,  
                     self.TO: batch_dict['batch_outcomes'][:, 0][:, None],
                     self.METRIC_BD:    batch_dict['batch_metric_bd'],
                     self._METRIC_TRAJ_DM: batch_dict['dist_trajectory'],
                     self._N_CLST: self.data_object.batch_object._n_clusters,
                     self._BS: batch_dict['batch_size']}

        # Select graph to use depending on the loop
        if self.recn_init and nloop==0:
            fetches = [self.op_optimizer_init, 
                       self.op_loss_init]
            _, loss = self.sess.run(fetches=fetches, feed_dict=feed_dict) 
            loss_other, loss_list = [], [loss]
        else:
            fetches = [self.op_optimizer, 
                       self.op_loss,
                       self.op_other,                                              
                       self.op_losses_list]
            _, loss, loss_other, loss_list = \
                self.sess.run(fetches=fetches, feed_dict=feed_dict)         

        return loss, loss_list # loss_other


    def _test_step(self, nloop):
        """ Evaluate the AE on the hold out data """
        if self.testset_ratio:
            # Get test data
            test_batch_dict = self.data_object.generate_batch(testdata=True)
            if test_batch_dict is None: 
                return [None]
            test_data = self.data_object.apply_normalisation(
                                            test_batch_dict['batch_params'])
            feed_dict = {
                self.X_PARAM:  test_data,  
                self.TO: test_batch_dict['batch_outcomes'][:, 0][:, None],
                self.METRIC_BD:    test_batch_dict['batch_metric_bd'],
                self._METRIC_TRAJ_DM: test_batch_dict['dist_trajectory'],
                self._N_CLST: self.data_object.batch_object._n_clusters,
                self._BS: test_batch_dict['batch_size']}
            if self.recn_init and nloop==0:
                return self.sess.run([self.op_loss_init], feed_dict=feed_dict)
            else:
                return self.sess.run([self.op_losses_list],
                                     feed_dict=feed_dict)[0]
        else:
            return [None]


    """ INTERFACING REPRESENTATIONS """

    # @timing
    def get_encoder_jacobian(self, orig_batch, bs=None):
        """ Calculate the Jacobian of the decoder for a given batch """
        if bs is None:
            bs = orig_batch.shape[0]
        output_size = self.latentdim_param_ae
        if self.parameter_arch is not None:
            input_size = sum([np.prod(pa) for pa in self.parameter_arch])
            # slice the 
        else:
            input_size = self.inputdim_param_ae[-1]

        jacobian_matrix = np.zeros((bs, output_size, input_size))
        # We iterate over the output vectors dimensions 
        for jc in range(output_size):    
            gradients = self.sess.run(self.enc_jac_grad_op, 
                                      feed_dict={self.JIDX: jc, 
                                                 self.X_PARAM: orig_batch})
            jacobian_matrix[:, jc, :] = np.array(gradients)
        return jacobian_matrix


    # @timing
    def get_decoder_jacobian(self, latent_batch, bs=None):
        """ Calculate the Jacobian of the decoder for a given batch """
        if bs is None:
            bs = latent_batch.shape[0]
        input_size = self.latentdim_param_ae
        if self.parameter_arch is not None:
            output_size = sum([np.prod(pa) for pa in self.parameter_arch])
        else:
            output_size = self.inputdim_param_ae[-1]
        jacobian_matrix = np.zeros((bs, output_size, input_size))
        # We iterate over the output vectors dimensions 
        for jc in range(output_size):    
            gradients = self.sess.run(self.dec_jac_grad_op, 
                                      feed_dict={self.JIDX: jc, 
                                                 self.Z_PARAM: latent_batch})
            jacobian_matrix[:, jc, :] = np.array(gradients)
        return jacobian_matrix


    @timing
    def get_enc_jac_stats(self, orig_batch):
        """
            Calculate a measure of transformation the jacobian applies at 
            the certain point.
            https://www.quora.com/How-do-we-calculate-the-determinant-of-a-non-square-matrix 
        """
        logger.info("ENCODER JACOBIAN {} samples".format(orig_batch.shape[0]))
        j_list = self.get_encoder_jacobian(orig_batch)
        # jvol_list = [np.sqrt(np.linalg.det(np.dot(j.T, j))) for j in j_list]
        # jvol_list = [np.linalg.norm(j, axis=0).mean() for j in j_list]
        jnorm_list = [np.linalg.norm(j) for j in j_list]
        # jcond_list = [np.linalg.cond(j) for j in j_list]   
        jcond_list = [np.linalg.cond(j) if jnorm_list[i] else 0. \
                        for i,j in enumerate(j_list)]      
        return np.vstack([jnorm_list, jcond_list]).T


    @timing
    def get_dec_jac_stats(self, latent_batch):
        """
            Calculate a measure of transformation the jacobian applies at 
            the certain point.
            https://www.quora.com/How-do-we-calculate-the-determinant-of-a-non-square-matrix 
        """
        logger.info("DECODER JACOBIAN {} samples".format(latent_batch.shape[0]))
        j_list = self.get_decoder_jacobian(latent_batch)
        # determinant of the transformation
        jvol_list = [np.linalg.det(np.dot(j.T, j)) for j in j_list]
        # matrix norm
        # jnorm_list = [np.linalg.norm(j) for j in j_list]
        # jnorm_list = [np.linalg.norm(j, axis=0).mean() for j in j_list]
        # condition number  
        jcond_list = [np.linalg.cond(j) if jvol_list[i] else 0. \
                        for i,j in enumerate(j_list)]      
        return np.vstack([jvol_list, jcond_list]).T


    def _get_out_prediction(self, z_input):
        # return self.sess.run(x_hat, feed_dict={self.Z: z_input})
        return self.sess.run(self.branch_op, feed_dict={self.Z_PARAM: z_input})


    """ INTERFACING REPRESENTATIONS """

    def get_param_embedding(self, x_input):
        # return self.sess.run(z_mean, feed_dict={self.X: x_input})
        return self.sess.run(self.encoder_op, feed_dict={
                  self.X_PARAM: self.data_object.apply_normalisation(x_input)})


    def get_param_reconstruction(self, z_input):
        # return self.sess.run(x_hat, feed_dict={self.Z: z_input})
        return self.data_object.apply_denormalisation(
            self.sess.run(self.decoder_op, feed_dict={self.Z_PARAM: z_input}))



###############################################################################
###############################################################################
###############################################################################



class TrainPCA(NNutil):
    """ 
        Perform PCA for parameter embedding and reconstruction.
    """

    def __init__(self, data_object, seed_model, dim_latent, **kwargs):
        # Data
        self.data_object = data_object
        self.dirname = self.data_object.dirname
        self.load_model_path = self.data_object.load_model_path
        self.get_out_prediction = None
        self.num_epochs = 1

        self.parameter_dims = self.data_object.datagen_object.parameter_dims
        self.parameter_arch = self.data_object.datagen_object.parameter_arch
        self.latent_dim = dim_latent

        self.metric_name = self.data_object.datagen_object.metric_name
        self.metric_dim = np.prod(self.data_object.datagen_object.metric_dim)

        # self.get_dec_jac_stats = None
        # self.get_decoder_jacobian = None

        # Initialise PCA
        if self.load_model_path is None:
            self.pca = PCA(n_components=self.latent_dim, random_state=seed_model)
            _init_data = np.random.randn(self.latent_dim, *self.parameter_dims)
            self.pca.fit(self._remove_padding(_init_data))
        else:
            self.restore_model()

      
    """ TRAINING """

    def run_training(self, nloop,
                     save_training_model=True, save_training_info=True,
                     save_dataset=False, verbose=True, **kwargs):
        # Cut out the padding before fitting
        data = self._remove_padding(self.data_object.param_original)
        if data.shape[0] < self.latent_dim:
            data = np.tile(data, (self.latent_dim, 1))
        self.pca.fit(data)
        var_explained = self.pca.explained_variance_ratio_.sum()*100
        # Update all the data with the latest model
        self.data_object.update_representations(
                            ae_name='param',
                            embedding_fn=self.get_param_embedding, 
                            recn_fn=self.get_param_reconstruction, 
                            verbose=verbose, 
                            save_dataset=save_dataset)
        # Save PCA statistics
        pca_stats = {'variance': self.pca.explained_variance_ratio_.sum()*100, 
                     'reconstruction': np.mean(self.data_object.recn_error)}
        self.data_object.training_loss_dict['pca'] = pca_stats

        # Log PCA training statistics
        if verbose:
            logger.info("PCA FITTING DONE; {}-D -> {}-D;\n{}"
                        "- Datapoints:               {:4};\n{}"
                        "- PCA variance:          {:4.2f}%;\n{}"
                        "- PCA reconstruction: {:4.4e};\n".format(
                        self.parameter_arch if self.parameter_arch is not None \
                                            else self.parameter_dims, 
                        self.latent_dim, _TAB, self.data_object.num_datapoints, 
                        _TAB, pca_stats['variance'], _TAB, 
                        pca_stats['reconstruction']))

        if save_training_model:
            self.save_model(nloop)
        if save_training_info:
            self.save_training_data([[nloop]+[[None]]+[[var_explained]]])


    """ INTERFACING REPRESENTATIONS """

    def get_param_embedding(self, x_input):
        # https://intellipaat.com/community/22811/pca-projection-and-reconstruction-in-scikit-learn
        # Cut out the padding and flatten
        x_data = self._remove_padding(x_input)
        # Apply PCA transformation
        return self.pca.transform(x_data)


    def get_param_reconstruction(self, z_input):
        # Apply inverse PCA transform
        z_data = self.pca.inverse_transform(z_input)
        # Add the padding and reshape
        return self._add_padding(z_data)


    """ SAVE/LOAD """

    # @timing
    def save_training_data(self, loop_train_data):
        """ Appends epoch losses after each loop """
        savepath = self.dirname+"/saved_models/"
        if not os.path.isdir(savepath): os.makedirs(savepath)
        filepath = savepath + "training_pca_var.csv".format(self.dirname)
        with open(filepath, 'a') as outfile: 
            writer = csv.writer(outfile) 
            writer.writerows(loop_train_data)


    # @timing
    def save_model(self, num_loop):
        savepath = self.dirname+"/saved_models/"
        modelpath = savepath + "loop_{num:05d}".format(num=num_loop)
        if not os.path.isdir(modelpath): os.makedirs(modelpath)
        joblib.dump(self.pca, modelpath+'/model_info.joblib')


    # @timing
    def restore_model(self):
        model_dir = os.path.join(self.load_model_path, 'saved_models')
        # Load last saved model
        model_last = sorted([mn for mn in os.listdir(model_dir) \
                                if 'loop' in mn])[-1]
        # path = self.load_model_path \
        #     + "saved_models/loop_{num:05d}/model_info-{num}".format(num=nloop)
        model_num = int(model_last.split('loop_')[1])
        path = os.path.join(model_dir, model_last, 'model_info.joblib')
        self.pca = joblib.load(path)
        logger.info("RESTORED PCA MODEL; loop {}"
                    "\n{}\tLocation: {}".format(model_num, _TAB, path))


    # @timing
    def get_decoder_jacobian(self, latent_batch, bs=None):
        """
            The Jacobian of the decoder is basically the inverse 
            transformation matrix (pca.components_) the constant (pca.mean_) 
            which is added is neglected.
        """
        return [self.pca.components_.T] * latent_batch.shape[0]

    @timing
    def get_dec_jac_stats(self, latent_batch):
        """
            Calculate a measure of transformation the jacobian applies at 
            the certain point.
            https://www.quora.com/How-do-we-calculate-the-determinant-of-a-non-square-matrix 
        """
        n_smp = latent_batch.shape[0]
        logger.info("DECODER JACOBIAN {} samples".format(n_smp))
        j_dec = self.get_decoder_jacobian(latent_batch)[0]
        # determinant of the transformation
        jvol = np.linalg.det(np.dot(j_dec.T, j_dec))
        jcond = np.linalg.cond(j_dec) if jvol else 0.
    
        return np.tile([jvol, jcond], [n_smp,1])



###############################################################################
###############################################################################
###############################################################################



    # def get_embedding_chunked(self, x_input, chunk_size=400):
    #     num_inputs = x_input.shape[0]
    #     chunks = int(np.ceil(float(num_inputs)/chunk_size))
    #     reps = []
    #     with tf.name_scope("get_embedding"):
    #         z, z_mean, z_var = self._encoder(self.X)
    #     for i in range(chunks):
    #         start = i * chunk_size
    #         stop = min(start + chunk_size, num_inputs)
    #         chunk_reps = self.sess.run(z_mean, 
    #                                    feed_dict={self.X: x_input[start:stop]})
    #         reps.append(chunk_reps)
    #     return np.vstack(reps) 


    # def get_reconstruction_chunked(self, z_input, chunk_size=400):
    #     num_inputs = z_input.shape[0]
    #     chunks = int(np.ceil(float(num_inputs)/chunk_size))
    #     reps = []
    #     with tf.name_scope("get_reconstruction"):
    #         x_hat = self._decoder(self.Z)
    #     for i in range(chunks):
    #         start = i * chunk_size
    #         stop = min(start + chunk_size, num_inputs)
    #         chunk_reps = self.sess.run(x_hat,
    #                                    feed_dict={self.Z: z_input[start:stop]})
    #         reps.append(chunk_reps)
    #     return np.vstack(reps) 

