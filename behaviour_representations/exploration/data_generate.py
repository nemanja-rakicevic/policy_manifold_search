
"""
Author:         Anonymous
Description:
                Data generation functions
                - static data (synthetic and MNIST)
                - dynamic data, control policy outputs (displacement and NN)

                Here the sampling/exploration strategy is implemented
"""

import logging
import itertools
import numpy as np

from operator import itemgetter

from behaviour_representations.utils.utils import _TAB, _SEED
from behaviour_representations.utils.utils import info_outcome, timing


from behaviour_representations.exploration.data_generate_ps import DisplacementParamSampling
from behaviour_representations.exploration.data_generate_ls import LatentSampling
from behaviour_representations.exploration.data_generate_mape import SampleMAPE


logger = logging.getLogger(__name__)



class GenerateData(object):
    """
        Base class for data generation
    """
    def __init__(self, seed_data, **kwargs):
        np.random.seed(seed_data)
        self.initial = True


    def _shuffle(self, *args):
        idx = np.random.permutation(len(args[0]))
        out_args = [] 
        for a in args:
            if type(a) == np.ndarray:
                out_args.append(a[idx])
            elif type(a) == list:
                out_args.append(list(itemgetter(*idx)(a)))
        return tuple(out_args)


    def generate_data(self, **kwargs):
        raise NotImplementedError



class GenerateDisplacement(GenerateData, SampleMAPE,
                           DisplacementParamSampling, LatentSampling):
    """
        Wrapper around agent environment evaluations.

        Generate data by running the brick-hitting-the-ball environment, 
        with a displacement-based policy, or a nn policy.
    """
    
    def __init__(self, task_object, exploration, experiment_directory,
                       **kwargs):
        super().__init__(**kwargs)
        self.dirname = experiment_directory
        # Unpack task details
        self.task = task_object
        self.num_outcomes = self.task.num_outcomes
        self.parameter_dims = self.task.controller_obj.parameter_dims
        self.parameter_arch = self.task.controller_obj.parameter_arch
        self.traj_dim = self.task.traj_dim

        self.metric_dim = self.task.environment.bd_metric.metric_dim
        self.metric_name = self.task.environment.bd_metric.metric_name
        self.env_info = self.task.environment.env_info
        self.env_name = self.task.env_name
        self.ctrl_name = self.task.ctrl_name
        # Unpack exploration details
        self.stype = exploration['scale_type']
        self.srange = exploration['limit_range']
        self.irange = exploration['init_range'] \
            if exploration['init_range'] is not None else (-1, 1)
        self.exploration_normal = exploration['normal']
        self.exploration_initial = \
                exploration['initial'] if exploration['initial'] is not None \
                                       else self.exploration_normal 
        self.num_samples = exploration['num_samples']
        self.k_init = exploration['k_init']
        if '_mape' in exploration['normal']:
            self.mape_dict = {'niter': exploration['mape_niter'],
                              'decay': exploration['mape_decay'],
                              'sigma': exploration['mape_sigma'],
                              'use_fit': int(exploration['mape_use_fitness'])}


    def _eval_population(self, next_param_population, 
                         from_latent, recn_fn=None, convert=True,
                         shuffle=False, balance=False, **kwargs):
        """ 
            Reconstruct samples from ebedded space and evaluate 
            the policy parameters
        """
        assert len(next_param_population)>0, "Empty population!"
        if from_latent and recn_fn is not None: 
            if len(next_param_population.shape) == 1:
                next_param_population = next_param_population.reshape(
                                            -1, next_param_population.shape[0])
            next_param_population_hat = recn_fn(next_param_population)
        elif convert:
            next_param_population_hat = \
                self.task.controller_obj.convert_params(
                                            uniforms=next_param_population)
        else:
            next_param_population_hat = next_param_population.copy()
        # Evaluate the params
        out_dict = self.task.evaluate_population(
                                  param_population=next_param_population_hat)
        # # Balance data to avoid disproportionate labels
        # if balance: 
        #     out_tuple = self.task.balance_outcomes(*out_tuple)       
        # if shuffle: 
        #     out_tuple = self._shuffle(*out_tuple)
        if from_latent:
            out_dict.update({'param_embedded': next_param_population})
        # else:
        #     out_dict.update({'param_embedded': None})
        return out_dict


    @timing
    def generate_data(self, **kwargs):
        """ 
            Generate a parameter population using a given exploration_strategy
        """
        if kwargs['nloop']==0:
            self.exploration_type = self.exploration_initial
            _k, _name = self.k_init, 'INITIAL'
            _mape_iter = ''
        else:
            self.exploration_type = self.exploration_normal
            _k, _name = 1, 'NORMAL'
            _mape_iter = '{} iters'.format(self.mape_dict['niter']) \
                            if 'mape' in self.exploration_normal else ''
        logging.info("EXPLORATION START - {} '{}' ({}@{:.0f} samples);".format(
                                          _name, self.exploration_type, 
                                          _mape_iter, _k*self.num_samples))

        # Call function to generate data population 
        low, high = self.irange
        trial_outputs = getattr(self, self.exploration_type)(k=_k, low=low,
                                                                   high=high, 
                                                                   **kwargs)

        # Organise the returned data in a dict for main dataset
        if isinstance(trial_outputs, tuple):
            trial_outputs = dict(param_original=trial_outputs[0], 
                                 outcomes=trial_outputs[1], 
                                 metric_bd=trial_outputs[2], 
                                 traj_main=trial_outputs[3], 
                                 traj_aux=trial_outputs[4])
            if len(trial_outputs)==6:
                trial_outputs.update({'param_embedded': trial_outputs[5]})
            else:
                trial_outputs.update({'param_embedded': None})

        # Log exploration progress
        trial_outs = trial_outputs['outcomes']
        logger.info("EXPLORATION DONE '{}' generated {} [{}-dim] new samples:"
                    "\n{}{}".format(self.exploration_type, 
                    len(trial_outs) if trial_outs is not None else 0, 
                    self.parameter_dims, _TAB, info_outcome(trial_outs)))
        return trial_outputs

            