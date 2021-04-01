
"""
Author:         Anonymous
Description:
                Data generation using parameter space

"""


import csv
import logging
import itertools
import numpy as np

from operator import itemgetter
from inspect import stack 




class SampleBasic(object):
    """ 
        Collection of basic methods for initial parameter space sampling.
    """

    def ps_uniform(self, k, low, high, **kwargs):
        """ Uniform population sampling """
        sampled_population =  np.random.uniform(low=low, high=high,
                                                size=(int(k*self.num_samples), 
                                                      *self.parameter_dims))
        return self._eval_population(sampled_population, from_latent=False, 
                                     shuffle=False, **kwargs)


    def ps_gaussian(self, k, mean=0., std=1., **kwargs):
        """ Uniform population sampling """
        sampled_population = np.random.normal(loc=mean, scale=std,
                                                size=(int(k*self.num_samples), 
                                                      *self.parameter_dims))
        return self._eval_population(sampled_population, from_latent=False, 
                                     shuffle=False, **kwargs)

    def ps_upwards(self, k, **kwargs):
        """ Samples with y-motion fixed to go upwards """
        sampled_population =  np.random.uniform(size=(int(k*self.num_samples), 
                                                      *self.parameter_dims))
        sampled_population[:,1] = 0
        sampled_population[:,4] = 1
        return self._eval_population(sampled_population, from_latent=False,
                                     shuffle=False, **kwargs)


    def ps_uniform_upwards(self, k, **kwargs):
        """ Samples of upward movements from constrained uniform regions """
        low_range, high_range = \
            np.zeros((self.parameter_dims)), np.ones((self.parameter_dims))
        low_range[4] = 0.25
        high_range[1] = 0.1
        sampled_population = np.random.uniform(low=low_range, high=high_range,
                                               size=(int(k*self.num_samples),
                                                     *self.parameter_dims))
        return self._eval_population(sampled_population, from_latent=False,
                                     shuffle=False, **kwargs)
        

    def ps_gaussian_upwards(self, k, **kwargs):
        """ Samples of Gaussian-based upward movements """
        means_start = np.zeros(int((self.parameter_dims-1)/2))
        means_start[1] = 0.05
        means_end = np.ones(int((self.parameter_dims-1)/2))
        means_end[1] = 0.5
        means = np.hstack([means_start, means_end, [0.5]])
        stds = 0.5*np.ones(self.parameter_dims)
        stds[1] = 0.05
        stds[4] = 0.15
        sampled_population = np.random.normal(loc=means, scale=stds,
                                              size=(int(k*self.num_samples),
                                                    *self.parameter_dims))
        return self._eval_population(sampled_population, from_latent=False,
                                     shuffle=False, **kwargs)


    def ps_gaussian_upwards_clipped(self, k, **kwargs):
        """ Samples of Gaussian-based upward movements with constraints """
        means_start = np.zeros(int((self.parameter_dims-1)/2))
        means_start[1] = 0.05
        means_end = np.ones(int((self.parameter_dims-1)/2))
        means_end[1] = 0.5
        means = np.hstack([means_start, means_end, [0.5]])
        stds = 0.5*np.ones(self.parameter_dims)
        stds[1] = 0.05
        stds[4] = 0.15
        sampled_population = np.random.normal(
            loc=means, scale=stds, size=(int(k*self.num_samples), 
                                         *self.parameter_dims)).clip(0,1)
        return self._eval_population(sampled_population, from_latent=False,
                                     shuffle=False, **kwargs)



class SamplePF(object):
    """ 
        Collection of Particle Filter based methods for parameter space 
        sampling.
    """

    def _ps_partfilt(self, fitness_fn, k, sigma=0.01, balance=False, **kwargs):
        """ Base PF method, samples selected based on fitness_fn """
        # Initial uniform-ish sample
        low_range = np.zeros(self.parameter_dims)
        high_range = np.ones(self.parameter_dims)
        # low_range[4] = 0.25
        # high_range[1] = 0.1
        new_population =  np.random.uniform(low=low_range, high=high_range, 
                                            size=(self.num_samples, #//4, 
                                                  self.parameter_dims))
        new_population = self.task.controller_obj.convert_params(uniforms=new_population)

        _sample_size = 100
        param_population = []
        iter_ = 0

        while len(param_population) < int(k*self.num_samples):
            print("\n\n WHILE:ITER:size", iter_, len(param_population))

            # Evaluate population and use outputs
            # tmp_data_dict = \
            #     self.task.evaluate_population(param_population=new_population)
            tmp_data_dict = self._eval_population(
                                        next_param_population=new_population,
                                        from_latent=False, convert=False)

            tmp_population = tmp_data_dict['param_original']
            tmp_outcomes = tmp_data_dict['outcomes']
            tmp_metric_bd = tmp_data_dict['metric_bd']
            tmp_ball_traj = tmp_data_dict['traj_main'] 
            tmp_striker_traj = tmp_data_dict['traj_aux']

            # Keep all the data
            param_population = tmp_population if iter_==0 \
                          else np.vstack([param_population, tmp_population])
            outcome_list = tmp_outcomes if iter_==0 \
                           else np.append(outcome_list, tmp_outcomes) 
            metric_bd_list = tmp_metric_bd if iter_==0 \
                           else np.vstack([metric_bd_list, tmp_metric_bd]) 
            if iter_==0: 
                ball_traj = tmp_ball_traj
                striker_traj = tmp_striker_traj 
            else: 
                ball_traj.extend(tmp_ball_traj) 
                striker_traj.extend(tmp_striker_traj) 

            # FITNES CRITERIA: parameters that lead to successful trial
            tmp_dict = {'tmp_outcomes': tmp_outcomes, 
                        'tmp_ball_traj':tmp_ball_traj}
            sel_idx = fitness_fn(**tmp_dict)
            new_population = tmp_population[sel_idx,:]

            # Perturb parameters that fit the criteria otherwise random
            if len(new_population):
                _tmp_sz = max(1, int(_sample_size / new_population.shape[0]))
                new_population = np.tile(new_population, (_tmp_sz, 1))
                new_population += np.random.normal(scale=sigma, 
                                                   size=new_population.shape)
            else:
                _rem_sz = max(0, int(k*self.num_samples)-len(param_population))
                new_population = \
                    np.random.uniform(low=low_range, high=high_range, 
                                      size=(min(_sample_size, _rem_sz), 
                                            self.parameter_dims))
                new_population = \
                    self.task.controller_obj.convert_params(uniforms=new_population)
            iter_ += 1
        # Balance the complete data
        if balance:
            param_population, outcome_list,\
            metric_bd_list, \
            ball_traj, striker_traj = \
                self.task.balance_outcomes(param_population, outcome_list, 
                                           metric_bd_list,
                                           ball_traj, striker_traj)   

        trial_outputs = dict(param_original=param_population, 
                             param_embedded=None, 
                             outcomes=outcome_list, 
                             metric_bd=metric_bd_list, 
                             traj_main=ball_traj, 
                             traj_aux=striker_traj)
        return trial_outputs


    def ps_partfilt_success(self, **kwargs):
        """ Applying PF, with fitness being successful outcomes """
        def fitness_fn(tmp_outcomes, **kwargs):
            return np.where(tmp_outcomes>-1)[0]
        return self._ps_partfilt(fitness_fn, **kwargs)


    def ps_partfilt_moved(self, thrsh=0.05, **kwargs):
        """ Applying PF, with fitness being ball moved by threshold """
        def fitness_fn(tmp_ball_traj, **kwargs):
            return np.array([np.linalg.norm(t[0]-t[-1])>thrsh \
                            for t in tmp_ball_traj])
        return self._ps_partfilt(fitness_fn, **kwargs)



###############################################################################
###############################################################################
###############################################################################


class SampleNN(object):


    def ps_nn_uniform(self, k, low, high, 
                      nloop, nloop_ofst, flag_continue_data, **kwargs):
        """ Sample NN weights uniformly between -1 and 1 """

        # Skip used rand if continuing previous experiment
        if flag_continue_data and nloop==nloop_ofst:
            total_samples = int(nloop_ofst*k*self.num_samples)
            total_param = sum([np.prod(pa) for pa in self.parameter_arch])
            np.random.uniform(low=low, high=high, 
                              size=(total_samples, total_param))

        # Make a parameter_dim placeholder
        layer_samples = np.inf * np.ones((int(k*self.num_samples), 
                                          *self.parameter_dims))
        # Fill each layer up to weight matrix shape
        for l, pa in enumerate(self.parameter_arch):
            low_range = low * np.ones((int(k*self.num_samples), *pa))
            high_range = high * np.ones((int(k*self.num_samples), *pa))
            layer_samples[:,:pa[0],:pa[1], l] = \
                np.random.uniform(low=low_range, high=high_range, 
                                  size=(int(k*self.num_samples), *pa))
        return self._eval_population(layer_samples, from_latent=False, 
                                     shuffle=False, **kwargs)


    def ps_nn_normal(self, k, 
                     nloop, nloop_ofst, flag_continue_data, **kwargs):
        """ Sample NN weights from a Gaussian with (0, 0.1) """

        # Skip used rand if continuing previous experiment
        if flag_continue_data and nloop==nloop_ofst:
            total_samples = int(nloop_ofst*k*self.num_samples)
            total_param = sum([np.prod(pa) for pa in self.parameter_arch])
            np.random.normal(scale=0.1, size=(total_samples, total_param))

        # Make a parameter_dim placeholder
        layer_samples = np.inf * np.ones((int(k*self.num_samples), 
                                          *self.parameter_dims))
        # Fill each layer up to weight matrix shape
        for l, pa in enumerate(self.parameter_arch):
            layer_samples[:,:pa[0],:pa[1], l] = \
                np.random.normal(scale=0.1, size=(int(k*self.num_samples), *pa))
        return self._eval_population(layer_samples, from_latent=False, 
                                     shuffle=False, **kwargs)



    def ps_nn_glorot(self, k, 
                     nloop, nloop_ofst, flag_continue_data, **kwargs):
        """ 
            Xavier-Glorot initialisation of weights, tries to make the 
            variance of the layer outputs, equal to the variance of its inputs
        """

        # Skip used rand if continuing previous experiment
        if flag_continue_data and nloop==nloop_ofst:
            total_samples = int(nloop_ofst*k*self.num_samples)
            total_param = sum([np.prod(pa) for pa in self.parameter_arch])
            np.random.normal(scale=0.1, size=(total_samples, total_param))

        # Make a parameter_dim placeholder
        layer_samples = np.inf * np.ones((int(k*self.num_samples), 
                                          *self.parameter_dims))
        # Fill each layer up to weight matrix shape
        for l, pa in enumerate(self.parameter_arch):
            var_scale = np.sqrt(1/np.sum(pa))
            layer_samples[:,:pa[0],:pa[1], l] = \
                np.random.normal(scale=var_scale, 
                                 size=(int(k*self.num_samples), *pa))

        return self._eval_population(layer_samples, from_latent=False, 
                                     shuffle=False, **kwargs)


###############################################################################
###############################################################################
###############################################################################


class DisplacementParamSampling(SampleBasic, SamplePF, SampleNN):
    """ 
        Collection of methods that sample the parameter space 
    """
    pass