  
"""
Author:         Anonymous
Description:
                Data generation using MAP-Elites
                - parameter space
                - latent space
                - mixing strategies
"""


import time
import os
import csv
import logging
import itertools
import inspect
import numpy as np

from functools import partial
from operator import itemgetter
from inspect import stack 

from behaviour_representations.utils.utils import _TAB

logger = logging.getLogger(__name__)




def _iso_dd(x, y, normalize_distance=False):
    """ ISO_DD helper function """
    assert(x.shape == y.shape)
    ninf_idx = x<np.inf
    x_ninf = x[ninf_idx]
    y_ninf = y[ninf_idx]
    sigma_iso = 0.01 
    sigma_line = 0.2
    gauss_iso = np.random.normal(0, sigma_iso, size=len(x_ninf))
    gauss_line = np.random.normal(0, sigma_line)
    direction = (x_ninf - y_ninf)
    if normalize_distance:
        norm = np.linalg.norm(direction)
        direction = direction / norm
    z = x_ninf.copy() + gauss_iso + gauss_line * direction
    new_sample = x.copy()
    new_sample[ninf_idx] = z
    return new_sample


class SampleMAPE(object):
    """ 
        Collection of methods that sample the parameter space using 
        variations of the MAP-E approach.
        - mutation_frac: fraction of parameter dimensions to mutate
        - container_frac: fraction of container cells to sample from
    """

    def _get_sample_recn_error(self, new_evals, embedded,
                               recn_fn=None, embedding_fn=None,
                               **kwargs):
        """ Evaluate recn error per sample if latent space is used """
        if recn_fn is not None and embedding_fn is not None:
            if embedded is not None:
                new_evals['param_embedded'] = embedded
            else:
                new_evals['param_embedded'] = \
                    embedding_fn(new_evals['param_original'])

            # Get reconstruction errors on the fly
            param_recn = recn_fn(new_evals['param_embedded'])
            param_orig = new_evals['param_original']
            tmp_orig = param_orig.reshape(param_orig.shape[0], -1)
            tmp_recn = param_recn.reshape(param_recn.shape[0], -1)
            ninf_idx = np.where(tmp_recn[0]<np.inf)[0]
            errors = tmp_orig[:, ninf_idx] - tmp_recn[:, ninf_idx]
            new_evals['recn_error'] = np.linalg.norm(errors, axis=1)
        return new_evals


    def _add_samples_to_grid(self, new_evals, use_fit):
        """ 
            Assign new individual if their bd is not in grid, 
            or they have larger fitness. If equal decide randomly.
            If not using fitness, assign randomly new/old.
        """
        # List of used indices
        nbd_list = []
        idx_list = []
        # Loop over new evaluations
        for idx, new_bd in enumerate(new_evals['metric_bd']):
            nbd = np.argmax(new_bd)
            # Evaluate assignment criteria
            if nbd in self.main_grid.keys():
                old_fit = np.max(self.main_grid[nbd]['outcomes'], 
                                 axis=0)[use_fit]
                new_fit = new_evals['outcomes'][idx][use_fit]
                if old_fit == new_fit:
                    assign_flag = np.random.choice([True, False])
                elif old_fit < new_fit:
                    assign_flag = True
                else:
                    assign_flag = False
            else:
                assign_flag = True
            # Enter values in main bd grid if criteria are met
            if assign_flag:
                nbd_list.append(nbd)
                idx_list.append(idx)
                self.main_grid[nbd] = {}
                for key_ in new_evals.keys():
                    if 'traj' in key_:
                        self.main_grid[nbd][key_] = [new_evals[key_][idx]]
                    elif 'embedded' in key_ and new_evals[key_] is None:
                        self.main_grid[nbd][key_] = new_evals[key_]
                    else: 
                        _data = new_evals[key_][idx]
                        if 'param_original' in key_:
                            _data = _data.reshape(-1, *self.parameter_dims)
                        if 'outcomes' in key_:
                            _data = _data.reshape(-1, 2)
                        self.main_grid[nbd][key_] = _data
        return nbd_list, idx_list


    def _get_mape_output(self, new_evals, new_bds):
        """ Extract data from grid to pass as results """
        mape_output = {}
        for k in new_evals.keys():
            # vals_list = [v[k] for v in self.main_grid.values()]
            vals_list = [self.main_grid[b][k] for b in new_bds]
            if len(new_bds)==0:
                mape_output[k] = None
            else:
                if 'traj' in k:
                    mape_output[k] = \
                        list(itertools.chain.from_iterable(vals_list))
                elif 'recn_error'==k:
                    mape_output[k] = np.hstack(vals_list)
                else:
                    mape_output[k] = np.vstack(vals_list) 
        if 'param_embedded' not in mape_output.keys():
            mape_output['param_embedded'] = None
 
        # Log data statistics
        logging.info("MAP-E '{}' found {} new behaviours "
                     "({} successful)".format(stack()[1][3], len(new_bds), 
                        sum([(self.main_grid[v]['outcomes'][:,0]>-1).any() \
                                        for v in new_bds])))
        return mape_output


    def _log_info(self, nloop, it_, ratio_list, iter_info):
        """ Log profiling for parts of MAPE iteration """
        if not it_ % 10:
            avg_ratio = np.vstack(ratio_list[-10:])
            assert np.min(avg_ratio, axis=0)[1] > 0
            avg_ratio = np.mean(avg_ratio[:,0]/avg_ratio[:,1], axis=0)
            info = "MAP-E PROFILING nloop: {}; "\
                   "iter: {}; avg ps/nsmp {:.4}".format(nloop, it_, avg_ratio)
            for k in sorted(iter_info.keys()):
                val = iter_info[k]
                indent = '' if k=='reconstruction' else '\t'
                info += "\n{}===== {}:{}{:4} total;  ({:.4})".format(
                        _TAB, k.upper(), indent+'\t', val['len'], val['time'])
            logging.info(info+'\n')


    def _log_save(self, nloop, it_, ratio_list, new_bds):
        """ Log final results of MAPE search """
        if not (it_+1)%100: #and 'ps_mape' in self.exploration_type:
            logging.info("MAP-E iter {} done [{} new]".format(
                            it_+1, len(new_bds))) # len(self.main_grid.keys())))
        # Save MAPE exploration progress
        bd_outcomes = [v['outcomes'] for v in self.main_grid.values()]
        bd_outcomes = np.vstack(bd_outcomes)
        # mape_data = [nloop]+[it_]+[self.num_samples] \
        #              + [list(self.main_grid.keys())] \
        #              + [list(bd_outcomes[:, 1])] \
        #              + [list(bd_outcomes[:, 0])] \
        #              + [ratio_list[-1]]
        mape_data = [nloop]+[it_]+[self.num_samples]+[0] \
                     + [len(self.main_grid.keys())] \
                     + [max(bd_outcomes[:, 1])] \
                     + [sum(bd_outcomes[:, 0]==0)] \
                     + [ratio_list[-1][0]/ratio_list[-1][1]]

        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
        filepath = '{}/ref_data_{}_{}.csv'.format(self.dirname, 
                                                  self.ctrl_name,
                                                  self.exploration_normal)
        with open(filepath, 'a') as outfile: 
            writer = csv.writer(outfile) 
            writer.writerows([mape_data])


################################################################################


    def _mape_search(self, nloop, nloop_ofst, flag_continue_data,
                     param_original, param_embedded, 
                     outcomes, recn_error, metric_bd, traj_data,
                     ucb_size=2,
                     **kwargs):
        """ 
            Core MAPE selection and perturbation process
        """

        # Extract MAPE parameters
        sigma = self.mape_dict['sigma']
        niter = self.mape_dict['niter']
        use_fit = self.mape_dict['use_fit']
        # Apply num_iter decay
        if self.mape_dict['decay']:
            self.mape_dict['niter'] *= self.mape_dict['decay']

        # Load data generated in loop 0 by random search
        if (nloop-nloop_ofst)==1 or (nloop==nloop_ofst and flag_continue_data):
            self.main_grid = {}
            # initialize counters for UCB
            self.ucb_a_vals = np.zeros(ucb_size)
            self.ucb_a_freq = np.ones(ucb_size, dtype=np.int)
            # if metric_bd is not None
            prev_data = dict(param_original=param_original,
                             param_embedded=param_embedded,
                             outcomes=outcomes,
                             recn_error=recn_error,
                             metric_bd=metric_bd,
                             traj_main=traj_data['main'],
                             traj_aux=traj_data['aux'])

            unique_bds = np.unique(metric_bd, axis=0) 
            unique_bds_num = np.argmax(unique_bds, axis=1) 
            container_idxs = np.arange(len(unique_bds))
            for cont_idx_ in container_idxs:
                bd_num = unique_bds_num[cont_idx_]
                sel_idx = np.where(
                          (metric_bd==unique_bds[cont_idx_]).all(axis=1))[0]
                sel_idx = [sel_idx[0]]
                self.main_grid[bd_num] = {}
                for k in prev_data.keys():
                    if prev_data[k] is None:
                        continue
                    if 'traj' in k:
                        if len(sel_idx)==1: 
                            self.main_grid[bd_num][k] = \
                                [prev_data[k][sel_idx[0]]]
                        else:
                            self.main_grid[bd_num][k] = \
                                list(itemgetter(*sel_idx)(prev_data[k])) 
                    # elif 'outcomes' in k and len(sel_idx)==1: 
                    #     self.main_grid[bd_num][k] = prev_data[k][sel_idx][0]
                    else: 
                        self.main_grid[bd_num][k] = prev_data[k][sel_idx]
                        
        # Prepare vars
        if 'recn_error' in list(self.main_grid.values())[0].keys():
            list_recn_error = [x['recn_error'] for x in self.main_grid.values()]
        else:
            list_recn_error = []
        iter_info = {'sampling':{}, 'evaluation':{}, 
                     'reconstruction':{}, 'assignment': {}}
        num_old_bds = 0
        new_bds = []
        ratio_list = []
        # Run MAPE for num_iter
        for it_ in range(0, niter):
            # Sample population new to evaluate
            tt1 = time.time()
            (new_population, embedded), ratio_ps_nsmp = self._fn_sampling_mape(
                                            population_dict=self.main_grid, 
                                            list_recn_error=list_recn_error,
                                            nloop=nloop, niter=it_,   
                                            nbd_diff=len(new_bds)-num_old_bds,
                                            ucb_a_vals=self.ucb_a_vals,
                                            ucb_a_freq=self.ucb_a_freq,
                                            sigma=sigma, **kwargs)
            ratio_list.append(list(ratio_ps_nsmp))
            #
            iter_info['sampling']['len'] = len(self.main_grid)
            iter_info['sampling']['time'] = time.time()-tt1

            # Evaluate population and use outputs
            tt2 = time.time()
            new_evals = self._eval_population(
                                        next_param_population=new_population,
                                        from_latent=False)
            #
            iter_info['evaluation']['len'] = len(new_evals['param_original'])
            iter_info['evaluation']['time'] = time.time()-tt2

            # Evaluate recn error per sample if latent space is used
            tt3 = time.time()
            new_evals = self._get_sample_recn_error(new_evals, embedded, 
                                                    **kwargs)
            #
            iter_info['reconstruction']['len'] = len(new_evals['param_original'])
            iter_info['reconstruction']['time'] = time.time()-tt3

            # Add new samples to main bd grid
            tt4 = time.time()
            num_old_bds = len(new_bds)
            nbd_list, idx_list = self._add_samples_to_grid(new_evals, use_fit)
            new_bds.extend(nbd_list)
            #
            iter_info['assignment']['len'] = len(new_bds)
            iter_info['assignment']['time'] = time.time()-tt4

            # Increase strategy - new behaviour counter
            new_a_vals = np.zeros(ucb_size)
            if ucb_size==2:
                s_idx = 1-ratio_ps_nsmp[0]//ratio_ps_nsmp[1]
            else:
                s_idx = ratio_ps_nsmp[0]

            new_a_vals[s_idx] = len(nbd_list)/self.num_samples

            self.ucb_a_vals = self.ucb_a_vals + \
                                (new_a_vals - self.ucb_a_vals)/self.ucb_a_freq
            self.ucb_a_freq[s_idx] += 1 

            # Log MAPE exploration progress
            self._log_info(nloop, it_, ratio_list, iter_info)
            self._log_save(nloop, it_, ratio_list, new_bds)


        # Return only new discoveries
        mape_output = self._get_mape_output(new_evals, new_bds)

        return mape_output


################################################################################


    def _fn_sampling_mape(self, population_dict, error_search=False, **kwargs):
        """ Samples and perturbs a population from the grid """

        new_population = []
        new_recn_error = []

        # Selection of individuals - based on reconstruction error or randomly 
        if error_search and len(population_dict)>self.num_samples:
            container_idxs = sorted(
                                population_dict.keys(), 
                                key=lambda x: population_dict[x]['recn_error'], 
                                reverse=True)
            container_samples = container_idxs[:self.num_samples]
        else:
            container_idxs = list(population_dict.keys())
            replace_cond = len(container_idxs)<self.num_samples
            container_samples = np.random.choice(container_idxs, 
                                                 size=self.num_samples, 
                                                 replace=replace_cond)
        # Extract parameters of selected individuals
        for cont_idx_ in container_samples:
            v = population_dict[cont_idx_]
            new_population.append(v['param_original'])
            if 'recn_error' in v.keys(): new_recn_error.append(v['recn_error'])
        new_population = np.vstack(new_population)
        new_recn_error = np.array(new_recn_error)

        # Perturbation of individuals - in latent or original parameter space
        return self._parameter_mutation(new_population, 
                                        new_recn_error, **kwargs)



    def _parameter_mutation(self, new_population, new_recn_error, niter,
                                  list_recn_error, nbd_diff, sigma,
                                  ucb_a_vals, ucb_a_freq, **kwargs):
        """ Applies a perturbation to the selected individuals """
        
        nsmp = new_population.shape[0]
        
        # Get MAPE variant ### TODO - split this into functions / operators ???
        mape_fn = inspect.stack()[3].function

        if mape_fn == 'ps_mape':
            return self._mutate_ps(new_population, sigma=sigma, **kwargs), \
                   (nsmp, nsmp)

        elif 'ps_mape_directional' in mape_fn:
            return self._mutate_iso_line(new_population, **kwargs), (nsmp, nsmp)

        elif 'ls_mape_dde' in mape_fn:
            # If first iteration do coin-flip otherwise MAB-UCB
            if niter==0:
                a_idx = np.random.choice(len(ucb_a_vals))
            else:
                a_idx = np.argmax(ucb_a_vals + \
                              np.sqrt(2*np.log(sum(ucb_a_freq)) / ucb_a_freq))
            a_sel = self.ucb_actions[a_idx]
            # Select ratio based on UCB criteria
            idx_p1 = int(a_sel[0]*nsmp)
            idx_p2 = idx_p1 + int(a_sel[1]*nsmp)
            idx_p3 = idx_p2 + int(a_sel[2]*nsmp)
            assert idx_p3 <= nsmp
            p1, e1 = self._mutate_recn_xover(new_population[:idx_p1], **kwargs)
            p2, e2 = self._mutate_iso_line(new_population[idx_p1:idx_p2], 
                                           **kwargs)
            p3, e3 = self._mutate_ps(new_population[idx_p2:], 
                                     sigma=self.sigma_iso, **kwargs)
            return (np.vstack([p1, p2, p3]), np.vstack([e1, e2, e3])), \
                   (a_idx, len(ucb_a_vals))

        elif mape_fn == 'ls_mape_mix_fixed':
            idx_mid = nsmp//2
            p1, e1 = self._mutate_ps(new_population[:idx_mid], sigma=sigma, 
                                     **kwargs)
            p2, e2 = self._mutate_ls(new_population[idx_mid:], sigma=sigma, 
                                     **kwargs)
            ratio_ps_nsmp = (idx_mid, nsmp)
            return (np.vstack([p1, p2]), np.vstack([e1, e2])), (idx_mid, nsmp)

        elif mape_fn == 'ls_mape_mix_region':
            rthrs = 0 if list_recn_error is None else np.mean(list_recn_error) # np.mean(new_recn_error)
            idx_high = np.where(new_recn_error>rthrs)[0]
            if len(idx_high)==0: idx_high = np.array([0])
            if len(idx_high)==nsmp: idx_high = idx_high[:-1]
            idx_low = np.setdiff1d(np.arange(nsmp), idx_high)
            p1, e1 = self._mutate_ps(new_population[idx_high], sigma=sigma, 
                                     **kwargs)
            p2, e2 = self._mutate_ls(new_population[idx_low], sigma=sigma, 
                                     **kwargs)
            ratio_ps_nsmp = (len(idx_high), nsmp)
            return (np.vstack([p1, p2]), np.vstack([e1, e2])), ratio_ps_nsmp

        elif mape_fn == 'ls_mape_mix_improvement':
            # If first iteration do coin-flip
            if niter==0:
                nbd_diff = 0 if np.random.random()<0.5 else 1
            # Select based on previous iterations discoveries
            if nbd_diff == 0:
                return self._mutate_ps(new_population, sigma=sigma, **kwargs), \
                       (nsmp, nsmp)
            else:
                return self._mutate_ls(new_population, sigma=sigma, **kwargs), \
                       (0, nsmp)


        elif mape_fn == 'ls_mape_mix_ucb':
            # If first iteration do coin-flip otherwise MAB-UCB
            if niter==0:
                a_sel = 0 if np.random.random()<0.5 else 1
            else:
                a_sel = np.argmax(ucb_a_vals + \
                              np.sqrt(2*np.log(sum(ucb_a_freq)) / ucb_a_freq))
            # Select based on UCB criteria
            if a_sel == 0:
                return self._mutate_ps(new_population, sigma=sigma, **kwargs), \
                       (nsmp, nsmp)
            else:
                return self._mutate_ls(new_population, sigma=sigma, **kwargs), \
                        (0, nsmp)

        elif 'ls_mape_' in mape_fn:
            return self._mutate_ls(new_population, sigma=sigma, **kwargs), \
                   (0, nsmp)

        else:
            raise ValueError('No output for \'{}\' strategy!'.format(mape_fn))



    def _mutate_ps(self, new_population, sigma, param_stats, embedding_fn,
                         mutation_frac=1, **kwargs):
        """ Applies a perturbation in the original (high-dim) space """
        nsmp = new_population.shape[0]

        if mutation_frac < 1:
            # Mutate only along some of the dimensions
            dim_param = new_population.shape[1]
            mutation = np.zeros_like(new_population)
            sample_sz = max(1, int(mutation_frac * dim_param))
            mutate_idxs = np.random.choice(np.arange(dim_param),
                                           size=sample_sz, replace=False)
            mutation_mean = np.zeros(dim_param)
            mutation_var = sigma*np.ones(dim_param)
            mutation[:, mutate_idxs] = \
                np.random.normal(loc=mutation_mean[mutate_idxs], 
                                 scale=mutation_var[mutate_idxs], 
                                 size=(nsmp, len(mutate_idxs)))
        else:
            # Mutate all the dimensions
            mutation = np.random.normal(loc=0, scale=sigma, 
                                        size=new_population.shape)
            # mutation = np.random.normal(loc=0, scale=param_stats['std'])
            # mutation = np.random.multivariate_normal(
            #                         mean=np.zeros(dim_param),
            #                         cov=sigma*param_stats['cov'],
            #                         size=nsmp)
        new_population += mutation
        # Clip to range
        if self.srange is not None:
            n_idx = new_population<np.inf
            new_population[n_idx] = np.clip(new_population[n_idx], *self.srange)
        # Get Latent space if applicable
        new_latent = None if embedding_fn is None \
                          else embedding_fn(new_population)
        return new_population, new_latent



    def _mutate_ls(self, new_population, sigma, embedding_fn, recn_fn, 
                         jacobian_fn, latent_stats, **kwargs):
        """ Applies a perturbation in the latent (low-dim) space """
        nsmp = new_population.shape[0]

        new_latent = embedding_fn(new_population)
        dim_orig = np.count_nonzero(new_population[0]<np.inf)  # hack, good
        # assert dim_orig == sum([np.prod(pa) for pa in self.parameter_arch]), \
        #         "Some of the dimensions are inf!"
        # dim_orig = sum([np.prod(pa) for pa in self.parameter_arch])
        dim_lat = new_latent.shape[1]

        if jacobian_fn is not None:
            j_list = jacobian_fn(new_latent)
            sigma_ps = np.diag(sigma * np.ones(dim_orig))
            
            # scale - transpose
            sigma_ls = [np.dot(np.dot(jj.T, sigma_ps), jj) for jj in j_list]

            new_latent = np.vstack([np.random.multivariate_normal(
                                    mean=m, cov=c) for m, c in zip(new_latent, 
                                                                   sigma_ls)])
        else:
            new_latent = np.random.normal(loc=new_latent, 
                                          scale=latent_stats['std'])

        # Reconstruct and return population
        new_population = recn_fn(new_latent)

        # Clip to range
        if self.srange is not None:
            n_idx = new_population<np.inf
            new_population[n_idx] = np.clip(new_population[n_idx], *self.srange)

        return new_population, new_latent


    def _iso_line(self, x, y):
        """ ISO_DD helper function """
        assert(x.shape == y.shape)
        ninf_idx = x<np.inf
        x_ninf = x[ninf_idx]
        y_ninf = y[ninf_idx]
        gauss_iso = np.random.normal(0, self.sigma_iso, size=len(x_ninf))
        gauss_line = np.random.normal(0, self.sigma_line)
        direction = (x_ninf - y_ninf)
        if not self.use_distance:
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 0 else 1
            z = x_ninf.copy() + gauss_line * direction
        else:
            z = x_ninf.copy() + gauss_iso + gauss_line * direction
        new_sample = x.copy()
        new_sample[ninf_idx] = z
        return new_sample


    def _mutate_iso_line(self, new_population, param_stats, latent_stats,
                            embedding_fn, **kwargs):
        """ Applies a perturbation in the original (high-dim) space """
        nsmp = new_population.shape[0]
        if nsmp==0: return np.empty((0,*new_population.shape[1:])), \
                           np.empty((0, len(latent_stats['mu'])))

        # Generate #nsmp different pairs for ISO_DD
        x_list = new_population.copy()
        y_idx = np.arange(nsmp)
        np.random.shuffle(y_idx)
        cnt = 0
        while (y_idx == np.arange(nsmp)).any():
            same_idx = np.where(y_idx == np.arange(nsmp))[0]
            swap_idx = (same_idx+np.random.randint(1, nsmp, size=len(same_idx)))
            swap_idx = swap_idx % nsmp
            tmp = y_idx[same_idx]
            y_idx[same_idx] = y_idx[swap_idx]
            y_idx[swap_idx] = tmp
            cnt += 1
            if cnt > nsmp:
                break
        y_list = new_population[y_idx]

        # Generate new perturbed samples by applying ISO_DD
        res = list(map(self._iso_line, y_list, x_list))
        new_perturbed = np.stack(res, axis=0)

        # Clip to range
        if self.srange is not None:
            n_idx = new_perturbed<np.inf
            new_perturbed[n_idx] = np.clip(new_perturbed[n_idx], *self.srange)
        # Get Latent space if applicable
        new_latent = None if embedding_fn is None \
                          else embedding_fn(new_perturbed)
        return new_perturbed, new_latent



    def _mutate_recn_xover(self, new_population, embedding_fn, recn_fn, 
                           latent_stats, **kwargs):
        """ Applies a perturbation in the latent (low-dim) space """
        new_latent = embedding_fn(new_population)
        recn_population = recn_fn(new_latent)

        final_population = 0.5*(new_population + recn_population)

        # Clip to range
        if self.srange is not None:
            n_idx = final_population<np.inf
            final_population[n_idx] = np.clip(final_population[n_idx], 
                                              *self.srange)
        return final_population, new_latent




################################################################################

    # Directional variation operator, introduced in:
    # Vassiliades and Mouret, "Discovering the Elite Hypervolume by 
    # Leveraging Interspecies Correlation", GECCO 2018.

    # Implementation:
    # https://github.com/resibots/pymap_elites/blob/edff034f06697c4906fc60154ceac5d716c1b828/map_elites/common.py#L145

    def ps_mape_directional(self, embedding_fn, recn_fn, **kwargs):
        """ 
            Directional: Iso+DDLine
        """
        self.use_distance = True
        self.sigma_iso = 0.01 
        self.sigma_line = 0.2
        return self._mape_search(use_distance=True, 
                                 embedding_fn=None, recn_fn=None, **kwargs)


    def ps_mape_directional_norm(self, embedding_fn, recn_fn, **kwargs):
        """ 
            Directional: Line
        """
        self.use_distance = False
        self.sigma_iso = 0.01 
        self.sigma_line = 0.2
        return self._mape_search(use_distance=False, 
                                 embedding_fn=None, recn_fn=None, **kwargs)


    # Data-Driven Encoding (DDE) variation operator, introduced in:
    # Gaier, Asteroth and Mouret, "Discovering Representations for Black-box 
    # Optimization", GECCO 2020.

    def ls_mape_dde(self, **kwargs):
        """ 
            Data-Driven Encoding
        """
        self.use_distance = True
        self.sigma_iso = 0.003
        self.sigma_line = 0.1
        self.ucb_actions = [[0.00, 0.00, 1.00], [0.25, 0.00, 0.75],
                            [0.50, 0.00, 0.50], [0.75, 0.00, 0.25],
                            [1.00, 0.00, 0.00], [0.00, 0.25, 0.75],
                            [0.00, 0.50, 0.50], [0.00, 0.75, 0.25],
                            [0.00, 1.00, 0.00]]
        return self._mape_search(ucb_size=len(self.ucb_actions), **kwargs)



    def ps_mape(self, embedding_fn, recn_fn, **kwargs):
        """ Keep all samples with priority to succ ones """
        return self._mape_search(embedding_fn=None, recn_fn=None, **kwargs)



    def ls_mape_standard(self, jacobian_fn, **kwargs):
        """ Keep all samples with priority to succ ones """
        return self._mape_search(jacobian_fn=None, error_search=False, **kwargs)


    def ls_mape_recn(self, jacobian_fn, **kwargs):
        """ Keep all samples with priority to succ ones """
        return self._mape_search(jacobian_fn=None, error_search=True, **kwargs)


    def ls_mape_jacobian(self, **kwargs):
        """ Keep all samples with priority to succ ones """
        return self._mape_search(error_search=False, **kwargs)


    def ls_mape_full(self, **kwargs):
        """ Keep all samples with priority to succ ones """
        return self._mape_search(error_search=True, **kwargs)




    def ls_mape_mix_fixed(self, **kwargs):
        """   """
        return self._mape_search(error_search=False, **kwargs)


    def ls_mape_mix_region(self, **kwargs):
        """   """
        return self._mape_search(error_search=False, **kwargs)


    def ls_mape_mix_improvement(self, **kwargs):
        """   """
        return self._mape_search(error_search=False, **kwargs)


    def ls_mape_mix_ucb(self, **kwargs):
        """   """
        return self._mape_search(error_search=False, ucb_size=2, **kwargs)




    def ls_mape_mix_fixed_standard(self, jacobian_fn, **kwargs):
        """   """
        return self._mape_search(jacobian_fn=None, error_search=False, **kwargs)


    def ls_mape_mix_region_standard(self, jacobian_fn, **kwargs):
        """   """
        return self._mape_search(jacobian_fn=None, error_search=False, **kwargs)


    def ls_mape_mix_improvement_standard(self, jacobian_fn, **kwargs):
        """   """
        return self._mape_search(jacobian_fn=None, error_search=False, **kwargs)

