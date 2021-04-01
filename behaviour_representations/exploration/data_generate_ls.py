
"""
Author:         Anonymous
Description:
                Data generation using latent space clusters

"""


import logging
import itertools
import numpy as np

from operator import itemgetter
from scipy.stats import multivariate_normal




class SampleParticles(object):

    
    def _get_cluster_idx_sz(self, cluster_to_data_dict, mode='all'):
        num_clusters = len(cluster_to_data_dict)
        if mode == 'all':
            cluster_idxs = range(num_clusters)
        elif mode == 'successful':
            cluster_idxs = range(1, num_clusters)
        elif mode == 'failed':
            cluster_idxs = [0]
        else:
            raise ValueError("Selection not defined")
        cluster_sz = {k:len(v) for (k,v) in cluster_to_data_dict.items() \
                      if k in cluster_idxs}
        logging.info("{} cluster sizes - {}".format(mode.upper(), cluster_sz))
        return cluster_idxs, cluster_sz


    def _sample_particles(self, noise, param_embedded, cluster_object,
                          **kwargs):
        """ 
            Collection of methods that sample the latent space using 
            particles uniformly sampled from clusters.
        """

        cluster_idxs, _ = \
            self._get_cluster_idx_sz(cluster_object.cluster_to_datapoints)
        # Determine number of samples per cluster 
        ### TODO: proportional to sizes?
        num_samples_per_cluster = max(1, int(np.floor(
                                    self.num_samples / len(cluster_idxs))))
        logging.info("Sampling {} particles from clusters.".format(
                     num_samples_per_cluster))

        # Sample existing datapoints from clusters, 
        # and add a small noise  proportional to the cluster_std
        next_param_population = []
        for i, cidx in enumerate(cluster_idxs):
            cluster_data = cluster_object.cluster_to_datapoints[cidx]
            datapoint_idxs = np.random.choice(
                cluster_data, 
                size=num_samples_per_cluster, 
                replace=num_samples_per_cluster>len(cluster_data))
            sampled_datapoints = param_embedded[datapoint_idxs] 
            sampled_datapoints = sampled_datapoints + noise * \
                cluster_object.cluster_to_centroids[cidx][0].dev * \
                np.random.randn(*sampled_datapoints.shape)            
            next_param_population = sampled_datapoints\
                                    if i==0 \
                                    else np.vstack([next_param_population,
                                                    sampled_datapoints])
        return self._eval_population(next_param_population, from_latent=True,
                                     shuffle=True, **kwargs)

    def _sample_clusters(self, cluster_idxs, cluster_object, k,
                             outcomes, **kwargs):
        """ 
            Collection of methods that sample the latent space using 
            cluster info.
        """
        sampled_class_idx = cluster_object.cluster_to_class[cluster_idxs]
        mean_embedded =\
            np.array(cluster_object.cluster_to_centroids[cluster_idxs][0].point)
        cov_embedded = \
            np.array(cluster_object.cluster_to_centroids[cluster_idxs][0].dev)
        # next_param_std = \
        #   np.minimum(np.array(embedded_centroids_std[smallest_cluster]), 1.0)
        _local_clstr = cluster_idxs - \
                       np.where(cluster_object.cluster_to_class[cluster_idxs] \
                                == cluster_object.cluster_to_class)[0][0]
        logging.info("Sampling: class {} --> "
                     "cluster {} ({});"
                     "\n{}\t- cluster mean: {}\n{}\t- cluster std: {}".format(
                      np.unique(outcomes)[sampled_class_idx],
                      _local_clstr, cluster_idxs,
                      _TAB, mean_embedded, _TAB, cov_embedded))
        # Sample behaviour population from the cluster with appropriate cov
        if isinstance(cov_embedded, float):
            cov = cov_embedded * np.eye(len(mean_embedded))
        elif isinstance(cov_embedded, np.ndarray) \
            and len(cov_embedded.shape) == 1:
            cov = np.diag(cov_embedded)
        elif isinstance(cov_embedded, np.ndarray) \
            and len(cov_embedded.shape) == 2:
            cov = cov_embedded
        else:
            raise ValueError("Covariance matrix is invalid. "
                             "({})".format(type(cov_embedded)))
        sampled_population = np.random.multivariate_normal(
                                mean_embedded, cov, size=(k*self.num_samples))
        return self._eval_population(sampled_population, from_latent=True,
                                     shuffle=True, **kwargs)
   

    def ls_cluster_rand(self, cluster_object, **kwargs):
        """ Sample normally from a random clusters. """
        cluster_idxs, _ = \
            self._get_cluster_idx_sz(cluster_object.cluster_to_datapoints)
        sampled_cluster_idx = np.random.choice(cluster_idxs)
        return self._sample_clusters(sampled_cluster_idx, cluster_object,
                                     **kwargs)


    def ls_cluster_min(self, cluster_object, **kwargs):
        """ Sample normally from cluster with least datapoints """
        _, cluster_sz = \
            self._get_cluster_idx_sz(cluster_object.cluster_to_datapoints)
        sampled_cluster_idx = min(cluster_sz, key=cluster_sz.get)
        return self._sample_clusters(sampled_cluster_idx, cluster_object,
                                     **kwargs)


    def ls_particles_successful(self, k, outcomes, param_embedded, **kwargs):
        """ Sample from the successful parameters. No double sampling. """
        succ_idxs = np.where(outcomes>-1)[0]
        datapoint_idxs = np.random.choice(succ_idxs, replace=False, 
                                          size=min(succ_idxs.shape[0], 
                                                   k*self.num_samples))
        sampled_population = param_embedded[datapoint_idxs, :] 
        return self._eval_population(sampled_population, 
                        from_latent=True, shuffle=True, **kwargs)


    def ls_particles_exact(self, **kwargs):
        """ Sample exact particles from clusters """
        return self._sample_particles(noise=0, **kwargs)


    def ls_particles_rand(self, **kwargs):
        """ Sample particles from clusters and add noise"""
        return self._sample_particles(noise=0.1, **kwargs)


###############################################################################
###############################################################################
###############################################################################


class SampleOutside(object):


    def ls_outside_uniform(self, cluster_object, out_std=1., in_std=1, 
                           sz_iter_max=1000, accept_trsh=None, **kwargs):
        # Get all cluster centroids and devs
        ccent = np.vstack([cc[0].point for cc \
                           in cluster_object.cluster_to_centroids.values()])
        cdevs = np.vstack([cc[0].dev for cc \
                           in cluster_object.cluster_to_centroids.values()])
        ldim = ccent.shape[1]
        # Get multivariate normal functions of each cluster
        mv_objs = [multivariate_normal(mean=c, cov=d) for c,d \
                                                     in zip(ccent, cdevs)]  
        # Get sampling ranges based on cluster centres and std devs
        cmins, cmaxs = np.min(ccent, axis=0), np.max(ccent, axis=0)
        dmaxs_low = cdevs[np.argmin(ccent, axis=0), np.arange(ldim)]
        dmaxs_up = cdevs[np.argmax(ccent, axis=0), np.arange(ldim)]
        lower_bound = cmins - dmaxs_low * out_std
        upper_bound = cmaxs + dmaxs_up * out_std
        # Acceptance thrsh by default 1 standard deviation from mean
        if accept_trsh is None:
            _mv = multivariate_normal(mean=np.zeros(ldim))
            accept_trsh = _mv.pdf(in_std*np.ones(ldim))/_mv.pdf(np.zeros(ldim))
        # Generate samples outside dev
        smpl_accepted = []
        nsamp = 0
        sz_iter = min(sz_iter_max, 10*self.num_samples)
        while nsamp < self.num_samples:
            smpl_uni = np.random.uniform(low=lower_bound, high=upper_bound,
                                         size=(sz_iter, ldim))
            # calculate likelihood of sample coming from the clusters
            smpl_lkhs = [mv.pdf(smpl_uni)/mv.pdf(mv.mean) for mv in mv_objs]  
            smpl_lkhs = np.vstack(smpl_lkhs).T
            # accept if more than 1 standard deviation away from all clusters
            accepted_idx = np.where((smpl_lkhs < accept_trsh).all(axis=1))[0]
            smpl_accepted.append(smpl_uni[accepted_idx, :])
            nsamp += len(accepted_idx)
        smpl_accepted = np.vstack(smpl_accepted)
        smpl_accepted = smpl_accepted[:self.num_samples,:]
        return self._eval_population(smpl_accepted, 
                                    from_latent=True, shuffle=True, **kwargs)


    def ls_outside_stds(self, cluster_object, out_std=2., in_std=1, 
                           sz_iter_max=1000, accept_trsh=None, **kwargs):
        # Get all cluster centroids and devs
        ccent = np.vstack([cc[0].point for cc \
                           in cluster_object.cluster_to_centroids.values()])
        cdevs = np.vstack([out_std * cc[0].dev for cc \
                           in cluster_object.cluster_to_centroids.values()])
        ldim = ccent.shape[1]
        # Get multivariate normal functions of each cluster
        mv_objs = [multivariate_normal(mean=c, cov=d) for c,d \
                                                     in zip(ccent, cdevs)]  
        # Acceptance thrsh by default 1 standard deviation from mean
        if accept_trsh is None:
            _mv = multivariate_normal(mean=np.zeros(ldim))
            accept_trsh = _mv.pdf(in_std*np.ones(ldim))/_mv.pdf(np.zeros(ldim))
        # Generate samples outside dev
        smpl_accepted = []
        nsamp = 0
        sz_iter = min(sz_iter_max, 10*self.num_samples)
        while nsamp < self.num_samples:
            sz_smpl = max(1, int(sz_iter/len(mv_objs)))
            smpl_stds = np.vstack([mv.rvs(sz_smpl) for mv in mv_objs]) 
            # calculate likelihood of sample coming from the clusters
            smpl_lkhs = [mv.pdf(smpl_stds)/mv.pdf(mv.mean) for mv in mv_objs]  
            smpl_lkhs = np.vstack(smpl_lkhs).T
            # accept if more than 1 standard deviation away from all clusters
            accepted_idx = np.where((smpl_lkhs < accept_trsh).all(axis=1))[0]
            # add samples
            smpl_accepted.append(smpl_stds[accepted_idx, :])
            nsamp += len(accepted_idx)
        smpl_accepted = np.vstack(smpl_accepted)
        smpl_accepted = smpl_accepted[:self.num_samples,:]
        return self._eval_population(smpl_accepted, 
                                    from_latent=True, shuffle=True, **kwargs)


    def ls_outside_w_outcome(self, cluster_object, out_brach_fn, 
                             out_std=2.0, in_std=0.5, 
                             sz_iter_max=1000, accept_trsh=None, **kwargs):
        # Get all cluster centroids and devs
        ccent = np.vstack([cc[0].point for cc \
                           in cluster_object.cluster_to_centroids.values()])
        cdevs = np.vstack([out_std * cc[0].dev for cc \
                           in cluster_object.cluster_to_centroids.values()])
        ldim = ccent.shape[1]
        # Get multivariate normal functions of each cluster
        mv_objs = [multivariate_normal(mean=c, cov=d) for c,d \
                                                     in zip(ccent, cdevs)]  
        # Acceptance thrsh by default 1 standard deviation from mean
        if accept_trsh is None:
            _mv = multivariate_normal(mean=np.zeros(ldim))
            accept_trsh = _mv.pdf(in_std*np.ones(ldim))/_mv.pdf(np.zeros(ldim))
        # Generate samples outside dev
        smpl_accepted = []
        nsamp = 0
        sz_iter = min(sz_iter_max, 10*self.num_samples)
        while nsamp < self.num_samples:
            sz_smpl = max(1, int(sz_iter/len(mv_objs)))
            smpl_stds = np.vstack([mv.rvs(sz_smpl) for mv in mv_objs]) 
         
            # calculate likelihood of sample coming from the clusters
            smpl_lkhs = [mv.pdf(smpl_stds)/mv.pdf(mv.mean) for mv in mv_objs]  
            smpl_lkhs = np.vstack(smpl_lkhs).T
            # accept if more than 1 standard deviation away from all clusters
            accepted_idx = np.where((smpl_lkhs < accept_trsh).all(axis=1))[0]
            smpl_stds = smpl_stds[accepted_idx, :]
            # evaluate the outcomes of selected samples and filter out
            outcome_lkhs = out_brach_fn(smpl_stds)[:, -1]
            _rnd_draw = np.random.uniform(low=0.2, high=0.8)
            # _tier1 = np.where(outcome_lkhs >= 0.8)[0]
            # _tier2 = np.where(outcome_lkhs >= _rnd_draw)[0]
            # accepted_idx = np.concatenate(_tier1, _tier2)
            accepted_idx = np.where(outcome_lkhs >= _rnd_draw)[0]
            smpl_stds = smpl_stds[accepted_idx, :]
            # add samples
            smpl_accepted.append(smpl_stds)
            nsamp += len(accepted_idx)
        smpl_accepted = np.vstack(smpl_accepted)
        smpl_accepted = smpl_accepted[:self.num_samples,:]
        return self._eval_population(smpl_accepted, 
                                    from_latent=True, shuffle=True, **kwargs)



###############################################################################
###############################################################################
###############################################################################


class SampleBetween(object):
    """ 
        Collection of methods that sample the latent space using 
        cluster descriptors.
    """

    def _sample_means(self, mtype, noise, cluster_object, **kwargs):
        """ Takes samples within cluster mean support """
        if mtype == 'medoid':
            mtype_dict = cluster_object.cluster_to_medoids
        elif mtype == 'centroid':
            mtype_dict = cluster_object.cluster_to_centroids
        else:
            raise ValueError("Unsupported mtype!")
        # Get number of samples to take
        num_clusters = cluster_object.num_clusters
        if noise:
            num_samples_per_cluster = max(1, int(np.floor(
                                          self.num_samples / num_clusters)))
        else:
            num_samples_per_cluster = 1
        # Log
        logging.info("Sampling {} particles around each "
                     "cluster {}.".format(num_samples_per_cluster, mtype))
        # Sample existing datapoints from clusters, and add a small noise 
        # proportional to the medoid avg distance
        next_param_population = []
        for i, cidx in enumerate(range(cluster_object.num_clusters)):
            mean_centre = mtype_dict[cidx][0].point
            mean_dev = mtype_dict[cidx][0].dev
            sampled_datapoints = mean_centre 
            if noise:
                if len(mean_dev.shape)==1:
                    mean_dev = np.diag(mean_dev)
                sampled_datapoints = np.random.multivariate_normal(
                                        mean=sampled_datapoints,
                                        cov=mean_dev,
                                        size=num_samples_per_cluster)
            next_param_population = sampled_datapoints if i==0 else \
                                    np.vstack([next_param_population, 
                                               sampled_datapoints])
        return self._eval_population(next_param_population, 
                        from_latent=True, shuffle=True, **kwargs)
   

    def _interpolate_means(self, mtype, noise, cluster_object, **kwargs):
        """ Interpolates - between cluster mean descriptors """
        if mtype == 'medoid':
            mtype_dict = cluster_object.cluster_to_medoids
        elif mtype == 'centroid':
            mtype_dict = cluster_object.cluster_to_centroids
        else:
            raise ValueError("Unsupported mtype!")
        # Get number of samples to take
        next_latent_population = []
        num_clusters = cluster_object.num_clusters
        if num_clusters > 1:
            num_mean_interp = num_clusters * (num_clusters - 1) / 2
            if noise:
                num_samples_per_cluster = \
                    max(1, int(np.floor(self.num_samples / num_mean_interp)))
            else:
                num_samples_per_cluster = 1
            # Log
            logging.info("Sampling {} particles between each "
                         "cluster {}.".format(num_samples_per_cluster, mtype))
            # Get geometric mean between each medoid pair
            for ii, (i1, i2) in enumerate(itertools.combinations(
                                                      range(num_clusters), 2)):
                interp_point = \
                    (mtype_dict[i1][0].point + mtype_dict[i2][0].point) / 2
                interp_dev = \
                    (mtype_dict[i1][0].dev + mtype_dict[i2][0].dev) / 2
                if noise:
                    if len(interp_dev.shape)==1:
                        interp_dev = np.diag(interp_dev)
                    interp_point = np.random.multivariate_normal(
                                        mean=interp_point,
                                        cov=interp_dev,
                                        size=num_samples_per_cluster)
                next_latent_population = interp_point if ii==0 else \
                                        np.vstack([next_latent_population, 
                                                   interp_point])
        else:
            interp_point = mtype_dict[0][0].point
            interp_dev = mtype_dict[0][0].dev
        # If empty or not enough clusters/samples - add random ones
        if len(next_latent_population) < self.num_samples:
            num_add = self.num_samples - len(next_latent_population)
            new_add = np.random.multivariate_normal(mean=interp_point,
                                                    cov=noise*interp_dev,
                                                    size=num_add)

            next_latent_population = new_add \
                                     if not len(next_latent_population) \
                                     else np.vstack([next_latent_population, 
                                                     new_add])
        # Select num_samples
        # next_latent_population = next_latent_population[:self.num_samples]
        # Sort them by inter-cluster distance
        return self._eval_population(next_latent_population,
                        from_latent=True, shuffle=True, **kwargs)


    def _epsilon_means(self, mtype, noise, epsilon, recn_fn, cluster_object, 
                       shuffle=True, balance=False, **kwargs):
        """ Sample cluster center descriptors and add noise """
        if mtype == 'medoid':
            mtype_dict = cluster_object.cluster_to_medoids
        elif mtype == 'centroid':
            mtype_dict = cluster_object.cluster_to_centroids
        else:
            raise ValueError("Unsupported mtype!")
        # Get number of samples to take
        num_clusters = cluster_object.num_clusters
        num_samples_per_cluster = max(1, int(np.floor(
                                      self.num_samples / num_clusters)))
        # Log
        logging.info("Sampling {} particles around each "
                     "cluster {}.".format(num_samples_per_cluster, mtype))
        # Sample medoids and add noise in latent space
        next_latent_population = []
        for i, cidx in enumerate(range(num_clusters)):
            mean_centre = mtype_dict[cidx][0].point
            mean_dist = mtype_dict[cidx][0].dev
            latent_samples = mean_centre 
            latent_samples = latent_samples + noise * mean_dist * \
                             np.random.randn(num_samples_per_cluster, 
                                             len(mean_centre))
            next_latent_population = latent_samples if i==0 else \
                                     np.vstack([next_latent_population, 
                                               latent_samples])
        # Select num_samples
        next_latent_population = next_latent_population[:self.num_samples]
        num_samples =  int(epsilon * len(next_latent_population))
        latent_samples = next_latent_population[
          np.random.choice(next_latent_population.shape[0], num_samples)]
        next_latent_population = np.vstack([next_latent_population, 
                                            latent_samples])
        # Reconstruct params and add noise in param space (to each point)
        next_param_population = recn_fn(next_latent_population)
        # next_param_population = next_param_population[-num_samples:,:] + \
        #   noise * np.random.randn(num_samples, next_param_population.shape[1])

        # Evaluate the population
        # out_dict = self.task.evaluate_population(
        #                           param_population=next_param_population)
        out_dict = self._eval_population(
                                next_param_population=next_param_population,
                                from_latent=False, convert=False)

        # Balance data to avoid disproportionate labels
        # if balance: 
        #     out_tuple = self.task.balance_outcomes(*out_tuple)       
        # if shuffle: 
        #     out_tuple = self._shuffle(*out_tuple)
        out_dict.update({'param_embedded': next_latent_population})
        return out_dict


    def _extrapolate_means(self, mtype, noise, cluster_object, **kwargs):
        """ Extrapolates - takes samples outside cluster mean support """
        if mtype == 'medoid':
            mtype_dict = cluster_object.cluster_to_medoids
        elif mtype == 'centroid':
            mtype_dict = cluster_object.cluster_to_centroids
        else:
            raise ValueError("Unsupported mtype!")
        # Get number of samples to take
        num_clusters = cluster_object.num_clusters
        num_samples_per_cluster = max(1, int(np.floor(
                                      self.num_samples / num_clusters)))
        # Log
        logging.info("Sampling {} particles "
                     "outside data support".format(self.num_samples))

        # GENERATE SAMPLES
        # DO REJECTIONS SAMPLING
        for i, cidx in enumerate(range(num_clusters)):
            mean_centre = mtype_dict[cidx][0].point
            mean_dist = mtype_dict[cidx][0].dev



    def _extrapolate_and_prediciton(self, mtype, noise, cluster_object, 
                                    out_brach_fn, **kwargs):
        """ Extrapolates - takes samples outside cluster mean support and combines """
        pass


    ### MEDOIDS

    def ls_medoids_exact(self, **kwargs):
        """ Sample exact cluster medoids """
        return self._sample_means(mtype='medoid', noise=0, **kwargs)


    def ls_medoids_rand(self, **kwargs):
        """ Sample cluster medoids and add noise in latent space """
        return self._sample_means(mtype='medoid', noise=0.1, **kwargs)


    def ls_medoids_interp_exact(self, **kwargs):
        """ Sample between medoids """
        return self._interpolate_means(mtype='medoid', noise=0, **kwargs)


    def ls_medoids_interp_rand(self, **kwargs):
        """ Sample between medoids and add noise in latent space  """
        return self._interpolate_means(mtype='medoid', noise=0.1, **kwargs)


    def ls_medoids_epsilon(self, **kwargs):
        """ 
            Sample cluster medoids and add noise both in latent space.
            Epsilon gives the fraction of this samples which are added with 
            parameter space noise.
        """
        return self._epsilon_means(mtype='medoid', noise=0.1, epsilon=0.5, 
                                   **kwargs)

    ### CENTROIDS

    def ls_centroids_exact(self, **kwargs):
        """ Sample exact cluster centroids """
        return self._sample_means(mtype='centroid', noise=0, **kwargs)


    def ls_centroids_rand(self, **kwargs):
        """ Sample cluster centroids and add noise in latent space """
        return self._sample_means(mtype='centroid', noise=0.1, **kwargs)


    def ls_centroids_interp_exact(self, **kwargs):
        """ Sample between centroids """
        return self._interpolate_means(mtype='centroid', noise=0, **kwargs)


    def ls_centroids_interp_rand(self, **kwargs):
        """ Sample between centroids and add noise in latent space """
        return self._interpolate_means(mtype='centroid', noise=0.1, **kwargs)


    def ls_centroids_epsilon(self, **kwargs):
        """ 
            Sample cluster medoids and add noise both in latent space.
            Epsilon gives the fraction of this samples which are added with 
            parameter space noise.
        """
        return self._epsilon_means(mtype='centroid', noise=0.1, epsilon=0.5, 
                                   **kwargs)



###############################################################################
###############################################################################
###############################################################################



class LatentSampling(SampleParticles, SampleBetween, SampleOutside):
    """ 
        Collection of methods that sample the latent space 
    """
    pass
    
