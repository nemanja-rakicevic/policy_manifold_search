
"""
Author:         Anonymous
Description:
                Managing clustering in the latent space
                - create clusters
                - get cluster loss (used for embedding losses)

"""

import logging
import numpy as np 

from collections import namedtuple
from itertools import product

import behaviour_representations.clustering.cluster_core as cclst
from behaviour_representations.utils.utils import _TAB, _SEED

logger = logging.getLogger(__name__)



Centre = namedtuple('Centre', 'point dev')



def get_data_centroids(data, labels):
    """ 
        Returns the geometric centre of a cluster (centroid),
        and the standard deviation of distances to other elements.
    """
    assert len(data)==len(labels), \
        "Needs to be same length! data: {}, labels: {}".format(len(data),
                                                               len(labels))
    outcome_vals = np.unique(labels)
    centroid_means = []
    centroid_stds = []
    for tag in outcome_vals: 
        current_vals = data[labels==tag]
        centroid_means.append(np.mean(current_vals, axis=0))
        centroid_stds.append(np.std(current_vals, axis=0))
    return np.array(centroid_means), np.array(centroid_stds)


def get_data_medoids(data, labels):
    """ 
        Returns the cluster element equally distant to other 
        members of the same cluster (medoid), and the average distance.
    """
    assert len(data)==len(labels), \
        "Needs to be same length! data: {}, labels: {}".format(len(data),
                                                               len(labels))
    unique_labels = np.unique(labels)
    medoid_means = []
    medoic_avgdist = []
    for tag in unique_labels: 
        current_vals = data[labels==tag]

        if len(current_vals.shape) == 1:
            medoid_means.append(current_vals)
            medoic_avgdist.append(0)

        elif current_vals.shape[0] == 2:
            medoid_means.append(current_vals[np.random.choice(2)])
            medoic_avgdist.append(
                np.linalg.norm(current_vals[0] - current_vals[1]))

        if current_vals.shape[0] > 2:
            tag_means = []
            tag_avgdist = []
            num_vals = current_vals.shape[0] 
            for i, v1 in enumerate(current_vals):
                _other_vals = current_vals[np.arange(num_vals)!=i]
                _dists = np.linalg.norm(v1-_other_vals, axis=1)
                tag_avgdist.append(np.mean(_dists))

            medoid_means.append(current_vals[np.argmin(tag_avgdist)])
            medoic_avgdist.append(np.min(tag_avgdist))

    return np.array(medoid_means), np.array(medoic_avgdist)


###############################################################################
###############################################################################
###############################################################################


class ClusterManager(object):
    """
        Cluster manager class
    """

    def __init__(self, clustering, 
                       **kwargs):

        self.data_labels = None
        self._n_samples = 1 # clustering['n_samples_in_cluster']
        self.max_clusters_per_class = clustering['max_clusters']

        # Initialise clustering method
        kw = dict(min_obs=self._n_samples, k_max=self.max_clusters_per_class)
        if clustering['algorithm'] == 'kmeans':
            self.clustering_algo = cclst.ClusterKMeans(**kw) 

        elif clustering['algorithm'] == 'bic':
            self.clustering_algo = cclst.ClusterBIC(cluster_algo='kmeans', **kw) 
        
        elif clustering['algorithm'] == 'gmeans':
            self.clustering_algo = \
                cclst.ClusterGMeans(strictness=clustering['strictness'], **kw) 
        else:
            logger.error("Undefined clustering method")
            raise ValueError("Undefined clustering method")
        # Initialise cluster statistics
        self._clear_info()


    def _clear_info(self):
        """ Initilise the cluster statistics """
        self.num_clusters = None
        # array holding cluster affiliation for each datapoint
        self.datapoints_to_cluster = None
        # dictionary {cluster index: indices of data in cluster}
        self.cluster_to_datapoints = {} 
        # dictionary {cluster index: number of datapoints in cluster}   
        self.cluster_sizes = {}
        # dictionary {cluster index: class of each datapoint}  
        self.cluster_to_class = {}
        self.class_to_cluster = {} 
        # magnet loss for each example datapoint
        self.datapoint_losses = None
        # magnet loss averaged over each cluster
        self.cluster_losses = None
        # dictionary {cluster index: Centre namedtuple} 
        self.cluster_to_centroids = {} 
        self.cluster_to_medoids = {} 

   
    def update_clusters(self, data_embedded, data_labels, metric_dict=None,
                        verbose=False):
        """
            Create clusters on the whole dataset, and update vars
        """
        def _global_clst_idx(class_, clst_idx_, ctc_map):
            return np.where(ctc_map==class_)[0][0] + clst_idx_
        # Take all the data and cluster it
        if data_embedded.shape[0] > 1:
            cluster_labels = self.clustering_algo.cluster_data(data_embedded)
        else:
            cluster_labels = np.zeros(data_embedded.shape[0], dtype=int)
        # Update cluster info
        self._clear_info()
        self.num_clusters = len(np.unique(cluster_labels))
        self.datapoints_to_cluster = cluster_labels
        for clst_idx in np.unique(cluster_labels):
            data_idx = np.flatnonzero(cluster_labels == clst_idx)
            self.cluster_to_datapoints[clst_idx] = data_idx
            self.cluster_to_class[clst_idx] = data_labels[data_idx]
            self.cluster_sizes[clst_idx] = len(data_idx)
        # for lab_ in np.unique(data_labels):
        #     self.class_to_clusters[lab_] = NOT YET!
        # Update centroids and medoids
        self.get_cluster_centroids(data_embedded)
        self.get_cluster_medoids(data_embedded, dist_matrix=None)

        # Log cluster distribution
        logger.info("CLUSTER: {}; {} clusters ({} max)\n{}\t\t- sizes: {}".format(
                    self.clustering_algo.__class__.__name__, 
                    len(np.unique(cluster_labels)),
                    self.max_clusters_per_class, _TAB, self.cluster_sizes))


    def update_cluster_losses(self, batch_indices, batch_losses):
        """
            Given a list of batch examples indexes and corresponding losses,
            store the new losses and update corresponding cluster losses.
        """
        # Lazily allocate structures for losses
        if self.datapoint_losses is None:
            self.has_loss = np.zeros_like(self.data_labels, bool)
            self.datapoint_losses = np.zeros_like(self.data_labels, float)
        # Cluster losses are resetted every time the cluster distribution changes
        if self.cluster_losses is None:
            self.cluster_losses = np.zeros_like(self.cluster_to_class, float)
        # Update losses for each datapoint in the minibatch
        _indexes = np.array(batch_indices)
        self.datapoint_losses[_indexes] = batch_losses
        self.has_loss[_indexes] = True
        # Find affected clusters (containing the indexes) and update the corresponding cluster losses
        affected_clusters = np.unique(self.datapoints_to_cluster[_indexes])
        for ac_idx_ in affected_clusters:
            cluster_datapoint_idxs = self.datapoints_to_cluster == ac_idx_
            cluster_datapoint_losses = self.datapoint_losses[cluster_datapoint_idxs]
            # Take the average cluster loss in the cluster of examples for which a loss has been measured
            self.cluster_losses[ac_idx_] = np.mean(cluster_datapoint_losses[self.has_loss[cluster_datapoint_idxs]])


    def get_cluster_sizes(self):
        return [len(self.cluster_to_datapoints[i]) \
                    for i in self.cluster_to_datapoints]


    def get_cluster_medoids(self, data, dist_matrix=None):
        """ 
            Returns the cluster element equally distant to other 
            members of the same cluster (medoid), and the average distance.
        """
        def _get_medoid(d_idx, data, info, dist_matrix_fn):
            if len(d_idx) > 2:
                d_idx_combos = np.array(list(set(product(d_idx, d_idx)) - \
                                        set(zip(d_idx, d_idx))))
                costs = np.zeros_like(d_idx)
                for e, i in enumerate(d_idx):
                    pair_idx = d_idx_combos[
                                  np.where(d_idx_combos[:,0] == i)[0]]
                    pair_idx = np.array(list(map(lambda x: x[::-1] \
                                                    if x[0]>x[1] else x, 
                                                 pair_idx)))
                    costs[e] = np.mean(dist_matrix_fn(pair_idx[:,0], 
                                                      pair_idx[:,1],
                                                      info=info))
                sel_idx = d_idx[np.argmin(costs)]
                medoid = data[sel_idx]
                avg_dist = np.diag(np.mean(medoid-data, axis=0))
                return (Centre(medoid, avg_dist), sel_idx)
            elif len(d_idx) == 2:
                sel_idx = np.random.choice(d_idx)
                pair_cov = np.diag(data[d_idx[0]]-data[d_idx[1]])
                return (Centre(data[sel_idx], pair_cov), sel_idx)
            elif len(d_idx) == 1:
                return (Centre(data[d_idx[0]], 
                               0.0001*np.eye(data.shape[1])), d_idx[0])
            else:
                raise ValueError("Cluster is empty!")

        # define distance metric based on given info
        if dist_matrix is None:
            info = data
            def dist_matrix_fn(i1, i2, info):
                return np.linalg.norm((info[i1] - info[i2]).reshape(
                                       -1, info.shape[1]), axis=1)
        else:
            info = dist_matrix
            def dist_matrix_fn(i1, i2, info):
                return info[i1, i2]

        for c_idx, d_idx in self.cluster_to_datapoints.items():
            self.cluster_to_medoids[c_idx] = _get_medoid(d_idx, data, info,
                                                         dist_matrix_fn)


    def get_cluster_centroids(self, data):
        """ 
            Returns the geometric centre of a cluster (centroid),
            and the standard deviation of distances to other elements.
        """
        for c_idx, d_idx in self.cluster_to_datapoints.items():
            c_mean = np.mean(data[d_idx], axis=0)
            if len(d_idx)==1:
                c_cov = 0.001 * np.eye(len(c_mean))
            else:
                c_cov  = np.cov(data[d_idx], rowvar=False)
            # check if some std is zero
            zero_std_idxs = np.where(c_cov == 0)
            if len(zero_std_idxs[0]):
                c_cov[zero_std_idxs] = 0.0001 #* np.ones_like(zero_std_idxs)
            self.cluster_to_centroids[c_idx] = (Centre(c_mean, c_cov), None)



    def load_cluster_data(self):
        raise NotImplementedError

