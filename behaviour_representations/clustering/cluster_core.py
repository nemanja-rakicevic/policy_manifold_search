
"""
Author:         Anonymous
Description:
                k-Means clustering methods, with k evaluation
                - stanrdard k-Means
                - BIC; Using Bayesian Information Criteria 
                [REF: https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf]
                [CODE: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans/251169#251169; ALT:https://github.com/mynameisfiber/pyxmeans/blob/master/pyxmeans/xmeans.py]
                - G-Means; Using Anderson-Darling test to estimate if the cluster is Gaussian 
                [REF: https://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf]
                [CODE: https://github.com/flylo/g-means/blob/master/gmeans.py]
"""


import logging
import warnings
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import scale, StandardScaler

from scipy.spatial import distance
from scipy.stats import anderson

from behaviour_representations.utils.utils import organise_labels, _SEED


logger = logging.getLogger(__name__)



class ClusterBase(object):
    """ Clustering base class. Chooses k-Means algorithm """

    def __init__(self, min_obs=1, cluster_algo='kmeans',
                 verbose=False, **kwargs):
        if cluster_algo == 'kmeans':
            self.cluster_algo = KMeans 
        elif cluster_algo == 'gmm':
            self.cluster_algo = GaussianMixture 
        else:
            raise ValueError("Undefined kmeans method")
        self.min_obs = min_obs
        self.verbose = verbose


    def _check_singularity(self, data):
        # if data.shape[0] > len(np.unique(data, axis=0)):
        unique_data = np.unique(data, axis=0)
        if np.abs(np.mean(unique_data-unique_data[0]))<1e-5:
            if self.verbose: 
                logger.warning("Clustering: Datapoint latent projection" \
                               "collapsed (diff: {:.4e})!".format(
                               np.abs(np.mean(unique_data-unique_data[0]))))
            return True
        else:
            return False


    def _init_model(self, num_clusters):
        try:
            return self.cluster_algo(n_clusters=num_clusters, 
                                     init="k-means++", random_state=_SEED)
        except:
            return self.cluster_algo(n_components=num_clusters,
                                     random_state=_SEED)


    def _get_labels(self, model, datapoints):
        try:
            return model.labels_
        except:
            return model.predict(datapoints)


    def _get_centers(self, model):
        try:
            return model.cluster_centers_
        except:
            return model.means_


    def _get_nclusters(self, model):
        try:
            return model.n_clusters
        except:
            return model.n_components


    def cluster_data(self, datapoints):
        raise NotImplementedError



class ClusterGMM(ClusterBase):
    """ Basic Gausian Mixture Model clustering """

    def __init__(self, k_max=4, num_retries=10, **kwargs):
        super().__init__(**kwargs)
        self.cluster_algo = GaussianMixture 
        self.k_max = k_max
        self.num_retries = num_retries


    def cluster_data(self, datapoints, num_tries=10):
        if datapoints.shape[0] < self.k_max:
            self.k_max = datapoints.shape[0]
        for k_ in range(self.k_max, 0, -1):
            gmm = self._init_model(num_clusters=k_)
            for t_ in range(num_tries):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gmm.fit(datapoints)
                outputs = self._get_labels(gmm) 
                # check that each cluster assignment of this class 
                # has at least _n_samples points
                if min(np.bincount(outputs)) >= self.min_obs:
                    if self.verbose: 
                        logger.info("GMM: clustering with k={} "
                                    "(k_max={}) in {} tries.".format(
                                    k_, self.k_max, t_))
                    return outputs
        if self.verbose: 
              logger.warning("GMM: Could not find more than "
                             "{} cluster(s).".format(k_))
        print("\nHEERE find more than {} cluster.".format(k_) )
        return outputs



class ClusterKmeans(ClusterBase):
    """ Basic KMeans clustering """

    def __init__(self, k_max=4, num_retries=10, **kwargs):
        super().__init__(**kwargs)
        self.cluster_algo = KMeans 
        self.k_max = k_max
        self.num_retries = num_retries


    def cluster_data(self, datapoints, num_tries=10):
        if datapoints.shape[0] < self.k_max:
            self.k_max = datapoints.shape[0]
        for k_ in range(self.k_max, 0, -1):
            kmeans = self.cluster_algo(n_clusters=k_, init="k-means++", 
                                       random_state=_SEED)
            for t_ in range(num_tries):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans.fit(datapoints)
                outputs = kmeans.labels_ 
                # check that each cluster assignment of this class 
                # has at least _n_samples points
                if min(np.bincount(outputs)) >= self.min_obs:
                    if self.verbose: 
                        logger.info("KMeans: clustering with k={} "\
                                    "(k_max={}) in {} tries.".format(
                                    k_, self.k_max, t_))
                    return outputs
        if self.verbose: 
              logger.warning("KMeans: Could not find more than "\
                             "{} cluster(s).".format(k_))
        print("\nHEERE find more than {} cluster.".format(k_) )
        return outputs




class ClusterGMeans(ClusterBase):
    """ 
        Clustering approach using G-Means algorithm. 
        Breaks clusters down until the data in it is Gaussian
    """

    def __init__(self, max_depth=5, strictness=2, **kwargs):
        super().__init__(**kwargs)
        self.cluster_algo = KMeans 
        self.max_depth = max_depth
        if strictness not in range(5):
            raise ValueError("GMeans: Strictness parameter must be integer" + \
                             "from 0 (not strict) to 4 (very strict)")
        self.strictness = strictness
        self.stopping_criteria = []
        

    def _get_index(self):
        new_idx = len(self._index)
        self._index.append(new_idx)
        return new_idx


    def _check_gaussian(self, vector):
        """ 
            Anderson-Darling test to see if samples come from a 
            Gaussian distribution 
        """
        output = anderson(vector)
        return True  if output[0] <= output[1][self.strictness] else False
        

    def _recursive_clustering(self, data, depth, index):
        """ Recursively run kmeans with k=2 until stopping criteria """
        depth += 1
        # Check max_depth criterion
        if self._check_singularity(data):
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('collapsed_data')
            return
        # Check max_depth criterion
        if depth == self.max_depth:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('max_depth')
            return
        kmeans = self._init_model(num_clusters=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans.fit(data)
        centers = self._get_centers(kmeans)
        v = centers[0] - centers[1]
        # x_prime = scale(data.dot(v) / (v.dot(v)))
        x_prime = data.dot(v) / (v.dot(v))
        isgaussian = self._check_gaussian(x_prime)
        # Check if Gaussian criterion
        if isgaussian == True:
            self.data_index[index[:, 0]] = index
            self.stopping_criteria.append('gaussian')
            return
        labels = self._get_labels(kmeans, data)
        for k in set(labels):
            current_data = data[labels==k]
            # Check min_obs criterion
            if current_data.shape[0] <= self.min_obs:
                self.data_index[index[:, 0]] = index
                self.stopping_criteria.append('min_obs')
                return
            current_index = index[labels==k]
            current_index[:, 1] = self._get_index()
            self._recursive_clustering(data=current_data, depth=depth,
                                       index=current_index)


    def cluster_data(self, datapoints):
        """ Fit the recursive clustering model to the data """
        data_index = np.array([(i, False) for i in range(datapoints.shape[0])])
        self.data_index = data_index
        self._index = []
        self.stopping_criteria = []
        self._recursive_clustering(data=datapoints, depth=0, index=data_index)
        # Log clustering stopping criteria
        def log_stopping(c_name):
            logger.warning("GMeans: cluster(s) {} stopped ".format(
                np.where(np.array(self.stopping_criteria)==c_name)[0]) + \
                "because '{}' reached.".format(c_name))
        if self.verbose:
            if 'min_obs' in self.stopping_criteria: 
                  log_stopping('min_obs')
            if 'max_depth' in self.stopping_criteria: 
                  log_stopping('max_depth')
            if 'collapsed_data' in self.stopping_criteria: 
                  log_stopping('collapsed_data')
        return organise_labels(self.data_index[:, 1])



class ClusterBIC(ClusterBase):
    """ 
        Clustering approach in which appropriate k is choesen
        using Bayesion Information Criterion
    """

    def __init__(self, k_max=10, **kwargs):
        super().__init__(**kwargs)
        self.k_max = k_max
        self.k_range = list(range(1, self.k_max))


    def _get_model(self, datapoints):
        # Run KMeans k_max times and save each result in the KMeans object
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            list_models = \
                [self._init_model(i).fit(datapoints) for i in self.k_range]
        # Calculate BIC for each model
        list_BIC = \
            [self._compute_bic(model, datapoints) for model in list_models]
        # return the best model and the optimal 
        best_bic = np.argmax(list_BIC)
        return list_models[best_bic], self.k_range[best_bic]


    def _compute_bic(self, model, datapoints):
        """ Computes the BIC metric for a given clusters """
        N, d = datapoints.shape
        # assign centers, number of clusters and labels
        centers = self._get_centers(model)
        m = self._get_nclusters(model)
        labels  = self._get_labels(model, datapoints)
        # size of the clusters
        n = np.bincount(labels)
        # disregard if some clusters do not have enough samples
        if min(n) < self.min_obs:
            return -np.inf
        if m!=len(n):
            return -np.inf
        # compute variance for all clusters beforehand
        cl_var = (1.0 / (N - m) / d) * \
                 sum([sum(distance.cdist(datapoints[np.where(labels == i)], 
                                         [centers[i]], 
                                         'euclidean')**2) for i in range(m)])
        const_term = 0.5 * m * np.log(N) * (d+1)
        BIC = np.sum([n[i] * np.log(n[i]) -
                     n[i] * np.log(N) -
                     ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                     ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
        return BIC


    def cluster_data(self, datapoints):
        if datapoints.shape[0] < self.k_max:
            self.k_range = list(range(1, datapoints.shape[0]))

        if self._check_singularity(datapoints):
            return np.zeros(datapoints.shape[0], dtype=int)
        else:
            best_model, best_k = self._get_model(datapoints)
            best_model.fit(datapoints)
            if self.verbose: 
                logger.info("ClusterBIC: Best clustering with " + \
                            "k={} (k_max={}).".format(best_k, self.k_range))
            return self._get_labels(best_model, datapoints)
