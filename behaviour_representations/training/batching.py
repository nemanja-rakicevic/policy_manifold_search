
"""
Author:         Anonymous
Description:
                BatchManager class containing implementations of batching 
                - standard: batching data by traversing the whole dataset

"""

import logging
import pickle
import numpy as np 

from itertools import combinations
from scipy.spatial import distance_matrix

from behaviour_representations.utils.utils import timing, shuffle


logger = logging.getLogger(__name__)



class BatchManager(object):
    """
        Generates the batching setup for AE training based on batching_mode
    """
    def __init__(self, batching, clustering,
                       test_batches=2, 
                       **kwargs):
        # Initialize batch size
        self._batchsize = None
        self.batchsize = batching['batchsize_normal']
        self.batchsize_init = batching['batchsize_init'] \
                                  if batching['batchsize_init'] is not None \
                                  else self.batchsize
        # Initialize batch mode
        self._batchmode = None
        self.batchmode = batching['batchmode_normal']
        self.batchmode_init = batching['batchmode_init'] \
                                  if batching['batchmode_init'] is not None \
                                  else self.batchmode
        # Adjustable number of batch iterations within epoch
        self.n_iter_fixed = batching['fixed_iters']                           
        # Magnet loss batching info
        self._n_samples = clustering['n_samples_in_cluster']
        self._n_clusters = clustering['n_clusters_to_sample']
        # Leave data for testing
        self._n_test_batches = test_batches


    # @timing
    def init_batching(self, nloop, nepoch, data_orig, labels_orig, data_recn, 
                      datagen_object, testset_ratio=None, verbose=False,
                      **kwargs):
        """ 
            Runs at the beginning of a loop, oraganizes batching by creating
            a list of sample indices that generate_batch() iteratively 
            traverses and returns when called at certain iteration number.
        """
        assert len(data_orig) and len(labels_orig), "Cannot batch if no data!"
        # Evaluate on the test set or no                                                                               
        self._testset_ratio = testset_ratio if testset_ratio is not None \
                                               and nloop else 0
        # Selects batchmode and batchsize based on the loop
        self._batchmode = self.batchmode if nloop else self.batchmode_init
        self._batchsize = self.batchsize if nloop else self.batchsize_init

        # Initialise batch indices
        batch_dict = dict(_data_size=len(labels_orig),
                          _data_idx=np.arange(len(labels_orig)),
                          _success_idx=np.where(labels_orig>-1)[0],
                          _fail_idx=np.where(labels_orig==-1)[0])

        # Organize data indices based on batchmode and other variables
        self.indices_train, self.indices_test = \
            getattr(self, '_{}_batching'.format(self._batchmode))(**batch_dict)
        
        # Get number of iterations based on number of indices and batchsize
        self.num_iter_train = len(self.indices_train) // self._batchsize
        if len(self.indices_train) % self._batchsize:
            self.num_iter_train += 1

        # Log the batch organisation
        if verbose:
            logger.info("BATCH: '{}' size: {}; train datapoints {} "
                        "({} unique/{} total)\n".format(
                        self._batchmode, self._batchsize, 
                        len(self.indices_train),
                        len(np.unique(self.indices_train)), len(labels_orig)))

        # Return num train iterations
        return self.num_iter_train



    def generate_batch(self, data_orig, labels_orig, traj_main, metric_bd,
                        iteration=None, testdata=False, **kwargs):
        """ 
            Returns the batch from the pre-defined index arrangement,
            based on current num_iter and data given.
        """
        # Select between train or test data batch
        if testdata:
            if len(self.indices_test) == 0: return None
            self.batch_indices = self.indices_test
        else:
            bidx_start = iteration * self._batchsize
            bidx_stop = min(iteration * self._batchsize + self._batchsize,
                            len(self.indices_train))
            self.batch_indices = self.indices_train[bidx_start:bidx_stop]
        assert len(self.batch_indices), "No batch indices selected!"
        return dict(batch_size=len(self.batch_indices),
                    batch_indices=self.batch_indices,
                    batch_params=data_orig[self.batch_indices, :],
                    batch_outcomes=labels_orig[self.batch_indices],
                    batch_trajectory=traj_main[self.batch_indices, ...],
                    batch_metric_bd=metric_bd[self.batch_indices],
                    dist_trajectory=self._dist_traj(self.batch_indices,
                                                    **kwargs)
                    )


    """ BATCHING MODES """


    def _standard_batching(self, _data_size, _data_idx, **kwargs):
        """
            Stochastic batch sampling
        """
        # Split dataset on train/test based on testset_ratio
        test_data_size = int(self._testset_ratio * _data_size)
        _idx_test = _data_idx[:test_data_size]
        _idx_train = _data_idx[test_data_size:]

        # Get final training set size based on n_iter_fixed
        if self.n_iter_fixed is not None:
            _n_train_samples = self.n_iter_fixed  * self._batchsize
        else:
            _n_train_samples = len(_idx_train)

        # Fill up quota for training set and re-shuffle
        # _data_idx = shuffle(_data_idx)
        if len(_idx_test) > 0:
            _idx_test = shuffle(_idx_test)
        _idx_train = shuffle(np.resize(_idx_train, _n_train_samples))

        return _idx_train, _idx_test


    def _repeat_one_batching(self, _data_idx, **kwargs):
        """
            Whole batch is just one point
        """
        one_sample = np.random.choice(_data_idx)
        batching_indices = np.tile(one_sample, self._batchsize)
        return batch_indices


    def _succ_only_batching(self, _success_idx, **kwargs):
        """
            Focus only on mismatched successful samples
        """
        return _success_idx


    def _balanced_batching(self, _success_idx, _fail_idx, **kwargs):
        """ 
            Each batch has the same number of successful and failed samples 
        """
        num_fail = len(_fail_idx)
        num_succ = len(_success_idx)
        if num_succ > num_fail:
            # tile _fail_idx and shuffle
            diff_x = int(np.floor(num_succ/num_fail))
            # diff_r = num_succ%num_fail
            _fail_idx = np.tile(_fail_idx, diff_x)
            _fail_idx = shuffle(_fail_idx)
        elif num_fail > num_succ:
            if num_succ == 0:
                return self._standard_batching(**kwargs)
            # tile _success_idx and shuffle
            diff_x = int(np.floor(num_fail/num_succ))
            # diff_r = num_succ%num_fail
            _success_idx = np.tile(_success_idx, diff_x)
            _success_idx = shuffle(_success_idx)
        # get all successful indices, divide into batchsize/2 chunks
        # get all failed indices, divide into batchsize/2 chunks
        num_fail = len(_fail_idx)
        num_succ= len(_success_idx)
        min_elems = min(num_fail, num_succ)
        # num_chunks = max(1, int(min_elems / (self._batchsize/2)))
        num_chunks = max(1, min_elems)
        split_succ = np.array_split(_success_idx, num_chunks)
        split_fail = np.array_split(_fail_idx, num_chunks)
        # stack alternately fail succ
        batching_indices = [None]*(len(split_succ)+len(split_fail))
        batching_indices[::2] = split_fail if num_fail > num_succ \
                                           else split_succ
        batching_indices[1::2] = split_fail if num_fail <= num_succ \
                                            else split_succ
        return np.concatenate(batching_indices)


##############################################################################
##############################################################################
##############################################################################


    def _dist_traj(self, indices, metric_traj_dtw, **kwargs):
        idx_ = np.array(list(combinations(indices, 2)))
        if metric_traj_dtw is not None:
            return metric_traj_dtw[idx_[:,0], idx_[:,1]].reshape(-1, 1)
        else:
            return np.array([None] * len(idx_)).reshape(-1,1)

