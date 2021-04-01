
"""
Author:         Anonymous
Description:
                Data management class
                Some important components:
                - Attributes:   
                    - original data (parameter & collision info), labels, 
                      embedded and redonctructed data
                    - cluster assignments and cluster mean and std
                    - (TODO) trajectories (state & action)
                - Methods:
                    - update_clusters: performs clusterring in the latent space
                    - update_representations: updates embedding and 
                      reconstructed data
                    - generate_data: calls dategen object method to get a new 
                      population and add it
                    - generate_batch: samples batches from the stored data
                    - plot_spaces: data visualisation
"""

import os
import csv
import logging
import pickle
import numpy as np

from functools import partial
from itertools import product, combinations
from operator import itemgetter 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import behaviour_representations.exploration.manage_datagen as mdgen
import behaviour_representations.clustering.manage_cluster as mclstr
import behaviour_representations.training.batching as mbatch

from behaviour_representations.utils.utils import (timing,
                                                  info_outcome,
                                                  pad_traj,
                                                  uniquify_by_key)
from behaviour_representations.utils.utils import _TAB, _SEED
from behaviour_representations.utils.utils import transform_PCA_TSNE


logger = logging.getLogger(__name__)




class DataManager(object):
    """ 
        Class that manages data storage, generation and batching.
    """

    def __init__(self, load_dataset_path, load_model_path,
                       dimred_mode, filter_mbd=True,
                       **taskargs_dict):
        self.dirname = taskargs_dict['experiment_directory']
        self.load_dataset_path = load_dataset_path
        self.load_model_path = load_model_path
        self.ds_rate = 20
        # Managers
        self.datagen_object = mdgen.DataGenManager(**taskargs_dict)
        self.skip_clust = False
        self.cluster_object = mclstr.ClusterManager(**taskargs_dict)
        self.batch_object = mbatch.BatchManager(**taskargs_dict)
        # Initialise data containers
        self._init_data()
        # Metrics containers
        self._uniquify_metric = filter_mbd
        # self._uniquify_metric = filter_mbd if filter_mbd is not None else \
        #     '_mape_' in taskargs_dict['exploration']['normal']
        # Dimensionalities
        self.num_outcomes = self.datagen_object.num_outcomes

        self.inputdim_param_ae = self.datagen_object.parameter_dims
        self.inputdim_traj_ae = self.datagen_object.traj_dim
        # self.latentdim_param_ae = self.datagen_object.latent_dim
        # self.latentdim_traj_ae = self.datagen_object.latent_dim
        self.traj_len = self.inputdim_traj_ae[0]

        # Training
        self.training_loss_dict = {}

        # Initialise statistics
        self.standard_save = True
        # Dimensionality reduction technique for visualisation
        self.dimred = dimred_mode if isinstance(dimred_mode, list) \
                                  else [dimred_mode]
        # Normalise data before embedding flag
        if taskargs_dict['experiment']['type']=='displacement':
            self._normalise_flag = 'gaussian' 
        elif taskargs_dict['experiment']['type']=='nn_policy' \
             and self.datagen_object.stype is not None:
            self._normalise_flag = 'scaled'
        else:
            self._normalise_flag = False
        logger.info("DATASET PRE-PROCESSING: scaling: '{}'; range: {}".format(
                        self.datagen_object.stype, self.datagen_object.srange))

        # Initialise data discovery writer csv if new experiment
        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)

        filepath = '{}/ref_data_{}_{}.csv'.format(
                        self.dirname, 
                        self.datagen_object.ctrl_name,
                        self.datagen_object.exploration_normal)
        if not os.path.isfile(filepath):
            with open(filepath, 'a') as outfile: 
                writer = csv.DictWriter(outfile, 
                    fieldnames = ["nloop", "niter", "nsmp", "nstep", "coverage", 
                                  "fitness", "outcomes", "ratios"])
                writer.writeheader()

    """ MANAGE DATASET """

    def _init_data(self):
        # Parameter containers
        self.outcomes = None             # [n_samples, 1]
        self.param_original = None       # [n_samples, parameters_dim]
        self.param_embedded = None       # [n_samples, latentdim_param_ae]
        self.param_reconstructed = None  # [n_samples, parameters_dim]
        self.recn_error = None  
        self.new_data = None
        self.param_stats = {}
        self.latent_stats = {}
        # Trajectory 
        self.traj_original = None
        self.traj_embedded = None
        self.traj_reconstructed = None
        self.traj_data = {}
        self.traj_data['main'] = None             # [n_samples, traj_len, traj_dim]
        self.traj_data['aux'] = None          # [n_samples, traj_len, traj_dim]       
        # self.traj_main = None             # [n_samples, traj_len, traj_dim]
        # self.traj_aux = None          # [n_samples, traj_len, traj_dim]
        # Metrics containers
        self.metric_bd = None
        self.metric_traj_dtw = None       # [n_samples, n_samples]
        self.metric_traj_dtw_norm = None
        # Initialise statistics
        self.num_datapoints = 0
        self.classes_per_loop = []


    def _add_data(self, new_data_dict, embedding_fn=None,
                        **kwargs):
        # Add generated data
        if new_data_dict is not None:
            if self.param_original is None:
                self.param_original = new_data_dict['param_original']
                self.outcomes = new_data_dict['outcomes']
                self.traj_data['main'] = new_data_dict['traj_main']
                self.traj_data['aux'] = new_data_dict['traj_aux']
                # make a padded version of trajectories (max 1000 steps)
                self.traj_original = pad_traj(new_data_dict['traj_main'], 
                                              max_len=self.traj_len)
                self.metric_bd = new_data_dict['metric_bd']
            else:
                self.param_original = np.vstack([self.param_original, 
                                                 new_data_dict['param_original']])
                self.outcomes = np.vstack([self.outcomes, 
                                           new_data_dict['outcomes']])
                # self.outcomes = np.append(self.outcomes, new_data_dict['outcomes'])
                self.traj_data['main'].extend(new_data_dict['traj_main'])
                self.traj_data['aux'].extend(new_data_dict['traj_aux'])
                # make a padded version of trajectories (max 1000 steps)
                self.traj_original = np.vstack([self.traj_original, 
                                                pad_traj(new_data_dict['traj_main'], 
                                                         max_len=self.traj_len)])
                self.metric_bd = np.vstack([self.metric_bd, 
                                            new_data_dict['metric_bd']])
            # Log new data
            logger.info("DATASET: ({} + {}): {}\n".format(
                    self.num_datapoints, len(new_data_dict['outcomes']) \
                                         if new_data_dict is not None else 0,
                    info_outcome(self.outcomes)))
            # Update the data manager info
            self._update_info(**kwargs)
            # Update embedding and reconstruction data
            if embedding_fn is not None :
                self.update_representations(embedding_fn=embedding_fn,**kwargs)
        else:
            logger.info("DATASET ({}): No new unique points! {}\n".format(
                        self.num_datapoints, info_outcome(self.outcomes)))
        # Log added data
        # Save class label progress
        class_sz = {0:0, -1:0}                                                
        class_sz.update({cc:np.count_nonzero(self.outcomes[:,0]==cc) \
                   for cc in np.unique(self.outcomes[:,0])})
        self.classes_per_loop.append(class_sz)
        if self.standard_save: self.save_dataset()


    def _update_info(self, loaded=False, **kwargs):
        """ Update statistics and remove duplicate parameters """
        # Update data sizes
        self.unique_classes = np.unique(self.outcomes[:,0])
        self.num_datapoints = len(self.outcomes)
        # self.num_outcomes = len(self.unique_classes)
        # Update dataset statistics
        all_param = self.param_original.reshape(self.num_datapoints, -1)
        all_param = all_param[:, np.invert(np.isinf(all_param[0,:]))]
        self.param_stats['mu'] = np.mean(all_param, axis=0)
        self.param_stats['std'] = np.std(all_param, axis=0)
        self.param_stats['cov'] = np.cov(all_param, rowvar=False)
        # self.param_stats['mu'] = np.zeros(self.param_original.shape[1:])
        # self.param_stats['std'] = np.ones(self.param_original.shape[1:])
        # self.param_stats['cov'] = np.eye(np.prod(
        #                                     self.param_original.shape[1:]))
         # Refresh datapoint-cluster information (prepare for reclustering)
        # CLUSTER STATISTICS make as a method
        if not self.skip_clust:
            if self.cluster_object.datapoint_losses is not None:
                new_data_zeros = np.zeros(self.outcomes.shape[0] - \
                        self.cluster_object.datapoint_losses.shape[0])
                self.cluster_object.datapoint_losses = \
                          np.append(self.datapoint_losses, new_data_zeros)
                self.cluster_object.has_loss = \
                          np.append(self.has_loss, new_data_zeros.astype(int))
            # reset datapoints to cluster
            self.cluster_object.datapoints_to_cluster = \
                                    np.zeros_like(self.outcomes, int)


    def apply_normalisation(self, data):
        """ Transform data based on normalisation flag """

        if self._normalise_flag=='gaussian':
            zero_std_idxs = np.where(self.param_stats['std'] == 0)
            if len(zero_std_idxs[0]):
                self.param_stats['std'][zero_std_idxs] = 1.
            return (data - self.param_stats['mu'])/self.param_stats['std'] 

        elif self._normalise_flag=='scaled':
            scale = max(self.datagen_object.srange)
            assert scale > 0
            return data / scale

        else:
            return data


    def apply_denormalisation(self, data):
        """ Apply inverse transform to data based on normalisation flag """
            
        if self._normalise_flag=='gaussian':
            return data * self.param_stats['std'] + self.param_stats['mu']

        elif self._normalise_flag=='scaled':
            scale = max(self.datagen_object.srange)
            assert scale > 0
            return data * scale

        else:
            return data


    def update_representations(self, embedding_fn, recn_fn=None, 
                               ae_name='param', 
                               verbose=False, save_dataset=False, **kwargs):
        # Update Autoencoder latent and reconstruction space
        if ae_name=='param' or ae_name=='pca':
            self.param_embedded = embedding_fn(self.param_original)
            # self.latentdim_param_ae = self.param_embedded.shape[1]
            if recn_fn is not None:
                self.param_reconstructed = recn_fn(self.param_embedded)

                tmp_orig = self.param_original.reshape(self.param_original.shape[0], -1)
                tmp_recn = self.param_reconstructed.reshape(self.param_reconstructed.shape[0], -1)
                ninf_idx = np.where(tmp_recn[0]<np.inf)[0]
                errors = tmp_orig[:, ninf_idx] - tmp_recn[:, ninf_idx]
                self.recn_error = np.linalg.norm(errors, axis=1)
                self.training_loss_dict[ae_name] = {'reconstruction': 
                                                    np.mean(self.recn_error)}
# ADD also plot reconstructed trajectories
            # Update clustering and info
            if not self.skip_clust:
                self.cluster_object.update_clusters(self.param_embedded,
                                                    self.outcomes, 
                                                    verbose=verbose)
            else:
                logger.info(">>> skipping clustering.\n")

        elif ae_name=='traj':
            self.traj_embedded = embedding_fn(self.traj_original)
            # self.latentdim_traj_ae = self.traj_embedded.shape[1]
            if recn_fn is not None:
                self.traj_reconstructed = recn_fn(self.traj_embedded)
        # Update dataset statistics
        self.latent_stats['mu'] = np.mean(self.param_embedded, axis=0)
        self.latent_stats['std'] = np.std(self.param_embedded, axis=0)
        self.latent_stats['cov'] = np.cov(self.param_embedded.reshape(
                                    self.num_datapoints,-1), rowvar=False)

        if save_dataset: self.save_dataset()


    def save_dataset(self):
        task_data_dict = {
          # data 
          "outcomes": self.outcomes, 
          "param_original": self.param_original,
          "param_embedded": self.param_embedded,
          "param_reconstructed": self.param_reconstructed, 
          "recn_error": self.recn_error, 
          # trajectories and behaviours
          "traj_original": self.traj_original,
          "traj_embedded": self.traj_embedded,
          "traj_reconstructed": self.traj_reconstructed,

          "traj_main": self.traj_data['main'],
          "traj_aux": self.traj_data['aux'],
          "metric_bd": self.metric_bd,       
          "metric_traj_dtw": self.metric_traj_dtw,
          # cluster stuff
          "classes_per_loop": self.classes_per_loop,
          "datapoints_to_cluster": self.cluster_object.datapoints_to_cluster, 
          "cluster_to_class": self.cluster_object.cluster_to_class, 
          "cluster_to_datapoints": self.cluster_object.cluster_to_datapoints, 
          "cluster_to_centroids": self.cluster_object.cluster_to_centroids, 
          # plotting samples
          # "sample_emb": self._sample_emb,
          # "sample_recn":self._sample_recn, 
        }
        if not os.path.isdir(self.dirname):
            logger.warning("No directory {}; Creating...".format(self.dirname))
            os.makedirs(self.dirname)
        with open(self.dirname+"/experiment_dataset.dat", "wb") as f:
            pickle.dump(task_data_dict, f)


    # @timing
    def load_dataset(self, write_loaded=True):   
        """
            Load data from file and use it as first loop 0
        """                                                                               
        self.datagen_object.initial = False
        with open(self.load_dataset_path+"/experiment_dataset.dat", "rb") as f:
            task_data_dict = pickle.load(f)
        self.outcomes  = task_data_dict['outcomes']
        self.param_original = task_data_dict['param_original']
        self.param_embedded = task_data_dict['param_embedded']
        self.param_reconstructed = task_data_dict['param_reconstructed']
        self.recn_error = task_data_dict['recn_error']

        self.traj_original = task_data_dict['traj_original']
        self.traj_embedded = task_data_dict['traj_embedded']
        self.traj_reconstructed = task_data_dict['traj_reconstructed']

        self.traj_original = task_data_dict['traj_original']
        self.traj_data['main'] = task_data_dict['traj_main']
        self.traj_data['aux'] = task_data_dict['traj_aux']

        if 'metric_bd' in task_data_dict.keys():
            self.metric_bd = task_data_dict['metric_bd']
        else:
            self.metric_bd = task_data_dict['metric_conbin']
        self.metric_traj_dtw = task_data_dict['metric_traj_dtw']
        
        self.cluster_object.datapoints_to_cluster = \
            task_data_dict['datapoints_to_cluster']
        self.cluster_object.cluster_to_class = \
            task_data_dict['cluster_to_class']
        self.cluster_object.cluster_to_datapoints = \
            task_data_dict['cluster_to_datapoints']
        self.cluster_object.num_clusters = \
            len(task_data_dict['cluster_to_datapoints'])
        self.cluster_object.get_cluster_centroids(self.param_embedded)
        self.cluster_object.get_cluster_medoids(self.param_embedded)

        # Update dataset statistics
        self.unique_classes = np.unique(self.outcomes)
        self.num_datapoints = len(self.outcomes)
        self.metric_traj_dtw_norm = \
                None if self.metric_traj_dtw is None \
                else self.metric_traj_dtw / np.max(self.metric_traj_dtw)

        if self.load_model_path is None:
            class_sz = {0:0, -1:0}                                                 
            class_sz.update({cc:np.count_nonzero(self.outcomes==cc) \
                       for cc in np.unique(self.outcomes)})
            self.classes_per_loop.append(class_sz)
        else:
            self.classes_per_loop = task_data_dict['classes_per_loop']
            class_sz = self.classes_per_loop[-1]
        # Log info
        logging.info("RESTORED saved dataset; {} datapoints; "
                     "Classes: {}\n{}\tLocation: {}".format(
                          self.outcomes.shape[0], class_sz, _TAB, 
                          self.load_dataset_path+"/experiment_dataset.dat"))

        # Save as data generated in Loop 0
        if write_loaded:
            unique_bds = np.argmax(np.unique(self.metric_bd, axis=0), axis=1)
            self._write_discovery(nloop=0, n_new=len(self.outcomes), 
                                  unique_bds=unique_bds, outcomes=self.outcomes)
        # Update info
        self._update_info(loaded=True)


    """ MANAGE BATCHING """

    def init_batching(self, nloop, nepoch=None, update_dict=None, **kwargs):
        """ Syncronise the Batch Manager and get num_iter"""
        if update_dict is not None:
            self.update_representations(**update_dict)
        return self.batch_object.init_batching(
                                    nloop=nloop, nepoch=nepoch,
                                    data_orig=self.param_original, 
                                    labels_orig=self.outcomes, 
                                    data_recn=self.param_reconstructed, 
                                    datagen_object=self.datagen_object,
                                    **kwargs)


    def generate_batch(self, **kwargs):
        """ Interface to the Batch Manager """
        return self.batch_object.generate_batch(
                                    data_orig=self.param_original, 
                                    labels_orig=self.outcomes, 
                                    data_recn=self.param_reconstructed,
                                    traj_main=self.traj_original, 
                                    metric_traj_dtw=self.metric_traj_dtw_norm,
                                    metric_bd=self.metric_bd,
                                    cluster_object=self.cluster_object, 
                                    datagen_object=self.datagen_object, 
                                    **kwargs)


    """ MANAGE DATA GENERATION """

    def generate_data(self, nloop, 
                            sample_plot_fn=None, aux_title=None, aux_save=None,
                            **kwargs):
        """ 
            Generate new data for the dataset, using a _datagen_fn function.
            Update the representations of the new data with current model.
        """
        new_data_dict = self.datagen_object.generate_data(
                                nloop=nloop,
                                param_original=self.param_original, 
                                outcomes=self.outcomes, 
                                param_embedded=self.param_embedded,
                                recn_error=self.recn_error,
                                metric_bd=self.metric_bd,
                                traj_data=self.traj_data,
                                latent_stats=self.latent_stats,
                                param_stats=self.param_stats,
                                cluster_object=self.cluster_object,
                                batch_mode=self.batch_object._batchmode,
                                **kwargs)
        if new_data_dict['outcomes'] is not None:    
            # Filter repeating data and remove duplicates before adding 
            n_new = len(new_data_dict['outcomes'])
            new_data_dict = uniquify_by_key(self, 'param_original', 
                                                  new_data_dict)
            if self._uniquify_metric:
                new_data_dict = uniquify_by_key(self, 'metric_bd', 
                                                      new_data_dict)
            # overlay new samples
            self.new_data = new_data_dict

            if new_data_dict is not None:
                if 'param_embedded' not in self.new_data.keys():
                    self.new_data['param_embedded'] = None

                if sample_plot_fn is not None: #and 'ls' in  in self.exp_normal:
                    spec_title = nloop if aux_title is None else aux_title
                    if 'spaces' in sample_plot_fn.keys():
                        self.plot_spaces(show_samples=True,
                                         spec_title=spec_title, 
                                         plot_fn=sample_plot_fn['spaces'],
                                         **kwargs)  
                    if 'traj' in sample_plot_fn.keys():
                        self.plot_statistics(show_samples=True,
                                               spec_title=spec_title,
                                               plot_fn=sample_plot_fn['traj'])
                # add data to dataset 
                if aux_save is None:    
                    self._add_data(new_data_dict, **kwargs)
            else:
                self._add_data(None, **kwargs)
        else:
            self.new_data = None
            self._add_data(None, **kwargs)

        # Save exploration progress
        exp_name = self.datagen_object.exploration_type
        n_samples = self.datagen_object.num_samples if nloop else \
            int(self.datagen_object.num_samples * self.datagen_object.k_init)
        if 'mape' not in exp_name and 'partfilt' not in exp_name:
            unique_bds = np.argmax(np.unique(self.metric_bd, axis=0), axis=1)
            self._write_discovery(nloop=nloop, n_new=n_samples, 
                      unique_bds=unique_bds, outcomes=self.outcomes)
            

    def _write_discovery(self, nloop, n_new, unique_bds, outcomes):
        """ Save trial output info """
        exploration_data = [nloop]+[0]+[n_new]+[0] \
                           + [len(unique_bds)] \
                           + [max(outcomes[:, 1])] \
                           + [sum(outcomes[:, 0]==0)] \
                           + [-1]

        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
        filepath = '{}/ref_data_{}_{}.csv'.format(
                        self.dirname, 
                        self.datagen_object.ctrl_name,
                        self.datagen_object.exploration_normal)
        with open(filepath, 'a') as outfile: 
            writer = csv.writer(outfile) 
            writer.writerows([exploration_data])


    """ PLOTTING WRAPPERS """

    def _get_plot_samples(self, mapping_fn, use_dims, num_samples=10):
        """ 
            Create a sample grid to visualise the transformation 
            from latent to reconstructed space.
            Depending on the latent space dimensionality, either sample across
            all dimensions, or only high variance ones.
        """
        rmin = np.min(self.param_embedded, axis=0)
        rmax = np.max(self.param_embedded, axis=0)
        r = []
        # Fix other dimensions to mid-range and sample high variance dims
        if self.param_embedded.shape[1]>10:
            rmid = np.mean(self.param_embedded, axis=0)
            if use_dims is not None:
                dims_emb = use_dims['emb']
            else:
                dims_emb = np.argsort(np.std(self.param_embedded, axis=0))[-2:]
            for i in range(self.param_embedded.shape[1]):
                if i in dims_emb:
                    r.append(np.linspace(rmin[i], rmax[i], num_samples))
                else:
                    r.append([rmid[i]])
        # Sample across all dimensions                    
        else:
            pwr = 1 / self.param_embedded.shape[1]
            stp = num_samples if self.param_embedded.shape[1]==2 \
                              else max(2, int(np.ceil(np.power(100, pwr))))
            for mn, mx in zip(rmin, rmax):
                r.append(np.linspace(mn, mx, stp))
        # Extract and transform the selected samples
        sample_emb = np.array(list(product(*r)))
        sample_mapped = mapping_fn(sample_emb)
        return sample_emb, sample_mapped


    # @timing
    def plot_spaces(self, plot_fn, spec_title, ae_type_traj,
                    recn_fn=None, aux_plot_fn=None, 
                    use_dims=None, test_idx=None,
                    show_samples=False, show_plots=False, save_path=None, 
                    **kwargs):
        dimred_set = set(['pca', 'tsne', 'combined', 'all', 'original'])
        savepath = save_path if save_path else self.dirname + '/plots'
        dimred_together = True
        show_samples = False if self.new_data is None else show_samples

        # Data argumets
        original_kwargs = {
            'param_original': self.param_original, 
            'param_embedded': self.param_embedded, 
            'param_reconstructed': self.param_reconstructed,
            'metric_bd': self.metric_bd,
            'metric_dim': \
                self.datagen_object.metric_dim,

            'traj_original': self.traj_original, 
            'traj_embedded': self.traj_embedded, 
            'traj_reconstructed': self.traj_reconstructed,

            'transform_latent': self.param_embedded.shape[1] > 2, 
            'transform_together': dimred_together,
            'latentdim_param_ae': self.param_embedded.shape[1],
            'latentdim_traj_ae': self.traj_embedded.shape[1] \
                if self.traj_embedded is not None else None,
            'cluster_object': self.cluster_object,
            'new_data': self.new_data if show_samples else None,

            'param_covmat': self.new_data['param_original'] if show_samples else self.param_original,
            'use_dims': use_dims,
            }
        # Plot arguments
        plot_kwargs = {
            'parameter_arch':self.datagen_object.parameter_arch,
            'outcomes': self.outcomes, 
            'loss_dict': self.training_loss_dict, 
            'ae_type_traj': ae_type_traj, 
            'cluster_to_class': self.cluster_object.cluster_to_class,
            'env_info': self.datagen_object.env_info,
            'show_plots': show_plots, 
            'savepath': savepath, 
            'spec_title': spec_title}

        if test_idx is not None:
            plot_kwargs.update({'test_idx': test_idx})


        # Show latent-recn space mapping samples
        if recn_fn is not None:
            sample_emb, sample_recn = self._get_plot_samples(recn_fn, use_dims)
            original_kwargs.update({'samples_recn': (sample_emb, sample_recn)})
            if show_samples and self.new_data['param_embedded'] is not None:
                self.new_data['param_recn'] = \
                    recn_fn(self.new_data['param_embedded'])
                original_kwargs.update({'new_data': self.new_data})

        # Show latent-outcome prediction samples
        if aux_plot_fn is not None:
            sample_aux_emb, samples_aux = \
                self._get_plot_samples(aux_plot_fn, use_dims, num_samples=20)
            original_kwargs.update(
                {'samples_aux': (sample_aux_emb, samples_aux, 
                                 aux_plot_fn.__name__)})

        # Make high-dimensional data plottable, reduce input dimension 
        if np.prod(self.inputdim_param_ae) > 2:
            # check if it is flat or images, and then flatten
            data_input_flat = self.param_original  
            # data_input_flat = self.param_original.reshape(self.num_datapoints, -1)
            original_kwargs.update({'param_original': data_input_flat})

            if 'original' in self.dimred or 'all' in self.dimred:
                # plot_dims = np.array([0,-1])
                plot_fn(dimred_mode='original', 
                        **original_kwargs, **plot_kwargs)

            if 'pca' in self.dimred or 'all' in self.dimred:
                mod_kwargs = transform_PCA_TSNE(transform_type=['pca'], 
                                                **original_kwargs)
                plot_fn(dimred_mode='pca', **mod_kwargs, **plot_kwargs)
            if 'tsne' in self.dimred or 'all' in self.dimred:
                mod_kwargs = transform_PCA_TSNE(transform_type=['tsne'],
                                                **original_kwargs)
                plot_fn(dimred_mode='tsne', **mod_kwargs, **plot_kwargs)
            if 'combined' in self.dimred or 'all' in self.dimred:
                mod_kwargs = transform_PCA_TSNE(transform_type=['combined'],
                                                **original_kwargs)
                plot_fn(dimred_mode='combined', **mod_kwargs, **plot_kwargs)        
            if not dimred_set.intersection(set(self.dimred)):
                err_text = "Appropriate dimensionality "\
                           "reduction method not provided"
                logger.error(err_text)
                raise Exception(err_text)
        else:
            plot_fn(dimred_mode='original', **original_kwargs, **plot_kwargs)


    def plot_statistics(self, plot_fn, show_samples=False,
                        show_plots=False, save_path=None, **kwargs):
        savepath = save_path if save_path else self.dirname+'/plots'
        plot_fn(traj_main=self.traj_data['main'], 
                param_original=self.param_original, 
                outcomes=self.outcomes, 
                new_data=self.new_data if show_samples else None,
                cluster_object=self.cluster_object,
                metric_bd=self.metric_bd, 
                metric_dim=self.datagen_object.metric_dim, 
                metric_name=self.datagen_object.metric_name,
                env_info=self.datagen_object.env_info,
                env_name=self.datagen_object.env_name, 
                savepath=savepath, show_plots=show_plots, **kwargs)


    # @timing
    def plot_training(self, plot_fn, save_path=None, load_path=None, **kwargs):
        savepath = save_path if save_path is not None else self.dirname+'/plots'
        loadpath = load_path if load_path is not None else self.dirname
        plot_fn(loadpath=loadpath, savepath=savepath, **kwargs)


    # @timing
    def plot_bd_data(self, plot_fn, save_path=None, **kwargs):
        savepath = save_path if save_path else self.dirname+'/plots'
        refpath = 'analysis/ref_files_{}_{}'.format(
                      self.datagen_object.env_name,
                      self.datagen_object.ctrl_name)
        plot_fn(loadpath=self.dirname+'/ref_data_{}_{}.csv'.format(
                            self.datagen_object.ctrl_name,
                            self.datagen_object.exploration_normal), 
                refpath=refpath, savepath=savepath, 
                metric_dim=self.datagen_object.metric_dim, 
                metric_name=self.datagen_object.metric_name, **kwargs)

