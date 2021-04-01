
"""
Author:         Anonymous
Description:
                Main experiment manager
"""

import os
import json
import glob
import logging
import numpy as np
import pandas as pd

import behaviour_representations.tasks.manage_data as mdata
import behaviour_representations.training.manage_training as mmodel

import behaviour_representations.utils.plotting as uplot
import behaviour_representations.analysis.plot_ae_training as aetrain
import behaviour_representations.analysis.plot_bd_fromfile as bdfile


from behaviour_representations.utils.utils import _SEED
from behaviour_representations.utils.utils import get_dict_item, set_dict_item

logger = logging.getLogger(__name__)




class ExperimentManager(object):
    """ 
        Class that manages the whole experiment execution based on given 
        specifications. 
        Includes: data storage, generation and batching, 
                  model selection and training.
    """

    def __init__(self, taskargs_dict):
        # Check if load data overwrites some settings:
        ### metric dim, data_dim -> raise error or change
        taskargs_dict = self._preprocess_arguments(taskargs_dict)
        # Create Data Manager object (includes datagen, clusters and batch)
        self.data_object = mdata.DataManager(**taskargs_dict)
        # Select the model and training routine
        self.model_object = mmodel.TrainingManager(data_object=self.data_object, 
                                                   **taskargs_dict)
        # Define which plots to save 
        self.num_loops = taskargs_dict['training']['num_loops']
        self.env_name = taskargs_dict['experiment']['environment']['id']
        self.exp_init = taskargs_dict['exploration']['initial']
        self.exp_normal = taskargs_dict['exploration']['normal']
        self.no_plots = False
        self.save_training_model = True

        # Graphs to plot at each loop
        self._what_to_plot = ['spaces', 'training'] 
        if self.model_object is None:
            self._what_to_plot = []
            self.data_object.skip_clust = True
        elif self.pca_param is not None:
            self._what_to_plot = ['spaces'] #, 'trajectories', 'training']
            self.data_object.skip_clust = True
        elif self.ae_param is not None and self.ae_traj is None:
            self._what_to_plot = ['spaces'] #, 'trajectories', 'training']
        # Just testing
        if not taskargs_dict['test_experiment']:
            self._what_to_plot = []
        self.data_object.skip_clust = True


    def _preprocess_arguments(self, taskargs_dict):
        """ 
            Perform checks to make sure experiment arguments are consistent,
            in order to avoid silent bugs 
        """
        self.ae_traj = taskargs_dict['training']['ae_traj']
        self.ae_param = taskargs_dict['training']['ae_param']
        self.pca_param = taskargs_dict['training']['pca_param']

        # Check config_file vs command line (priority) arguments
        if taskargs_dict['load_dataset_path'] is not None:
            with open(taskargs_dict['load_dataset_path']\
                      +'/experiment_metadata.json', 'r') as f:
                load_dict = json.load(f)
            check_list = [['experiment','type'],['experiment','metric','dim'],
                          ['experiment','environment','id']]
            for cl_ in check_list:
                if get_dict_item(taskargs_dict, cl_) != \
                   get_dict_item(load_dict, cl_):
                    input("\n>>> CLASH: {}, ENTER to overwrite.\n".format(cl_))
                    set_dict_item(taskargs_dict, cl_, 
                                  get_dict_item(load_dict, cl_))
                    logger.info("CHECK: overwriting {} by load.".format(cl_))
        
        # Force model and data seeds to be equal
        if taskargs_dict['seed_model'] != taskargs_dict['seed_data']:
            # input("\n>>> CLASH: 'seed_model'!='seed_data', "
            #       "ENTER to overwrite both to {}.\n".format(
            #                         taskargs_dict['seed_data']))
            taskargs_dict['seed_model'] = taskargs_dict['seed_data']

        # Check experiment argument inconsistencies
        if taskargs_dict['experiment']['type'] in ['synthetic', 'mnist']:
            taskargs_dict['training']['num_loops'] = 1
            logger.info("CHECK: static experiments run 1 loop.")

        # Check AE training argument inconsistencies
        if self.ae_param is not None and 'hyperplane' not \
           in taskargs_dict['experiment']['controller']['type']:
            # Match agent and AE network input
            if taskargs_dict['experiment']['controller']['type'] == \
               "displacement" and self.ae_param['type'] != 'fc':
                input("\n>>> ERROR: 'displacement' works only with 'fc' AE."
                      "ENTER to overwrite.\n")
                self.ae_param['type'] = 'fc'
                logger.info("CHECK: 'displacement' works only with 'fc' AE.")
            # Match size of loss and coeff
            llen = len(self.ae_param['loss_fn'])
            clen = len(self.ae_param['loss_coeff'])
            klen = len(self.ae_param['loss_kwargs'])
            if llen != clen or llen != klen:
                if clen==0: 
                    self.ae_param['loss_coeff'] = [1.]*llen
                elif clen<llen:
                    self.ae_param['loss_coeff'] = \
                        self.ae_param['loss_coeff'][:llen]
                if klen==0: 
                    self.ae_param['loss_kwargs'] = [1.]*llen
                elif clen<llen:
                    self.ae_param['loss_kwargs'] = \
                        self.ae_param['loss_kwargs'][:llen]

        # Check batching fixed_iterations is positive if defined
        if taskargs_dict['batching']['fixed_iters'] is not None and \
           taskargs_dict['batching']['fixed_iters'] <= 0 :
            taskargs_dict['batching']['fixed_iters'] = None
            logger.info("CHECK: batching fixed_iterations must be positive, "
                        "changing to None.")

        # Loading previous model, dataset or whole experiment
        self.load_model_path = taskargs_dict['load_model_path']
        self.load_dataset_path = taskargs_dict['load_dataset_path']

        self.flag_continue_data = False
        self.flag_continue_model = False
        self.nloop_ofst = 0

        # Load only the model and start with data generation from scratch
        if self.load_model_path is not None and self.load_dataset_path is None:
            # model_dir = os.path.join(self.load_model_path, 'saved_models')
            # self.nloop_ofst = len([mn for mn in os.listdir(model_dir) \
            #                        if 'loop' in mn])
            # Fit to new data
            self.flag_continue_model = True
            # # Just show new representation
            # self.flag_continue_model = False

        # Load only the dataset and start a new model from scratch
        elif self.load_model_path is None and self.load_dataset_path is not None:
            # data = pd.read_csv(filename)
            # self.nloop_ofst = data.nloop.iloc[-1]
            # Fit to new data
            self.flag_continue_data = False

        elif self.load_model_path is not None and self.load_dataset_path is not None:
            # Same experiment, continue where it left off
            if self.load_model_path == self.load_dataset_path:
                self.flag_continue_data = True
                self.flag_continue_model = True
                csvfile = glob.glob(
                    os.path.join(self.load_model_path,'*.csv'))[0]
                data = pd.read_csv(csvfile)
                # Get last loop and niter, 
                # if less than half re-start, otherwise new loop
                self.nloop_ofst = data.nloop.iloc[-1]

                if 'mape_niter' in taskargs_dict['exploration'].keys():
                    niter_goal = taskargs_dict['exploration']['mape_niter']
                    niter_done = (data.nloop == self.nloop_ofst).sum()
                    rmidx = data[data.nloop==self.nloop_ofst].index.tolist() 
                    data = data.drop(rmidx)
                    data.to_csv(csvfile, index=False)
                    # if niter_done != niter_goal:  ### not saving every iteration
                    #     self.niter_ofst = niter_done 
                    # if niter_done < niter_goal//2:
                    #     rmidx = data[data.nloop==self.nloop_ofst].index.tolist() 
                    #     data = data.drop(rmidx)
                    #     data.to_csv(csvfile, index=False)
                    # else:
                    #     self.nloop_ofst += 1 
                else:
                    self.nloop_ofst += 1 
            # Combine the model and dataset from different experiments - just normal load 
            else:
                self.flag_continue_data = False
                self.flag_continue_model = False

        # Save the modified arguments
        filename = '{}/experiment_metadata.json'.format(
                        taskargs_dict['experiment_directory'])
        with open(filename, 'w') as outfile:  
            json.dump(taskargs_dict, outfile, sort_keys=True, indent=4)

        return taskargs_dict


    def generate_data(self, nloop, save_dataset=True, verbose=True):
        """ 
            Data Generate wrapper function, modify generation based on loop
        """

        # Continue previous session
        if self.flag_continue_data and \
                                self.data_object.load_dataset_path is not None:
            self.data_object.load_dataset(write_loaded=False)
            if self.model_object is not None:
                self.data_object.update_representations(
                        embedding_fn=self.model_object.get_param_embedding, 
                        recn_fn=self.model_object.get_param_reconstruction)
            self.data_object.load_dataset_path = None

        # Load data from file in current loop
        if self.data_object.load_dataset_path is not None:
            self.data_object.load_dataset(write_loaded=True)
            self.data_object.load_dataset_path = None
        
        # Generate new data
        else:
            sample_plot_fn = {}
            if 'spaces' in self._what_to_plot:
                sample_plot_fn['spaces'] = uplot.plot_ae_spaces
            if 'trajectories' in self._what_to_plot:
                if 'striker' in self.env_name:
                    sample_plot_fn['traj'] = uplot.plot_traj_striker
                elif 'walker' in self.env_name:
                    sample_plot_fn['traj'] = uplot.plot_traj_walker

            ### IMPLEMENT LOADING THE TRAINING DATA CSV
            if nloop == self.nloop_ofst and not self.flag_continue_model:
                sample_plot_fn = None 

            # Check if the model is defined or not (PCA vs AE vs PS)
            if self.model_object is not None:
                genargs = dict(
                    embedding_fn=self.model_object.get_param_embedding, 
                    recn_fn=self.model_object.get_param_reconstruction,
                    out_brach_fn=self.model_object.get_out_prediction,
                    jacobian_fn=self.model_object.get_decoder_jacobian,
                    aux_plot_fn=self.model_object.get_dec_jac_stats,)
            else:
                genargs = dict(embedding_fn=None, recn_fn=None,
                               out_brach_fn=None, jacobian_fn=None, 
                               aux_plot_fn=None)

            # Generate new datapoints
            self.data_object.generate_data(
                sample_plot_fn=sample_plot_fn,
                nloop=nloop, 
                save_dataset=save_dataset, 
                verbose=verbose, 
                ae_type_traj=self.ae_traj, 
                nloop_ofst=self.nloop_ofst, 
                flag_continue_data=self.flag_continue_data,
                **genargs)


    def run_training(self, nloop, verbose=True):
        """ 
            Training step wrapper function.
        """

        # Continue previous session
        if self.flag_continue_model and self.load_model_path is not None:
            self.load_model_path = None

        # Load saved model from file in current loop
        if self.load_model_path is not None:
            self.no_plots = True
            self.load_model_path = None
            self.data_object.update_representations(
                    embedding_fn=self.model_object.get_param_embedding, 
                    recn_fn=self.model_object.get_param_reconstruction)

        # Fit the model to available data
        else:
            if self.model_object is not None:
                save_training_model = \
                    (nloop==self.num_loops-1) or self.save_training_model
                self.model_object.run_training(
                    nloop=nloop, 
                    save_training_model=save_training_model, 
                    verbose=verbose)
            else:
                logger.info(">>> skipping training.\n")


    def plot_data(self, nloop):
        """ 
            Plot step wrapper function, plot specified visualisations.
        """
        if self.no_plots:
            self.no_plots = False
            return
        if 'spaces' in self._what_to_plot:
            self.data_object.plot_spaces(
                    plot_fn=uplot.plot_ae_spaces, spec_title=nloop, 
                    ae_type_traj=self.ae_traj,
                    recn_fn=self.model_object.get_param_reconstruction,
                    # aux_plot_fn=self.model_object.get_out_prediction)
                    aux_plot_fn=self.model_object.get_dec_jac_stats)
        if 'training' in self._what_to_plot:
            try:
                self.data_object.plot_training(plot_fn=aetrain.plot_training, 
                                               spec_title=nloop)
            except:
                logger.warning(">>> skipping training plot [FIX BUG - NaN].\n")
        if 'trajectories' in self._what_to_plot:
            if 'striker' in self.env_name:
                plot_fn = uplot.plot_traj_striker
            elif 'walker' in self.env_name:
                plot_fn = uplot.plot_traj_walker
            else:
                plot_fn = bdfile.plot_bd_grid
            self.data_object.plot_statistics(plot_fn=plot_fn, 
                                             spec_title=nloop)
        # Finalise with grid coverage plot and training plot
        if nloop==self.num_loops-1:
            grid_type = 'outcome' #if 'walker' in self.env_name else 'outcome'
            self.data_object.plot_statistics(plot_fn=bdfile.plot_bd_grid,
                                             grid_type=grid_type, 
                                             save_path=self.data_object.dirname)
            self.data_object.plot_statistics(plot_fn=bdfile.plot_bd_traj, 
                                             save_path=self.data_object.dirname)
            self.data_object.plot_statistics(plot_fn=bdfile.plot_l2_dist, 
                                             save_path=self.data_object.dirname)
            if self.model_object is not None:
                self.data_object.plot_training(
                    plot_fn=aetrain.plot_training, 
                    save_path=self.data_object.dirname)
                    
