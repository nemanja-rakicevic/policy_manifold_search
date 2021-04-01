
"""
Author:         Anonymous
Description:
                Main file to run the algorithm.
"""

import os
import sys
import argparse
import datetime
import json
import logging
import warnings
warnings.filterwarnings('ignore')
import operator
from functools import reduce  # forward compatibility for Python 3

import behaviour_representations.tasks.experiment_manage as mm

from behaviour_representations.utils.utils import arch_type, bool_type, range_type, scale_type
from behaviour_representations.utils.utils import set_dict_item


logger = logging.getLogger(__name__) 


# STARTING POINT
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--config_file', required=True)

# Experiment arguments
parser.add_argument('--xname', default=None)
parser.add_argument('--xdir', default=None)

parser.add_argument('--experiment__type',
                    default='nn_policy', 
                    help="Select data/task to use:\n"
                         "'displacement' (7d policy)\n"
                         "'nn_policy' (NN d policy)\n")

parser.add_argument('--controller__architecture', nargs='+', type=int, 
                    default=[], help="Controller architecture blueprint.")
parser.add_argument('--controller__type', 
                    help="Controller type.")

parser.add_argument('--metric__dim', type=int, 
                    default=1, help="Bins for behaviour the container")
parser.add_argument('--metric__type',
                    default='contact_bins', 
                    choices=['contact_bins', 
                             'contact_vector', 
                             'contact_grid', 
                             'gait_descriptor', 
                             'gait_grid',
                             'gait_grid_small',
                             'simple_grid'])

parser.add_argument('--environment__id', 
                    default='striker',  # default='ls_particles_exact',
                    help="Environment to use:\n"
                         "'striker_augmented'\n"
                         "'striker_augmented_mix_scale'\n"
                         "'bipedal_walker_augmented'\n"
                         "'bipedal_walker_augmented_mix_scale'\n"
                         "'bipedal_kicker_augmented'\n"
                         "'bipedal_kicker_augmented_mix_scale'\n")

# Exploration arguments
parser.add_argument('--num_samples', type=int, 
                    default=1000, 
                    help="Number of population samples per iteration")
parser.add_argument('--k_init', type=float, 
                    default=2, 
                    help="Multiplier for the initial population sample size.")

parser.add_argument('--init_range', type=range_type, 
                    help="Initial range for the parameter search")
parser.add_argument('--limit_range', type=range_type, 
                    help="Allowed range for the parameter search")
parser.add_argument('--scale_type', type=scale_type, 
                    help="Parameter normalisation type")


parser.add_argument('--exploration__initial', 
                    default='ps_nn_uniform',  # default='ps_uniform',
                    help="Exploration strategy to use:\n"
                         "'ps_uniform'\n"
                         "'ps_normal'\n"
                         "'ps_nn_uniform'\n"
                         "'ps_nn_glorot'\n")

parser.add_argument('--exploration__normal', 
                    default='ls_medoids_interp_exact',  # default='ls_particles_exact',
                    help="Exploration strategy to use:\n"
                         "'ls_mape_mix_region'\n"
                         "'ls_mape_standard'\n"
                         "'ls_mape_dde'\n"
                         "'ps_nn_uniform'\n"
                         "'ps_nn_glorot'\n")

parser.add_argument('--exploration__mape_use_fitness', type=bool_type, 
                    default=False, 
                    help="MAPE: use fitness information when assigning.")
parser.add_argument('--exploration__mape_niter', type=int, 
                    default=100, 
                    help="MAPE: algorithm iterations.")
parser.add_argument('--exploration__mape_decay', type=float, 
                    default=0, 
                    help="MAPE: decay coefficient.")
parser.add_argument('--exploration__mape_sigma', type=float, 
                    default=1, 
                    help="MAPE: sampling standard deviation.")

# Clustering arguments
parser.add_argument('--clustering__algorithm',
                    default='bic', choices=['kmeans', 'bic', 'gmeans'],
                    help="Exploration strategy to use:\n"
                         "'kmeans'\n"
                         "'bic'\n"
                         "'gmeans'\n")
parser.add_argument('--max_clusters', type=int,
                    default=16,
                    help="Maximum number of clusters")

parser.add_argument('--strictness', type=int, choices=range(5),
                    default=0,
                    help="Strictness factor for the Andreson-Darling test")

parser.add_argument('--n_clusters_to_sample', type=int,
                    default=4, help="Number of clusters to sample from")
parser.add_argument('--n_samples_in_cluster', type=int,
                    default=16, 
                    help="Number of samples to take from each cluster")


# AutoEncoder arguments 

### Parameter AE
parser.add_argument('--ae_param__type',
                    default='fc', choices=['fc', 'cnn'],
                    help="Parameter AutoEncoder type to use.")
parser.add_argument('--ae_param__architecture', nargs='+', type=arch_type, 
                    default=[], help="AutoEncoder architecture blueprint.")
parser.add_argument('--ae_param__lr', type=float,
                    default=0.0001, help="PARAM AE learning rate")
parser.add_argument('--ae_param__num_epochs', type=int,
                    default=10000, help="Number of epochs for AE training")
parser.add_argument('--ae_param__early_stop', type=bool_type,
                    default=False, 
                    help="Activate early stopping based on train/test loss")

parser.add_argument('--ae_param__branch', nargs='+', type=arch_type, 
                    default=[], help="AutoEncoder architecture blueprint.")
parser.add_argument('--ae_param__loss_fn', nargs='+',
                    default=['reconstruction'], 
                    help="Select data/task to use:\n"
                         "'reconstruction'\n")
parser.add_argument('--ae_param__loss_coeff', nargs='+', type=float, 
                    default=None,
                    help="Weights for the given loss components. "\
                         "Has to be the same length as --loss_fn")

### Trajectory AE
parser.add_argument('--ae_traj__type',
                    default='None', choices=['None', 'fc', 'conv1d', 'lstm'],
                    help="Trajectory AutoEncoder type to use.")
parser.add_argument('--ae_traj__architecture', nargs='+', type=arch_type, 
                    default=[], help="AutoEncoder architecture blueprint.")
parser.add_argument('--ae_traj__lr', type=float,
                    default=0.0001, help="TRAJ AE learning rate")
parser.add_argument('--ae_traj__num_epochs', type=int,
                    default=10000, help="Number of epochs for AE training")
parser.add_argument('--ae_traj__early_stop', type=bool_type,
                    default=False, 
                    help="Activate early stopping based on train/test loss")
parser.add_argument('--ae_traj__loss_fn', nargs='+',
                    default=['reconstruction'], 
                    help="Select data/task to use:\n"
                         "'reconstruction'\n")
parser.add_argument('--ae_traj__loss_coeff', nargs='+', type=float, 
                    default=None,
                    help="Weights for the given loss components. "\
                         "Has to be the same length as --loss_fn_traj_ae")

# Training arguments

parser.add_argument('--num_loops', type=int,
                    default=1, help="Number of trials to run")
parser.add_argument('--recn_init', type=int, 
                    default=0, help="RECONSTRUCTION only epochs coeff.")
parser.add_argument('--testset_ratio', type=float, 
                    default=0., help="Evaluate on test set during training")
parser.add_argument('--dim_latent', type=int, 
                    default=2, help="Size of the latent dimension")

# Batching arguments

parser.add_argument('--batching__fixed_iters', type=int,
                    default=None,  
                    help="Fixed or flexible number of iterations.")

parser.add_argument('--batchsize_normal', type=int,
                    default=64,  
                    help="Batch size for the optimisation. (Default: 64)")
parser.add_argument('--batchsize_init', type=int,
                    default=None, 
                    help="Batch size for the reconstruction phase. "\
                         "(Default: None)")

parser.add_argument('--batchmode_normal', 
                    default='standard',
                    help="Select data/task to use:\n"
                         "'standard'\n")
parser.add_argument('--batchmode_init', 
                    default=None, 
                    help="Initial batching mode, same options as batchmode.")

# Other arguments
parser.add_argument('--seed_model', type=int, 
                    default=100, help="Seed for the random number generator.")
parser.add_argument('--seed_data', type=int, 
                    default=100, help="Seed for the random number generator.")
parser.add_argument('--seed_task', type=int, 
                    default=100, help="Seed for the random number generator.")

parser.add_argument('--load_dataset_path', 
                    default=None, 
                    help="Path to the dataset file with initial population.")
parser.add_argument('--load_model_path', 
                    default=None, help="Path to load the last saved model.")
parser.add_argument('--test_experiment', type=bool_type,
                    default=True, 
                    help="Flag defining the name of the experiment as a test.")
parser.add_argument('--dimred_mode', nargs='+',
                    default=['original'],
                    choices=['pca', 'tsne', 'combined', 'all', 'original'], 
                    help="Select dimensionality reduction technique "\
                         "for visualisation:\n"
                         "'pca'\n"
                         "'tsne'\n"
                         "'combined' (pca+tsne)\n"
                         "'all'\n"
                         "'original'\n")



def recursive_items(mm, dictionary, curr_path):
    found = []
    for key, value in dictionary.items():
        if key == mm:
            found += [curr_path + [key]]
        elif type(value) is dict:
            path = curr_path + [key]
            found += recursive_items(mm, value, path)
    return found


def _load_args():
    """ 
        Load experiment args from config and overwrite from cmd line 
    """
    args = parser.parse_args()
    with open(args.config_file) as json_file:
        metadata = json.load(json_file)
    metadata['config_file'] = args.config_file
    # additional args to overwrite file
    if len(sys.argv) > 3:
        args = vars(args)
        for a in sys.argv[3:]:
            if '--' == a[:2]:
                k = a[2:]
                # k is the actual key
                if k in metadata.keys():
                    metadata[k] = args[k]
                else:
                    # k is lower level key, with path
                    keypath = k.split('__')
                    listpath = recursive_items(keypath[-1], metadata, [])
                    if len(listpath)==1:
                        path = listpath[0]
                    # multiple paths with same key
                    elif len(listpath)>1:
                        path = [i for i in listpath if keypath[-2] in i][0]
                    else:
                        raise ValueError("Argument '{}' not supported "
                                         "in this experiment!".format(k))
                    set_dict_item(metadata, path, args[k])
    return metadata

 
def _start_logging(taskargs):
    """ 
        Define the experiment name and start logging.
        - If continuing previous experiment, experiment metadata is copied
        - Otherwise, arguments are inputted and used to create dirname
    """
    # Continue previous experiment
    if taskargs['load_model_path'] == taskargs['load_dataset_path'] \
       and taskargs['load_model_path'] is not None:
        dirname = taskargs['load_dataset_path']
        num_loops = taskargs['training']['num_loops']
        with open(dirname + '/experiment_metadata.json', 'r') as f:
            taskargs = json.load(f)
        # taskargs['experiment_directory'] = dirname
        taskargs['load_model_path'] = dirname
        taskargs['load_dataset_path'] = dirname
        taskargs['training']['num_loops'] = num_loops
        logger.info('Continuing loaded session: {}\n'.format(dirname))
    # Create new experiment
    else:
        env_info = taskargs['experiment']['environment']['id']
        ctrl_info = taskargs['experiment']['controller']['type'].split('-')[-1]
        exp_start = taskargs['exploration']['initial'].split('_')[-1]
        seed_num = taskargs['seed_data']
        exp_info = taskargs['exploration']['normal']

        # Additional experiment flags
        xname = '---'+taskargs['xname'] if taskargs['xname'] is not None else ''
        xdir = '---'+taskargs['xdir'] if taskargs['xdir'] is not None else ''

        # Training type
        ae_info = ''
        if 'decay' in taskargs['config_file']:
            ae_info = ae_info + '-DECAY_'
        if taskargs['training']['ae_param'] is not None and \
           'elbo' in taskargs['training']['ae_param']['loss_fn']:
            ae_info = ae_info + '-ELBO_'
        if taskargs['training']['ae_param'] is not None and \
           'l2' in taskargs['training']['ae_param']['loss_fn']:
            ae_info = ae_info + '-L2_'
        if taskargs['training']['ae_param'] is not None and \
           'l1' in taskargs['training']['ae_param']['loss_fn']:
            ae_info = ae_info + '-L1_'

        # Latent representation learning type
        ld_info = '-LD{}'.format(taskargs['training']['dim_latent']) \
                        if 'ls' in taskargs['exploration']['normal'] else ''
        if '_pca.json' in taskargs['config_file']:
            ld_info = '-PCA' + ld_info 
        elif 'pms' in taskargs['config_file'] or 'dde' in taskargs['config_file']:
            arch_arg = taskargs['training']['ae_param']['architecture']
            ae_arch = '-'.join([str(aa[1]) for aa in arch_arg])
            ld_info = '-AE-' + ae_arch + ld_info 

        # Main experiment directory
        # dirname = 'experiment_data'
        dirname = os.path.join(os.getcwd(), 'experiment_data')
        dirname = os.path.join(dirname, 'env__{}{}'.format(env_info, xdir))
        dirname = os.path.join(dirname, 'ENV_{}_{}__CFG__{}{}{}{}'.format(
            env_info, ctrl_info, 
            exp_info, ld_info, ae_info,
            xname))
        dirname = os.path.join(dirname, 'S{}---{}'.format(
            seed_num, datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))
        taskargs['experiment_directory'] = dirname

        # Save session hyperparameters
        os.makedirs(dirname)
        logger.info('Starting session: {}\n'.format(dirname))

    # Start logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-52s '
                               '%(levelname)-8s %(message)s',
                        handlers=[
                            logging.FileHandler(
                                '{}/logging_data.log'.format(dirname)),
                            logging.StreamHandler()
                        ])
    return taskargs


def main_run():

    # Initialise arguments
    task_kwargs = _load_args()
    task_kwargs = _start_logging(task_kwargs)
    experiment = mm.ExperimentManager(task_kwargs)

    # Main loop 
    for nloop in range(experiment.nloop_ofst, experiment.num_loops):
        logger.info('--- Running loop: {} ---\n'.format(nloop))
        # Generate initial/new data
        experiment.generate_data(nloop=nloop)
        # Fit AE to new data and save current model
        experiment.run_training(nloop=nloop)
        # Save the visualisations
        experiment.plot_data(nloop=nloop)
    
    logger.info("\t>>> TRAINING DONE. [total: {} datapoints]".format(
        experiment.data_object.num_datapoints))



if __name__ == "__main__":
    try:
        main_run()
    except Exception as e:
        logging.fatal(e, exc_info=True)
