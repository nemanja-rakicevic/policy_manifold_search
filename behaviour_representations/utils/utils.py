
"""
Author:         Anonymous
Description:
                Helper functions and constants
"""

import time
import logging
import argparse
import warnings
import inspect
import numpy as np

from functools import reduce, wraps  # forward compatibility for Python 3
from operator import itemgetter, getitem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)



""" CONSTANTS """

_TAB        = ' '*86
_SEED       = 100
_MARKER_SZ  = 50
_ALPHA_PTS  = 0.6
PLOT_DPI    = 100


""" CONSTANTS ENVS """

STRIKER_REW = 0.2
VEL_THRSH_PYBULLET = .009
VEL_THRSH_BOX2D = .009
EPS = 1e-05


""" DECORATORS """

def ignore_warnings(f):
    def wrap(*args, **kwargs):   
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = f(*args, **kwargs)
        return ret
    return wrap


def timing(f):
    @wraps(f)
    def wrap(s, *args, **kwargs):
        time1 = time.time()
        ret = f(s, *args, **kwargs)
        time2 = time.time()
        hours, rem = divmod(time2-time1, 3600)
        minutes, seconds = divmod(rem, 60)
        time_args = [int(hours), int(minutes), int(seconds),
                     int(seconds % 1 * 1000)]
        if f.__name__ == 'evaluate_population':
            if inspect.stack()[1][3] != 'init_batching':
              logger.info("({:02}:{:02}:{:02},{}) "\
                          "Evaluating {} instances of '{}'".format(
                          *time_args, len(kwargs['param_population']), 
                          s.environment.env.unwrapped.__class__.__name__))
        elif f.__name__ == 'run_training':       
            loss_names = [l.__name__ for l in s.loss_functions] \
                         if s.recn_init == 1 else s.loss_functions[-1].__name__
            logger.info("({:02}:{:02}:{:02},{}) "\
                        "AE {} fitted;".format(*time_args, loss_names))

        elif f.__name__ == 'generate_data':
            logger.info("({:02}:{:02}:{:02},{}) "\
                        "'{}' ({} samples)\n".format(*time_args,
                          s.exploration_type, s.num_samples))

        else:
            logger.info("({:02}:{:02}:{:02},{}) "\
                        "Function '{:s}' execution".format(
                        *time_args, f.__name__))
        return ret
    return wrap



""" ARGUMENTS TYPES """

def arch_type(string):
    try:
        layer_type, arg1, arg2 = string.split('-')
        return [layer_type, int(arg1), int(arg2)]
    except:
        logger.error("Invalid architecture definition")
        raise argparse.ArgumentTypeError("Invalid architecture definition.")


def bool_type(string):
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def range_type(string):
    if string in ['None', 'none', 'null']:
        return None
    elif (string).isnumeric():
        return [-int(string), int(string)]
    else:
        raise argparse.ArgumentTypeError('Invalid range definition.')


def scale_type(string):
    if string in ['None', 'none', 'null']:
        return None
    elif string in ['tanh', 'clip']:
        return string
    else:
        raise argparse.ArgumentTypeError('Invalid scaling type.')


""" FUNCTIONS """


def downsample_traj(traj_list, ds):
    tmp_out = []
    for t_list in traj_list:
        tmp_traj = []
        for t in t_list:
            tmp_traj.append(t[::ds])
        tmp_out.append(tmp_traj)
    return tmp_out


def pad_traj(traj_list, max_len):
    """ Adds padding to traj_list to make it a nd_array """
    n_dim = traj_list[0].shape[1]
    padded_list = np.ones((len(traj_list), max_len, n_dim))
    for i, t in enumerate(traj_list):
        tmp_array = t[-1] * np.ones((max_len, n_dim))
        tmp_array[:len(t)] = t                                                                                                                                                                                                                                                                   
        padded_list[i] = tmp_array
    return padded_list


def info_outcome(labels):
  if labels is None: return None
  return {"class {}".format(int(l)): np.count_nonzero(labels[:,0]==l) \
            for l in np.unique(labels[:,0])}



def uniquify_by_key(obj, key, new_data_dict):
    """ 
        Removes duplicates from new data, assuming *_original is unique 
    """
    if new_data_dict is None:
        logger.info("DUPLICATES: Empty dict!")
        return None
    new_data = new_data_dict[key]
    axt = tuple(range(1,len(new_data.shape)))
    orig_data = getattr(obj, key)
    tmp_data = np.vstack([orig_data, new_data]) if orig_data is not None \
                                                else new_data
    n_duplicates = tmp_data.shape[0] - np.unique(tmp_data, axis=0).shape[0]
    # Find and remove duplicates
    if n_duplicates > 0:
        unique_new_data = np.unique(new_data, axis=0)
        if orig_data is not None:
            unique_orig_data = np.unique(orig_data, axis=0) # redundant
            # Find the first unique element in new data which is absent 
            # from the original data
            _uq_idxs = \
                np.array([np.where((row==new_data).all(axis=axt))[0][0] \
                for row in unique_new_data \
                if (row != unique_orig_data).any(axis=axt).all()])
        else:
            _uq_idxs = np.array([np.where((row==new_data).all(axis=axt))[0][0]\
                                 for row in unique_new_data])
        # Adjust the new_data_dict
        _uq_idxs = _uq_idxs.flatten().astype(int)
        if len(_uq_idxs)==0:
            logger.info("DUPLICATES '{}': Removing {}, Keeping {}.".format(
                            key, n_duplicates, len(_uq_idxs)))
            return None
        for k,v in new_data_dict.items():
            if type(v) == np.ndarray:
                new_data_dict[k] = v[_uq_idxs]
            elif type(v) == list:
                items = itemgetter(*_uq_idxs)(v)
                if len(_uq_idxs) == 1:
                    new_data_dict[k] = [items]
                else:
                    new_data_dict[k] = list(items)
        # Log 
        logger.info("DUPLICATES '{}': Removing {}, Keeping {}."
                    "\n{}{}".format(key, n_duplicates, len(_uq_idxs), _TAB, 
                    info_outcome(new_data_dict['outcomes'])))
    return new_data_dict



def get_dict_item(dict_name, itempath):
    """Access a nested object in root by item sequence."""
    return reduce(getitem, itempath, dict_name)


def set_dict_item(dict_name, itempath, value):
    """Set a value in a nested object in dict_name by item sequence."""
    get_dict_item(dict_name, itempath[:-1])[itempath[-1]] = value



def shuffle(args):
    idx = np.random.permutation(len(args))
    if type(args) == np.ndarray:
        return args[idx]
    elif type(args) == list:
        return list(itemgetter(*idx)(args))


def organise_labels(labels):
    real_labels = np.zeros_like(labels)
    for i, l in enumerate(np.unique(labels)):
        real_labels[labels==l] = i
    return real_labels


def transform_PCA_TSNE(transform_type, latent_dim,
                        param_original, param_embedded, param_reconstructed, 
                        original_centroids, embedded_centroids,  
                        reconstructed_centroids, samples, transform_latent, 
                        num_components=None, transform_together=True):  
    """ 
        Perform a combination of PCA and tSNE dimensionality reduction, e
        ither together on both (input & reconstructed) datasets or separately. 
        Nice resources: 
            - https://www.datacamp.com/community/tutorials/introduction-t-sne
            - https://distill.pub/2016/misread-tsne/
    """
    def _return_transf(transf, num_components, data):
        if 'pca' in transf or 'combined' in transf:
            if 'combined' not in transf: num_components = 2
            pca = PCA(n_components=num_components, random_state=100)
            data = pca.fit_transform(data)
        if 'tsne' in transf or 'combined' in transf:
            tsne = TSNE(n_components=2, random_state=100)
            data = tsne.fit_transform(data)
        return data

    num_components = num_components if num_components \
                                    else min(50, int(.5*latent_dim)) 
                                    #int(self.param_originalinal.shape[1]/2) 
    ld, ldc, lr, lrc = len(param_original), len(original_centroids),\
                       len(param_reconstructed),len(reconstructed_centroids)
    # Reduce either together or separately
    if together==True:
        _result = _return_transf(transform_type, num_components,
                                 np.vstack([param_original, 
                                            original_centroids, 
                                            param_reconstructed, 
                                            reconstructed_centroids, 
                                            samples[1]]))
        orig_pca_tsne = _result[:ld,:]
        original_centroids_pca_tsne = _result[ld:ld+ldc,:]
        recn_pca_tsne = _result[ld+ldc:ld+ldc+lr,:]
        reconstructed_centroids_pca_tsne = _result[ld+ldc+lr:ld+ldc+lr+lrc,:]
        samples_recn_pca_tsne = _result[ld+ldc+lr+lrc:,:]
        # samples_recn_pca_tsne = np.ones((samples[1].shape[0],13))
    else:        
        _result_orig = _return_transf(transform_type, num_components,
                                      np.vstack([param_original, 
                                                 original_centroids]))
        orig_pca_tsne = _result_orig[:ld,:]
        original_centroids_pca_tsne = _result_orig[ld:ld+ldc,:]
        _result_recn = _return_transf(transform_type, num_components,
                                      np.vstack([param_reconstructed, 
                                                 reconstructed_centroids, 
                                                 samples[1]]))
        recn_pca_tsne = _result_recn[:lr,:]
        reconstructed_centroids_pca_tsne = _result_recn[lr:lr+lrc,:]
        samples_recn_pca_tsne = _result_recn[lr+lrc:,:]
    # Check if the latent space also needs to be reduced
    if transform_latent:
        le,lec = len(param_embedded),len(embedded_centroids)
        _result_embed = _return_transf(transform_type, num_components,
                                       np.vstack([param_embedded, 
                                                  embedded_centroids, 
                                                  samples[0]]))
        embed_pca_tsne = _result_embed[:le,:]
        embedded_centroids_pca_tsne = _result_embed[le:le+lec,:]
        samples_embed_pca_tsne = _result_embed[le+lec:,:]
    else:
        embed_pca_tsne = param_embedded
        embedded_centroids_pca_tsne = embedded_centroids
        samples_embed_pca_tsne = samples[0]
    return {**dict(param_original=orig_pca_tsne,
                   param_embedded=embed_pca_tsne,
                   param_reconstructed=recn_pca_tsne, 
                   original_centroids=original_centroids_pca_tsne,  
                   embedded_centroids=embedded_centroids_pca_tsne,
                   reconstructed_centroids=reconstructed_centroids_pca_tsne, 
                   samples=(samples_embed_pca_tsne, samples_recn_pca_tsne)),
            **kwargs}



def sample_torus(center_xy, min_R=1, max_R=2, num_samples=1000):
    """ https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly """

    # Get uniform samples
    uni_sample = np.random.uniform(size=(num_samples, 2))
    # Convert to r, theta samples
    r_samples = min_R + max_R*np.sqrt(uni_sample[:,0])
    theta_samples = 2*np.pi*uni_sample[:,1]
    # Convert to x,y samples
    x_samples = center_xy[0] + r_samples * np.cos(theta_samples)
    y_samples = center_xy[1] + r_samples * np.sin(theta_samples)
    return np.vstack([x_samples, y_samples]).T

