
"""
Author:         Anonymous
Description:
                Contains several features for analyzing and comparing the 
                performance across multiple experiments:

                - perfloss  : Performance w.r.t. test/train loss ratio and the 
                          used AE architecture
                - perfratio : Showing performance w.r.t. test, train loss and the 
                          used AE architecture
                - bd    : Plots the behaviour coverage graph
                - fit   : Plots the fitness graph 
"""

import os
import sys
import ast
import csv
import logging
import pickle
import glob 
import argparse
import time

import numpy as np
import pandas as pd
import multiprocessing as mpi 
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import matplotlib.ticker as ticker
from itertools import combinations
from functools import partial

from behaviour_representations.analysis import load_metadata, load_dataset
from behaviour_representations.utils.utils import timing

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument('-load', '--load_path', 
                    default=None, required=True,
                    help="Path to directory to plot.")

parser.add_argument('-save', '--save_path', 
                    default=None, required=False,
                    help="Path to directory where to save.")

parser.add_argument('-t', '--plot_type', nargs='+',
                    default=['bd'], # , 'fit', 'perfloss', 'perfratio', 'perfl2'
                    help="Select plot type(s):\n"
                         "'perfloss'\n"
                         "'perfratio'\n"
                         "'perfl2'\n"
                         "'bd'\n"
                         "'fit'\n")

parser.add_argument('-f', '--filter_string', 
                    default='',
                    help="Take into account experiments that contain this.")



def mscatter(x,y,ax=None, m=None, **kw):
    if not ax: ax=plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def _smooth(data, w_len=10000):
    window = np.ones(w_len)/w_len
    pad = np.ones(w_len//2)
    data_pad = np.concatenate([pad*data[0], data, pad[:-1]*data[-1]])
    data_smooth = np.convolve(data_pad, window, mode='valid')
    assert len(data_smooth) == len(data), \
            "data_smooth: {}; data: {}; smooth {}".format(
                len(data_smooth), len(data), len(smooth))
    return data_smooth


def load_bddata(filename):
    data = pd.read_csv(filename)
    if data.shape[1]!=8: return load_bddata_old(data.values.T, filename)
    data_dict = dict(zip(data.columns, data.values.T))
    data_dict['name'] = filename.split('/')[-1][9:-4]
    if '_mix_' not in data_dict['name'] and '_dde' not in data_dict['name']:
        data_dict['ratios'] = None
    elif len(data_dict['ratios'])>1:
        data_dict['ratios'][0] = data_dict['ratios'][1]
    return data_dict


def load_bddata_old(data, filename):
    # data = pd.read_csv(filename, header=None)
    data_dict = {}
    # Get experiment name
    data_dict['name'] = filename.split('/')[-1][9:-4]
    # Get loop number
    data_dict['nloop'] = data[0]
    # Get iteration number
    data_dict['niter'] = data[1]
    # Get number of samples per iteration
    data_dict['nsmp'] = data[2]
    # Get behaviour descriptor lists
    with mpi.Pool(processes=7) as pool:
        data_nbd = list(pool.map(ast.literal_eval, data[3]))
        # data_dict['labs'] = list(pool.map(ast.literal_eval, data[4]))  
        try:
            data_fit = list(pool.map(ast.literal_eval, data[4]))
        except:
            data_fit = None
        try:
            # data_ratios = list(pool.map(ast.literal_eval, data[5][1:]))
            data_ratios = list(pool.map(ast.literal_eval, data[6][1:]))
        except:
            data_ratios = None
        # [len(bds) for bds in data_nbd])
    data_dict['coverage'] = np.array(list(map(len, data_nbd)))
    data_dict['fitness'] = np.array(list(map(max, data_fit))) \
                                if data_fit is not None else None
    # Get mixing ratios if available
    if data_ratios is None or len(data_ratios)==0 \
       or '_mix_' not in data_dict['name']:
        data_dict['ratios'] = None
    else:
        tmp = np.array(data_ratios)
        assert np.min(tmp, axis=0)[1] > 0
        ratios = tmp[:,0]/tmp[:,1]
        data_dict['ratios'] = np.concatenate([[ratios[0]], ratios])
    return data_dict


def load_lossdata(filepath):
    """ Extract the losses """
    filepath  = '/'.join(filepath.split('/')[:-1])
    filename = os.path.join(filepath,
                            'saved_models/training_losses_param_ae.csv')
    try:
        data = pd.read_csv(filename, 
                           header=None, usecols=[1, 2], names=['test', 'train'])
    except:
        return None, None
    if len(data['test'].values)==0: return None, None
    final_test = ast.literal_eval(data['test'].values[-1])
    final_test = None if final_test[0] is None else sum(final_test) 
    final_train = sum(ast.literal_eval(data['train'].values[-1]))
    return final_test, final_train


def load_metadata_info(filepath):
    """ Extract the representation learning approach and latent size """
    filepath = '/'.join(filepath.split('/')[:-1])
    metadata = load_metadata(filepath)
    search_algo = metadata['exploration']['normal']
    if 'ps_' in search_algo:
        return 'PS', 'PS', search_algo
    if metadata['training']['ae_param'] is not None:
        arch_arg = metadata['training']['ae_param']['architecture']
        representation_type = 'AE-'+'-'.join([str(aa[1]) for aa in arch_arg])
    else:
        representation_type = 'PCA'
    dim_latent = metadata['training']['dim_latent']
    return 'LD-{}'.format(dim_latent), representation_type, search_algo



def mpi_get_dist(ij, param_original):
    focus_param = param_original[ij[0]]
    compare_param = param_original[ij[1]]
    return np.linalg.norm(focus_param-compare_param)


def load_get_l2dist(filepath):
    """ Extract the mean of the pairwise l2-dist of parameters in archive """
    filepath = '/'.join(filepath.split('/')[:-1])
    dataset = load_dataset(filepath)
    param_original = dataset['param_original']
    del dataset
    non_inf = param_original[0]<np.inf
    comb_idx = list(combinations(range(len(param_original)), 2))
    param_flat = param_original[:, non_inf]

    # ### indexing - cannot fit in memory
    # try:
    #     mm = np.linalg.norm(param_flat[None,...]-param_flat[:,None,:], axis=2)
    #     mean_dist = np.triu(mm).sum() / (np.triu(mm)>0).sum()
    #     return mean_dist
    # except Exception as e:
    #     print("\nNUMPY VERSION FAILED:", e)

    ### parallelizing with multiprocessing
    with mpi.Pool(mpi.cpu_count()-1) as pool:
        dist_list = pool.map(partial(mpi_get_dist, 
                                     param_original=param_flat), comb_idx)
    mean_dist = np.mean(dist_list)

    # ### bruteforce
    # dist_list = []
    # for i, pp in enumerate(param_original):
    #     focus_param = pp[non_inf]
    #     for j, qq in enumerate(param_original):
    #         if i != j:
    #             compare_param = qq[non_inf]
    #             dist_list.append(np.linalg.norm(focus_param-compare_param))
    # mean_dist = np.mean(dist_list)
    return mean_dist


################################################################################


def plot_performance_v_l2dist(refpath, graph_name, metric_name, metric_dim, 
                              filter_string, 
                              savepath=None, show_plots=False, spec_title=None, 
                              img_format='jpg', dpi=300, **kwargs):
    """ Get all .csv data files of same experiment type """
    if len(filter_string): graph_name = graph_name+'__'+filter_string
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    mlist = ['o', 'v', '^', '*','P', 's', 'X', '<', '>', 'p', 'D', 'd']
    colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    fname = os.path.join(refpath, 'saved_performance_v_l2dist.pkl')
    if os.path.exists(fname):
        print("\nLOADING:", fname)
        with open(fname, "rb") as f:
            experiments_dict = pickle.load(f)
    else:
        # Organise experiments to consider
        experiments_dict = {}
        if '.csv' in refpath:
            # If plotting only one experiment
            show_xloop = True
            exp_name = '_'.join(refpath.split('/')[-2].split('__')[:2])
            experiments_dict[exp_name] = refpath
        else:
            # Organise plotting multiple experiments
            filter_include = []
            filter_exclude = []
            for fterm in filter_string.split('+'):
                if len(fterm) and fterm[0]=='^':
                    filter_exclude += glob.glob('{}/ENV_*{}*'.format(refpath, 
                                                                     fterm[1:]))
                else:
                    filter_include += glob.glob('{}/ENV_*{}*'.format(refpath, 
                                                                     fterm))
            filter_exp = np.setdiff1d(filter_include, filter_exclude)
            # Extract only AE-based experiments
            for d in filter_exp:
                # Define the filters
                exp_name = d.split('/')[-1].split('__')[2]
                # Group csv files accordimg to filters
                if '__S' not in d.split('/')[-1]:
                    csv_file = glob.glob(d+'/S*/ref_data_*.csv')
                else:
                    csv_file = glob.glob(d+'/ref_data_*.csv')
                if exp_name in experiments_dict.keys():
                    experiments_dict[exp_name]['csv'] += csv_file
                else:
                    experiments_dict[exp_name] = dict(csv=csv_file)

        # Load and plot points for each experiment
        experiments_to_plot = sorted(experiments_dict.keys(), reverse=True)
        n_exp = len(experiments_to_plot)
        print("\n\n=== starting: L2-DIST GRAPH ===\n- {}".format(
                                            '\n- '.join(experiments_to_plot)))
        all_points = []
        for i, klab in enumerate(experiments_to_plot):
            print("\n> Extracting performance ({}/{}): '{}'".format(
                                                i+1, n_exp, klab))
            # Get all seeds of this experiment and average values
            l2dist, num_bd = [], []
            for cv in experiments_dict[klab]['csv']:
                print("   > Loading:", cv)
                latent_dim, latent_type, search_algo = load_metadata_info(cv)
                data_dict = load_bddata(cv)
                if len(data_dict['coverage']):
                    final_num_bd = data_dict['coverage'][-1]
                    num_bd.append(final_num_bd)
                    seedl2dist = load_get_l2dist(cv)
                    l2dist.append(seedl2dist)
                else:
                    print("   > EMPTY!")
                del data_dict
            if len(num_bd) == 0:
                print("> ALL EMPTY!")
                continue
            else:
                experiments_dict[klab]['num_bd'] = np.median(num_bd) 
                experiments_dict[klab]['l2dist'] = np.mean(l2dist) 
                experiments_dict[klab]['latent_type'] = latent_type
                experiments_dict[klab]['latent_dim'] = latent_dim
                experiments_dict[klab]['search_algo'] = search_algo
        # save experiments_dict
        with open(fname, "wb") as f:
            print("\nSAVING:", fname)
            pickle.dump(experiments_dict, f)

    # Plot graph
    total_l2 = [ed['l2dist'] for ed in experiments_dict.values()]
    total_nbd = [ed['num_bd'] for ed in experiments_dict.values()]
    l2min, l2max = min(total_l2), max(total_l2)
    bdmin, bdmax = min(total_nbd), max(total_nbd)
    total_ltype = [ed['latent_type'] for ed in experiments_dict.values()]
    total_ldim = [ed['latent_dim'] for ed in experiments_dict.values()]
    total_search = [ed['search_algo'] for ed in experiments_dict.values()]
    uniq_ltype = sorted(np.unique(total_ltype), reverse=True)
    uniq_ldim = sorted(np.unique(total_ldim), reverse=True)
    uniq_search = sorted(np.unique(total_search))

    szdict = dict(zip(uniq_ltype, (5*np.arange(1,len(uniq_ltype)+1))**2))
    mkdict = dict(zip(uniq_ldim, mlist[:len(uniq_ldim)]))
    expdict = dict(zip(uniq_search, colors[:len(uniq_search)]))

    plot_ltype = [szdict[rtl] for rtl in total_ltype]
    plot_ldim = [mkdict[rtd] for rtd in total_ldim]
    plot_search = [expdict[rts] for rts in total_search]
    plot_hatch = ['....' if 'PCA' in rtl else '' for rtl in total_ltype]
    
    # Plot experimant points
    for i in range(len(experiments_dict)):
        ax.scatter(total_l2[i], total_nbd[i],
                   s=plot_ltype[i], marker=plot_ldim[i], c=plot_search[i], 
                   hatch=plot_hatch[i],
                   label=total_search[i], 
                   edgecolor='k', lw=.4, alpha=0.5)

    # Plot lines to PS versions
    ax.set_xlim(0.1, 10*l2max)
    ax.set_ylim(0.8*bdmin, 1.05*bdmax)
    ps_search = [ek for ek in experiments_dict.keys() if 'ps_' in ek]
    for psexp in ps_search:
        sa = experiments_dict[psexp]['search_algo'] 
        xcoord = experiments_dict[psexp]['l2dist']
        ycoord = experiments_dict[psexp]['num_bd']
        # Add linear line
        ax.vlines(xcoord, ax.get_ylim()[0], ycoord, alpha=0.6, linestyles='--', 
                  lw=1, colors=expdict[sa], zorder=0)
        # Add ps_mape line
        ax.hlines(ycoord, ax.get_xlim()[0], xcoord, alpha=0.6, linestyles='--',
                  lw=1, colors=expdict[sa], zorder=0)

    # Labels
    max_bd = np.prod(metric_dim)
    ylabel = 'discovered behaviours (max {})'.format(max_bd)
    xlabel = 'mean L2-distance'
    # Add labels
    num_exp = len(experiments_dict)
    plt.minorticks_on()
    ax.set_title('{} (total: {} experiments)'.format(graph_name, num_exp))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", nonposx='clip')
    ax.grid(b=True, which='minor', alpha=0.2)
    ax.grid(b=True, which='major', alpha=0.5)

    # Add legends
    ldim_sorted = np.vstack([(plt.scatter([], [], c='k', marker=mkdict[ld]), 
                            ld) for ld in sorted(mkdict.keys(), reverse=True)])
    ltype_sorted = np.vstack([(plt.scatter([], [], c='w', edgecolor='k', 
                    s=szdict[lt], hatch='....' if 'PCA' in lt else ''), lt) \
                    for lt in sorted(szdict.keys(), reverse=True)])
    search_sorted = np.vstack([(plt.scatter([], [], c=expdict[lt]), lt) \
                                for lt in sorted(expdict.keys())])
    lgd_search = ax.legend(search_sorted[:,0], search_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(1., 1.01,), ncol=1)
    nsalg = 1 - len(search_sorted)*0.065
    lgd_ldim = ax.legend(ldim_sorted[:,0], ldim_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(1., nsalg,), ncol=1) #len(uniq_ldim))
    lgd_ltype = ax.legend(ltype_sorted[:,0], ltype_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(1.27, nsalg,), ncol=1) #len(uniq_ltype))


    ax.add_artist(lgd_ldim)
    ax.add_artist(lgd_ltype)
    ax.add_artist(lgd_search)

    # Save/show figure
    savepath = refpath if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/peformance_v_l2dist_analysis_loop_{:05d}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=(lgd_ldim, lgd_ltype), 
                    bbox_inches='tight', dpi=dpi) 
    else:
        plt.savefig('{}/{}__peformance_v_l2dist_analysis.{}'.format(
                    savepath, graph_name, img_format), 
                    format=img_format, bbox_extra_artists=(lgd_ldim, lgd_ltype),
                    # bbox_inches=mpl.transforms.Bbox([[0,0],[10,10.1]]),
                    bbox_inches='tight',
                    # pad_inches=0.3, 
                    dpi=dpi) 

    if show_plots:
        plt.show()
    else:
        plt.cla()



################################################################################


def plot_performance_v_ratio(refpath, graph_name, metric_name, metric_dim, 
                            filter_string, 
                            savepath=None, show_plots=False, spec_title=None, 
                            img_format='jpg', dpi=300, **kwargs):
    """ Get all .csv data files of same experiment type """
    if len(filter_string): graph_name = graph_name+'__'+filter_string
    # nrow, ncol = 1, 4
    nrow, ncol = 2, 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol,5*nrow))
    ax = ax.reshape((nrow, ncol))
    cm = plt.cm.get_cmap('jet')  # RdYlBu summer viridis_r
    # mdict = {2:'o', 5:'v', 10:'^', 20:'*', 50:'P', 100:'s'}
    mlist = ['o', 'v', '^', '*','P', 's', 'x', '<', '>', 'p', 'D', 'd']
    search_list = ['ls_mape_jacobian', 'ls_mape_standard', 
                   'ls_mape_mix_region', 'ls_mape_mix_ucb']

    # Organise experiments to consider
    experiments_dict = {}
    if '.csv' in refpath:
        # If plotting only one experiment
        show_xloop = True
        exp_name = '_'.join(refpath.split('/')[-2].split('__')[:2])
        experiments_dict[exp_name] = refpath
    else:
        # Organise plotting multiple experiments
        filter_include = []
        filter_exclude = []
        for fterm in filter_string.split('+'):
            if len(fterm) and fterm[0]=='^':
                filter_exclude += glob.glob('{}/ENV_*AE*{}*'.format(refpath, 
                                                                    fterm[1:]))
            else:
                filter_include += glob.glob('{}/ENV_*AE*{}*'.format(refpath, 
                                                                    fterm))
        filter_exp = np.setdiff1d(filter_include, filter_exclude)
        # Extract only AE-based experiments
        for d in filter_exp:
            # Define the filters
            exp_name = d.split('/')[-1].split('__')[2]
            # Group csv files accordimg to filters
            if '__S' not in d.split('/')[-1]:
                csv_file = glob.glob(d+'/S*/ref_data_*.csv')
            else:
                csv_file = glob.glob(d+'/ref_data_*.csv')
            if exp_name in experiments_dict.keys():
                experiments_dict[exp_name]['csv'] += csv_file
            else:
                experiments_dict[exp_name] = dict(csv=csv_file)

    # Load and plot points for each experiment
    experiments_to_plot = sorted(experiments_dict.keys(), reverse=True)
    n_exp = len(experiments_to_plot)
    print("\n\n=== starting: ANALYSIS GRAPH ===\n- {}".format(
                                            '\n- '.join(experiments_to_plot)))
    all_points = []
    for i, klab in enumerate(experiments_to_plot):
        print("\n> Extracting performance ({}/{}): '{}'".format(
                                            i+1, n_exp, klab))
        # Get all seeds of this experiment and average values
        loss_test, loss_train, num_bd = [], [], []
        for cv in experiments_dict[klab]['csv']:
            print("   > Loading:", cv)
            latent_dim, latent_type, _ = load_metadata_info(cv)
            data_dict = load_bddata(cv)
            data_coverage = data_dict['coverage']
            final_test_loss, final_train_loss = load_lossdata(cv)
            if len(data_coverage) and final_test_loss is not None:
                loss_test.append(final_test_loss)
                loss_train.append(final_train_loss)
                final_num_bd = data_coverage[-1]
                num_bd.append(final_num_bd)
            else:
                print("   > EMPTY!")
        if len(num_bd) == 0 or len(loss_test) == 0:
            print("> ALL EMPTY!")
            continue
        else:
            experiments_dict[klab]['loss_test'] = np.mean(loss_test)
            experiments_dict[klab]['loss_train'] = np.mean(loss_train)
            experiments_dict[klab]['num_bd'] = np.median(num_bd) 
            experiments_dict[klab]['latent_type'] = latent_type
            experiments_dict[klab]['latent_dim'] = latent_dim

    # Extract ps_mape as a reference
    ps_vals = []
    for ps in glob.glob('{}/ENV_*ps_mape*/S*/ref_data_*.csv'.format(refpath)):
            data_dict = load_bddata(ps)
            ps_vals.append(data_dict['coverage'][-1])
    psmedian = np.median(ps_vals)

    # Plot separate graphs
    total_test = [ed['loss_test'] for ed in experiments_dict.values() \
                                  if 'loss_test' in ed.keys()]
    total_train = [ed['loss_train'] for ed in experiments_dict.values() \
                                    if 'loss_test' in ed.keys()]
    total_nbd = [ed['num_bd'] for ed in experiments_dict.values() \
                              if 'loss_test' in ed.keys()]
    tsmin, tsmax = min(total_test), max(total_test)
    trmin, trmax = min(total_train), max(total_train)
    bdmin, bdmax = min(total_nbd), max(total_nbd + [psmedian])
    total_ltype = [ed['latent_type'] for ed in experiments_dict.values() \
                                     if 'loss_test' in ed.keys()]
    total_ldim = [ed['latent_dim'] for ed in experiments_dict.values() \
                                   if 'loss_test' in ed.keys()]
    uniq_ltype = sorted(np.unique(total_ltype))
    uniq_ldim = sorted(np.unique(total_ldim))
    szdict = dict(zip(uniq_ltype, (5*np.arange(1,len(uniq_ltype)+1))**2))
    mkdict = dict(zip(uniq_ldim, mlist[:len(uniq_ldim)]))
    for i, ss in enumerate(search_list):
        if sum([ss in ek for ek in experiments_dict.keys()])==0: 
            continue
        # Exctract values from dictionary
        plot_test = [vv['loss_test'] for kk, vv in experiments_dict.items() \
                                     if ss in kk and 'loss_test' in vv.keys()]  # if 'loss_test' in ed.keys()
        plot_train = [vv['loss_train'] for kk, vv in experiments_dict.items() \
                                       if ss in kk and 'loss_test' in vv.keys()]
        plot_nbd = [vv['num_bd'] for kk, vv in experiments_dict.items() \
                                 if ss in kk and 'loss_test' in vv.keys()]
        raw_ltype = [vv['latent_type'] for kk, vv in experiments_dict.items() \
                                       if ss in kk and 'loss_test' in vv.keys()]
        raw_ldim = [vv['latent_dim'] for kk, vv in experiments_dict.items() \
                                     if ss in kk and 'loss_test' in vv.keys()]
        plot_ltype = [szdict[rtl] for rtl in raw_ltype]
        plot_ldim = [mkdict[rtd] for rtd in raw_ldim]
        
        # # Plot experimant points
        ratios = np.array(plot_test)/np.array(plot_train)
        scatter = mscatter(ratios, plot_nbd,
                           ax=ax[i//ncol, i%ncol], 
                           cmap=cm, edgecolor='k', lw=.4, alpha=0.5, 
                           vmin=tsmin, vmax=tsmax,
                           c=plot_test, s=plot_ltype, m=plot_ldim,
                           norm=mpl.colors.LogNorm())

        ax[i//ncol, i%ncol].set_xlim(min(0.9, 0.5*min(ratios)), 
                                     max(1.1, 1.5*max(ratios)))

    # Fix figure
    if nrow == 1:
        fig.tight_layout(rect=[0.01, 0.1, 0.955, 0.92])
        cbar_ax = fig.add_axes([0.95, 0.21, 0.01, 0.65])
        # plt.subplots_adjust(wspace=-0.1)
        ty = 0.98
    else:
        fig.tight_layout(rect=[0.012, 0.075, 0.915, 0.97])
        cbar_ax = fig.add_axes([0.91, 0.115, 0.015, 0.82])
        plt.subplots_adjust(hspace=0.15)
        ty = 0.998
    # Labels
    max_bd = np.prod(metric_dim)
    clabel = 'test loss final'
    ylabel = 'discovered behaviours (max {})'.format(max_bd)
    xlabel = 'test / train loss ratio'

    # Add colorbar
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')  #, format='%.0e'
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(clabel, rotation=270)
    # cbar.ax.set_yscale("log", nonposy='clip')
    # Add labels
    num_exp = len(experiments_dict)
    fig.suptitle('{} (total: {} experiments)'.format(graph_name, num_exp), 
                 fontsize=16, y=ty) # if nrow>1 else 1.012)
    plt.minorticks_on()
    for i, ss in enumerate(search_list):
        ax[i//ncol, i%ncol].set_title(ss)
        if nrow==1:
            ax[i//ncol, i%ncol].set_xlabel(xlabel)
            if i==0: ax[i//ncol, i%ncol].set_ylabel(ylabel)
        elif nrow>1:
            if not i%2: ax[i//ncol, i%ncol].set_ylabel(ylabel)
            if i>=ncol: ax[i//ncol, i%ncol].set_xlabel(xlabel)
        # ax[i//ncol, i%ncol].set_xlim(0, 1.1*rmax)
        ax[i//ncol, i%ncol].set_ylim(0.95*bdmin, 1.05*bdmax)
        ax[i//ncol, i%ncol].set_xscale("log") #, nonposx='clip')
        # ax[i//ncol, i%ncol].set_yscale("log", nonposy='clip')
        ax[i//ncol, i%ncol].grid(b=True, which='minor', alpha=0.2)
        ax[i//ncol, i%ncol].grid(b=True, which='major', alpha=0.5)
        # Add linear line
        ax[i//ncol, i%ncol].vlines(1, *ax[i//ncol, i%ncol].get_ylim(),
                                linestyles='--', lw=1, colors='gray', zorder=0)
        # Add ps_mape line
        ax[i//ncol, i%ncol].hlines(psmedian, *ax[i//ncol, i%ncol].get_xlim(),
                                linestyles='--', lw=1, colors='green', zorder=0)
        # ax[i//ncol, i%ncol].set_aspect('equal', adjustable='box')
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    # Add legends
    ldim_sorted = np.vstack([(plt.scatter([], [], c='k', marker=mkdict[ld]), 
                        ld) for ld in sorted(mkdict.keys())])
    ltype_sorted = np.vstack([(plt.scatter([], [], c='w', edgecolor='k', 
                        s=szdict[lt]), lt) for lt in sorted(szdict.keys())])
    lgd_ldim = ax[-1,0].legend(ldim_sorted[:,0], ldim_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(0., -.12,), ncol=len(uniq_ldim))
    lgd_ltype = ax[-1,0].legend(ltype_sorted[:,0], ltype_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(0., -.2,), ncol=len(uniq_ltype))
    ax[-1,0].add_artist(lgd_ldim)
    ax[-1,0].add_artist(lgd_ltype)

    # Save/show figure
    savepath = refpath if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/peformance_v_ratio_analysis_loop_{:05d}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=(lgd_ldim, lgd_ltype), 
                    bbox_inches='tight', dpi=dpi) 
    else:
        plt.savefig('{}/{}__peformance_v_ratio_analysis.{}'.format(
                    savepath, graph_name, img_format), 
                    format=img_format, bbox_extra_artists=(lgd_ldim, lgd_ltype),
                    # bbox_inches=mpl.transforms.Bbox([[0,0],[10,10.1]]),
                    # bbox_inches='tight',
                    # pad_inches=0.3, 
                    dpi=dpi) 

    if show_plots:
        plt.show()
    else:
        plt.cla()


################################################################################


def plot_performance_v_loss(refpath, graph_name, metric_name, metric_dim, 
                            filter_string, 
                            savepath=None, show_plots=False, spec_title=None, 
                            img_format='jpg', dpi=300, **kwargs):
    """ Get all .csv data files of same experiment type """
    if len(filter_string): graph_name = graph_name+'__'+filter_string
    # nrow, ncol = 1, 4
    nrow, ncol = 2, 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol,5*nrow))
    ax = ax.reshape((nrow, ncol))
    cm = plt.cm.get_cmap('jet')  # RdYlBu summer viridis_r
    # mdict = {2:'o', 5:'v', 10:'^', 20:'*', 50:'P', 100:'s'}
    mlist = ['o', 'v', '^', '*','P', 's', 'x', '<', '>', 'p', 'D', 'd']
    search_list = ['ls_mape_jacobian', 'ls_mape_standard', 
                   'ls_mape_mix_region', 'ls_mape_mix_ucb']

    # Organise experiments to consider
    experiments_dict = {}
    if '.csv' in refpath:
        # If plotting only one experiment
        show_xloop = True
        exp_name = '_'.join(refpath.split('/')[-2].split('__')[:2])
        experiments_dict[exp_name] = refpath
    else:
        # Organise plotting multiple experiments
        filter_include = []
        filter_exclude = []
        for fterm in filter_string.split('+'):
            if len(fterm) and fterm[0]=='^':
                filter_exclude += glob.glob('{}/ENV_*AE*{}*'.format(refpath, 
                                                                    fterm[1:]))
            else:
                filter_include += glob.glob('{}/ENV_*AE*{}*'.format(refpath, 
                                                                    fterm))
        filter_exp = np.setdiff1d(filter_include, filter_exclude)
        # Extract only AE-based experiments
        for d in filter_exp:
            # Define the filters
            exp_name = d.split('/')[-1].split('__')[2]
            # Group csv files accordimg to filters
            if '__S' not in d.split('/')[-1]:
                csv_file = glob.glob(d+'/S*/ref_data_*.csv')
            else:
                csv_file = glob.glob(d+'/ref_data_*.csv')
            if exp_name in experiments_dict.keys():
                experiments_dict[exp_name]['csv'] += csv_file
            else:
                experiments_dict[exp_name] = dict(csv=csv_file)

    # Load and plot points for each experiment
    experiments_to_plot = sorted(experiments_dict.keys(), reverse=True)
    n_exp = len(experiments_to_plot)
    print("\n\n=== starting: ANALYSIS GRAPH ===\n- {}".format(
                                            '\n- '.join(experiments_to_plot)))
    all_points = []
    for i, klab in enumerate(experiments_to_plot):
        print("\n> Extracting performance ({}/{}): '{}'".format(
                                            i+1, n_exp, klab))
        # Get all seeds of this experiment and average values
        loss_test, loss_train, num_bd = [], [], []
        for cv in experiments_dict[klab]['csv']:
            print("   > Loading:", cv)
            latent_dim, latent_type, _ = load_metadata_info(cv)
            data_dict = load_bddata(cv)
            data_coverage = data_dict['coverage']
            final_test_loss, final_train_loss = load_lossdata(cv)
            if len(data_coverage) and final_test_loss is not None:
                loss_test.append(final_test_loss)
                loss_train.append(final_train_loss)
                final_num_bd = data_coverage[-1]
                num_bd.append(final_num_bd)
            else:
                print("   > EMPTY!")
        if len(num_bd) == 0:
            print("> ALL EMPTY!")
            continue
        else:
            experiments_dict[klab]['loss_test'] = np.mean(loss_test)
            experiments_dict[klab]['loss_train'] = np.mean(loss_train)
            experiments_dict[klab]['num_bd'] = np.median(num_bd) # np.mean(num_bd)
            experiments_dict[klab]['latent_type'] = latent_type
            experiments_dict[klab]['latent_dim'] = latent_dim

    # Extract ps_mape as a reference
    ps_vals = []
    for ps in glob.glob('{}/ENV_*ps_mape*/S*/ref_data_*.csv'.format(refpath)):
            data_dict = load_bddata(ps)
            ps_vals.append(data_dict['coverage'][-1])
    psmedian = np.median(ps_vals)

    # Plot separate graphs
    total_test = [ed['loss_test'] for ed in experiments_dict.values() \
                                  if 'loss_test' in ed.keys()]
    total_train = [ed['loss_train'] for ed in experiments_dict.values() \
                                    if 'loss_test' in ed.keys()]
    total_nbd = [ed['num_bd'] for ed in experiments_dict.values() \
                              if 'loss_test' in ed.keys()]
    tsmin, tsmax = min(total_test), max(total_test)
    trmin, trmax = min(total_train), max(total_train)
    cmin, cmax = min(total_nbd), max(total_nbd + [psmedian])
    total_ltype = [ed['latent_type'] for ed in experiments_dict.values() \
                                     if 'loss_test' in ed.keys()]
    total_ldim = [ed['latent_dim'] for ed in experiments_dict.values() \
                                   if 'loss_test' in ed.keys()]
    uniq_ltype = sorted(np.unique(total_ltype))
    uniq_ldim = sorted(np.unique(total_ldim))
    szdict = dict(zip(uniq_ltype, (5*np.arange(1,len(uniq_ltype)+1))**2))
    mkdict = dict(zip(uniq_ldim, mlist[:len(uniq_ldim)]))
    for i, ss in enumerate(search_list):
        if sum([ss in ek for ek in experiments_dict.keys()])==0: 
            continue
        # Exctract values from dictionary
        plot_test = [vv['loss_test'] for kk, vv in experiments_dict.items() \
                                     if ss in kk and 'loss_test' in vv.keys()]  # if 'loss_test' in ed.keys()
        plot_train = [vv['loss_train'] for kk, vv in experiments_dict.items() \
                                       if ss in kk and 'loss_test' in vv.keys()]
        plot_nbd = [vv['num_bd'] for kk, vv in experiments_dict.items() \
                                 if ss in kk and 'loss_test' in vv.keys()]
        raw_ltype = [vv['latent_type'] for kk, vv in experiments_dict.items() \
                                       if ss in kk and 'loss_test' in vv.keys()]
        raw_ldim = [vv['latent_dim'] for kk, vv in experiments_dict.items() \
                                     if ss in kk and 'loss_test' in vv.keys()]
        plot_ltype = [szdict[rtl] for rtl in raw_ltype]
        plot_ldim = [mkdict[rtd] for rtd in raw_ldim]
        # Plot experimant points
        scatter = mscatter(plot_test, plot_train, ax=ax[i//ncol, i%ncol], 
                           cmap=cm, edgecolor='k', lw=.4, alpha=0.5, 
                           vmin=cmin, vmax=1.01*cmax,
                           c=plot_nbd, s=plot_ltype, m=plot_ldim)

    # Fix figure
    if nrow == 1:
        fig.tight_layout(rect=[0.01, 0.1, 0.955, 0.92])
        cbar_ax = fig.add_axes([0.95, 0.21, 0.01, 0.65])
        # plt.subplots_adjust(wspace=-0.1)
        ty = 0.98
    else:
        fig.tight_layout(rect=[0.012, 0.075, 0.915, 0.97])
        cbar_ax = fig.add_axes([0.91, 0.115, 0.015, 0.82])
        plt.subplots_adjust(hspace=0.15)
        ty = 0.995
    # Add colorbar
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    cbar.ax.get_yaxis().labelpad = 15
    max_bd = np.prod(metric_dim)
    cbar.ax.set_ylabel('discovered behaviours (max {})'.format(max_bd), 
                        rotation=270)
    cticks = list(cbar.get_ticks())
    ctlbs = [t.get_text() for t in cbar.ax.get_yticklabels()]
    cbar.set_ticks(cticks + [psmedian])
    cbar.set_ticklabels(ctlbs + ['ps_mape'])

    # Add labels
    num_exp = len(experiments_dict)
    fig.suptitle('{} (total: {} experiments)'.format(graph_name, num_exp), 
                 fontsize=16, y=ty) # if nrow>1 else 1.012)
    # ax[i].set_title('{} ({})'.format(graph_name, len(plot_test)), pad=50)
    plt.minorticks_on()
    lq, uq = min(tsmin,trmin), max(tsmax,trmax)
    for i, ss in enumerate(search_list):
        ax[i//ncol, i%ncol].set_title(ss)
        if nrow==1:
            ax[i//ncol, i%ncol].set_xlabel('test loss final')
            if i==0: ax[i//ncol, i%ncol].set_ylabel('training loss final')
        elif nrow>1:
            if not i%2: ax[i//ncol, i%ncol].set_ylabel('training loss final')
            if i>=ncol: ax[i//ncol, i%ncol].set_xlabel('test loss final')
        ax[i//ncol, i%ncol].set_xscale("log", nonposx='clip')
        ax[i//ncol, i%ncol].set_yscale("log", nonposy='clip')
        ax[i//ncol, i%ncol].set_xlim(0.01*tsmin, 10*tsmax)
        ax[i//ncol, i%ncol].set_ylim(0.01*trmin, 10*trmax)
        ax[i//ncol, i%ncol].grid(b=True, which='minor', alpha=0.2)
        ax[i//ncol, i%ncol].grid(b=True, which='major', alpha=0.5)
        # Add linear line
        ax[i//ncol, i%ncol].plot(ax[i//ncol, i%ncol].get_xlim(), 
                                 ax[i//ncol, i%ncol].get_xlim(), 
                                 ls='--', lw=0.5, c='gray', zorder=0)
        # ax[i//ncol, i%ncol].set_aspect('equal', adjustable='box')
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    # Add legends
    ldim_sorted = np.vstack([(plt.scatter([], [], c='k', marker=mkdict[ld]), 
                        ld) for ld in sorted(mkdict.keys())])
    ltype_sorted = np.vstack([(plt.scatter([], [], c='w', edgecolor='k', 
                        s=szdict[lt]), lt) for lt in sorted(szdict.keys())])
    
    lgd_ldim = ax[-1,0].legend(ldim_sorted[:,0], ldim_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(0., -.12,), ncol=len(uniq_ldim))
    lgd_ltype = ax[-1,0].legend(ltype_sorted[:,0], ltype_sorted[:,1], 
            loc='upper left', bbox_to_anchor=(0., -.2,), ncol=len(uniq_ltype))
    ax[-1,0].add_artist(lgd_ldim)
    ax[-1,0].add_artist(lgd_ltype)

    # Save/show figure
    savepath = refpath if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/peformance_v_loss_analysis_loop_{:05d}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=(lgd_ldim, lgd_ltype), 
                    bbox_inches='tight', dpi=dpi) 
    else:
        plt.savefig('{}/{}__peformance_v_loss_analysis.{}'.format(
                    savepath, graph_name, img_format), 
                    format=img_format, bbox_extra_artists=(lgd_ldim, lgd_ltype),
                    # bbox_inches=mpl.transforms.Bbox([[0,0],[10,10.1]]),
                    # bbox_inches='tight',
                    # pad_inches=0.3, 
                    dpi=dpi) 

    if show_plots:
        plt.show()
    else:
        plt.cla()


################################################################################

def _get_klab(k):
    if 'ls_mape_mix_region-AE' in k:
        klab = 'PMS'
    elif 'ls_mape_mix_region-PCA' in k:
        klab = 'PMS-PCA'
    elif 'ls_mape_standard' in k:
        klab = 'PMS-no-jacobian'
    elif 'ls_mape_dde' in k:
        klab = 'DDE'
    elif 'ps_mape_directional' in k:
        klab = 'MAPE-IsoLineDD'
    elif 'ps_nn_uniform' in k:
        klab = 'ps_uniform'
    elif 'ps_nn_glorot' in k:
        klab = 'ps_glorot'
    elif 'ps_mape' == k:
        klab = 'MAPE-Iso'
    else:
        klab = k
    return klab


def plot_bd_graph(refpath, graph_name, metric_name, metric_dim, graph_type,
                  filter_string, show_xloop=False,
                  savepath=None, show_plots=False, spec_title=None, 
                  img_format='jpg', dpi=300, **kwargs):
    """ Get all .csv data files of same experiment type """
    if len(filter_string): graph_name = graph_name+'__'+filter_string
    # f, ax = plt.subplots(figsize=(15,10))
    f, ax = plt.subplots(2,1, figsize=(15,15), 
                              gridspec_kw={'height_ratios': [3, 1]})

    color_list = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyle_list = ["-","--","-.",":"]
    experiments_dict = {}
    plot_ratio_list = []

    markerdict = {'PMS':'o',
                  'PMS-PCA': 's',
                  'PMS-no-jacobian': 'X',
                  'MAPE-DDE': '^'}
    
    # Organise experiments to plot
    if '.csv' in refpath:
        # If plotting only one experiment
        show_xloop = True
        exp_name = '_'.join(refpath.split('/')[-2].split('__')[:2])
        experiments_dict[exp_name] = refpath
    else:
        # Organise plotting multiple experiments
        filter_include = []
        filter_exclude = []
        for fterm in filter_string.split('+'):
            if len(fterm) and fterm[0]=='^':
                filter_exclude += glob.glob('{}/ENV_*{}*'.format(refpath, 
                                                                 fterm[1:]))
            else:
                filter_include += glob.glob('{}/ENV_*{}*'.format(refpath, 
                                                                 fterm))
        filter_exp = np.setdiff1d(filter_include, filter_exclude)
        for d in filter_exp:
            # Define the filters
            exp_name = d.split('/')[-1].split('__')[2]
            # Group csv files accordimg to filters
            if '__S' not in d.split('/')[-1]:
                csv_file = glob.glob(d+'/S*/ref_data_*.csv')
            else:
                csv_file = glob.glob(d+'/ref_data_*.csv')
            if exp_name in experiments_dict.keys():
                experiments_dict[exp_name] += csv_file
            else:
                experiments_dict[exp_name] = csv_file

    # Load and plot bd coverage line of each experiment
    experiments_to_plot = sorted(experiments_dict.keys(), reverse=True)
    n_exp = len(experiments_to_plot)
    print("\n\n=== starting: {} GRAPH ===\n- {}".format(
                graph_type.upper(), '\n- '.join(experiments_to_plot)))


    # remap experiment order
    # new_idx = np.array([6,5,4,7,2,3,0,1])
    # experiments_to_plot = np.array(experiments_to_plot)[new_idx]


    color_dict = {}
    max_nsmp = 0
    for i, k in enumerate(experiments_to_plot):
        # klab = k[8:] if 'uniform' in k else k
        print("\n> Plotting {} ({}/{}): '{}'".format(graph_type,i+1,n_exp,k))

        klab = k # _get_klab(k)

        color = color_list[i % len(color_list)]
        # linestyle  = linestyle_list[i // len(color_list)]
        linestyle  = ':' if 'PCA-' in klab else '-'
        color_dict[k] = color

        # Extract experiment data
        plot_ratio = False
        num_smp, num_bd, mix_ratio, m_xloop = [], [], [], []
        for cv in experiments_dict[k]:
            print("   > Loading:", cv)
            data_dict = load_bddata(cv)
            data_nsmp = data_dict['nsmp'].astype(int)
            if len(data_nsmp):
                xl = np.where(data_dict['niter']==0)[0]
                # data_nsmp = data_dict['nsmp'].astype(int)
                m_xloop.append(xl * data_nsmp[xl])
                num_smp.append(np.arange(sum(data_nsmp)))
                num_bd.append(np.repeat(data_dict[graph_type], data_nsmp))
                if data_dict['ratios'] is not None:
                    plot_ratio = True
                    mix_ratio.append(np.repeat(data_dict['ratios'], data_nsmp))
            else:
                print("   > EMPTY!")
            del data_dict

        if sum([len(ns) for ns in num_smp]) == 0:
            print("> ALL EMPTY!")
            continue

        plot_ratio_list.append(plot_ratio)
        

        # Get median and quartiles if multiple experiments (seeds)
        if len(num_smp)>1:
            print("   >>>> merging", len(num_smp))
            out_bd_rng = [[], []]
            out_ratio_rng = [[], []]
            lens = sorted(np.unique([len(s) for s in num_bd]))
            if len(lens) > 1:
                # cut lists in chunks based on data lengths
                cut_idx = list(zip([0]+lens[:-1], lens))

                # Organize chunks for behaviour discovery
                tmp_bd = []
                for ndb in num_bd:
                    tmp_bd.append([ndb[s:e] for s,e in cut_idx])
                # collect the appropriate chunks if non empty
                tmp_bd_chunks = []
                for i in range(len(lens)):
                    tmp_bd_chunks.append([t[i] for t in tmp_bd if len(t[i])])
                # stack chunks of same size and get their statistics
                out_bd = []
                for tc in tmp_bd_chunks:
                    q1, qm, q3 = np.percentile(np.vstack(tc), [25, 50, 75], 
                                          # [0, 50, 100],  
                                          interpolation='nearest', axis=0)
                    out_bd.extend(qm)
                    out_bd_rng[0].extend(q1)
                    out_bd_rng[1].extend(q3)
                out_bd = np.array(out_bd)

                # Organize chunks for mixing ratios
                if plot_ratio: 
                    tmp_ratio = []
                    for mr in mix_ratio:
                        tmp_ratio.append([mr[s:e] for s,e in cut_idx])
                    # collect the appropriate chunks if non empty
                    tmp_mr_chunks = []
                    for i in range(len(lens)):
                        tmp_mr_chunks.append([t[i] for t in tmp_ratio \
                                                                if len(t[i])])
                    # stack chunks of same size and get their statistics
                    out_ratio = []
                    for tmr in tmp_mr_chunks:
                        # q1, qm, q3 = np.percentile(np.vstack(tmr), [25, 50, 75], 
                        #                           # [0, 50, 100],  
                        #                           interpolation='nearest', axis=0)

                        qm = np.mean(np.vstack(tmr), axis=0)
                        std = np.std(np.vstack(tmr), axis=0)
                        q1, q3 = qm-std, qm+std

                        out_ratio.extend(qm)
                        out_ratio_rng[0].extend(q1)
                        out_ratio_rng[1].extend(q3)
                    out_ratio = np.array(out_ratio)
            else:
                out_bd_rng[0], out_bd, out_bd_rng[1] = \
                    np.percentile(np.vstack(num_bd), [25, 50, 75],  # [0, 50, 100],  
                                  interpolation='nearest', axis=0)
                if plot_ratio: 
                    # out_ratio_rng[0], out_ratio, out_ratio_rng[1] = \
                    #     np.percentile(np.vstack(mix_ratio), [25, 50, 75], # [0, 50, 100],  
                    #                   interpolation='nearest', axis=0)
                    out_ratio = np.mean(np.vstack(mix_ratio), axis=0)
                    ratio_std = np.std(np.vstack(mix_ratio), axis=0)
                    out_ratio_rng = [out_ratio-ratio_std, out_ratio+ratio_std]

            # Get x-axis length
            out_smp = np.array(num_smp[np.argmax([len(ns) for ns in num_smp])])
            # Get y-axis values
            out_bd_rng = np.array(out_bd_rng)
            ax[0].fill_between(out_smp, out_bd_rng[0], out_bd_rng[1], 
                               color=color, alpha=0.2)
            if plot_ratio: 
                out_ratio_rng = np.clip(out_ratio_rng, 0, 1)
                ax[1].fill_between(out_smp, 
                                   _smooth(out_ratio_rng[0]), 
                                   _smooth(out_ratio_rng[1]), 
                                   color=color, alpha=0.2)
            # Get x-axis loop locations
            m_xloop = m_xloop[np.argmax(list(map(len, num_bd)))]
        else:
            out_bd = np.array(num_bd[0])
            out_smp = np.array(num_smp[0])
            if plot_ratio:
                out_ratio = np.array(mix_ratio[0])

        # Plot coverage curve of experiment type k
        if show_xloop:
            xloop = m_xloop[0]
            ax.plot(out_smp, out_bd, 
                    alpha=0.5, label="__"+k, color='k', linestyle='--')
            [ax[0].axvline(ln, c='b', ls='--', alpha=0.2) for ln in xloop]
            # make plot nice
            ax[0].set_xticks(list(ax[0].get_xticks())[1:] \
                             +list(xloop) + list(xloop))
            xlabels = ax[0].get_xticks().astype(int).tolist()
            xlabels[-len(xloop):] = ['\n     L{}'.format(i) \
                                          for i in range(len(xloop))]
            ax[0].set_xticklabels(xlabels)
            npts = len(out_smp)
            ax[0].set_xlim(-int(0.02*npts), npts+int(0.02*npts))
        else:
            if 'ls_mape_' in klab:
            # if klab in markerdict.keys():
                ax[0].scatter(m_xloop, out_bd[np.array(m_xloop)], 
                              s=10, color=color)
                              # marker=markerdict[klab])
            ax[0].plot(out_smp, out_bd, label=klab, color=color, 
                       linestyle=linestyle, alpha=0.5)

        # Check if mixing ratios plot is needed
        if plot_ratio:
            out_ratio = _smooth(out_ratio)
            if 'ls_mape_' in klab:
                ax[1].scatter(m_xloop, out_ratio[np.array(m_xloop)], 
                              s=10, color=color, alpha=0.5)
            ax[1].plot(out_smp, out_ratio, label=klab, color=color, 
                       linestyle=linestyle, alpha=0.5)
        # Align graphs
        if len(out_smp) > max_nsmp: max_nsmp = len(out_smp) 

    # Add labels and legends
    legends = []
    ax[0].set_title(graph_name)
    # ax[0].set_ylabel('# {} ( /{})'.format(metric_name, np.prod(metric_dim)))
    max_bd = np.prod(metric_dim)
    if graph_type=='coverage':
        ax[0].set_ylabel('discovered behaviours (max {})'.format(max_bd))
    elif graph_type=='fitness':
        ax[0].set_ylabel('best behaviour fitness')
    ax[0].set_xlabel('total trial evaluations')
    ax[0].grid(which='minor', alpha=0.2)
    ax[0].grid(which='major', alpha=0.5)
    # ax[0].set_ylim(5000,7000)
    ax[0].set_xlim(right=int(1.01*max_nsmp))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    h, l = ax[0].get_legend_handles_labels()
    legends.append(ax[0].legend(h, l, loc='upper left', 
                                bbox_to_anchor=(1, 1), ncol=1))
    if np.any(plot_ratio_list): # and graph_type != 'fitness':
        ax[1].set_ylabel('mixing ratio (ps/nsmp)')
        ax[1].set_xlabel('total trial evaluations')
        ax[1].minorticks_on()
        ax[1].grid(which='minor', alpha=0.5, linestyle=':')
        ax[1].grid(which='major', alpha=0.5)
        ax[1].set_xlim(right=int(1.01*max_nsmp))
        ax[1].set_ylim(-0.03, 1.03)
        ax[1].set_yticks([0, 0.5, 1])
        ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        h, l = ax[1].get_legend_handles_labels()
        legends.append(ax[1].legend(h, l, loc='upper left', 
                                    bbox_to_anchor=(1, 1), ncol=1))
    else:
        f.delaxes(ax[1]) 

    # Save/show figure
    plt.subplots_adjust(hspace = 0.1)
    savepath = refpath if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/bd_{}_loop_{:05d}.{}'.format(
                    savepath, graph_type, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=tuple(legends), 
                    bbox_inches='tight', dpi=dpi) 
    else:
        plt.savefig('{}/{}__bd_{}.{}'.format(
                    savepath, graph_name, graph_type, img_format), 
                    format=img_format, bbox_extra_artists=tuple(legends),
                    bbox_inches='tight', dpi=dpi) 
    if show_plots:
        plt.show()
    else:
        plt.cla()


################################################################################


if __name__ == "__main__":
    args = parser.parse_args()

    refpath=args.load_path
    if 'hyperplane' in refpath or 'HYPERPLANE' in refpath:
        dim = 100
        metric_dim = [dim, dim]
        metric_name = 'simple_grid'
    elif 'striker' in refpath or 'STRIKER' in refpath:
        dim = 30
        metric_dim = [17, dim, dim]
        metric_name = 'contact_grid'
    elif 'bipedal_walker' in refpath or 'BIPEDAL_WALKER' in refpath:
        dim = 10
        metric_dim = [dim//2, dim**2, dim//2, dim//2]
        metric_name = 'gait_grid'
    elif 'bipedal_kicker' in refpath or 'BIPEDAL_KICKER' in refpath:
        dim = 10
        metric_dim = [5*dim, 2*dim**2]
        metric_name = 'simple_grid'
    elif 'quadruped_walker' in refpath or 'QUADRUPED_WALKER' in refpath:
        dim = 10
        metric_dim = [dim**2, dim**2]
        metric_name = 'simple_grid'
    elif 'quadruped_kicker' in refpath or 'QUADRUPED_KICKER' in refpath:
        dim = 30
        metric_dim = [17, dim, dim]
        metric_name = 'contact_grid'

    idx = -2 if refpath[-1]=='/' else -1
    graph_name = refpath.split('/')[idx]

    if 'perfl2' in args.plot_type:
        plot_performance_v_l2dist(refpath, graph_name, metric_name, metric_dim,  
                                  filter_string=args.filter_string, 
                                  savepath=args.save_path)
    if 'perfratio' in args.plot_type:
        plot_performance_v_ratio(refpath, graph_name, metric_name, metric_dim,  
                                 filter_string=args.filter_string, 
                                 savepath=args.save_path)
    if 'perfloss' in args.plot_type:
        plot_performance_v_loss(refpath, graph_name, metric_name, metric_dim,  
                                filter_string=args.filter_string, 
                                savepath=args.save_path)
    if 'bd' in args.plot_type:
        plot_bd_graph(refpath, graph_name, metric_name, metric_dim, 
                      filter_string=args.filter_string, graph_type='coverage',
                      savepath=args.save_path)
    if 'fit' in args.plot_type:
        plot_bd_graph(refpath, graph_name, metric_name, metric_dim, 
                      filter_string=args.filter_string, graph_type='fitness',
                      savepath=args.save_path)

                         