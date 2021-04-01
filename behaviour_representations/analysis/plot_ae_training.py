
"""
Author:         Anonymous
Description:
                Loads the dataset from a given path,
                plots the minibatch losses
"""


import os
import argparse
import ast
import csv
import json
import pickle
import logging
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

from behaviour_representations.analysis import load_metadata, load_dataset
from behaviour_representations.utils.utils import bool_type


logger = logging.getLogger(__name__)



parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-load', '--loadpath', 
                    default=None, help="Path to the experiment files.")
parser.add_argument('-save', '--savepath', 
                    default=None, help="Path to save the plots.")

parser.add_argument('-avg', '--epoch_mean', type=bool_type, 
                    default=True, 
                    help="Plot only last minibatch loss of epoch.")
parser.add_argument('-win',  '--smth_win', type=int, 
                    default=1, help="Smoothing window size.")
parser.add_argument('-logy', '--logy', type=bool_type, 
                    default=True, help="Y axis in log scale.")


def moving_average(a, n=3):
    D = pd.Series(a)
    data_mean = np.array(D.rolling(n, min_periods=1).mean())
    data_std = np.array(D.rolling(n, min_periods=1).std())
    nan_idxs = np.where(np.isnan(data_mean))[0]
    data_mean[nan_idxs] = a[nan_idxs]
    nan_idxs = np.where(np.isnan(data_std))[0]
    data_std[nan_idxs] = data_std[nan_idxs+1]
    return data_mean, data_std


def load_lossdata(datapath, ae_name, epoch_mean):
    """ Extract the losses """
    epoch_data = []
    test_losses = []
    recn_data = []
    nloop_data = []
    param_ae_cum_loop_sizes = [0]

    filename = '{}/saved_models/training_{}.csv'.format(datapath, ae_name)
    # try:
    with open(filename) as csvfile: 
        reader = csv.reader(csvfile) 
        for i, row in enumerate(reader):
            row_data = list(map(ast.literal_eval, row)) 
            nloop_data.append(row_data[0])
            test_losses.append(row_data[1])
            if type(row_data[1][0])==list:
                if epoch_mean:
                    epoch_val = np.mean(row_data[2], axis=0) 
                else:
                    epoch_val = [row_data[-1]] 
                epoch_data.append(epoch_val)
            else:
                epoch_data.append(row_data[2])

    # Extract the loop lengths (epochs per loop)
    for nl in np.unique(nloop_data):
        param_ae_cum_loop_sizes.append(np.sum(np.array(nloop_data)==nl))
    # Stack the losses together
    test_losses = np.column_stack((itertools.zip_longest(*test_losses, 
                                                          fillvalue=0)))
    train_losses = np.column_stack((itertools.zip_longest(*epoch_data, 
                                                          fillvalue=0)))
    return train_losses, test_losses, param_ae_cum_loop_sizes
    # except:
    #     return None, None, None



def plot_training(loadpath, epoch_mean=True, loadmeta=None, **kwargs):

    loss_data_dict = {}
    loadmeta = loadpath if loadmeta is None else loadmeta
    all_data = load_dataset(loadpath)
    args_dict = load_metadata(loadmeta)
    classes_per_loop = all_data['classes_per_loop']
    has_recn = args_dict['training']['recn_init']
    # get param ae trainig data
    loss_data_dict['param'] = {}
    if args_dict['training']['pca_param'] is not None:
        loss_data_dict['param']['param_title'] = 'pca_var' 
        loss_data_dict['param']['loss_fn_param_ae'] = ['pca_var'] 
    else: 
        loss_data_dict['param']['param_title'] = 'losses_param_ae'
        loss_data_dict['param']['loss_fn_param_ae'] = \
            args_dict['training']['ae_param']['loss_fn']
    loss_data_dict['param']['param_ae_train_losses'], \
    loss_data_dict['param']['param_ae_test_losses'], \
    loss_data_dict['param']['param_ae_cum_loop_sizes'] = \
        load_lossdata(loadpath, 
                      loss_data_dict['param']['param_title'], epoch_mean)
    # get trajectory ae trainig data
    loss_data_dict['traj'] = {}
    if args_dict['training']['ae_traj'] is not None:
        # if args_dict['training']['ae_traj'] is None:
        #     loss_data_dict['traj']['loss_fn_traj_ae'] = 'None'
        # else:
        loss_data_dict['traj']['loss_fn_traj_ae'] = \
                args_dict['training']['ae_traj']['loss_fn'] 
        loss_data_dict['param']['traj_title'] = 'losses_traj_ae'
        loss_data_dict['traj']['traj_ae_train_losses'], \
        loss_data_dict['traj']['traj_ae_test_losses'], \
        loss_data_dict['traj']['traj_ae_cum_loop_sizes'] = \
            load_lossdata(loadpath, 
                          loss_data_dict['param']['traj_title'], epoch_mean)

    # print("\n\n=====================================================")
    # print(loss_data_dict['param']['param_ae_cum_loop_sizes'])
    # print(classes_per_loop)

    # call data plotting
    plot_training_data(has_recn, classes_per_loop, epoch_mean=epoch_mean, 
                       **loss_data_dict['param'], **loss_data_dict['traj'], 
                       **kwargs)


def plot_training_data(has_recn, classes_per_loop,
                       param_ae_train_losses, param_ae_test_losses, 
                       param_ae_cum_loop_sizes, loss_fn_param_ae, epoch_mean,
                       param_title, traj_title=None, 
                       traj_ae_train_losses=None, traj_ae_test_losses=None, 
                       traj_ae_cum_loop_sizes=None, loss_fn_traj_ae=None,
                       smth_win=None, logy=True, show_plots=False, dpi=200, 
                       spec_title=None, savepath=None, img_format='png', 
                       **kwargs):
    """ Plot the training loss and the data used """
    def moving_average(a, n=3):
        # D = pd.Series(a)
        # data_mean = np.array(D.rolling(n, min_periods=1).mean())
        # data_std = np.array(D.rolling(n, min_periods=1).std())
        # nan_idxs = np.where(np.isnan(data_mean))[0]
        # data_mean[nan_idxs] = a[nan_idxs]
        # nan_idxs = np.where(np.isnan(data_std))[0]
        # data_std[nan_idxs] = data_std[nan_idxs+1]
        # return data_mean, data_std
        return a, np.zeros_like(a)
        
    if param_ae_cum_loop_sizes is not None:
        cutoff = min(len(param_ae_cum_loop_sizes)-1, len(classes_per_loop))
    else:
        cutoff = len(classes_per_loop)
    classes_per_loop = classes_per_loop[:cutoff]

    classes_vals = np.fliplr([list(d.values()) for d in classes_per_loop])
    classes_vals = np.column_stack((itertools.zip_longest(*classes_vals, 
                                                          fillvalue=0)))
    classes_keys = np.fliplr([list(d.keys()) for d in classes_per_loop])

    num_epochs, num_losses = param_ae_train_losses.shape
    smth_win = smth_win if smth_win else max(100, int(num_epochs/10))

    # Figure
    f, axs = plt.subplots(2 + 1*(traj_ae_train_losses is not None), 1, 
                          sharex=False, # True, 
                          figsize=(18.6, 14 + 1*(loss_fn_traj_ae is not None)))

    # Plot PARAM AE losses
    _offset_idxs = []
    for nloss_ in range(num_losses):
        color = next(axs[1]._get_lines.prop_cycler)['color']
        # take care of recn_init offset in data
        _idxs = np.arange(param_ae_cum_loop_sizes[1], num_epochs) \
                       if has_recn and nloss_!=0 \
                       else np.arange(num_epochs)
        _offset_idxs.append(_idxs)
        # smooth out data
        data_mean, data_std = \
            moving_average(param_ae_train_losses[_idxs, nloss_], smth_win)
        n_epochs = np.arange(len(data_mean))

        axs[1].plot(_idxs, data_mean, c=color, label=loss_fn_param_ae[nloss_])
        axs[1].fill_between(_idxs, data_mean-data_std, data_mean+data_std,
                            color=color, alpha=0.4)
        # show min vals
        axs[1].axhline(min(param_ae_train_losses[_idxs, nloss_]), 
                      c=color, ls=':', alpha=0.4)
        [axs[1].axvline(l, c=color, ls='--', alpha=0.4) for l in \
            np.cumsum(param_ae_cum_loop_sizes)]
        # plot test data
        if param_ae_test_losses.any() != None:
            axs[1].plot(_idxs, param_ae_test_losses[_idxs, nloss_],
                        c=color, ls='--') #, label=loss_fn[nloss_])

    # Plot TRAJ AE losses
    if traj_ae_train_losses is not None:
        num_epochs_traj, num_losses_traj = traj_ae_train_losses.shape
        _traj_offset_idxs = []
        for nloss_ in range(num_losses_traj):
            color = next(axs[2]._get_lines.prop_cycler)['color']
            # take care of recn_init offset in data
            _idxs = np.arange(num_epochs_traj)
            _traj_offset_idxs.append(_idxs)
            # smooth out data
            data_mean, data_std = \
                moving_average(traj_ae_train_losses[_idxs, nloss_], smth_win)
            n_epochs = np.arange(len(data_mean))

            axs[2].plot(_idxs, data_mean, 
                        c=color, label=loss_fn_traj_ae[nloss_])
            axs[2].fill_between(_idxs, data_mean-data_std, data_mean+data_std,
                                color=color, alpha=0.4)
            # show min vals
            axs[2].axhline(min(traj_ae_train_losses[_idxs, nloss_]), 
                          c=color, ls=':', alpha=0.4)
            [axs[2].axvline(l, c=color, ls='--', alpha=0.4) \
                        for l in np.cumsum(traj_ae_cum_loop_sizes)]
            # plot test data
            if traj_ae_test_losses.any() != None:
                axs[2].plot(_idxs, traj_ae_test_losses[_idxs, nloss_], 
                            c=color, ls='--') #, label=loss_fn[nloss_])
        # Plot settings
        axs[2].set_title('[{}]'.format(traj_title.upper()))
        axs[2].set_ylabel('{} minibatch loss'.format(
                              'mean' if epoch_mean else 'last')) 
        if logy==True: axs[2].set_yscale("log", nonposy='clip')
        axs[2].grid(which='minor', alpha=0.2)
        axs[2].grid(which='major', alpha=0.5)
        axs[2].set_yticks(list(axs[2].get_yticks())[1:] + \
                          [min(traj_ae_train_losses[_traj_offset_idxs[l], l]) \
                          for l in range(num_losses_traj)])
        ylabels = axs[2].get_yticks().tolist()
        ylabels[-num_losses_traj:] = \
        ['min {}      \n({:.2e})     \n'.format(
            l, min(traj_ae_train_losses[_traj_offset_idxs[i], i])) \
            for i,l in enumerate(loss_fn_traj_ae)]
        axs[2].set_yticklabels(ylabels)
        axs[2].set_ylim(top=10*traj_ae_train_losses.max())
        handles, labels = axs[2].get_legend_handles_labels()
        lgd = axs[2].legend(handles, labels, loc='upper left')


    # Plot the data sample ratios
    width = max(10, int(0.3*min(param_ae_cum_loop_sizes[1:])))
    for d in range(len(classes_vals[-1])):
        axs[0].bar(np.cumsum(param_ae_cum_loop_sizes[:-1]), 
                             classes_vals[:,d], width, 
                             align='edge',
                             bottom=0 if d==0 else classes_vals[:, d-1])
                             # label="class: {}".format(classes_keys[0, d]))

    # Fix axes and labels
    axs[0].set_title('Class distribution over epochs')
    axs[1].set_title('[{}]'.format(param_title.upper()))
    axs[1].set_ylabel('{} minibatch loss'.format(
                          'mean' if epoch_mean else 'last')) 
    if logy==True: axs[1].set_yscale("log", nonposy='clip')
    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='upper left')
    axs[1].grid(which='minor', alpha=0.2)
    axs[1].grid(which='major', alpha=0.5)
    # make ticks nicer
    axs[1].set_xticks(list(axs[1].get_xticks())[1:] + \
                      list(np.cumsum(param_ae_cum_loop_sizes)))
    axs[1].set_yticks(list(axs[1].get_yticks())[1:] + \
                      [min(param_ae_train_losses[_offset_idxs[l], l]) \
                      for l in range(num_losses)])
    xlabels = axs[1].get_xticks().astype(int).tolist()
    xlabels[-len(np.cumsum(param_ae_cum_loop_sizes)):] = \
    ['\n      Loop {}'.format(i+1) for i \
        in range(len(np.cumsum(param_ae_cum_loop_sizes)))]
    axs[1].set_xticklabels(xlabels)
    ylabels = axs[1].get_yticks().tolist()
    ylabels[-num_losses:] = \
    ['min {}      \n({:.2e})     \n'.format(
        l, min(param_ae_train_losses[_offset_idxs[i], i])) \
        for i,l in enumerate(loss_fn_param_ae)]
    axs[1].set_yticklabels(ylabels)
    axs[0].set_ylabel('Class label distribution [num samples]')
    axs[0].set_xticks(np.cumsum(param_ae_cum_loop_sizes), \
                      ['\nloop {}'.format(i) for i in \
                      range(len(np.cumsum(param_ae_cum_loop_sizes)))])
    custom_lgd = [axs[0].scatter([], [], color='C0', lw=1), 
                  axs[0].scatter([], [], color='C1', lw=1)]
    custom_lab = ['class: -1', 'class: 0']
    axs[0].legend(custom_lgd, custom_lab, loc='upper left')
    axs[0].yaxis.grid(which="both", linestyle='--', alpha=0.5)
    axs[1].set_xlim(left=0, right=num_epochs)
    axs[0].set_xlim(left=0, right=num_epochs)
    axs[1].set_ylim(top=1.5*param_ae_train_losses.max())
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[-1].set_xlabel('Epochs')
    # axs[0].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:.e}'.format(x)))
    # Save/show figure
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if type(spec_title)==int:
            plt.savefig('{}/minibatch_losses_loop_{:05d}.{}'.format(
                        savepath, spec_title, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), 
                        bbox_inches='tight', dpi=dpi) 
        else:
            plt.savefig('{}/plot_minibatch_losses.{}'.format(
                        savepath, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), 
                        bbox_inches='tight', dpi=dpi) 
    if show_plots:
        plt.show()
    else:
        plt.cla()



if __name__ == "__main__":
    args = parser.parse_args()
    args.savepath = args.savepath if args.savepath \
                                  else args.loadpath #+'plots_additional'
    try:
        plot_training(**vars(args), show_plots=True)
    except Exception as e:
        logging.fatal(e, exc_info=True)


