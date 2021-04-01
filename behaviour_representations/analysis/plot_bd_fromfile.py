
"""
Author:         Anonymous
Description:
                Plots for analysing individual experiment progress:
                
                - traj    : Plot trajectories achieved during the experiment 
                - gridout : Plot behaviour grid coverage (color by outcome)
                - gridfit : Plot behaviour grid coverage (color by fitness)
                - l2dist  : Plot similarity grid based on l2-dist of r neighbors
                - video   : Save video
"""

import os
import logging
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.patches as mpatches
from itertools import product

import behaviour_representations.tasks.task_control as ctrl
from behaviour_representations.analysis import load_metadata, load_dataset

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-load', '--dataset_path',
                    default=None, required=True,
                    help="Path to the dataset file.")
parser.add_argument('-t', '--plot_type', nargs='+',
                    default=['gridfit', 'l2dist', 'traj'],
                    help="Select plot type(s):\n"
                         "'l2dist'\n"
                         "'traj'\n"
                         "'gridout'\n"
                         "'gridfit'\n"
                         "'video'\n")
parser.add_argument('-ring', '--ring_size',
                    default=2, type=int,
                    help="Path to the dataset file.")
parser.add_argument('-max', '--max_val',
                    default=None, type=int,
                    help="Path to the dataset file.")


MAX_VALS = {"bipedal_walker": 300,
            "striker": 100}


def get_figname(refpath, nsmp):
    refpath = refpath if refpath[-1]=='/' else refpath+'/'
    # figname = '__'.join(refpath.split('/')[-2].split('__')[:3]) \
    #            + " ({})".format(nsmp)
    task_name = refpath.split('/')[-3]
    seed_num = refpath.split('/')[-2].split('---')[0]
    figname = '__'.join([task_name, seed_num]) + " ({})".format(nsmp)
    return figname


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def load_spaces_video(dataset_path, *kwargs):
    imgs = dataset_path+'/plots/ae_spaces_original_loop_*'
    gifs = dataset_path+'/plots/ae_spaces_all.gif'
    vids = dataset_path+'/plots/ae_spaces_all.mp4'
    # convert images to gif
    os.system('convert -delay 200 -loop 0 {} {}'.format(imgs, gifs))
    # convert gif to video # -crf
    os.system('ffmpeg -f gif -i {} {}'.format(gifs, vids))
    os.system('rm {}'.format(gifs))


def load_plot_bd_traj(dataset_path, **kwargs):
    task_data_dict = load_dataset(dataset_path)
    args_dict = load_metadata(dataset_path)
                                                                                ### REMOVE THIS PART
    taskobj = ctrl.Experiment(disp=False, run_distributed=False,
                              **args_dict['experiment'], **args_dict)
    plot_bd_traj(env_info=taskobj.environment.env_info,
                 env_name=args_dict['experiment']['environment']['id'], 
                 metric_name=args_dict['experiment']['metric']['type'],
                 metric_dim=args_dict['experiment']['metric']['dim'], 
                 savepath=dataset_path, **task_data_dict, **kwargs)


def load_plot_bd_grid(dataset_path, **kwargs):
    task_data_dict = load_dataset(dataset_path)
    args_dict = load_metadata(dataset_path)
    plot_bd_grid(outcomes=task_data_dict['outcomes'],
                 env_name=args_dict['experiment']['environment']['id'], 
                 metric_bd=task_data_dict['metric_bd'],
                 metric_name=args_dict['experiment']['metric']['type'],
                 metric_dim=args_dict['experiment']['metric']['dim'], 
                 savepath=dataset_path, **kwargs)


def load_plot_l2_dist(dataset_path, **kwargs):
    task_data_dict = load_dataset(dataset_path)
    args_dict = load_metadata(dataset_path)
    plot_l2_dist(outcomes=task_data_dict['outcomes'],
                 param_original=task_data_dict['param_original'],
                 env_name=args_dict['experiment']['environment']['id'], 
                 metric_bd=task_data_dict['metric_bd'],
                 metric_name=args_dict['experiment']['metric']['type'],
                 metric_dim=args_dict['experiment']['metric']['dim'], 
                 savepath=dataset_path, **kwargs)


################################################################################


def get_fig_specs(env_name, metric_name, metric_dim):
    """
        Define plot grid dimensions based on env and metric
    """
    ldim1 = 0.08
    cbar_list = [0.92, 0.11, 0.015, 0.77]
    if metric_name=='contact_grid':
        metric_dim = metric_dim if type(metric_dim)==list \
                                else [17, metric_dim, metric_dim]
        figsize = (8,12)
        ldim0 = 0.08
        nrow, ncol = 6, 4
        ### add titles to each ax
        ax_names = [['no contact', None, None, None], ['S','E','N','W'],
                    ['S-E','S-N','S-W', None], ['E-S','E-N','E-W', None], 
                    ['N-S','N-E','N-W', None], ['W-S','W-E','W-N', None]]
        ij_map = np.zeros((nrow, ncol), dtype=int)
        ij_map[0, 1:] = -1*np.ones(ncol-1)
        ij_map[1, :] = np.arange(1, 5)
        ij_map[2:,-1] = -1*np.ones(ncol)
        ij_map[2:,:-1] = np.arange(5, 17).reshape([ncol,ncol-1])

    elif metric_name=='gait_grid':
        if "bipedal_walker" in env_name:
            metric_dim = metric_dim if type(metric_dim)==list \
                                    else [metric_dim//2, metric_dim**2, 
                                          metric_dim//2, metric_dim//2]
            figsize = (30, 3)
            cbar_list = [0.13, 0.015, 0.77, 0.05]
            ldim0 = 0.11
            nrow, ncol = metric_dim[:2]
            ax_names = np.array([[None]*ncol]*nrow)
            ij_map = np.reshape(list(product(range(nrow), range(ncol))),
                                [nrow, ncol, 2])
        elif env_name == "quadruped_walker":
            metric_dim = metric_dim if type(metric_dim)==list \
                else [metric_dim*2]*2 + [metric_dim//3]*4
            figsize = (10, 10)
            cbar_list = [0.13, 0.015, 0.77, 0.01]
            ldim0 = 0.11
            nrow, ncol = metric_dim[:2]
            ax_names = np.array([[None]*ncol]*nrow)
            ij_map = np.reshape(list(product(range(nrow), range(ncol))),
                                [nrow, ncol, 2])

    elif metric_name=='simple_grid':
        if "quadruped_walker" in env_name or "quadruped_kicker" in env_name:
            metric_dim = metric_dim if type(metric_dim)==list \
                                    else [metric_dim**2, metric_dim**2]
            figsize = (10, 10)
            nrow, ncol = 1,1
            cbar_list = [0.13, 0.015, 0.77, 0.01]
            ldim0 = 0.11
            ax_names = np.array([[None]*ncol]*nrow)
            ij_map = np.zeros((1,1), dtype=int)
        elif "bipedal_kicker" in env_name:
            metric_dim = metric_dim if type(metric_dim)==list \
                                    else [5*metric_dim, 2*metric_dim**2]
            figsize = (10, 2)
            nrow, ncol = 1,1
            ldim0 = 0.099
            ldim1 = 0.02
            cbar_list = [0.13, -0.100, 0.77, 0.05]
            ax_names = np.array([[None]*ncol]*nrow)
            ij_map = np.zeros((1,1), dtype=int)

    elif metric_name=='gait_descriptor':
        metric_dim = metric_dim if type(metric_dim)==list \
                                else [metric_dim, metric_dim, 
                                      metric_dim, metric_dim]
        figsize = (10, 10)
        ldim0 = 0.099
        ldim1 = 0.02
        nrow, ncol = metric_dim[:2]
        ax_names = np.array([[None]*ncol]*nrow)
        # ax_names[1:,-1] = np.arange(1,nrow)*np.prod(metric_dim[1:])
        ij_map = np.reshape(list(product(range(nrow), range(ncol))),
                            [nrow, ncol, 2])

    elif 'hyperplane' in metric_name:
        metric_dim = metric_dim if type(metric_dim)==list \
                                else [metric_dim, metric_dim]
        figsize = (6,6)
        ldim0 = 0.095
        nrow, ncol = 1,1
        ax_names = np.array([[None]*ncol]*nrow)
        ij_map = np.zeros((1,1), dtype=int)
    else:
        raise ValueError("Metric name '{}' not defined!".format(metric_name))

    return metric_dim, figsize, nrow, ncol, ldim0, ldim1, cbar_list, ax_names, ij_map


################################################################################


def plot_bd_traj(traj_main, outcomes, metric_bd, 
                 env_info, env_name, metric_name, metric_dim,
                 savepath=None, show_plots=False, spec_title=None,
                 img_format='png', dpi=300, **kwargs):
    """ Plots the motion trajectories colored by their cluster color """
    nsmp = len(outcomes)
    f, ax = plt.subplots(figsize=(10,5))
    figname = get_figname(savepath, nsmp) if savepath is not None \
                    else "Trajectories ({})".format(nsmp)
    ax.set_title(figname)

    # if metric_name=='contact_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [17, metric_dim, metric_dim]
    # elif metric_name=='gait_grid' or metric_name=='gait_grid_small':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #         else [metric_dim//2, metric_dim**2, metric_dim//2, metric_dim//2]
    # elif metric_name=='gait_descriptor':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #         else [metric_dim//2, metric_dim*2, metric_dim, metric_dim]
    # elif metric_name=='simple_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #         else [metric_dim**2, metric_dim**2]
    # else:
    #     raise ValueError("Incorrect metric_name: '{}'".format(metric_name))


    # if env_name in ['striker', 'striker_target', 'striker_augmented', 
    #                 'quadruped_kicker_bounded'] and metric_name=='contact_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [17, metric_dim, metric_dim]
    # elif 'bipedal_walker' in env_name and metric_name=='gait_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [metric_dim//2, metric_dim**2, 
    #                                   metric_dim//2, metric_dim//2]
    # elif env_name=='bipedal_kicker' and metric_name=='simple_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [5*metric_dim, 2*metric_dim**2]
    # elif 'quadruped_walker' in env_name and metric_name=='simple_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [metric_dim**2, metric_dim**2]
    # elif 'quadruped_walker' in env_name and metric_name=='gait_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [metric_dim*2]*2 + [metric_dim//3]*4
    # elif 'quadruped_kicker' in env_name and metric_name=='simple_grid':
    #     metric_dim = metric_dim if type(metric_dim)==list \
    #                             else [metric_dim**2, metric_dim**2]
    # else:
    #     raise ValueError("Incorrect metric_name: '{}'".format(metric_name))

    # Define plot metric
    metric_dim, _, _, _, _, _, _, _, _ = \
        get_fig_specs(env_name, metric_name, metric_dim)

    outs_marker = {0: '*', -1: 'o'}
    clst_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Plot trajectories
    min_x, max_x = np.inf, -np.inf
    init_pos = round(14/30*100,2) if 'bipedal' in env_name else 0
    for data_idx, (t, m, l) in enumerate(zip(traj_main, 
                                             metric_bd, 
                                             outcomes[:,0])):
        traj_x = t[:, 0] - init_pos
        if min(traj_x) < min_x: min_x = min(traj_x)
        if max(traj_x) > max_x: max_x = max(traj_x)
        # Show clusters along 2nd dimension
        clusters_idx = np.argmax(m)//np.prod(metric_dim[2:]) \
                                                if len(metric_dim) > 2 else 0
        color = clst_colors[clusters_idx % len(clst_colors)]
        ax.plot(traj_x, t[:, 1],
                alpha=0.1, linewidth=1., linestyle='-', color=color)
        ax.scatter(traj_x[-1], t[-1, 1],
                 alpha=0.3, marker=outs_marker[l], edgecolor=color,
                 c='w' if l==-1 else color, s=2 if l==-1 else 10)
    custom_lgd, custom_lab = ax.get_legend_handles_labels()
    if len(custom_lgd)>0:
        lgd = ax.legend(custom_lgd, custom_lab,
                        loc='upper left', bbox_to_anchor=(1, 1),
                        ncol=int(np.ceil(len(custom_lab)/20))) 
    else:
        lgd = None

    # Add targets
    if metric_name=='contact_grid':
        if 'target_info' in env_info.keys() and env_name == 'striker': 
            for ti in env_info['target_info']:
                target = mpatches.Circle(color='g', alpha=0.1, linewidth=2, 
                                         zorder=nsmp, **ti)
                ax.add_artist(target)
        if 'ball_ranges' in env_info.keys():
            ball_ranges = env_info['ball_ranges']
            ax.set_xlim(ball_ranges[0])
            ax.set_ylim(ball_ranges[1])
        else:
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-.6, 1.6) 
    
    elif env_name in ['bipedal_walker', 'bipedal_kicker', 'half_cheetah']:
        # if 'target_info' in env_info.keys():
        #     goal_line = env_info['target_info'][0]['xy'][0]
        #     ax.axvline(goal_line, c='g', ls='--', alpha=0.4)
        #     ax.axvline(-goal_line, c='g', ls='--', alpha=0.4)
        ax.set_xlim(min_x-0.2, max_x+0.2)
        ax.axvline(0, c='gray', ls='--', alpha=0.4)
        if env_name == 'bipedal_walker':
            ax.set_ylim(3.2, 7.2) 
        elif env_name == 'bipedal_kicker':
            ax.set_ylim(3.5, 10.1) 
        else:
            ax.set_ylim(0, 2.2)

    elif env_name in ['humanoid', 'quadruped']:
        if 'target_info' in env_info.keys():
            tradius = env_info['target_info'][0]['xy'][0]
            target = mpatches.Circle(color='g', alpha=0.4, linewidth=2, fill=0,
                                     xy=(0, 0), radius=tradius, zorder=nsmp)
            ax.add_artist(target)
        ax.set_xticks(np.arange(-20, 20, 2), minor=True)
        ax.set_yticks(np.arange(-20, 20, 2), minor=True)
        ax.grid(which='minor', color='gray', ls='-', lw=0.5, alpha=0.5)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20) 

    ax.set_aspect('equal')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    # Save/show figure
    plt.tight_layout()
    savepath = '.' if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/bd_traj_loop_{:05d}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                    # bbox_inches='tight'
                    )  # 300
        logger.info("Figure saved: '{}/traj_{:05d}.{}'".format(
                    savepath, spec_title, img_format))
    else:
        plt.savefig('{}/plot_bd_traj.{}'.format(
                savepath, img_format), 
                format=img_format, 
                bbox_extra_artists=(lgd,) if lgd is not None else None, 
                bbox_inches='tight', dpi=dpi) 
    if show_plots:
        plt.show()
    else:
        plt.cla()


################################################################################


def plot_bd_grid(outcomes, metric_bd, metric_name, metric_dim, env_name,
                 grid_type='outcome',
                 savepath=None, show_plots=False, spec_title=None, 
                 img_format='png', dpi=300, **kwargs):
    """ Plots the grid visualisation of behaviour descriptors """
    nsmp = len(outcomes)
    if grid_type=='outcome':
        outcomes = outcomes[:,0] + 2
        vmin, vmax = 0, 2
        norm = None
        cmap = mpl.colors.ListedColormap(['white', 
                                          'cornflowerblue', 
                                          'darkorange'])
    elif grid_type=='fitness':
        outcomes = outcomes[:,1]
        sample_min = round(min(outcomes), 2)
        sample_max = round(max(outcomes), 2)
        max_val = sample_max
        vmin, vmax = sample_min-1, max(sample_max, max_val)
        norm = MidpointNormalize(midpoint=sample_min, vmin=vmin, vmax=vmax)
        # cmap = mpl.cm.hot_r
        cmap = mpl.cm.CMRmap_r

    # Define plot grid dimensions based on env and metric
    metric_dim, figsize, nrow, ncol, ldim0, ldim1, cbar_list, \
    ax_names, ij_map = get_fig_specs(env_name, metric_name, metric_dim)

    # Reshape the descriptors for grid plot and mask based on outcome
    bd_grid = metric_bd.reshape([-1]+metric_dim)
    idx_empty = np.sum(bd_grid, axis=0)==0
    bd_grid = bd_grid * (outcomes).reshape([-1]+[1]*len(metric_dim))
    bd_grid = np.sum(bd_grid, axis=0)
    bd_grid[idx_empty] = vmin

    # Additional manipulation for 2D plotting
    # (TODO) think of a better way
    if 'hyperplane' in metric_name:
        bd_grid = bd_grid.reshape([-1]+metric_dim)
        im_grid_x, im_grid_y, im_step = metric_dim[1], metric_dim[0], 1

    elif metric_name=='simple_grid' and "bipedal_kicker" in env_name:
        bd_grid = bd_grid.reshape([-1]+metric_dim)
        im_grid_x, im_grid_y, im_step = metric_dim[1], metric_dim[0], 1

    elif metric_name=='simple_grid' and "quadruped_walker" in env_name\
        or metric_name=='simple_grid' and "quadruped_kicker":
        bd_grid = bd_grid.T
        bd_grid = bd_grid.reshape([-1]+metric_dim)
        im_grid_x, im_grid_y, im_step = metric_dim[1], metric_dim[0], 1

    elif metric_name=='gait_grid' and "quadruped_walker" in env_name:
        bd_grid = np.transpose(bd_grid, [0,1,2,4,3,5])
        bd_grid = bd_grid.reshape(metric_dim[:2] \
                  +[int(np.sqrt(np.prod(metric_dim[2:])))]*2)
        bd_grid = np.transpose(bd_grid, (1, 0, 2, 3))
        bd_grid = bd_grid[::-1,:,:,:]
        im_grid_x, im_grid_y = metric_dim[-1]**2, metric_dim[-1]**2
        im_step = metric_dim[-1]

    elif metric_name=='gait_grid' and "bipedal_walker" in env_name:
        bd_grid = bd_grid[::-1,:,:,:]
        im_grid_x, im_grid_y, im_step = metric_dim[-1], metric_dim[-1], 1

    else:
        im_grid_x, im_grid_y, im_step = metric_dim[-1], metric_dim[-1], 1

    # Main figure
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
    figname = get_figname(savepath, nsmp) if savepath is not None \
                    else "Coverage '{}' ({})".format(metric_name, nsmp)

    plt.subplots_adjust(wspace=0.1, hspace=0.1) #, left=0.5, bottom=0.5)
    ax = np.array(ax).reshape([nrow, ncol])
    ax[0, ncol//2].set_title(figname, pad=25)
    # Loop over subplots and add to fig
    for i,j in product(range(nrow),range(ncol)):
        idx = ij_map[i,j]
        if (np.isscalar(idx) and idx!=-1) or not np.isscalar(idx):
            vals = bd_grid[idx, ...] if np.isscalar(idx) \
                                    else bd_grid[idx[0],idx[1],:,:] 
            im = ax[i,j].imshow(vals, 
                           interpolation='none', aspect='auto', origin='lower',
                           cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
            ax[i,j].set_xticks(np.arange(-.5, im_grid_x, im_step), minor=True)
            ax[i,j].set_yticks(np.arange(-.5, im_grid_y, im_step), minor=True)
            ax[i,j].grid(which='minor', color='gray', 
                         linestyle='-', linewidth=0.3)
            ax[i,j].tick_params(axis='both', which='both', length=0,
                labelbottom=False, labelleft=False)
            if env_name != "bipedal_kicker":
                ax[i,j].set_aspect('equal')
            if ax_names is not None and ax_names[i][j] is not None:
                ax[i,j].set_title(ax_names[i][j])
        else:
            ax[i,j].axis('off')

    # Axis titles  
    if len(metric_dim) > 2:
        ax[-1,0].set_xlabel("dim 2")
        ax[-1,0].set_ylabel("dim 3")    
    fig.text(0.5, ldim1, 'dim 1', ha='center')
    fig.text(ldim0, 0.5, 'dim 0', va='center', rotation='vertical')
    # Add legend
    legend = []
    if grid_type=='outcome':
        h = [plt.scatter([], [], color='white', lw=1, edgecolors='gray'), 
             plt.scatter([], [], color='cornflowerblue',  lw=1),
             plt.scatter([], [], color='darkorange', lw=1)]
        l = ['EMPTY', 'FAIL', 'SUCCESS']
        legend.append(ax[0,-1].legend(handles=h, labels=l, loc='upper left', 
                                      bbox_to_anchor=(1, 1), ncol=1))
    elif grid_type=='fitness':
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(cbar_list)
        if metric_name in ['gait_grid','simple_grid'] \
           or "bipedal_kicker" in env_name:
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.ax.get_xaxis().labelpad = 0
            cbar.ax.set_xlabel(grid_type)
        else:
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
            cbar.ax.get_yaxis().labelpad = 5
            cbar.ax.set_ylabel(grid_type, rotation=270)
        cbar.set_ticks([sample_min, sample_max, max_val])
        cbar.set_ticklabels(['worst: {}'.format(sample_min), 
                             'best: {}'.format(sample_max), 
                             str(max_val) if max_val>sample_max else '' ])
    # Save/show figure
    savepath = '.' if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/bd_grid_loop_{:05d}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=tuple(legend), 
                    bbox_inches='tight', pad_inches=0.25, dpi=dpi) 
    else:
        plt.savefig('{}/plot_bd_grid.{}'.format(
                    savepath, img_format), 
                    format=img_format, bbox_extra_artists=tuple(legend), 
                    bbox_inches='tight', pad_inches=0.25, dpi=dpi) 
    if show_plots:
        plt.show()
    else:
        plt.cla()


################################################################################


def plot_l2_dist(outcomes, param_original, metric_bd, metric_name, metric_dim, 
                 env_name, ring_size=2, max_val=None,
                 savepath=None, show_plots=False, spec_title=None, 
                 img_format='png', dpi=300, **kwargs):
    """ 
        Plots the grid containing l2-distances of policy parameters 
        corresponding to the behaviour descriptor at a certain location.
    """
    cmap = mpl.cm.gist_earth_r
    nsmp = len(outcomes)
    # get non-infinity indices
    non_inf = param_original[0]<np.inf
    # neighbours list
    offset_ij = list(product(range(-ring_size, ring_size+1), 
                             range(-ring_size, ring_size+1)))
    offset_ij.remove((0,0))

    # Define plot grid dimensions based on env and metric
    metric_dim, figsize, nrow, ncol, ldim0, ldim1, cbar_list, \
    ax_names, ij_map = get_fig_specs(env_name, metric_name, metric_dim)

    # Filled cell indices
    bd_grid = metric_bd.reshape([-1]+metric_dim)
    idx_empty = np.sum(bd_grid, axis=0)==0
    bd_grid = np.sum(bd_grid, axis=0)
    # Make grid with actual indices
    idx_dataset = np.arange(nsmp)[:, None] * metric_bd
    idx_dataset_grid = idx_dataset.reshape([-1] + metric_dim)
    idx_dataset_grid = np.sum(idx_dataset_grid, axis=0)
    idx_dataset_grid[idx_empty] = -1

    # make sure grid is iterable
    if len(bd_grid.shape) == 2:
        bd_grid = bd_grid[None, ...]
        idx_dataset_grid = idx_dataset_grid[None, ...]
    elif len(bd_grid.shape) == 3:
        pass
    elif len(bd_grid.shape) == 4:
        reorder = np.array([0,2,1,3])
        bd_grid = np.transpose(bd_grid, reorder)
        bd_grid = bd_grid.reshape((np.prod(metric_dim[::2]), 
                                   np.prod(metric_dim[1::2])))
        bd_grid = bd_grid[None, ...]
        idx_dataset_grid = np.transpose(idx_dataset_grid, reorder)
        idx_dataset_grid = idx_dataset_grid.reshape((np.prod(metric_dim[::2]), 
                                                     np.prod(metric_dim[1::2])))
        idx_dataset_grid = idx_dataset_grid[None, ...]
    elif len(bd_grid.shape) == 6:
        bd_grid = np.transpose(bd_grid, [0,1,2,4,3,5])
        bd_grid = bd_grid.reshape(metric_dim[:2] \
                                  +[int(np.sqrt(np.prod(metric_dim[2:])))]*2)
        bd_grid = np.transpose(bd_grid, (1, 0, 2, 3))
        bd_grid = np.transpose(bd_grid, [0,2,1,3])
        bd_grid = bd_grid.reshape((np.prod(bd_grid.shape[:2]), 
                                   np.prod(bd_grid.shape[:2])))
        bd_grid = bd_grid[None, ...]
        idx_dataset_grid = np.transpose(idx_dataset_grid, [0,1,2,4,3,5])
        idx_dataset_grid = idx_dataset_grid.reshape(metric_dim[:2] \
                                  +[int(np.sqrt(np.prod(metric_dim[2:])))]*2)
        idx_dataset_grid = np.transpose(idx_dataset_grid, (1, 0, 2, 3))
        idx_dataset_grid = np.transpose(idx_dataset_grid, [0,2,1,3])
        idx_dataset_grid = idx_dataset_grid.reshape((
            np.prod(idx_dataset_grid.shape[:2]), 
            np.prod(idx_dataset_grid.shape[:2])))
        idx_dataset_grid = idx_dataset_grid[None, ...]
    else:
        raise ValueError("Metric size '{}' not defined!".format(bd_grid.shape))

    # placeholder distances grid
    l2_dist_grid = np.full_like(idx_dataset_grid, -1)
    mean_list = []
    # loop over each image
    for gg_ in range(bd_grid.shape[0]):
        # non-empty cells
        idx_bd_grid = np.where(bd_grid[gg_])
        grid_h, grid_w = idx_dataset_grid[gg_].shape
        # loop over non-empty cells only
        for i,j in np.vstack(idx_bd_grid).T:
            focus_idx = idx_dataset_grid[gg_][i,j]
            focus_param = param_original[focus_idx][non_inf]
            dist_list = []
            # loop over all non-empty neighbours
            for oi, oj in offset_ij:
                mm = i + oi
                nn = j + oj
                if mm>=0 and mm<grid_h and nn>=0 and nn<grid_w \
                         and idx_dataset_grid[gg_][mm,nn]>-1:
                    compare_idx = idx_dataset_grid[gg_][mm, nn]
                    compare_param = param_original[compare_idx][non_inf]
                    dist_list.append(np.linalg.norm(focus_param-compare_param))
            if len(dist_list):
                l2_dist_grid[gg_][i,j] = np.mean(dist_list)
                mean_list.append(np.mean(dist_list))
    mean_diff = np.mean(mean_list)

    # Re-shape and re-orient for plotting 
    im_grid_x, im_grid_y, im_step = metric_dim[-1], metric_dim[-1], 1
    # im_grid_x, im_grid_y, im_step = *l2_dist_grid.shape[-2:], 1
    if 'hyperplane' in metric_name:
        l2_dist_grid = l2_dist_grid.reshape([-1]+metric_dim)
    
    elif metric_name=='simple_grid' and "bipedal_kicker" in env_name:
        l2_dist_grid = l2_dist_grid.reshape([-1]+metric_dim)
        im_grid_x, im_grid_y, im_step = metric_dim[-1], metric_dim[-2], 1
    
    elif metric_name=='simple_grid' and "quadruped_walker" in env_name \
        or metric_name=='simple_grid' and "quadruped_kicker" in env_name:
        l2_dist_grid = l2_dist_grid[0].T
        l2_dist_grid = l2_dist_grid.reshape([-1]+metric_dim)

    elif metric_name=='gait_grid' and "quadruped_walker" in env_name:
        # separated graph
        l2_dist_grid = l2_dist_grid.reshape(metric_dim[0], metric_dim[-1]**2, 
                                       metric_dim[0], metric_dim[-1]**2)
        l2_dist_grid = np.transpose(l2_dist_grid, [0, 2, 1, 3])
        l2_dist_grid = l2_dist_grid[::-1,:,:,:]
        im_grid_x, im_grid_y = metric_dim[-1]**2, metric_dim[-1]**2
        im_step = metric_dim[-1]
        # # continuous graph
        # nrow, ncol = 1,1
        # ij_map = np.zeros((1,1), dtype=int)
        # im_grid_x, im_grid_y, im_step = *l2_dist_grid.shape[-2:], 1
    
    elif metric_name=='gait_grid' and "bipedal_walker" in env_name:
        # separated graph
        l2_dist_grid = l2_dist_grid.reshape(np.array(metric_dim)[reorder])
        l2_dist_grid = np.transpose(l2_dist_grid, reorder)
        l2_dist_grid = l2_dist_grid[::-1,:,:,:]
        # # continuous graph
        # l2_dist_grid = l2_dist_grid[::-1,:]
        # nrow, ncol = 1,1
        # ij_map = np.zeros((1,1), dtype=int)
        # im_grid_x, im_grid_y = l2_dist_grid.shape[-1], l2_dist_grid.shape[-2]
        # im_step = 1

    # Main figure
    vmin, vmax = -1, np.max(l2_dist_grid) if max_val is None else max_val
    norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize)

    figname = get_figname(savepath, nsmp) if savepath is not None \
                    else "L2-distaces '{}' ({})".format(metric_name, nsmp)

    plt.subplots_adjust(wspace=0.1, hspace=0.1) #, left=0.5, bottom=0.5)
    ax = np.array(ax).reshape([nrow, ncol])
    ax[0, ncol//2].set_title(figname, pad=25)
    for i,j in product(range(nrow),range(ncol)):
        idx = ij_map[i,j]
        if (np.isscalar(idx) and idx!=-1) or not np.isscalar(idx):
            vals = l2_dist_grid[idx, ...] if np.isscalar(idx) \
                                    else l2_dist_grid[idx[0],idx[1],...] 
            im = ax[i,j].imshow(vals, 
                           interpolation='none', aspect='auto', origin='lower',
                           cmap=cmap, vmin=-1, vmax=vmax, norm=norm)
            ax[i,j].set_xticks(np.arange(-.5, im_grid_x, im_step), minor=True)
            ax[i,j].set_yticks(np.arange(-.5, im_grid_y, im_step), minor=True)
            ax[i,j].grid(which='minor', color='gray', 
                         linestyle='-', linewidth=0.3)
            ax[i,j].tick_params(axis='both', which='both', length=0,
                labelbottom=False, labelleft=False)
            if env_name != "bipedal_kicker":
                ax[i,j].set_aspect('equal')
            if ax_names is not None and ax_names[i][j] is not None:
                ax[i,j].set_title(ax_names[i][j])
        else:
            ax[i,j].axis('off')
    # Axis titles  
    if len(metric_dim) > 2:
        ax[-1,0].set_xlabel("dim 2")
        ax[-1,0].set_ylabel("dim 3")    
    fig.text(0.5, ldim1, 'dim 1', ha='center')
    fig.text(ldim0, 0.5, 'dim 0', va='center', rotation='vertical')
    # Add legend
    legend = []
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes(cbar_list)
    if metric_name in ['gait_grid','simple_grid'] or "bipedal_kicker" in env_name:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.get_xaxis().labelpad = 8
        cbar.ax.set_xlabel('parameter L2 similarity (r={})'.format(ring_size))
        ctlbs = [t.get_text() for t in cbar.ax.get_xticklabels()]
    else:
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.get_yaxis().labelpad = 12
        cbar.ax.set_ylabel('parameter L2 similarity (r={})'.format(ring_size),
                            rotation=270)
        ctlbs = [t.get_text() for t in cbar.ax.get_yticklabels()]
    cticks = list(cbar.get_ticks())
    cbar.set_ticks(cticks + [mean_diff])
    cbar.set_ticklabels(ctlbs + ['\nmean: {:.2f}'.format(mean_diff)])
    # Save/show figure
    savepath = '.' if savepath is None else savepath
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if type(spec_title)==int:
        plt.savefig('{}/l2_dist_grid_loop_{:05d}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=tuple(legend), 
                    bbox_inches='tight', pad_inches=0.25, dpi=dpi) 
    else:
        plt.savefig('{}/plot_l2_dist_grid_r{}.{}'.format(
                    savepath, ring_size, img_format), 
                    format=img_format, bbox_extra_artists=tuple(legend), 
                    bbox_inches='tight', pad_inches=0.25, dpi=dpi) 
    if show_plots:
        plt.show()
    else:
        plt.cla()


################################################################################

if __name__ == "__main__":
    args = parser.parse_args()
    if 'traj' in args.plot_type:
        load_plot_bd_traj(dataset_path=args.dataset_path)
    if 'gridout' in args.plot_type:
        load_plot_bd_grid(dataset_path=args.dataset_path, grid_type='outcome')
    if 'gridfit' in args.plot_type:
        load_plot_bd_grid(dataset_path=args.dataset_path, grid_type='fitness')
    if 'l2dist' in args.plot_type:
        load_plot_l2_dist(dataset_path=args.dataset_path, 
                          ring_size=args.ring_size,
                          max_val=args.max_val)
    if 'video' in args.plot_type:
        load_spaces_video(dataset_path=args.dataset_path)

