
"""
Author:         Anonymous
Description:    
                Functions for plotting 
"""

import os
import logging

import numpy as np 
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import mpl_toolkits.axes_grid1 as ax_grid

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, MultipleLocator

# %matplotlib notebook
from behaviour_representations.utils.utils import ignore_warnings
from behaviour_representations.utils.utils import _MARKER_SZ, _ALPHA_PTS, PLOT_DPI

logger = logging.getLogger(__name__)



class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


@ignore_warnings
def plot_ae_spaces(param_original, param_embedded, param_reconstructed,
                   traj_original, traj_embedded, traj_reconstructed, 
                   latentdim_param_ae, latentdim_traj_ae, ae_type_traj,
                   outcomes, loss_dict, cluster_object, parameter_arch,
                   metric_bd, metric_dim, param_covmat=None, use_dims=None,
                   samples_aux=None, samples_recn=None, new_data=None,
                   dimred_mode='original', emphasize_means=None, test_idx=None,
                   savepath=None, show_plots=False, spec_title=None, 
                   img_format='png', dpi=PLOT_DPI, alpha_c=0.2, **kwargs):

    _alpha_samples = alpha_c * _ALPHA_PTS
    # mpl.rcParams.update({'font.size': 12})
    f, axs = plt.subplots(nrows=1 \
                            + 1*(ae_type_traj is not None) \
                            + 1*(samples_aux is not None), ncols=3, 
                          figsize=(15, 5 + 5*(ae_type_traj is not None)\
                                         + 5*(samples_aux is not None)), 
                          sharey=False, dpi=dpi)
    axs = axs.flatten()
    outcomes = outcomes[:,0]
    unique_outcomes = np.unique(outcomes)
    num_data = outcomes.shape[0]
    # Make sure clusters don't affect plots
    if cluster_object is not None and \
       cluster_object.datapoints_to_cluster is None:
        cluster_obj = None
    else:
        cluster_obj = cluster_object

    # Modify multidim inputs
    param_dim = param_original.shape[1:] if parameter_arch is None \
                                         else parameter_arch
    latent_dim = param_embedded.shape[1:]

    dims_orig = np.array([0,1])
    dims_emb = np.array([0,1])
    # dims_recn = np.array([0,1])
    if use_dims is not None:
        dims_orig = use_dims['orig']
        dims_emb = use_dims['emb']
        param_original = param_original.reshape(num_data, -1)
        param_reconstructed = param_reconstructed.reshape(num_data, -1)
        param_embedded = param_embedded.reshape(num_data, -1)
        param_original = param_original[:, dims_orig]
        param_reconstructed = param_reconstructed[:, dims_orig]
        param_embedded = param_embedded[:, dims_emb]

    elif dimred_mode=='original': 
        # Original
        if np.prod(param_dim) > 2:
            ### TODO check if flat
            param_original = param_original.reshape(num_data, -1)

            param_reconstructed = param_reconstructed.reshape(num_data, -1)
            stds = np.std(param_original, axis=0)
            num_nan = sum(np.isnan(stds))
            if num_nan:
                dims_orig = np.argsort(stds)[-2-num_nan:-num_nan]
            else:
                dims_orig = np.argsort(stds)[-2:]
            param_original = param_original[:, dims_orig]
            param_reconstructed = param_reconstructed[:, dims_orig]
            # dims_recn = np.argsort(np.std(param_reconstructed, axis=0))[-2:]
            # param_reconstructed = param_reconstructed[:, dims_recn]
        # Latent
        if np.prod(latent_dim) > 2:
            param_embedded = param_embedded.reshape(num_data, -1)
            dims_emb = np.argsort(np.std(param_embedded, axis=0))[-2:]
            param_embedded = param_embedded[:, dims_emb]


    # Plot outcome_branch predictions
    _ax_offset = 0
    if samples_aux is not None:
        _ax_offset = 3
        if param_covmat is not None:
            # Plot param covariance matrix
            all_pts = param_covmat.reshape(param_covmat.shape[0], -1)
            all_pts = all_pts[:, np.invert(np.isinf(all_pts[0,:]))]
            if all_pts.shape[0]>1:
                if test_idx is not None or new_data is not None:
                    covmat = np.cov(all_pts, rowvar=False)
                    vmin, vmax = covmat.min(), covmat.max()
                    matname = "Covariance"
                else:
                    covmat = np.corrcoef(all_pts, rowvar=False)
                    vmin, vmax, vcenter = -1, 1, 0
                    matname = "Correlation"
            else:
                covmat = np.ones((all_pts.shape[1], all_pts.shape[1]))
                vmin, vmax, vcenter = -1, 1, 0
                matname = "N/A"
            vcenter = (vmax-vmin)/2 + vmin
            midnorm = MidpointNormalize(vmin=vmin, vcenter=vcenter, vmax=vmax)
            im0 = axs[0].imshow(covmat, 
                       interpolation='none', aspect='auto', origin='upper',
                       cmap='seismic', norm=midnorm, vmin=vmin, vmax=vmax)
            ax0_divider = ax_grid.axes_divider.make_axes_locatable(axs[0])
            cax0 = ax0_divider.append_axes("right", size="3%", pad=0.1) # size="7%", pad="2%"
            cbar2 = ax_grid.colorbar.colorbar(im0, cax=cax0)
            axs[0].grid(which='minor', color='gray', 
                        linestyle='-', linewidth=0.2)
            # axs[0].set_xlabel('parameter dims')
            # axs[0].set_ylabel('parameter dims')
            cax0.set_yticks([vmin, vcenter, vmax])
            axs[0].axis('equal')
            axs[0].set_title("{} matrix [{}-D; {} smpl]".format(matname,
                                                                len(covmat),
                                                                len(all_pts)))
        if samples_aux[2]=='get_out_prediction':
            sc1 = axs[1].scatter(samples_aux[0][:, dims_emb[0]], 
                                samples_aux[0][:, dims_emb[1]], 
                                c=samples_aux[1][:, -1], #edgecolor='k', 
                                s=0.1*_MARKER_SZ, cmap='viridis', 
                                vmin=0, vmax=1)
            axs[1].set_title('Outcome prediction '
                             '[outcome_branch loss: {:.4e}]'.format(
                              loss_dict['param']['outcome_branch']))
            axs[2].axis('off')
        else:
            sc1 = axs[1].scatter(samples_aux[0][:, dims_emb[0]], 
                                samples_aux[0][:, dims_emb[1]], 
                                c=samples_aux[1][:, 0], marker=',', 
                                s=0.5*_MARKER_SZ, cmap='viridis') # 'rainbow', 'coolwarm'
            # axs[1].set_xlabel('latent dim [{}]'.format(dims_emb[0]))
            # axs[1].set_ylabel('latent dim [{}]'.format(dims_emb[1]))
            axs[1].set_title("DEC_Jacobian: |J.T J|")
            # axs[1].set_title("DEC_Jacobian matrix norm")
            sc2 = axs[2].scatter(samples_aux[0][:, dims_emb[0]], 
                                samples_aux[0][:, dims_emb[1]], 
                                c=samples_aux[1][:, 1], marker=',', 
                                s=0.5*_MARKER_SZ, cmap='viridis') # 'rainbow', 'coolwarm'
            # axs[2].set_xlabel('latent dim [{}]'.format(dims_emb[0]))
            # axs[2].set_ylabel('latent dim [{}]'.format(dims_emb[1]))
            axs[2].set_title("DEC_Jacobian: matrix condition number")

    # Plot dataset in original, embedded and reconstructed space
    outs_marker = {0: '*', -1:'o'}
    clst_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for i_outs_, outs_ in enumerate(unique_outcomes):
        data_idx = np.flatnonzero(outcomes == outs_)
        if cluster_obj is None:
            # clusters_idx = 0
            clusters_idx = np.argmax(metric_bd[data_idx,:], axis=1)
            clusters_idx = clusters_idx//np.prod(metric_dim[1:])
        else:
            clusters_idx = cluster_obj.datapoints_to_cluster[data_idx]
        color_list = clst_colors[clusters_idx % len(clst_colors)]
        plot_kw = \
            dict(alpha=_ALPHA_PTS, marker=outs_marker[outs_], 
                 edgecolor=color_list, c='w' if outs_==-1 else color_list)
        # plot PARAM AE spaces
        axs[_ax_offset+0].scatter(param_original[data_idx, 0], 
                                  param_original[data_idx, 1], **plot_kw)
        axs[_ax_offset+1].scatter(param_embedded[data_idx, 0], 
                                  param_embedded[data_idx, 1], **plot_kw)
        axs[_ax_offset+2].scatter(param_reconstructed[data_idx, 0], 
                                  param_reconstructed[data_idx, 1], **plot_kw)
        # plot TRAJ AE spaces
        if ae_type_traj is not None:
            plot_paths_ax(axs[_ax_offset+3], traj_original, 
                          title='({}) Original trajectories [{}]'.format(
                              num_data, traj_original[0].shape),
                          emphasize_means=emphasize_means, outcomes=outcomes,
                          cluster_object=cluster_obj, new_data=new_data, **kwargs)
            recn_loss = "loss: {:.4e}".format(
                    loss_dict['traj']['reconstruction']) \
                    if 'reconstruction' in loss_dict['traj'].keys() \
                    else "loss: N/A"
            plot_paths_ax(axs[_ax_offset+5], traj_reconstructed, 
                          title='Reconstructed traj [{}]'.format(recn_loss), 
                          emphasize_means=emphasize_means, outcomes=outcomes, 
                          cluster_object=cluster_obj, new_data=None, **kwargs)
            _tmp = [ln for ln in loss_dict['traj'].keys() \
                    if ln != 'reconstruction']
            latent_loss = "{} loss: {:.4e}".format('+'.join(_tmp), 
                sum([lv for ln, lv in loss_dict['traj'].items() \
                  if ln != 'reconstruction'])) if len(_tmp)>0 else "loss: N/A"
            axs[_ax_offset+4].set_title(
              '{}D Latent traj [{}]'.format(latentdim_traj_ae, latent_loss))
            axs[_ax_offset+4].set_xlabel('latent dim [{}]'.format(dims_emb[0]))
            axs[_ax_offset+4].set_ylabel('latent dim [{}]'.format(dims_emb[1]))

            axs[_ax_offset+4].scatter(traj_embedded[data_idx, dims_emb[0]], 
                                      traj_embedded[data_idx, dims_emb[1]], 
                                      **plot_kw)

    # Overlay the image with cluster means (medoid or centroid)
    if emphasize_means is not None and cluster_obj is not None:
        for c_idx in range(cluster_obj.num_clusters):
            color_list = clst_colors[c_idx % len(clst_colors)]
            plot_mean_kw = plot_kw.copy()
            plot_mean_kw.update({'s': 200, 'edgecolor': 'k', 
                                 'c': color_list, 'alpha': 0.5})
            if emphasize_means == 'medoid':
                # get medoid indices
                param_means = cluster_obj.cluster_to_medoids[c_idx][0].point
                _idx = cluster_obj.cluster_to_medoids[c_idx][1]
                plot_mean_kw.update({'marker': outs_marker[outcomes[_idx]]})
            elif emphasize_means == 'centroid':
                # get centroids
                param_means = cluster_obj.cluster_to_centroids[c_idx][0].point

            # plot in param latent spaces
            axs[_ax_offset+1].scatter(param_means[dims_emb[0]], 
                                      param_means[dims_emb[1]], 
                                      **plot_mean_kw)
            if ae_type_traj is not None:
                axs[_ax_offset+4].scatter(traj_embedded[_idx, dims_emb[0]], 
                                          traj_embedded[_idx, dims_emb[1]], 
                                          **plot_mean_kw)
    elif emphasize_means == 'index' or test_idx is not None:
        plot_mean_kw = plot_kw.copy()
        plot_mean_kw.update({'s': 200, 'edgecolor': 'k', 'marker':'s',
                             'c': 'grey', 'alpha': 0.5})
        if type(test_idx)==int:
            test_emb = param_embedded[test_idx]
            test_recn = param_reconstructed[test_idx]
        else:
            test_emb = test_idx[0][0][dims_emb]
            test_recn = test_idx[1][0][dims_orig]

        axs[_ax_offset+1].scatter(*test_emb, **plot_mean_kw)
        axs[_ax_offset+2].scatter(*test_recn, **plot_mean_kw)


    if new_data is not None:
        # plot trajectories colored by their cluster color
        new_outcomes = new_data['outcomes'][:,0]
        unique_new = np.unique(new_outcomes)
        num_new = len(new_outcomes)
        new_embedded = new_data['param_embedded']
        new_params = new_data['param_recn'] if new_embedded is not None \
                                            else new_data['param_original']
        if np.prod(param_dim) > 2:
            new_params = new_params.reshape(num_new, -1)
        new_params = new_params[:, dims_orig]

        if new_embedded is not None:
            if np.prod(latent_dim) > 2:
                new_embedded = new_embedded.reshape(num_new, -1)
                new_embedded = new_embedded[:, dims_emb]

        for i_outs_, outs_ in enumerate(unique_new):
            data_idx = np.where(new_outcomes == outs_)[0]
            plot_kw = dict(alpha=_ALPHA_PTS, marker='x' if outs_==-1 else '*', 
                           edgecolor='k', c='k')
            if new_embedded is not None:
                axs[_ax_offset+1].scatter(new_embedded[data_idx, 0], 
                                          new_embedded[data_idx, 1], **plot_kw)
                if new_data['param_original'] is not None:
                    new_orig = new_data['param_original'].reshape(num_new, -1)
                    new_orig = new_orig[:, dims_orig]
                    # plot_kw.update({'alpha':alpha_c})
                    axs[_ax_offset].scatter(new_orig[data_idx, 0], 
                                            new_orig[data_idx, 1], **plot_kw)
            axs[_ax_offset+2].scatter(new_params[data_idx, 0], 
                                      new_params[data_idx, 1], **plot_kw)


    # Display sample grid mapping from latent to reconstructed space
    if samples_recn is not None:
        num_samples = samples_recn[0].shape[0]
        sample_ls, sample_rs = samples_recn
        sample_rs = sample_rs.reshape(num_samples, -1)
        color_list = []
        for i,s in enumerate(range(num_samples)):
            color = next(axs[_ax_offset+1]._get_lines.prop_cycler)['color']
            color_list.append(color)
            xy_latent = (sample_ls[s, dims_emb[0]], sample_ls[s, dims_emb[1]])
            xy_reconst = (sample_rs[s, dims_orig[0]], 
                          sample_rs[s, dims_orig[1]])
            con = mpl.patches.ConnectionPatch(xyA=xy_reconst, xyB=xy_latent, 
                                              coordsA='data', coordsB='data',
                                              axesA=axs[_ax_offset+2], 
                                              axesB=axs[_ax_offset+1], 
                                              color=color, linewidth=1, 
                                              alpha=_alpha_samples, zorder=0)
            axs[_ax_offset+2].add_artist(con)
        sample_dict = dict(s=_MARKER_SZ, marker='s', edgecolor='w',  # c='g'
                           c=color_list, alpha=_alpha_samples, zorder=0)
        axs[_ax_offset+1].scatter(sample_ls[:, dims_emb[0]], 
                                  sample_ls[:, dims_emb[1]],
                                  **sample_dict)
        axs[_ax_offset+2].scatter(sample_rs[:, dims_orig[0]], 
                                  sample_rs[:, dims_orig[1]],
                                  **sample_dict)

    # Calculate plot limits
    limits_latent = param_embedded
    okrow = np.all(np.invert(np.isnan(limits_latent)), axis=1) 
    limits_latent = limits_latent[okrow]
    limits_original = np.vstack([param_original, param_reconstructed])
    okrow = np.all(np.invert(np.isnan(limits_original)), axis=1) 
    limits_original = limits_original[okrow]
    if new_data is not None:
        if new_embedded is not None:
            limits_latent = np.vstack([limits_latent, new_embedded])
        limits_original = np.vstack([limits_original, new_params])
    if len(limits_latent):
        x_min, y_min = np.min(limits_latent, axis=0)
        x_max, y_max = np.max(limits_latent, axis=0)
        ofstx = 0.05*(x_max-x_min)
        ofsty = 0.05*(y_max-y_min)
        axs[_ax_offset+1].set_xlim(x_min-ofstx, x_max+ofstx)
        axs[_ax_offset+1].set_ylim(y_min-ofsty, y_max+ofsty)
        # axs[_ax_offset+1].axis('equal')
        if samples_aux is not None:
            axs[1].set_xlim(x_min-ofstx, x_max+ofstx)
            axs[1].set_ylim(y_min-ofsty, y_max+ofsty)
            # axs[1].set_aspect('equal', 'box')
            ax1_divider = ax_grid.axes_divider.make_axes_locatable(axs[1])
            cax1 = ax1_divider.append_axes("right", size="3%", pad=0.1) # size="7%", pad="2%"
            cbar = ax_grid.colorbar.colorbar(sc1, cax=cax1)
            if samples_aux[2]=='get_out_prediction':
                cax1.set_yticks([0.01, 0.99])
                cax1.set_yticklabels(['FAIL', 'SUCCESS']) 
            else:
                axs[2].set_xlim(x_min-ofstx, x_max+ofstx)
                axs[2].set_ylim(y_min-ofsty, y_max+ofsty)
                # axs[1].set_aspect('equal', 'box')
                ax2_divider = ax_grid.axes_divider.make_axes_locatable(axs[2])
                cax2 = ax2_divider.append_axes("right", size="3%", pad=0.1) # size="7%", pad="2%"
                cbar2 = ax_grid.colorbar.colorbar(sc2, cax=cax2)
    # align original and reconstructed space limits
    if len(limits_original):
        x_min, y_min = np.min(limits_original, axis=0)
        x_max, y_max = np.max(limits_original, axis=0)
        ofstx = 0.1*(x_max-x_min)
        ofsty = 0.1*(y_max-y_min)
        axs[_ax_offset+0].set_xlim(x_min-ofstx, x_max+ofstx)
        axs[_ax_offset+0].set_ylim(y_min-ofsty, y_max+ofsty)
        axs[_ax_offset+2].set_xlim(x_min-ofstx, x_max+ofstx)
        axs[_ax_offset+2].set_ylim(y_min-ofsty, y_max+ofsty)
        # x_min, y_min = np.min(reconstructed_x[:,:2], axis=0)
        # x_max, y_max = np.max(reconstructed_x[:,:2], axis=0)
        # ofstx = 0.1*(x_max-x_min)
        # ofsty = 0.1*(y_max-y_min)
        # axs[2].set_xlim(x_min-ofstx, x_max+ofstx)
        # axs[2].set_ylim(y_min-ofsty, y_max+ofsty)
        # f.text(0.06, 0.5, 'PARAMETER AE', ha='center', va='center', rotation='vertical')

    if 'pca' in loss_dict.keys():
        latent_loss = "PCA variance {:2.2f}%".format(loss_dict['pca']['variance'])
        recn_loss = "PCA recn loss: {:2.2f}%".format(loss_dict['pca']['reconstruction'])
    elif 'param' in loss_dict.keys():
        _tmp = [ln for ln in loss_dict['param'].keys() \
                if ln != 'reconstruction' and ln != 'outcome_branch']
        latent_loss = "{} loss: {:.4e}".format('+'.join(_tmp), 
                        sum([lv for ln, lv in loss_dict['param'].items() \
                        if ln!='reconstruction' and ln!='outcome_branch'])) \
                        if len(_tmp)>0 else "loss: N/A"
        recn_loss = "loss: {:.4e}".format(loss_dict['param']['reconstruction']) \
                        if 'reconstruction' in loss_dict['param'].keys() \
                        else "loss: N/A"
    else:
        latent_loss = 'NONE'
        recn_loss = 'NONE'

    nl = '[loop: {}] '.format(spec_title) if type(spec_title)==int else ''
    space_names = ['{}Original space [{}D]'.format(nl, param_dim), 
                   'Latent space [{}D][{}]'.format(latentdim_param_ae, 
                                                  latent_loss), 
                   'Reconstructed space [{}]'.format(recn_loss)]
    for i, s in enumerate(space_names):
        axs[_ax_offset+i].set_title(s)
        if 'Latent' in s:
            axs[_ax_offset+i].set_xlabel('latent dim [{}]'.format(dims_emb[0]))
            axs[_ax_offset+i].set_ylabel('latent dim [{}]'.format(dims_emb[1]))
        else:
            axs[_ax_offset+i].set_xlabel('orig dim [{}]'.format(dims_orig[0]))
            axs[_ax_offset+i].set_ylabel('orig dim [{}]'.format(dims_orig[1]))
    # Add legend
    custom_lgd = [plt.scatter([], [], color='w', lw=1, marker='o', 
                              edgecolor='k'), 
                  plt.scatter([], [], color='k', lw=1, marker='*')]
    custom_lab = ['class: -1', 'class: 0']  
    if cluster_obj is not None:
        unique_clusters = np.unique(cluster_obj.datapoints_to_cluster)
        clst_lgd = [plt.scatter([], [], lw=1, marker='s', 
                    color=clst_colors[cc % len(clst_colors)]) \
                    for cc in unique_clusters]
        clst_lab = ['cluster {}'.format(cc) for cc in unique_clusters]
        custom_lgd.extend(clst_lgd)
        custom_lab.extend(clst_lab)   
    ncol = int(np.ceil(len(custom_lab)/(15*(1+1*(ae_type_traj is not None)))))      
    lgd = axs[_ax_offset+2].legend(custom_lgd, custom_lab,
                        loc='upper left', bbox_to_anchor=(1, 1), ncol=ncol)
    # Save/show figure
    plt.subplots_adjust(hspace = 0.2)
    if type(dimred_mode) != str: dimred_mode = 'dims{}'.format(dimred_mode)
    new_tmp = '' if new_data is not None else '_new'
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if type(spec_title)==int:
            fig_name = 'ae_spaces_{}_loop_{:05d}{}.{}'.format(
                        dimred_mode, spec_title, new_tmp, img_format)
            plt.savefig('{}/{}'.format(savepath, fig_name), 
                        format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                        bbox_inches='tight')  # 300
            logger.info("Figure saved: '{}/{}'".format(savepath, fig_name))
        else:
            plt.savefig('{}/{}.{}'.format(
                        savepath, spec_title, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                        bbox_inches='tight') 
    if show_plots:
        plt.show()
    else:
        plt.cla()



@ignore_warnings
def plot_traj_walker(traj_main, outcomes, metric_bd, 
                     metric_name, metric_dim,
                     cluster_object=None, env_info=None,
                     emphasize_means=None, #'medoid',
                     title=None, savepath=None, show_plots=False, 
                     spec_title=None, img_format='png', dpi=PLOT_DPI, **kwargs):
    """ Plots the ball trajectories """

### TODO : integrate actual image!!!

    _alpha = 0.2 * _ALPHA_PTS
    outcomes = outcomes[:,0]
    num_trajectories = len(outcomes)
    outs_marker = {0: '*', -1:'o'}
    clst_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    f, ax = plt.subplots()

    # Make sure clusters don't affect plots
    if cluster_object is not None and \
       cluster_object.datapoints_to_cluster is None:
        cluster_obj = None
    else:
        cluster_obj = cluster_object

    # plot trajectories colored by their cluster color
    min_x, max_x = 0, 20
    for data_idx, (t, m, l) in enumerate(zip(traj_main, 
                                             metric_bd, 
                                             outcomes)):
        if min(t[:, 0]) < min_x: min_x = min(t[:, 0])
        if max(t[:, 0]) > max_x: max_x = max(t[:, 0])
        clusters_idx = np.argmax(m)//np.prod(metric_dim[1:]) \
                            if cluster_obj is None \
                            else cluster_obj.datapoints_to_cluster[data_idx]
        color = clst_colors[clusters_idx % len(clst_colors)]
        ax.plot(t[:,0], t[:,1], 
                alpha=_alpha, linewidth=1., linestyle='-', color=color)
        ax.scatter(t[-1,0], t[-1,1], 
                 alpha=_ALPHA_PTS, marker=outs_marker[l], edgecolor=color,
                 c='w' if l==-1 else color, s=2 if l==-1 else 10)
    # Emphasize cluster medoids or centroids
    if emphasize_means is not None and cluster_obj is not None:
        for c_idx in range(cluster_obj.num_clusters):
            color = clst_colors[c_idx % len(clst_colors)]
            if emphasize_means == 'medoid':
                idx_med = cluster_obj.cluster_to_medoids[c_idx][1]
                t_med = traj_main[idx_med]
                l_med = outcomes[idx_med]
                ax.plot(t_med[:,0], t_med[:,1], zorder=num_trajectories-1,
                        alpha=1, linewidth=3, linestyle='--', color=color)
                ax.scatter(t_med[-1,0], t_med[-1,1], 
                           zorder=num_trajectories-1, 
                           alpha=_ALPHA_PTS, marker=outs_marker[l_med],
                           c='w' if l_med==-1 else color, edgecolor='k', 
                           s=5 if l_med==-1 else 15)
            elif emphasize_means == 'centroid':
                pass
    # Add goal line
    if 'target_info' in env_info.keys():
        goal_line = env_info['target_info'][0]['xy'][0]
        ax.axvline(goal_line, c='k', ls='--', alpha=0.4)

    if title is not None:
        ax.set_title('{}'.format(title, num_trajectories))
    else:
        nl = '[loop: {}] '.format(spec_title) if type(spec_title)==int else ''
        ax.set_title("{}Walker hull trajectories "
                     "({})".format(nl, num_trajectories))
        
    ax.set_aspect('equal')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    # Add legend
    if cluster_obj is not None:
        custom_lgd = [plt.scatter([], [], 
                      color='w', lw=1, marker='o', edgecolor='k'), 
                      plt.scatter([], [], color='k', lw=1, marker='*')]
        custom_lab = ['class: -1', 'class: 0']  
        unique_clusters = np.unique(cluster_obj.datapoints_to_cluster)
        clst_lgd = [mlines.Line2D([], [], linestyle='-', linewidth=2, 
                    color=clst_colors[cc % len(clst_colors)]) \
                    for cc in unique_clusters]
        clst_lab = ['cluster {}'.format(cc) for cc in unique_clusters]
        custom_lgd.extend(clst_lgd)
        custom_lab.extend(clst_lab) 
    else:
        custom_lgd, custom_lab = ax.get_legend_handles_labels()
    lgd = ax.legend(custom_lgd, custom_lab,
                    loc='upper left', bbox_to_anchor=(1, 1),
                    ncol=int(np.ceil(len(custom_lab)/15)))

# ADD also plot reconstructed trajectories

    # Save/show figure
    ax.set_xlim(min_x-0.2, max_x+0.2)
    ax.set_ylim(3, 13) 
    # plt.tight_layout()
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if type(spec_title)==int:
            plt.savefig('{}/traj_loop_{:05d}.{}'.format(
                        savepath, spec_title, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                        # bbox_inches='tight'
                        )  # 300
            logger.info("Figure saved: '{}/traj_{:05d}.{}'".format(
                        savepath, spec_title, img_format))
        else:
            plt.savefig('{}/{}.{}'.format(
                        savepath, spec_title, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                        bbox_inches='tight') 
    if show_plots:
        plt.show()
    else:
        plt.cla()



@ignore_warnings
def plot_traj_striker(traj_main, outcomes, 
                      emphasize_means=None, new_data=None,
                      savepath=None, show_plots=False, spec_title=None, 
                      img_format='png', dpi=PLOT_DPI, **kwargs):
    """ Plots the ball trajectories """
    outcomes = outcomes[:,0]
    num_trajectories = len(outcomes)
    unique_outcomes = np.unique(outcomes)
    f, ax = plt.subplots()
    lgd = plot_paths_ax(ax, traj_main, outcomes=outcomes, new_data=new_data, 
                        spec_title=spec_title, legend=True,
                        emphasize_means=emphasize_means, **kwargs)

# ADD also plot reconstructed trajectories
### SHOW only after re-fitting AE


    # Save/show figure
    plt.tight_layout()
    new_tmp = '' if new_data is not None else '_new'
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if type(spec_title)==int:
            plt.savefig('{}/traj_loop_{:05d}{}.{}'.format(
                        savepath, spec_title, new_tmp, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                        # bbox_inches='tight'
                        )  # 300
            logger.info("Figure saved: '{}/traj_{:05d}.{}'".format(
                        savepath, spec_title, img_format))
        else:
            plt.savefig('{}/{}.{}'.format(
                        savepath, spec_title, img_format), 
                        format=img_format, bbox_extra_artists=(lgd,), dpi=dpi, 
                        bbox_inches='tight') 
    if show_plots:
        plt.show()
    else:
        plt.cla()


@ignore_warnings
def plot_point_pairs(orig_dict, recn_dict, recn_errors,
                     param_ranges, target_info, ball_ranges, striker_ranges,
                     spec_title, savepath=None, show_plots=False, 
                     img_format='png', dpi=PLOT_DPI):
    """ 
        Plots the pairwise comparison of original and reconstructed points;
        Each pair has different color;
        - In the parameter space (orig: circle, recn: square)
        - Ball and striker trajectories for comparison 
            (orig: solid, recn: dashed)
    """
    _alpha = 0.3*_ALPHA_PTS
    _w, _h = .1, .05
    _d = 0.5*np.sqrt(_w**2+_h**2)
    _phi = np.arctan2(-_h, -_w)
    def get_bl(x, y, theta):
        alpha = _phi + theta
        return (x+_d*np.cos(alpha), y+_d*np.sin(alpha)-0.3)

    def add_striker(x, y, theta=None, is_orig=True, color=None):
      """ adds a striker as a circle or rectangle based on params """
      # choose if original or reconstructed path
      if is_orig:
          kws = dict(linestyle='-', linewidth=1, alpha=_alpha, 
                     edgecolor='none',facecolor=color)
      else:
          kws = dict(linestyle='--', linewidth=1, alpha=_alpha, 
                     edgecolor=color, facecolor='none')
      # choose striker shape
      if theta:
          return mpatches.Rectangle(get_bl(x, y, theta), 
                                    angle=np.rad2deg(theta), 
                                    width=_w, height=_h, **kws)
      else:
          return mpatches.Circle((x,y-0.3), _w, **kws)


    num_trajectories = len(orig_dict['labels'])
    # mpl.rcParams.update({'font.size': 16})

    f, axs = plt.subplots(1, 2,figsize=(18.6, 14))
    for od, rd, r_err, ol, rl, obt, rbt, ost, rst in \
        zip(orig_dict['data'], recn_dict['data'], recn_errors, \
            orig_dict['labels'], recn_dict['labels'], \
            orig_dict['ball_traj'], recn_dict['ball_traj'], \
            orig_dict['striker_traj'], recn_dict['striker_traj']):
        # original-reconstruction pairs have the same color
        color = next(axs[0]._get_lines.prop_cycler)['color']
        # Left plot: scatterplot of orig vs recn points (circle vs square)
        axs[0].scatter(od[0], od[1], 
                       s=_MARKER_SZ, c=color, marker='o', alpha=_ALPHA_PTS)
        axs[0].scatter(rd[0], rd[1], label='r_err: {:.2e}'.format(r_err),
                       s=_MARKER_SZ, c=color, marker='s', alpha=_ALPHA_PTS)
        # Right plot: ball trajectories
        axs[1].plot(obt[:,0], obt[:,1], 
                    alpha=0.5, linewidth=2, color=color, linestyle='-')
        axs[1].plot(rbt[:,0], rbt[:,1], 
                    alpha=0.5, linewidth=2, color=color, linestyle='--')
        axs[1].scatter(obt[-1,0], obt[-1,1], s=5, c='C0' if ol==-1 else 'C1')
        axs[1].scatter(rbt[-1,0], rbt[-1,1], s=5, c='C0' if rl==-1 else 'C1')
        # Right plot: striker trajectories
        n_skip = max(1, int(len(rst)/100))
        for coord in rst[::n_skip]:
            striker = add_striker(*coord, is_orig=False, color=color)
            axs[1].add_patch(striker)
        n_skip = max(1, int(len(rst)/100))
        for coord in ost[::n_skip]:
            striker = add_striker(*coord, is_orig=True, color=color)
            axs[1].add_patch(striker)

    # Add target and striker range
    ball = plt.Circle(color='g', alpha=_alpha, linewidth=0, 
                        zorder=num_trajectories, xy=(0,0), radius=0.025)
    striker = plt.Rectangle(tuple(striker_ranges[:,0]-np.array([0,0.3])),
                            abs(striker_ranges[0,1]-striker_ranges[0,0]),
                            abs(striker_ranges[1,1]-striker_ranges[1,0]),
                            linewidth=2, linestyle='--', edgecolor='k', 
                            facecolor='none', alpha=_alpha, 
                            zorder=num_trajectories)
    for ti in target_info:
        target = plt.Circle(color='g', alpha=_alpha, linewidth=2, 
                            zorder=num_trajectories, **ti)
        axs[1].add_artist(target)

    # ax = plt.gca()
    axs[1].add_artist(ball)
    axs[1].add_artist(striker) 
    # plt.axis('square')
    # axs[0].set_aspect('equal')
    axs[0].axis('square')
    axs[0].set_title('Original & reconstructed ({}) points'.format(
                     num_trajectories))
    axs[0].set_xlabel('dim 0')
    axs[0].set_ylabel('dim 1')
    axs[0].set_xlim(*striker_ranges[0])
    axs[0].set_ylim(*striker_ranges[1])
    # axs[1].set_aspect('equal')
    axs[1].axis('square')
    axs[1].set_title('Ball & striker ({}) trajectories'.format(
                     num_trajectories))
    axs[1].set_xlabel('x-axis')
    axs[1].set_ylabel('y-axis')
    axs[1].set_xlim(ball_ranges[0])
    axs[1].set_ylim(ball_ranges[1]) 
    # Add descriptive legend entry
    h, l = axs[0].get_legend_handles_labels()
    custom_lgd0 = [
        plt.scatter([], [], color='k', lw=1, marker='o', alpha=_ALPHA_PTS), 
        plt.scatter([], [], color='k', lw=1, marker='s', alpha=_ALPHA_PTS)]
    custom_lab0 = ['original','reconstructed' ]
    custom_lgd0.extend(h)
    custom_lab0.extend(l)
    custom_lgd1 = [
        plt.scatter([], [], color='C0', lw=1), 
        plt.scatter([], [], color='C1', lw=1),
        mlines.Line2D([], [], color='k', linestyle='-', linewidth=2),
        mlines.Line2D([], [], color='k', linestyle='--', linewidth=2)]
    custom_lab1 = ['class: -1', 'class: 0', 'original','reconstructed' ]
    # Add legend
    lgd = axs[0].legend(handles=custom_lgd0, 
                        labels=custom_lab0, 
                        loc='lower right', bbox_to_anchor=(1, 1), ncol=1)
    lgd = axs[1].legend(handles=custom_lgd1, labels=custom_lab1, 
                        loc='lower right', bbox_to_anchor=(1, 1), ncol=1)
    # Save/show figure
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        plt.savefig('{}/bad_trajs_{}.{}'.format(
                    savepath, spec_title, img_format), 
                    format=img_format, bbox_extra_artists=(lgd,), 
                    bbox_inches='tight', dpi=dpi) 
    if show_plots:
        plt.show()
    else:
        plt.cla()




###############################################################################
###############################################################################
###############################################################################


"""  BASIC PLOTS with AX"""

# @ignore_warnings
def plot_paths_ax(ax, trajectories, env_info,
                  outcomes=None, cluster_object=None,
                  new_data=None,
                  metric_bd=None, metric_dim=None,
                  emphasize_means='medoid',
                  title=None, spec_title=None,
                  legend=False, alpha_c=0.2, **kwargs):
    """ Plots the ball trajectories """
    _alpha = alpha_c * _ALPHA_PTS
    num_trajectories = len(outcomes)
    outs_marker = {0: '*', -1:'o'}
    clst_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])


    # Make sure clusters don't affect plots
    if cluster_object is not None and \
       cluster_object.datapoints_to_cluster is None:
        cluster_obj = None
    else:
        cluster_obj = cluster_object

    if new_data is not None:
        # plot trajectories colored by their cluster color
        for data_idx, (t, l) in enumerate(zip(new_data['traj_main'], 
                                              new_data['outcomes'][:,0])):
            ax.plot(t[:,0], t[:,1], 
                    alpha=_ALPHA_PTS, linewidth=1., linestyle='--', color='k')
            ax.scatter(t[-1,0], t[-1,1], 
                     alpha=_alpha, marker=outs_marker[l], edgecolor='k',
                     c='w' if l==-1 else 'k', s=2 if l==-1 else 10)

    if cluster_obj is not None:
        # plot trajectories colored by their cluster color
        for data_idx, (t, l) in enumerate(zip(trajectories, outcomes)):
            clusters_idx = cluster_obj.datapoints_to_cluster[data_idx]
            color = clst_colors[clusters_idx % len(clst_colors)]
            ax.plot(t[:,0], t[:,1], 
                    alpha=_alpha, linewidth=1., linestyle='-', color=color)
            ax.scatter(t[-1,0], t[-1,1], 
                     alpha=_ALPHA_PTS, marker=outs_marker[l], edgecolor=color,
                     c='w' if l==-1 else color, s=2 if l==-1 else 10)
        # Emphasize cluster medoids or centroids
        if emphasize_means is not None:
            for c_idx in range(cluster_obj.num_clusters):
                color = clst_colors[c_idx % len(clst_colors)]
                if emphasize_means == 'medoid':
                    idx_med = cluster_obj.cluster_to_medoids[c_idx][1]
                    t_med = trajectories[idx_med]
                    l_med = outcomes[idx_med]
                    ax.plot(t_med[:,0], t_med[:,1], zorder=num_trajectories-1,
                            alpha=1, linewidth=3, linestyle='--', color=color)
                    ax.scatter(t_med[-1,0], t_med[-1,1], 
                               zorder=num_trajectories-1, 
                               alpha=_ALPHA_PTS, marker=outs_marker[l_med],
                               c='w' if l_med==-1 else color, edgecolor='k', 
                               s=5 if l_med==-1 else 15)
                elif emphasize_means == 'centroid':
                    pass
    elif metric_bd is not None and metric_dim is not None:
        for data_idx, (t, m, l) in enumerate(zip(trajectories, 
                                                 metric_bd, 
                                                 outcomes)):
            # Show clusters along 0th dimension
            clusters_idx = np.argmax(m)//np.prod(metric_dim[1:])
            color = clst_colors[clusters_idx % len(clst_colors)]
            ax.plot(t[:,0], t[:,1],
                    alpha=0.2, linewidth=1., linestyle='-', color=color)
            ax.scatter(t[-1, 0], t[-1, 1],
                     alpha=0.5, marker=outs_marker[l], edgecolor=color,
                     c='w' if l==-1 else color, s=2 if l==-1 else 10)

    elif outcomes is not None:
        # plot trajectory with colors depending on outcome
        for t, l in zip(trajectories, outcomes):
            if len(t.shape) == 1: t = t.reshape(-1, 2)
            color = 'C0' if l==-1 else 'C1'
            ax.plot(t[:,0], t[:,1], 
                    alpha=_alpha, linewidth=1, linestyle='-', color=color)
            ax.scatter(t[-1,0], t[-1,1], s=2, c=color)
    else:
        # plot each trajectory in different colour
        for i, t in enumerate(trajectories):
            color = next(ax._get_lines.prop_cycler)['color']
            if len(t.shape) == 1: t = t.reshape(-1, 2)
            ax.plot(t[:,0], t[:,1], label=i, 
                    alpha=_alpha, linewidth=1, linestyle='-', color=color)
            ax.scatter(t[-1,0], t[-1,1], s=2, c=color)

    # Add target and striker range
    if 'target_info' in env_info.keys(): 
        for ti in env_info['target_info']:
            target = mpatches.Circle(color='g', alpha=0.1, linewidth=2, 
                                     zorder=num_trajectories, **ti)
            ax.add_artist(target)

    # if 'striker_ranges' in env_info.keys():
    #     striker_ranges = env_info['striker_ranges']
    #     striker = mpatches.Rectangle(tuple(striker_ranges[:,0]-np.array([0,0.3])),
    #                             abs(striker_ranges[0,1]-striker_ranges[0,0]),
    #                             abs(striker_ranges[1,1]-striker_ranges[1,0]),
    #                             linewidth=2, linestyle='--', edgecolor='k', 
    #                             facecolor='none', alpha=0.1, 
    #                             zorder=num_trajectories)
    #     ax.add_artist(striker) 
        
    if 'ball_ranges' in env_info.keys():
        ball_ranges = env_info['ball_ranges']
        ax.set_xlim(ball_ranges[0])
        ax.set_ylim(ball_ranges[1])
    else:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-.6, 1.6) 
    # plt.axis('equal')
    # ax.axis('square')
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title('{}'.format(title, num_trajectories))
    else:
        nl = '[loop: {}] '.format(spec_title) if type(spec_title)==int else ''
        ax.set_title('{}Ball trajectories ({})'.format(nl, num_trajectories))
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    # Add legend
    if legend == True:
        if cluster_obj is not None:
            custom_lgd = [plt.scatter([], [], 
                          color='w', lw=1, marker='o', edgecolor='k'), 
                          plt.scatter([], [], color='k', lw=1, marker='*')]
            custom_lab = ['class: -1', 'class: 0']  
            unique_clusters = np.unique(cluster_obj.datapoints_to_cluster)
            clst_lgd = [mlines.Line2D([], [], linestyle='-', linewidth=2, 
                        color=clst_colors[cc % len(clst_colors)]) \
                        for cc in unique_clusters]
            clst_lab = ['cluster {}'.format(cc) for cc in unique_clusters]
            custom_lgd.extend(clst_lgd)
            custom_lab.extend(clst_lab)
        elif metric_bd is not None and metric_dim is not None: 
            custom_lgd = [plt.scatter([], [], color='w', lw=1, marker='o', 
                                              edgecolor='k'), 
                          plt.scatter([], [], color='k', lw=1, marker='*')]
            custom_lab = ['class: -1', 'class: 0']  
        elif outcomes is not None:
            custom_lgd = [ax.scatter([], [], color='C0', lw=1), 
                          ax.scatter([], [], color='C1', lw=1)]
            custom_lab = ['class: -1', 'class: 0']
        else:
            custom_lgd, custom_lab = ax.get_legend_handles_labels()
        lgd = ax.legend(custom_lgd, custom_lab,
                        loc='upper left', bbox_to_anchor=(1, 1),
                        ncol=int(np.ceil(len(custom_lab)/15)))
        return lgd



@ignore_warnings
def distance_matrix_ax(ax, distances):
    im = ax.imshow(distances, interpolation='nearest', cmap='Reds') 
    ax.invert_yaxis()
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(-.5, len(distances), 1), minor=True);
    ax.set_yticks(np.arange(-.5, len(distances), 1), minor=True);
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
    ax1_divider = ax_grid.axes_divider.make_axes_locatable(ax)
    # add an axes to the right of the main axes.
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = ax_grid.colorbar.colorbar(im, cax=cax1)



