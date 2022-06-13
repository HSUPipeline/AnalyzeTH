"""Plotting functions for TH analysis."""

import matplotlib.pyplot as plt

from spiketools.plts.task import plot_task_structure as _plot_task_structure

###################################################################################################
###################################################################################################

def plot_task_structure(trials, ax=None, **plt_kwargs):
    """Plot the task structure for Treasure Hunt.

    Parameters
    ----------
    trials : pynwb.epoch.TimeIntervals
        The TreasureHunt trials structure from a NWB file.
    """

    _plot_task_structure([[trials.navigation_start[:], trials.navigation_stop[:]],
                          [trials.distractor_start[:], trials.distractor_stop[:]],
                          [trials.recall_start[:], trials.recall_stop[:]]],
                         [trials.start_time[:], trials.stop_time[:]],
                         shade_colors=['green', 'orange', 'purple'],
                         line_colors=['red', 'black'],
                         line_kwargs={'lw' : 1.25},
                         ax=ax, **plt_kwargs)

    
def plot_trial_position(t_spike_xs, t_spike_ys, chest_xs, t_mask, chest_ys, 
                        t_pos, ch_times, area_range, t_ind, ax):
    """Plot an encoding trial: with traversals, chest locations, and spike firing"""
    
    tspikes = {'positions' : np.array([t_spike_xs, t_spike_ys]), 'ms' : 10, 'alpha' : 0.6}
    landmarks = [{'positions' : np.array([chest_xs[t_mask], chest_ys[t_mask]]),
                  'color' : 'green', 'ms' : 40, 'alpha' : 0.25},
                 {'positions' : np.atleast_2d(t_pos[:, 0]).T, 'color' : 'blue', 'ms' : 15},
                 {'positions' : np.array(ch_times).T, 'color' : 'y', 'ms' : 15}]
    plot_positions(t_pos, tspikes, landmarks, alpha=0.75, figsize=(5, 7), ax=ax)
    title = 'Trial{}'.format(t_ind)
    ax.set_title(title, fontdict={'fontsize' : 16})
    
    # Set x & y limits to the terrain range, expanding z to include the towers
    ax.set_xlim(*area_range[0])
    ax.set_ylim(area_range[1][0], area_range[1][1])


def plot_trial_position_frs(t_spike_xs, t_spike_ys, chest_xs, t_mask, chest_ys, 
                            t_pos, ch_times, area_range, frs, count, t_ind):
    """Plot an encoding trial, with firing rates in each segment"""
    
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    tspikes = {'positions' : np.array([t_spike_xs, t_spike_ys]), 'ms' : 8, 'alpha' : 0.6}
    landmarks = [{'positions' : np.array([chest_xs[t_mask], chest_ys[t_mask]]),
                  'color' : 'green', 'ms' : 40, 'alpha' : 0.25},
                 {'positions' : np.atleast_2d(t_pos[:, 0]).T, 'color' : 'blue', 'ms' : 15},
                 {'positions' : np.array(ch_times).T, 'color' : 'y', 'ms' : 15}]
    plot_positions(t_pos, tspikes, landmarks, alpha=0.75, ax=ax1)
    plot_bar(frs[t_ind], count.keys(), ax=ax2)
    
    legend_elements = [Line2D([0], [0], color='C0', lw=2, alpha=0.75, label='Subject Position'),
                       Line2D([0], [0], marker='.', color='w', label='Chest Position',
                                        markerfacecolor='green', markersize=30, alpha=0.25),
                       Line2D([0], [0], marker='.', color='w', label='Starting Position',
                                        markerfacecolor='blue', markersize=18),
                       Line2D([0], [0], marker='.', color='w', label='Chest Opening',
                                        markersize=18, markerfacecolor='y'),
                       Line2D([0], [0], marker='.', color='w', label='Spikes', markersize=12,
                                        markerfacecolor='red', alpha=0.75)]
    ax1.set_xlim(*area_range[0])
    ax1.set_ylim(area_range[1][0], area_range[1][1])
    ax1.legend(handles=legend_elements, fontsize=8, loc='upper left')

    ax2.set_xticks=[0,1,2,3,4]
    ax2.set_xticklabels(['', '1', '2', '3', '4']) 
    ax2.tick_params(axis='x') 
    ax2.tick_params(axis='y')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("serial position ")
    ax2.set_ylabel("firing rate (Hz)")
    
    title = 'Trial{}'.format(t_ind+1)
    fig.suptitle(title, fontsize=18)


def bar_line_plot(all_frs, labels, task, subj, session, unid, n_trials):
    """Barplot of firing rate in each segment with lines from individual trials"""
    
    fig = plt.figure(figsize=(7,5))
    ax = plt.subplot(1,1,1)
    
    sns.barplot(data=all_frs, ci="sd", color='#8da0cb', alpha=0.6, errwidth=1.5)
    sns.despine()
    
    for t_ind in range(n_trials): 
        ax.plot(labels, all_frs[t_ind], color='#8da0cb', alpha=0.2)

    ax.set(title='{}-{}-S{} - U{}'.format(task, subj, session, uind), ylabel="Firing Rate")

    ax.set_xticks=[1,2,3,4]
    ax.set_xticklabels(labels) 


def boxplot_scatter(list_group_data, labels, task, subj, session, uind):
    """boxplot of firing rate in each segment with scatterplot from individual trials"""
    
    fig = plt.figure(figsize=(7,5))
    ax = plt.subplot(1,1,1)
    
    sns.boxplot(data=list_group_data, palette="Set2")
    sns.stripplot(data=list_group_data, linewidth=1, alpha=.45, palette="Set2")
    sns.despine()
    
    ax.set_xticklabels(labels) 
    ax.set(title='{}-{}-S{} - U{}'.format(task, subj, session, uind), ylabel="Firing Rate")
    
    
def violinplot_scatter(list_group_data, labels, task, subj, session, uind):
    """violinplot of firing rate in each segment with scatterplot from individual trials"""
    
    fig = plt.figure(figsize=(7,5))
    ax = plt.subplot(1,1,1)
    
    sns.violinplot(data=list_group_data, palette="Set2", inner="point")
    sns.despine()
    
    ax.set_xticklabels(labels) 
    ax.set(title='{}-{}-S{} - U{}'.format(task, subj, session, uind), ylabel="Firing Rate")
    

def plot_raster_segment(spikes_all_trials, n_trials, labels, vline, task, subj, session, uind):
    """Plot the raster of each segment with normalization """
    
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    plot_rasters(spikes_all_trials, vline=vline, show_axis=True, ax=ax)

    ax.set(title='{}-{}-S{} - U{}'.format(task, subj, session, uind), 
                  xlabel="Normalized Spike Times", ylabel="Trial number", 
                  xticks=[0.5, 1.5, 2.5, 3.5], yticks=range(0,n_trials))

    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', labelsize=18, pad=10) 


def plot_target_cells_heatmap(t_all_xs, t_all_ys, chest_x, chest_y, t_pos_all, 
                              area_range, target_bins, task, subj, session, uind):
    """Plot subject/spike position toward target bin and the fr by spatial target position"""
    
    target_bins_reshaped = reshape_target_bins(target_bins)
    target_bins_reshaped[target_bins_reshaped == 0.] = np.nan
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    tspikes = {'positions' : np.array([t_all_xs, t_all_ys]), 'ms' : 10, 'alpha' : 0.6}
    landmarks = [{'positions' : np.array([chest_x, chest_y]), 
                  'color' : 'green', 'ms' : 40, 'alpha' : 0.25}]
    plot_positions(t_pos_all, tspikes, landmarks, ax=ax1, 
                   xlim=area_range[0], ylim=area_range[1])
    img = ax2.imshow(target_bins_reshaped, aspect="auto", origin="lower")
    
    ax2.axis("on")
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax1.title.set_text('subject position')
    ax2.title.set_text('FR by spatial target position')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.13, 0.03, 0.75]) #colorbar at the right
    plt.colorbar(img, cax=cbar_ax)
    title = '{}-{}-S{} - U{} - ({:1.2f}-{:1.2f})'.format(task, subj, session, uind, *get_range(target_bins))
    fig.suptitle(title, fontsize=20)
    
    plt.show()
    
    
def plot_fr_by_target_subject_position(target_bins, bin_frs):
    """Plot the firing rate by spatial target and subject position"""
    
    vmin = np.min([np.nanmin(target_bins[np.nonzero(target_bins)]), np.nanmin(bin_frs[np.nonzero(bin_frs)])])
    vmax = np.max([np.nanmax(target_bins), np.nanmax(bin_frs)])
    target_bins = reshape_target_bins(target_bins)
    
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    plot_heatmap(target_bins, ignore_zero=True, aspect="auto", ax=ax1, vmin=vmin, vmax=vmax)
    img = ax2.imshow(bin_frs, aspect="auto", vmin=vmin, vmax=vmax)
    
    ax1.axis("on")
    ax2.axis("on")
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax1.title.set_text('FR by spatial target position')
    ax2.title.set_text('FR by subject position')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.13, 0.03, 0.75]) #colorbar at the right
    plt.colorbar(img, cax=cbar_ax)
    plt.show()
