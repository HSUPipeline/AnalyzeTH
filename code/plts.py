"""Plotting functions for TH analysis."""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec

from spiketools.plts.task import plot_task_structure as _plot_task_structure
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_bar, plot_barh, plot_hist
from spiketools.plts.spatial import create_heat_title
from spiketools.plts.utils import check_ax, savefig, make_axes, make_grid, get_grid_subplot
from spiketools.plts.style import set_plt_kwargs, drop_spines
from spiketools.plts.annotate import add_vlines, add_box_shades, add_hlines
from spiketools.plts.spatial import plot_positions, plot_heatmap


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
                         range_colors=['green', 'orange', 'purple'],
                         line_colors=['red', 'black'],
                         line_kwargs={'lw' : 1.25},
                         ax=ax, **plt_kwargs)


def plot_spikes_trial(spikes, spikes_trial, nav_stops_trial, nav_spikes_all, nav_starts, 
                      nav_stops, openings, name, frs, hlines, ax=None, **plt_kwargs):
    """Plot the spike raster for whole session, navigation periods and individual trials."""
    
    fig = plt.figure(figsize=(18,20))
    gs = gridspec.GridSpec(3, 2, width_ratios=[5, 1]) 
    
    ax0 = plt.subplot(gs[2,0])
    ax1 = plt.subplot(gs[2,1], sharey = ax0)
    ax2 = plt.subplot(gs[0,:])
    ax3 = plt.subplot(gs[1,:])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    ypos = (np.arange(len(spikes_trial))).tolist()
    plot_rasters(spikes_trial, show_axis=True, ax=ax0, xlabel='Spike times', 
                 ylabel="Trial number", yticks=range(0,len(spikes_trial)))
    add_box_shades(nav_stops_trial, np.arange(len(spikes_trial)), x_range=0.1, y_range=0.5, ax=ax0)
    add_hlines(hlines, ax=ax0, color='green', alpha=0.4)
    
    plot_barh(frs, ypos, ax=ax1, xlabel="FR")

    plot_rasters(spikes, ax=ax2, show_axis=True, ylabel='spikes from whole session', yticks=[])
    add_vlines(nav_stops, ax=ax2, color='purple') # navigation starts
    add_vlines(nav_starts, ax=ax2, color='orange')# navigation stops 

    plot_rasters(nav_spikes_all, vline=openings, show_axis=True, ax=ax3, 
                 ylabel='Spikes from navigation periods', yticks=[])

    ax1.tick_params(left=False)
    ax2.tick_params(left=False)
    ax3.tick_params(left=False)
    
    # Drop after spiketools update 
    ax0.spines.right.set_visible(False)
    ax0.spines.top.set_visible(False)
    ax1.spines.left.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax3.spines.right.set_visible(False)
    ax3.spines.left.set_visible(False)
    ax3.spines.top.set_visible(False)

    fig.suptitle(create_heat_title('{}'.format(name), frs), fontsize=20)
    

def plot_surrogates(surrogates, n_bin, data_value=None, p_value=None, ax=None, **plt_kwargs):
    """Plot a distribution of surrogate data.
    Parameters
    ----------
    surrogates : 1d array
        The collection of values computed on surrogates.
    n_bin : int
        The number of bins plotted in the histogram.
    data_value : float, optional
        The statistic value of the real data, to draw on the plot.
    p_value : float, optional
        The p-value to print on the plot.
    ax : Axes, optional
        Axis object upon which to plot.
    plt_kwargs
        Additional arguments to pass into the plot function.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    
    _, _, bars = ax.hist(surrogates, bins=n_bin, color='gray')
    for bar in bars:
        if bar.get_x() > data_value:
            bar.set_facecolor('lightgray')

    if data_value is not None:
        add_vlines(data_value, ax=ax, color='darkred', linestyle='solid', linewidth=4)
        ax.plot(data_value, 0, 'o', zorder=10, clip_on=False, color='darkred', markersize=15)

    if p_value is not None:
        text = 'p={:4.4f}'.format(p_value)
        ax.plot([], label=text)
        ax.legend(handlelength=0, edgecolor='white', loc='best', fontsize=16)
    
    
def plot_place_target_comparison(place_bin_frs, target_bin_frs, surr_place, surr_target, n_bin,
                                fval_place, favl_target, pval_place, pval_target, **plt_kwargs):
    """Plot firing rate maps of example place & target cells with their corresponding distribution of surrogate data"""
    grid = make_grid(3, 2, figsize=(10, 8), wspace=0.3, **plt_kwargs)
    plot_heatmap(place_bin_frs, ax=get_grid_subplot(grid, slice(0, 2), 0), 
                 title='FR by subject position', aspect='auto', cbar=True, **plt_kwargs)
    plt.axis("on")
    plot_heatmap(target_bin_frs, ignore_zero=True, ax=get_grid_subplot(grid, slice(0, 2), 1), 
                 title='FR by target position', aspect='auto', cbar=True, **plt_kwargs)
    plt.axis("on")
    
    plot_surrogates(surr_place, n_bin, fval_place, pval_place, ax=get_grid_subplot(grid, 2, 0))
    plot_surrogates(surr_target, n_bin, favl_target, pval_target, ax=get_grid_subplot(grid, 2, 1))
    
    get_grid_subplot(grid, 2, 0).set(xlabel='F-statistics', ylabel='count')
    drop_spines(get_grid_subplot(grid, 2, 0), ['top', 'right'])
    drop_spines(get_grid_subplot(grid, 2, 1), ['top', 'right'])
    

def plot_example_target(n_bins, target_bins, reshaped_target_bins, chests_per_bin, pos_per_bin, 
                        spikes_per_bin, chest_x, chest_y, area_range, name):
    """Plot navigation path and spikes associated with chests in each bin for selected example spatial target cells """
    
    fig=plt.figure(figsize=(20, 50))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.title(create_heat_title('{}'.format(name), target_bins), fontsize=25)
    plt.axis("off")

    for ind in range(n_bins):
        ax1 = fig.add_subplot(target_bins.shape[0], target_bins.shape[1], ind+1)
        ax2 = fig.add_subplot(target_bins.shape[0], target_bins.shape[1], ind+1, frame_on=False)
        ax1.axis("on")

        if chests_per_bin[(n_bins-1)-ind] == []:

            plot_heatmap(reshaped_target_bins, transpose=True, ignore_zero=True, 
                         ax=ax1, aspect='auto', origin='lower', alpha=0.5)
            plot_positions(pos_per_bin[(n_bins-1)-ind], ax=ax2, xlim=area_range[0], ylim=area_range[1])

        else: 

            tspikes = {'positions' : np.array([spikes_per_bin[(n_bins-1)-ind][0], spikes_per_bin[(n_bins-1)-ind][1]]), 
                       'color' : 'red', 'ms' : 10, 'alpha' : 0.9}
            landmarks = [{'positions' : np.array([chest_x[(n_bins-1)-ind], chest_y[(n_bins-1)-ind]]),
                          'color' : 'green', 'ms' : 30, 'alpha' : 0.7}]

            plot_heatmap(reshaped_target_bins, transpose=True, ignore_zero=True, 
                         ax=ax1, aspect='auto', origin='lower', alpha=0.5)
            plot_positions(pos_per_bin[(n_bins-1)-ind], tspikes, landmarks, ax=ax2, 
                           xlim=area_range[0], ylim=area_range[1])

        ax1.axis("on")
        
        
# def plot_surr_stats(surrogates, f_val, p_val, n_bin, ax=None, **plt_kwargs): 
    
#     ax = check_ax(ax, return_current=True)
    
#     _, _, bars = ax.hist(surrogates, bins=n_bin, color='gray', **plt_kwargs)
#     for bar in bars:
#         if bar.get_x() > f_val:
#             bar.set_facecolor('lightgray')
#     add_vlines(f_val, ax=ax, color='darkred', linestyle='solid', linewidth=4)
#     ax.plot(f_val, 0, 'o', zorder=10, clip_on=False, color='darkred', markersize=15)
    
#     text = 'p={:4.4f}'.format(p_val)
#     ax.plot([], label=text)
#     ax.legend(handlelength=0, edgecolor='white', loc='best', fontsize=16)
# #     ax.set(xlabel='F-statistics', ylabel='Count')