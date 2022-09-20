"""Plotting functions for TH analysis."""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec

from spiketools.plts.task import plot_task_structure as _plot_task_structure
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_bar, plot_barh, plot_hist
from spiketools.plts.spatial import create_heat_title
from spiketools.plts.utils import check_ax, savefig, make_axes
from spiketools.plts.style import set_plt_kwargs
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
    
    
def plot_distance_error(norm_error, norm_error_THF, norm_error_THO):
    """Plot the normalized distance error across sessions"""
    
    ax1, ax2, ax3 = make_axes(3, 3, figsize=(16, 3), wspace=0.4)
    
    plot_hist(norm_error, color='silver', ax=ax1, title='All recall distance error', 
              xlabel='Mean normalized \n distance error', ylabel='Number of sessions')
    plot_hist(norm_error_THF, color='silver', ax=ax2, title='THF recall distance error', 
              xlabel='Mean normalized \n distance error', ylabel='Number of sessions')
    plot_hist(norm_error_THO, color='silver', ax=ax3, title='THO recall distance error', 
              xlabel='Mean normalized \n distance error', ylabel='Number of sessions')
    
    add_vlines(0.5, linestyle='--', color='black', ax=ax1)
    add_vlines(0.5, linestyle='--', color='black', ax=ax2)
    add_vlines(0.5, linestyle='--', color='black', ax=ax3)

    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)

    

def plot_recall_correctness(correct_all, correct_THF, correct_THO):
    """Plot the percentage of recall correctness across sessions"""
    
    ax1, ax2, ax3 = make_axes(3, 3, figsize=(16, 3), wspace=0.4)

    plot_hist(correct_all, color='silver', ax=ax1, title='All recall correctness', 
              xlabel='% Recall', ylabel='Number of sessions')
    plot_hist(correct_THF, color='silver', ax=ax2, title='THF recall correctness', 
              xlabel='% Recall', ylabel='Number of sessions')
    plot_hist(correct_THO, color='silver', ax=ax3, title='THO recall correctness', 
              xlabel='% Recall', ylabel='Number of sessions')

    add_vlines(50, linestyle='--', color='black', ax=ax1)
    add_vlines(50, linestyle='--', color='black', ax=ax2)
    add_vlines(50, linestyle='--', color='black', ax=ax3)
    ax2.set_yticks(range(3))


    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)



def plot_confidence_response(conf_all, conf_THF, conf_THO):
    """Plot the number of confidence responsein each category across sessions"""
    ax1, ax2, ax3 = make_axes(3, 3, figsize=(16, 3), wspace=0.4)

    plot_bar(conf_all.values(), labels=conf_all.keys(), title='All confidence response',
             ylabel='count', ax=ax1, color='silver')
    plot_bar(conf_THF.values(), labels=conf_THF.keys(), title='THF confidence response', 
             ylabel='count', ax=ax2, color='silver')
    plot_bar(conf_THO.values(), labels=conf_THO.keys(), title='THO confidence response', 
             ylabel='count', ax=ax3, color='silver')

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    

def plot_example_target(n_bins, target_bins, chests_per_bin, tpos_per_bin, tspikes_pos_per_bin, 
                        chest_x, chest_y, reshaped_bins, area_range, name):
    """Plot individual bins in selected example spatial target cells """

    fig=plt.figure(figsize=(20, 50))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.title(create_heat_title('{}'.format(name), target_bins), fontsize=25)
    plt.axis("off")

    for ind in range(n_bins):
        ax1 = fig.add_subplot(target_bins.shape[0], target_bins.shape[1], ind+1)
        ax2 = fig.add_subplot(target_bins.shape[0], target_bins.shape[1], ind+1, frame_on=False)
        ax1.axis("on")

        if chests_per_bin[ind] == []:

            plot_heatmap(target_bins, ax=ax1, aspect='auto', alpha=0.6)
            ax2.axis("off")

        else: 
            
            tspikes = {'positions' : np.array([tspikes_pos_per_bin[ind][0], tspikes_pos_per_bin[ind][1]]), 
                       'color' : 'red', 'ms' : 10, 'alpha' : 0.7}
            landmarks = [{'positions' : np.array([chest_x[ind], chest_y[ind]]),
                          'color' : 'green', 'ms' : 40, 'alpha' : 0.7}]

            plot_heatmap(reshaped_bins, ignore_zero=True, ax=ax1, aspect='auto', alpha=0.6)
            plot_positions(tpos_per_bin[ind], tspikes, landmarks, ax=ax2, xlim=area_range[0], ylim=area_range[1])

        ax1.axis("on")
        ax1.set_yticks([])
        ax1.set_xticks([])