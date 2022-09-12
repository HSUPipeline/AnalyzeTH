"""Plotting functions for TH analysis."""

from matplotlib import gridspec
from spiketools.plts.task import plot_task_structure as _plot_task_structure
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_bar, plot_hist
from spiketools.plts.spatial import create_heat_title
from spiketools.plts.utils import check_ax, savefig, set_plt_kwargs
from spiketools.plts.annotate import _add_vlines, _add_box_shades, _add_hlines

import matplotlib.pyplot as plt
import numpy as np


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

    y_pos = np.arange(len(spikes_trial))
    plot_rasters(spikes_trial, show_axis=True, ax=ax0)
    _add_box_shades(nav_stops_trial, np.arange(len(spikes_trial)), x_range=0.1, y_range=0.5, ax=ax0)
    _add_hlines(hlines, ax=ax0, color='green', alpha=0.4)
    ax1.barh(y_pos, frs)

    plot_rasters(spikes, ax=ax2, show_axis=True, ylabel='spikes from whole session')
    _add_vlines(nav_stops, ax=ax2, color='purple') #navigation starts
    _add_vlines(nav_starts, ax=ax2, color='orange')#navigation stops 

    plot_rasters(nav_spikes_all, vline=openings, show_axis=True, ax=ax3, ylabel='Spikes from navigation periods')

    ax0.set(xlabel='Spike times', ylabel="Trial number", yticks=range(0,len(spikes_trial)))
    ax0.spines.right.set_visible(False)
    ax0.spines.top.set_visible(False)
    ax1.spines.left.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)

    ax0.xaxis.set_ticks_position('bottom')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.tick_params(left = False)
    ax1.set(xlabel="FR")

    ax2.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax3.spines.right.set_visible(False)
    ax3.spines.left.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax2.tick_params(left = False)
    ax3.tick_params(left = False)
    ax2.set(yticks=[])
    ax3.set(yticks=[])

    fig.suptitle(create_heat_title('{}'.format(name), frs), fontsize=20)
    
    
def plot_distance_error(norm_error, norm_error_THF, norm_error_THO):
    """Plot the normalized distance error across sessions"""
    
    fig=plt.figure(figsize=(16, 3))
    plt.subplots_adjust(wspace=0.4)
    ax = plt.subplot(132)
    ax1 = plt.subplot(133)
    ax2 = plt.subplot(131)

    plot_hist(norm_error, color='silver', ax=ax2)
    _add_vlines(0.5, linestyle='--', color='black', ax=ax2)
    ax2.set(title='All recall distance error', xlabel='Mean normalized \n distance error', ylabel='Number of sessions')
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plot_hist(norm_error_THF, color='silver', ax=ax)
    _add_vlines(0.5, linestyle='--', color='black', ax=ax)
    ax.set(title='THF recall distance error', xlabel='Mean normalized \n distance error', ylabel='Number of sessions')
    ax.set_yticks(range(3))
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plot_hist(norm_error_THO, color='silver', ax=ax1)
    _add_vlines(0.5, linestyle='--', color='black', ax=ax1)
    ax1.set(title='THO recall distance error', xlabel='Mean normalized \n distance error', ylabel='Number of sessions')
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    

def plot_recall_correctness(correct_all, correct_THF, correct_THO):
    """Plot the percentage of recall correctness across sessions"""
    
    fig=plt.figure(figsize=(16, 3))
    plt.subplots_adjust(wspace=0.4)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    plot_hist(correct_all, color='silver', ax=ax1)
    plot_hist(correct_THF, color='silver', ax=ax2)
    plot_hist(correct_THO, color='silver', ax=ax3)

    _add_vlines(50, linestyle='--', color='black', ax=ax1)
    _add_vlines(50, linestyle='--', color='black', ax=ax2)
    _add_vlines(50, linestyle='--', color='black', ax=ax3)
    ax2.set_yticks(range(3))

    ax1.set(title='All recall correctness', xlabel='% Recall', ylabel='Number of sessions')
    ax2.set(title='THF recall correctness', xlabel='% Recall', ylabel='Number of sessions')
    ax3.set(title='THO recall correctness', xlabel='% Recall', ylabel='Number of sessions')

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')


def plot_confidence_response(conf_all, conf_THF, conf_THO):
    """Plot the number of confidence responsein each category across sessions"""
    
    fig=plt.figure(figsize=(16, 3))
    plt.subplots_adjust(wspace=0.4)

    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    plot_bar(conf_all.values(), labels=conf_all.keys(), title='All confidence response',
             ylabel='count', ax=ax1, color='silver')
    plot_bar(conf_THF.values(), labels=conf_THF.keys(), title='THF confidence response', 
             ylabel='count', ax=ax2, color='silver')
    plot_bar(conf_THO.values(), labels=conf_THO.keys(), title='THO confidence response', 
             ylabel='count', ax=ax3, color='silver')

    ax3.spines.right.set_visible(False)
    ax3.spines.top.set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')