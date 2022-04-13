
import numpy as np
from pingouin import convert_angles, circ_rayleigh

from analyzeth.cmh.utils.subset_data import subset_period_event_time_data, subset_period_data
from analyzeth.analysis import get_spike_heading, bin_circular
from analyzeth.plts import plot_polar_hist

from spiketools.utils import restrict_range


# Plots
import matplotlib.pyplot as plt 
import seaborn as sns 
import analyzeth.cmh.settings_plots as PLOTSETTINGS
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# Analysis settings
import analyzeth.cmh.settings_analysis as SETTINGS


def plot_polar_hist_overlay_surrs(hd_degrees, mean_shuffle_hd_counts, ax=None):
    """Plot a polar histogram.

    Parameters
    ----------
    degrees : 1d array
        Data to plot in a circular histogram.
    """

    if not ax:
        ax = plt.subplot(111, polar=True)

    bin_edges, counts = bin_circular(hd_degrees)
    ax.bar(bin_edges[:-1], counts)
    
    #surr_edges, surr_counts = bin_circular(mean_surrogate_hd_degrees)
    #ax.bar(surr_edges[:-1], surr_counts, color = 'k', alpha = 0.4)
    ax.bar(bin_edges[:-1], mean_shuffle_hd_counts)

    return ax




def plot_hd_raster(
        nwbfile = None, 
        unit_ix = SETTINGS.UNIT_IX,
        ax = None,
        highlight = 'trial'
        ):
    
    # -- LOAD -- 
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder) 

    # -- SESSION DATA -- 
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    

    # -- TRIAL DATA -- 
    trial_starts = (nwbfile.trials['start_time'].data[:])/1e3       #convert to trial time in s
    trial_ends = (nwbfile.trials['stop_time'].data[:])/1e3          #convert to trial time in s

    # -- NAVIGATION DATA --
    navigation_start_times = nwbfile.trials['navigation_start'][:]/1e3
    navigation_end_times = nwbfile.trials['navigation_end'][:]/1e3
    
    # -- SPIKE DATA --
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)                            #get spikes in ms
    spikes = restrict_range(spikes, session_start, session_end)
    spikes = (spikes)/1e3                                          #convert to trial time in s  

    # -- HEAD DIRECTION DATA --
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = (head_direction.timestamps[:])/1e3                   #convert to trial time in s


    # -- PLOT --
    # if not ax:
    #     ax = plt.subplot(111)
    fig, ax = plt.subplots(figsize = [14, 5])

    # Add trial colors
    colors = ['r','y','b'] * 10
    if highlight == 'trial':
        for ix in range(len(trial_starts)):
            ax.axvspan(trial_starts[ix], trial_ends[ix], alpha=0.2, facecolor=colors[ix])

    else:
        for ix in range(len(navigation_start_times)):
            ax.axvspan(navigation_start_times[ix], navigation_end_times[ix], alpha=0.2, facecolor=colors[ix])

    # Add events
    ax.eventplot([spikes, hd_times], linelengths = [0.9, 0.9], colors = ['g', 'b'])
    
    # Format
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Spike Times', 'HD Times'])
    ax.set_xlabel('Time (s)')
    ax.set_title('Unit {} Raster Plot'.format(unit_ix))

    # Show plot
    plt.show()

    return fig, ax

def plot_line_hd_navigation(nwbfile):
    
    # Navigation period data
    navigation_start_times = nwbfile.trials['navigation_start'][:]/1e3
    navigation_end_times = nwbfile.trials['navigation_end'][:]/1e3

    # Head direction data
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:] / 1e3    
    hd_degrees = head_direction.data[:]
    
    # Get nav data
    hd_times_nav = subset_period_event_time_data(hd_times, navigation_start_times, navigation_end_times)
    hd_degrees_nav = subset_period_data(hd_degrees, hd_times, navigation_start_times, navigation_end_times)

    # Plot
    fig, ax = plt.subplots(figsize = [14, 5])
    ax.plot(hd_degrees_nav)

    return fig, ax

def plot_line_hd(nwbfile, unit_ix = None):
    
    # Navigation period data
    navigation_start_times = nwbfile.trials['navigation_start'][:]/1e3
    navigation_end_times = nwbfile.trials['navigation_end'][:]/1e3

    # Head direction data
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:] / 1e3    
    hd_degrees = head_direction.data[:]

    # -- PLOT --
    fig, ax = plt.subplots(figsize = [25, 5])

    # plot navigation periods
    colors = ['r','y','b'] * 10
    for ix in range(len(navigation_start_times)):
            ax.axvspan(navigation_start_times[ix], navigation_end_times[ix], alpha=0.2, facecolor=colors[ix])

    # plot hd
    ax.plot(hd_times, hd_degrees, marker ='o', markerfacecolor='g', markeredgecolor='g')
    #ax.scatter(hd_times, hd_degrees)

    if unit_ix != None:
        spikes = nwbfile.units.get_unit_spike_times(unit_ix) / 1e3
        ax.plot(spikes, len(spikes) * [365], 'rv')

    return fig, ax
