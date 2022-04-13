# -- IMPORTS --
# General
import os
import numpy as np
from pingouin import convert_angles, circ_rayleigh

# NWB
from pynwb import NWBHDF5IO

# Spike Tools
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.stats.permutations import zscore_to_surrogates, compute_empirical_pvalue
from spiketools.utils import restrict_range
from spiketools.plts.space import plot_positions
from spiketools.spatial.occupancy import (compute_occupancy, compute_spatial_bin_edges,
                                          compute_spatial_bin_assignment)

# Local
from analyzeth.cmh.load_nwb import load_nwb
from analyzeth.analysis import get_spike_positions, compute_bin_firing
from analyzeth.plts import plot_polar_hist

# Plots
import matplotlib.pyplot as plt 
import seaborn as sns 
import analyzeth.cmh.settings_plots as PLOTSETTINGS
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# Analysis settings
import analyzeth.cmh.settings_analysis as SETTINGS


def plot_positions_TH(
        nwbfile = None,
        bins = [7, 21],
        task = SETTINGS.TASK,
        subject = SETTINGS.SUBJECT,
        session = SETTINGS.SESSION,
        unit_ix = SETTINGS.UNIT_IX,
        trial_ix = SETTINGS.TRIAL_IX,
        data_folder = SETTINGS.DATA_FOLDER,
        shuffle_approach = SETTINGS.SHUFFLE_APPROACH,
        shuffle_n_surrogates = SETTINGS.N_SURROGATES,
        SHUFFLE = False,
        PLOT = False,
        SAVEFIG = False,
        VERBOSE = False
    ):

    """ Plot position map for session from NWB file """

    # -- LOAD & EXTRACT DATA --
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder)

    # Session data 
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    
    # Get spikes & restrict to session
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes = restrict_range(spikes, session_start, session_end)
    
    # Position data
    pos = nwbfile.acquisition['position']['xy_position']
    position_times = pos.timestamps[:]
    positions = pos.data[:]
    
    # Position for each spike
    spike_xs, spike_ys = get_spike_positions(spikes, position_times, positions)
    spike_positions = np.array([spike_xs, spike_ys])

    # Check binning
    x_bin_edges, y_bin_edges = compute_spatial_bin_edges(positions, bins)

    # -- PLOT --
    fig, ax = plt.subplots(figsize = [5,7])
    plot_positions(positions, spike_positions, x_bins=x_bin_edges, y_bins = y_bin_edges, ax = ax)

    return fig, ax

def plot_navigation_positions_TH(
        nwbfile = None,
        bins = [7, 21],
        task = SETTINGS.TASK,
        subject = SETTINGS.SUBJECT,
        session = SETTINGS.SESSION,
        unit_ix = SETTINGS.UNIT_IX,
        trial_ix = SETTINGS.TRIAL_IX,
        data_folder = SETTINGS.DATA_FOLDER,
        shuffle_approach = SETTINGS.SHUFFLE_APPROACH,
        shuffle_n_surrogates = SETTINGS.N_SURROGATES,
        SHUFFLE = False,
        PLOT = False,
        SAVEFIG = False,
        VERBOSE = False
    ):

    """ Plot position map for session from NWB file """

    # -- LOAD & EXTRACT DATA --
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder)

    # Session data 
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    
    # Get spikes & restrict to session
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes = restrict_range(spikes, session_start, session_end)
    
    # Position data
    pos = nwbfile.acquisition['position']['xy_position']
    position_times = pos.timestamps[:]
    positions = pos.data[:]
    print('Num positions: \t {}'.format(len(position_times)))

    # Check binning
    x_bin_edges, y_bin_edges = compute_spatial_bin_edges(positions, bins)


    # -- RESTRICT TO NAVIGATION PERIODS -- 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    
    # Get spikes during navigation period 
    spikes_navigation = np.array([])
    for ix in range(len(navigation_start_times)):
        spikes_navigation = np.append(
                            spikes_navigation,
                            spikes[(spikes > navigation_start_times[ix]) \
                                 & (spikes < navigation_end_times[ix])],   # <= ?
                            axis = 0)

    # Position for each spike
    spike_xs, spike_ys = get_spike_positions(spikes_navigation, position_times, positions)
    spike_positions = np.array([spike_xs, spike_ys])
    
    # -- PLOT --
    fig, ax = plt.subplots(figsize = [5,7])
    
    # Get and positions during each navigation period individually
    for ix in range(len(navigation_start_times)):
        navigation_ixs = np.where((navigation_start_times[ix] < position_times) \
                                & (position_times < navigation_end_times[ix]))
        navigation_positions = positions[:, navigation_ixs[0]] #np.where returns tuple even for 1d

        plot_positions(navigation_positions, 
                    spike_positions, 
                    x_bins=x_bin_edges, 
                    y_bins = y_bin_edges, 
                    ax = ax,
                    color = 'b')
        
    return fig, ax

def plot_navigation_positions_TH_OLD(
        nwbfile = None,
        bins = [7, 21],
        task = SETTINGS.TASK,
        subject = SETTINGS.SUBJECT,
        session = SETTINGS.SESSION,
        unit_ix = SETTINGS.UNIT_IX,
        trial_ix = SETTINGS.TRIAL_IX,
        data_folder = SETTINGS.DATA_FOLDER,
        shuffle_approach = SETTINGS.SHUFFLE_APPROACH,
        shuffle_n_surrogates = SETTINGS.N_SURROGATES,
        SHUFFLE = False,
        PLOT = False,
        SAVEFIG = False,
        VERBOSE = False
    ):

    """ Plot position map for session from NWB file 
    
    This will result in lines from end to start of each trial, which are not actual
    navigation points, updated above
    
    """

    # -- LOAD & EXTRACT DATA --
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder)

    # Session data 
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    
    # Get spikes & restrict to session
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes = restrict_range(spikes, session_start, session_end)
    
    # Position data
    pos = nwbfile.acquisition['position']['xy_position']
    position_times = pos.timestamps[:]
    positions = pos.data[:]
    print('Num positions: \t {}'.format(len(position_times)))

    # Check binning
    x_bin_edges, y_bin_edges = compute_spatial_bin_edges(positions, bins)


    # -- RESTRICT TO NAVIGATION PERIODS -- 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    
    # Get spikes during navigation period 
    spikes_navigation = np.array([])
    for ix in range(len(navigation_start_times)):
        spikes_navigation = np.append(
                            spikes_navigation,
                            spikes[(spikes > navigation_start_times[ix]) \
                                 & (spikes < navigation_end_times[ix])],   # <= ?
                            axis = 0)
    


    # Position for each spike
    spike_xs, spike_ys = get_spike_positions(spikes_navigation, position_times, positions)
    spike_positions = np.array([spike_xs, spike_ys])
    
    # Get positions during navigation period
    navigation_ixs = np.array([], dtype=int)
    for ix in range(len(navigation_start_times)):
        navigation_ixs_ix = np.where((navigation_start_times[ix] < position_times) \
                                        & (position_times < navigation_end_times[ix]))
        navigation_ixs = np.append(navigation_ixs, navigation_ixs_ix)
    positions_navigation = positions[:, navigation_ixs]

    print('Num navigation positions: \t {}'.format(len(navigation_ixs)))

    # -- PLOT --
    fig, ax = plt.subplots(figsize = [5,7])
    plot_positions(positions_navigation, spike_positions, x_bins=x_bin_edges, y_bins = y_bin_edges, ax = ax)

    return fig, ax







def plot_position_raster(
        nwbfile = None, 
        unit_ix = SETTINGS.UNIT_IX,
        ax = None,
        highlight = 'trial'
        ):
    
    # -- LOAD -- 
    if nwbfile == None:
        nwbfile = load_nwb(task, subj, session, data_folder) 

    # -- SESSION DATA -- 
    #session_start = nwbfile.trials['start_time'][0]
    #session_end = nwbfile.trials['stop_time'][-1]
    

    # -- TRIAL DATA -- 
    trial_starts = (nwbfile.trials['start_time'].data[:])/1e3       #convert to trial time in s
    trial_ends = (nwbfile.trials['stop_time'].data[:])/1e3          #convert to trial time in s

    # -- NAVIGATION DATA --
    navigation_start_times = nwbfile.trials['navigation_start'][:]/1e3
    navigation_end_times = nwbfile.trials['navigation_end'][:]/1e3
    
    # -- SPIKE DATA --
    #spikes = nwbfile.units.get_unit_spike_times(unit_ix)                            #get spikes in ms
    #spikes = restrict_range(spikes, session_start, session_end)
    #spikes = (spikes)/1e3                                          #convert to trial time in s  

    # Position data
    pos = nwbfile.acquisition['position']['xy_position']
    position_times = pos.timestamps[:] / 1e3
    #positions = pos.data[:]

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
        print('navi')
        print(navigation_start_times)
        print(navigation_end_times)

    # Add events
    ax.eventplot(position_times)
    
    # Format
    #ax.set_yticks([0,1])
    #ax.set_yticklabels(['Spike Times', 'HD Times'])
    ax.set_xlabel('Time (s)')
    ax.set_title('Position times')

    # Show plot
    plt.show()

    return fig, ax