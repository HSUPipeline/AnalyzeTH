
# -- IMPORTS -- 
# General
import os
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import sem, zscore
import itertools

# NWB
from pynwb import NWBHDF5IO

# Spike Tools
from spiketools.spatial.occupancy import compute_spatial_bin_edges, compute_spatial_bin_assignment
from spiketools.spatial.information import _compute_spatial_information
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.stats.permutations import zscore_to_surrogates, compute_empirical_pvalue
from spiketools.plts.space import plot_heatmap
from spiketools.plts.stats import plot_surrogates
from spiketools.plts.utils import make_axes
from spiketools.plts.data import plot_bar
from spiketools.utils import restrict_range
from spiketools.utils.data import get_value_by_time, get_value_by_time_range

# -- LOCAL -- 
# General
import analyzeth
from analyzeth.analysis import get_spike_positions
from analyzeth.target import compute_serial_position_fr
from analyzeth.cmh.utils.load_nwb import load_nwb
from analyzeth.cmh.utils.nwb_info import nwb_info
from analyzeth.cmh.utils.subset_data import subset_period_data, subset_period_event_time_data
# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import analyzeth.cmh.timeCells.settings_plots as PLOTSETTINGS 
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)

from analyzeth.cmh.timeCells.timeCellPlots import _plot_time_cell, _plot_trial_time_and_movement, _plot_time_cell_FR, _plot_trial_time_and_movement_FR


# Analysis Settings
import analyzeth.cmh.timeCells.settings_TC as SETTINGS


def time_cell_single_trial (nwbfile = None,
                            bin_len = SETTINGS.TIME_BIN_LENGTH_MS,  # default 1000 ms
                            task = SETTINGS.TASK,
                            subj = SETTINGS.SUBJ,
                            session = SETTINGS.SESSION,
                            unit_ix = SETTINGS.UNIT_IX,
                            trial_ix = SETTINGS.TRIAL_IX,
                            data_folder = SETTINGS.DATA_FOLDER,
                            date = SETTINGS.DATE,
                            PLOT = False,
                            PLOT_WITH_MOVEMENT = False,
                            GETPLOT = False,
                            SAVEFIG = False,
                            VERBOSE = False
                           ):

    """ Analyze Time Cell for Single Trial

    This will retun COUNT of spikes in given bin. See below for Firing Rate (Hz) per bin
    
    PARAMETERS
    ----------
    nwbfile: '.nwb' 
        NWB file for single session, if not provided it will load based on SETTINGS
    
    ...
    All parameters can be provided, if not provided they will be taken from settings


    RETURNS
    -------
    res : dict
        results dictionary containing info and figures of interest
    
    """

    # -- Load -- 
    if nwbfile == None:
        nwbfile = load_nwb(task, subj, session, data_folder) 
    if VERBOSE:
        nwb_info(nwbfile, unit_ix = unit_ix, trial_ix = trial_ix)
        
    # Navigation data
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    navigation_ix_start = navigation_start_times[trial_ix]
    navigation_ix_end = navigation_end_times[trial_ix]
    navigation_ix_len = navigation_ix_start - navigation_ix_end

    # Get spikes during navigation period of interest
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes_navigation_ix = restrict_range(spikes, navigation_ix_start, navigation_ix_end)
    n_spikes_navigation_ix = len(spikes_navigation_ix)
 
    # Time bins
    bin_edges = np.arange(navigation_ix_start, navigation_ix_end+bin_len, bin_len)   
    spike_bins = np.digitize(spikes_navigation_ix, bin_edges)       # Assign spikes to time bins
    spike_bin_counts = np.bincount(spike_bins)                      # Count per bin

    # Convert to trial (navigation) time
    bins_in_trial_time = bin_edges - navigation_ix_start
    spikes_in_trial_time = spikes_navigation_ix - navigation_ix_start

    if VERBOSE:
        print('\n -- TIME BIN DATA --')
        print('Total number of bins : \t\t\t {}'.format(len(bin_edges)-1))
        print('Bin length (ms): \t\t\t {}'.format(bin_len))
        print('Trial length (ms): \t\t\t {}'.format(np.round((navigation_ix_len),2))) 
        print('Number of bins with spikes: \t\t {}'.format(np.count_nonzero(spike_bin_counts)))
        print('Number of spiikes in trial: \t\t {}'.format(n_spikes_navigation_ix))
    
    # ----------------------------------------------------------------
    # -- MOVEMENT --
    
    if PLOT_WITH_MOVEMENT:
        # Extract the position data
        pos = nwbfile.acquisition['position']['xy_position']
        ptimes = pos.timestamps[:]
        positions = pos.data[:]

        # Get the chest positions & trial indices
        chest_xs, chest_ys = nwbfile.acquisition['chest_positions']['chest_positions'].data[:]
        chest_trials = nwbfile.acquisition['chest_trials']['chest_trials'].data[:]
        
        # Get the chests for the current trial
        t_mask = chest_trials == trial_ix

        # Select chest openings for the current trial
        ch_openings = nwbfile.trials['chest_opening'][trial_ix]
        t_time, t_pos = get_value_by_time_range(ptimes, positions, trial_ix_start, trial_ix_end)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        # Get spike positions within trial
        t_spike_xs, t_spike_ys = get_spike_positions(spikes_tix, t_time, t_pos)

            # @cmh - note that the total number of spike coords is one less
            # than the total numnber of spikes, reproduce with:
            # wv001, ses2, trial 10, unit 10
    
    # -----------------------------------------------------------------
    # -- PLOT --
    fig_time_cell = None
    if PLOT:
        title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', Trial ' + str(trial_ix)
        fig_time_cell, ax = plt.subplots(1,1, figsize =  [10,5])
        ax = _plot_time_cell (bin_len, spikes_in_trial_time, bins_in_trial_time,
                                spike_bin_counts, title, date, SAVEFIG) 

    fig_time_cell_movement = None                                
    if PLOT_WITH_MOVEMENT:
        title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', Trial ' + str(trial_ix)
        #fig_time_cell_movement = plt.subplots
        fig, axs = _plot_trial_time_and_movement(trial_ix,bin_len, spikes_in_trial_time, bins_in_trial_time,
                                                 spike_bin_counts, t_pos, t_spike_xs,t_spike_ys, ch_times,
                                                 chest_xs,chest_ys,t_mask, title, date, SAVEFIG)

        
    # -----------------------------------------------------------------
    # -- RETURN -- 
    res = {
        'spike_bin_counts'      : spike_bin_counts,
        'spikes_in_trial_time'  : spikes_in_trial_time,
        'bins_in_trial_time'    : bins_in_trial_time,

        'fig_time_cell'         : fig_time_cell,
        'fig_time_cell_movement': fig_time_cell_movement
    }

    return res
    


def time_cell_all_trials (
                        nwbfile = None,
                        bin_len = SETTINGS.TIME_BIN_LENGTH_MS,  # default 1000 ms
                        task = SETTINGS.TASK,
                        subj = SETTINGS.SUBJ,
                        session = SETTINGS.SESSION,
                        unit_ix = SETTINGS.UNIT_IX,
                        trial_ix = SETTINGS.TRIAL_IX,
                        data_folder = SETTINGS.DATA_FOLDER,
                        date = SETTINGS.DATE,
                        PLOT = False,
                        PLOT_WITH_MOVEMENT = False,
                        GETPLOT = False,
                        SAVEFIG = False,
                        VERBOSE = False
                        ):
    
    """ Analyze Time Cell across All Trials in session

    This will retun SUM of spikes in given bin. See below for Firing Rate (Hz) per bin
    
    PARAMETERS
    ----------
    nwbfile: '.nwb' 
        NWB file for single session, if not provided it will load based on SETTINGS
    
    ...
    All parameters can be provided, if not provided they will be taken from settings


    RETURNS
    -------
    if not GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time
    
    if GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, fig, axs
    
    """
    
    
    # -- LOAD NWB --
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subj, session, data_folder) 
    
    if VERBOSE:
        print('\n -- SUBJECT DATA --')
        print('West Virginia University')
        print('Subject {}'.format(subj))
        
    # -- SESSION DATA --                  
    # Get the number of trials & units
    n_trials = len(nwbfile.trials)
    n_units = len(nwbfile.units)

    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start
    
    if VERBOSE:
        print('\n -- SESSION DATA --')
        print('Chosen Session: \t\t\t {}'.format(session))
        print('Session Start Time: \t\t\t {}'.format(session_start))
        print('Session End Time: \t\t\t {}'.format(session_end))
        print('Total Session Length (ms): \t\t {}'.format(np.round(session_len,2))) 
        print('Total Session Length (sec): \t\t {}'.format(np.round((session_len)/1000,2))) 
        print('Total Session Length (min): \t\t {}'.format(np.round((session_len)/60000,2))) 
        print('Number of trials: \t\t\t {}'.format(n_trials))
        print('Number of units: \t\t\t {}'.format(n_units))
    
    
    # -- SPIKE DATA --
    # Get all spikes from unit across session 
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes = spikes / 1000  # @cmh fix this in converter 
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    if VERBOSE:
        print('\n -- UNIT DATA --')
        print('Chosen example unit: \t\t\t {}'.format(unit_ix))
        print('Total number of spikes: \t\t {}'.format(n_spikes_tot))
        print('Number of spikes within session: \t {}'.format(n_spikes_ses))
        
    # --------------------------------------------------------------
    # -- RUN ACROSS ALL TRIALS --
    
    unit_spikes_trial_time = np.array([])
    unit_bins_trial_time = []
    unit_bin_counts = []
    total_spike_count = 0
    
    for trial_ix in range(n_trials):
        spike_bin_counts, spikes_in_trial_time, bins_in_trial_time = time_cell_single_trial (
                                                                            nwbfile = nwbfile,
                                                                            bin_len = bin_len,  #ms
                                                                            task = task,
                                                                            subj = subj,
                                                                            session = session,
                                                                            unit_ix = unit_ix,
                                                                            trial_ix = trial_ix,
                                                                            data_folder = data_folder,
                                                                            date = date,
                                                                            PLOT = PLOT,
                                                                            PLOT_WITH_MOVEMENT = PLOT_WITH_MOVEMENT,
                                                                            GETPLOT = GETPLOT,
                                                                            SAVEFIG = SAVEFIG,
                                                                            VERBOSE = False)
        
        # Add spike data for trial
        unit_spikes_trial_time = np.append(unit_spikes_trial_time, spikes_in_trial_time)
        
        # Take longest trial length        
        if len(bins_in_trial_time) > len(unit_bins_trial_time):
            unit_bins_trial_time = bins_in_trial_time
        
        # Collect bin counts
        unit_bin_counts += [spike_bin_counts]
        
        # Collect total spike count
        total_spike_count += sum(spike_bin_counts)
    
    # Sum bin counts
    unit_bin_counts = [sum(x) for x in itertools.zip_longest(*unit_bin_counts, fillvalue=0)]
    
    if VERBOSE:
        print ('Number of spikes within trials: \t {} \n'.format(total_spike_count))
    
    # -- PLOT -- 
    if PLOT:
        title = title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', All Trials' 
        fig, ax = _plot_time_cell(
                                bin_len = bin_len, 
                                spikes_in_trial_time = unit_spikes_trial_time,
                                bins_in_trial_time = unit_bins_trial_time,
                                spike_bin_counts = unit_bin_counts,
                                title = title,
                                date = date,
                                SAVEFIG = SAVEFIG)
        
    
    # -- RETURN -- 
    if GETPLOT:
        return unit_spikes_trial_time, unit_bins_trial_time, unit_bin_counts, fig, ax

    if not GETPLOT:
        return unit_spikes_trial_time, unit_bins_trial_time, unit_bin_counts
        
        

# ----------------- FIRING RATE ANALYSIS ----------------------------------

# Essentially the same as above but with modifications to look at firing rate
# rather than summed counts. These functions call separate helper plotting funcitons
# to visualize firing rate





def time_cell_single_trial_FR (
                            nwbfile = None,
                            bin_len = SETTINGS.TIME_BIN_LENGTH_MS,  # default 1000 ms
                            task = SETTINGS.TASK,
                            subj = SETTINGS.SUBJ,
                            session = SETTINGS.SESSION,
                            unit_ix = SETTINGS.UNIT_IX,
                            trial_ix = SETTINGS.TRIAL_IX,
                            data_folder = SETTINGS.DATA_FOLDER,
                            date = SETTINGS.DATE,
                            PLOT = False,
                            PLOT_WITH_MOVEMENT = False,
                            GETPLOT = False,
                            SAVEFIG = False,
                            VERBOSE = False
                           ):

    """ Analyze Time Cell for Single Trial by FIRING RATE

    This will retun FR of spikes in given bin. See below for Firing Rate (Hz) per bin

    This may be modified to se a moving window for measuring FR, rather than set time bins

    
    PARAMETERS
    ----------
    nwbfile: '.nwb' 
        NWB file for single session, if not provided it will load based on SETTINGS
    
    ...
    All parameters can be provided, if not provided they will be taken from settings


    RETURNS
    -------
    if not GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time
    
    if GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, fig, axs
    
    """
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subj, session, data_folder) 

     # -- SESSION DATA --                  
    # Get the number of trials & units
    n_trials = len(nwbfile.trials)
    n_units = len(nwbfile.units)

    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start
    
    if VERBOSE:
        print('\n -- SESSION DATA --')
        print('Chosen Session: \t\t\t {}'.format(session))
        print('Session Start Time: \t\t\t {}'.format(session_start))
        print('Session End Time: \t\t\t {}'.format(session_end))
        print('Total Session Length (ms): \t\t {}'.format(np.round(session_len,2))) 
        print('Total Session Length (sec): \t\t {}'.format(np.round((session_len)/1000,2))) 
        print('Total Session Length (min): \t\t {}'.format(np.round((session_len)/60000,2))) 
        print('Number of trials: \t\t\t {}'.format(n_trials))
        print('Number of units: \t\t\t {}'.format(n_units))

    
    # -- TRIAL DATA --
    # Extract behavioural markers of interest
    trial_starts = nwbfile.trials['start_time'].data[:]
    chest_openings = nwbfile.trials.to_dataframe()['chest_opening'].values  #chest_openings = nwbfile.trials.chest_opening.data[:]

    # Trial start and end times
    trial_ix_start = trial_starts[trial_ix]
    trial_ix_end = chest_openings[trial_ix][-1]  # @cmh may want to modify this will see 
    trial_ix_len = trial_ix_end - trial_ix_start
    
    if VERBOSE:
        print('\n -- TRIAL DATA --')
        print('Chosen Trial: \t\t\t\t {}'.format(trial_ix))
        print('Trial Start Time: \t\t\t {}'.format(trial_ix_start))
        print('Trial End Time: \t\t\t {}'.format(trial_ix_end))
        print('Total Trial Length (ms): \t\t {}'.format(np.round(trial_ix_len,2))) 
        print('Total Trial Length (sec): \t\t {}'.format(np.round((trial_ix_len)/1000,2))) 
        print('Total Trial Length (min): \t\t {}'.format(np.round((trial_ix_len)/60000,2))) 


    # -- SPIKE DATA --
    # Get all spikes from unit across session 
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes = spikes / 1000  # @cmh fix this in converter 
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    # Extract spikes in trial_ix
    spikes_tix = restrict_range(spikes, trial_ix_start, trial_ix_end)
    n_spikes_tix = len(spikes_tix)

    if VERBOSE:
        print('\n -- UNIT DATA --')
        print('Chosen example unit: \t\t\t {}'.format(unit_ix))
        print('Total number of spikes: \t\t {}'.format(n_spikes_tot))
        print('Number of spikes within session: \t {}'.format(n_spikes_ses))
        print('Number of spikes within Trial {}: \t {}'.format(trial_ix, n_spikes_tix)) 
        
        
    # -- POSITION DATA -- 
    # Extract the position data
    pos = nwbfile.acquisition['position']['xy_position']
    ptimes = pos.timestamps[:]
    positions = pos.data[:]

    # Get the chest positions & trial indices
    chest_xs, chest_ys = nwbfile.acquisition['chest_positions']['chest_positions'].data[:]
    chest_trials = nwbfile.acquisition['chest_trials']['chest_trials'].data[:]
    
    if VERBOSE:
        print ('\n -- POSITION DATA --')
        print ('Extracted')
    
    
    # -----------------------------------------------------------------
    # -- TIME BIN --
    
    # Set bin edges
    bin_edges = np.arange(trial_ix_start, trial_ix_end+bin_len, bin_len)

    # Assign spikes to time bins
    spike_bins = np.digitize(spikes_tix, bin_edges)

    # Count per bin
    spike_bin_counts = np.bincount(spike_bins)

    # Convert to trial time
    bins_in_trial_time = bin_edges - trial_ix_start
    spikes_in_trial_time = spikes_tix - trial_ix_start

    if VERBOSE:
        print('\n -- TIME BIN DATA --')
        print('Total number of bins : \t\t\t {}'.format(len(bin_edges)-1))
        print('Bin length (ms): \t\t\t {}'.format(bin_len))
        print('Trial length (ms): \t\t\t {}'.format(np.round((trial_ix_len),2))) 
        print('Number of bins with spikes: \t\t {}'.format(np.count_nonzero(spike_bin_counts)))
        print('Number of spiikes in trial: \t\t {}'.format(n_spikes_tix))
        print()
    
    
    # ----------------------------------------------------------------
    # -- (NORMALIZED?) FIRING RATE --
    
        # @cmh not sure if normalized is the right way to go, the subj
        # would not know the length of each trail at the start...
        
    # Get firing rate in Hz per bin
    firing_rates_per_bin = spike_bin_counts / (bin_len/1000)
    
#     print('SPIKE BIN COUNTS')
#     print(spike_bin_counts)
#     print()
    
#     print('FIRING RATES')
#     print(firing_rates_per_bin)
    
    
    
    
    # ----------------------------------------------------------------
    # -- MOVEMENT --
    
    if PLOT_WITH_MOVEMENT:
        # Get the chests for the current trial
        t_mask = chest_trials == trial_ix

        # Select chest openings for the current trial
        ch_openings = nwbfile.trials['chest_opening'][trial_ix]
        t_time, t_pos = get_value_by_time_range(ptimes, positions, trial_ix_start, trial_ix_end)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        # Get spike positions within trial
        t_spike_xs, t_spike_ys = get_spike_positions(spikes_tix, t_time, t_pos)

            # @cmh - note that the total number of spike coords is one less
            # than the total numnber of spikes, reproduce with:
            # wv001, ses2, trial 10, unit 10
    
    
    
    # -----------------------------------------------------------------
    # -- PLOT --
    if PLOT:
        title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', Trial ' + str(trial_ix)
        fig, axs = _plot_time_cell_FR (bin_len, spikes_in_trial_time, bins_in_trial_time,
                                    spike_bin_counts, firing_rates_per_bin, title, date, SAVEFIG) 
                                    
    if PLOT_WITH_MOVEMENT:
        title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', Trial ' + str(trial_ix)
        fig, axs = _plot_trial_time_and_movement_FR (trial_ix,bin_len, spikes_in_trial_time, bins_in_trial_time,
                                                 spike_bin_counts, firing_rates_per_bin, t_pos, t_spike_xs,t_spike_ys, ch_times,
                                                 chest_xs,chest_ys,t_mask, title, date, SAVEFIG)

        
    # -----------------------------------------------------------------
    # -- RETURN -- 
        
    if not GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, firing_rates_per_bin
    
    if GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, firing_rates_per_bin, fig, axs
    

    
def time_cell_all_trials_FR (
                            nwbfile = None,
                            bin_len = SETTINGS.TIME_BIN_LENGTH_MS,  # default 1000 ms
                            task = SETTINGS.TASK,
                            subj = SETTINGS.SUBJ,
                            session = SETTINGS.SESSION,
                            unit_ix = SETTINGS.UNIT_IX,
                            trial_ix = SETTINGS.TRIAL_IX,
                            data_folder = SETTINGS.DATA_FOLDER,
                            date = SETTINGS.DATE,
                            PLOT = False,
                            PLOT_WITH_MOVEMENT = False,
                            GETPLOT = False,
                            SAVEFIG = False,
                            VERBOSE = False
                            ):

    """ Analyze Time Cell FIRING RATE across All Trials in session

    This will retun SUM of spikes in given bin. See below for Firing Rate (Hz) per bin
    
    PARAMETERS
    ----------
    nwbfile: '.nwb' 
        NWB file for single session, if not provided it will load based on SETTINGS
    
    ...
    All parameters can be provided, if not provided they will be taken from settings


    RETURNS
    -------
    if not GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time
    
    if GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, fig, axs
    
    """
    
    
    
    # -- LOAD NWB --
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subj, session, data_folder) 
    
    if VERBOSE:
        print('\n -- SUBJECT DATA --')
        print('West Virginia University')
        print('Subject {}'.format(subj))
        
    # -- SESSION DATA --                  
    # Get the number of trials & units
    n_trials = len(nwbfile.trials)
    n_units = len(nwbfile.units)

    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start
    
    if VERBOSE:
        print('\n -- SESSION DATA --')
        print('Chosen Session: \t\t\t {}'.format(session))
        print('Session Start Time: \t\t\t {}'.format(session_start))
        print('Session End Time: \t\t\t {}'.format(session_end))
        print('Total Session Length (ms): \t\t {}'.format(np.round(session_len,2))) 
        print('Total Session Length (sec): \t\t {}'.format(np.round((session_len)/1000,2))) 
        print('Total Session Length (min): \t\t {}'.format(np.round((session_len)/60000,2))) 
        print('Number of trials: \t\t\t {}'.format(n_trials))
        print('Number of units: \t\t\t {}'.format(n_units))
    
    
    # -- SPIKE DATA --
    # Get all spikes from unit across session 
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spikes = spikes / 1000  # @cmh fix this in converter 
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    if VERBOSE:
        print('\n -- UNIT DATA --')
        print('Chosen example unit: \t\t\t {}'.format(unit_ix))
        print('Total number of spikes: \t\t {}'.format(n_spikes_tot))
        print('Number of spikes within session: \t {}'.format(n_spikes_ses))
        
    # --------------------------------------------------------------
    # -- RUN ACROSS ALL TRIALS --
    
    unit_spikes_trial_time = np.array([])
    unit_bins_trial_time = []
    unit_bin_counts = []
    unit_bin_FR = []
    total_spike_count = 0
    
    for trial_ix in range(n_trials):
        spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, firing_rates_per_bin = \
            time_cell_single_trial_FR (
                                    nwbfile = nwbfile,
                                    bin_len = bin_len,  #ms
                                    task = task,
                                    subj = subj,
                                    session = session,
                                    unit_ix = unit_ix,
                                    trial_ix = trial_ix,
                                    data_folder = data_folder,
                                    date = date,
                                    PLOT = False,
                                    PLOT_WITH_MOVEMENT = PLOT_WITH_MOVEMENT,
                                    GETPLOT = GETPLOT,
                                    SAVEFIG = SAVEFIG,
                                    VERBOSE = False)
        
        # Add spike data for trial
        unit_spikes_trial_time = np.append(unit_spikes_trial_time, spikes_in_trial_time)
        
        # Take longest trial length        
        if len(bins_in_trial_time) > len(unit_bins_trial_time):
            unit_bins_trial_time = bins_in_trial_time
        
        # Collect bin counts
        unit_bin_counts += [spike_bin_counts]
        
        # Collect FR
        unit_bin_FR += [firing_rates_per_bin]
        
        # Collect total spike count
        total_spike_count += sum(spike_bin_counts)
    
    # Find Mean of Non-zero bin counts
    firing_rate_bin_means = [np.nanmean(x) for x in itertools.zip_longest(*unit_bin_FR, fillvalue=np.nan)]
    
    if VERBOSE:
        print ('Number of spikes within trials: \t {} \n'.format(total_spike_count))
    
    # -- PLOT -- 
    if PLOT:
        title = title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', All Trials' 
        fig, ax =_plot_time_cell_FR (
                     bin_len = bin_len, 
                     spikes_in_trial_time = unit_spikes_trial_time,
                     bins_in_trial_time = unit_bins_trial_time,
                     spike_bin_counts = unit_bin_counts,
                     firing_rates_per_bin = firing_rate_bin_means,
                     title = title,
                     date = date,
                     SAVEFIG = SAVEFIG)
    

    # -- RETURN --
    if GETPLOT:
        return unit_spikes_trial_time, unit_bins_trial_time, unit_bin_counts, firing_rate_bin_means, fig, ax
    
    if not GETPLOT:
        return unit_spikes_trial_time, unit_bins_trial_time, unit_bin_counts, firing_rate_bin_means
              


