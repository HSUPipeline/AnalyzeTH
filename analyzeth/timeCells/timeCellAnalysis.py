
# -- IMPORTS -- 
# General
import os
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import sem, zscore
import matplotlib.pyplot as plt
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
from analyzeth.timeCells.load_nwb import load_nwb

# Plots
import seaborn as sns
import analyzeth.timeCells.settings_plots as PLOTSETTINGS 
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)

# Analysis Settings
import analyzeth.timeCells.settings_TC as SETTINGS


def time_cell_single_trial (nwbfile = None,
                            bin_len = SETTINGS.TIME_BIN_LENGTH_MS,  # default 1000 ms
                            task = SETTINGS.TASK,
                            subj = SETTINGS.SUBJ,
                            session = SETTINGS.SESSION,
                            unit_ix = SETTINGS.UNIT_IX,
                            trial_ix = SETTINGS.TRIAL_IX,
                            data_folder = SETTINGS.DATA_FOLDER,
                            date = '',
                            PLOT = False,
                            PLOT_WITH_MOVEMENT = False,
                            GETPLOT = False,
                            SAVEFIG = False,
                            VERBOSE = False
                           ):

    """ Analyze Time Cell for Single Trial"""
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
        fig, axs = _plot_time_cell (bin_len, spikes_in_trial_time, bins_in_trial_time,
                                    spike_bin_counts, title, date, SAVEFIG) 
                                    
    if PLOT_WITH_MOVEMENT:
        title = task + ', ' + subj + ', Session ' + str(session) + ', Unit ' + str(unit_ix) + ', Trial ' + str(trial_ix)
        fig, axs = _plot_trial_time_and_movement(trial_ix,bin_len, spikes_in_trial_time, bins_in_trial_time,
                                                 spike_bin_counts, t_pos, t_spike_xs,t_spike_ys, ch_times,
                                                 chest_xs,chest_ys,t_mask, title, date, SAVEFIG)

        
    # -----------------------------------------------------------------
    # -- RETURN -- 
        
    if not GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time
    
    if GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, fig, axs
    

    
        

