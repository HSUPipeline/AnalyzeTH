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
from spiketools.plts.trials import plot_rasters

# Local
from analyzeth.cmh.utils.load_nwb import load_nwb
from analyzeth.cmh.utils.nwb_info import nwb_info
from analyzeth.cmh.utils.subset_data import subset_period_event_time_data, subset_period_data
from analyzeth.analysis import get_spike_heading, bin_circular
from analyzeth.plts import plot_polar_hist
from analyzeth.cmh.headDirection.headDirectionPlots import *

# Plots
import matplotlib.pyplot as plt 
import seaborn as sns 
import analyzeth.cmh.settings_plots as PLOTSETTINGS
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# Analysis settings
import analyzeth.cmh.settings_analysis as SETTINGS


def headDirection_cell(
    nwbfile = None,
    task = SETTINGS.TASK,
    subject = SETTINGS.SUBJECT,
    session = SETTINGS.SESSION,
    unit_ix = SETTINGS.UNIT_IX,
    trial_ix = SETTINGS.TRIAL_IX,
    data_folder = SETTINGS.DATA_FOLDER,
    date = SETTINGS.DATE,
    experiment_label = SETTINGS.ACQUISITION_LOCATION,
    shuffle_approach = SETTINGS.SHUFFLE_APPROACH,
    shuffle_n_surrogates = SETTINGS.N_SURROGATES,
    SHUFFLE = False,
    PLOT = False,
    PLOT_RASTER = False,
    SAVEFIG = False,
    VERBOSE = False
    ):

    """ Analyze Head Direction Cell for Single Session

    This will retun COUNT of spikes in given bin. See below for Firing Rate (Hz) per bin.

    This will only find significant UNIMODAL head direction cells using Circular Rayleigh
    
    PARAMETERS
    ----------
    nwbfile: '.nwb' 
        NWB file for single session, if not provided it will load based on SETTINGS
    
    ...
    All parameters can be provided, if not provided they will be taken from settings


    RETURNS
    -------
    res : dict
        dictionary containing data and figures of interest
    
    """

    # -------------------------------------------------------------------------------
    # -- LOAD & EXTRACT DATA --

    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder) 

    if VERBOSE:
        nwb_info(nwbfile, unit_ix = unit_ix, trial_ix = trial_ix)

    # -- SESSION DATA -- 
    # Get the number of trials & units
    n_trials = len(nwbfile.trials)
    n_units = len(nwbfile.units)

    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start

    # -- SPIKE DATA --
    # Get all spikes from unit across session 
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    # Get spikes during navigation period 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    spikes_navigation = subset_period_event_time_data(spikes, navigation_start_times, navigation_end_times)
    n_spikes_navigation = len(spikes_navigation)

    # -- HEAD DIRECTION DATA --
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]

    # -------------------------------------------------------------------------------
    # -- PLOT RASTER --
    raster_fig = None
    if PLOT_RASTER:
        #raster_fig = plt.figure()
        raster_ax = plot_hd_raster(nwbfile, unit_ix, highlight = 'nav')
    

    # -------------------------------------------------------------------------------
    # -- HEAD DIRECTION ANALYSIS -- 

    # -- HD ACROSS TRIAL --
    # Check for non-uniformity
    hd_z_val, hd_p_val = circ_rayleigh(convert_angles(hd_degrees))

    if VERBOSE:
        print('\n -- HEAD DIRECTION ANALYSIS | SESSION --')
        print('Circular Rayleigh:')
        print('\t z value: \t {}'.format(hd_z_val))
        print('\t p value: \t {}'.format(hd_p_val))

    hd_fig = None
    if PLOT:
        hd_fig = plt.figure()
        hd_ax = plot_polar_hist(hd_degrees)
        plt.title('Head Direction | Session')
        plt.show()

    # -- HD SPIKE COUNTS --
    # Get spike headings
    spike_hds = get_spike_heading(spikes_navigation , hd_times, hd_degrees)

    # Check for non-uniformity
    spike_hd_z_val, spike_hd_p_val = circ_rayleigh(convert_angles(spike_hds))

    if VERBOSE:
        print('\n -- HD SPIKE COUNTS --')
        print('Number of spikes with HD extracted: \t {}'.format(len(spike_hds)))
        print('Circular Rayleigh:')
        print('\t z value: \t {}'.format(spike_hd_z_val))
        print('\t p value: \t {}'.format(spike_hd_p_val))

    spike_hd_fig = None
    if PLOT:
        spike_hd_fig = plt.figure()
        hd_spike_ax = plot_polar_hist(spike_hds)
        plt.title('Spike HDs | Unit {}'.format(unit_ix))
        plt.show()


    # Occupancy
    

    # -------------------------------------------------------------------------------
    # -- STATISTICAL SHUFFLING -- 

    shuffled_spike_hds = []
    shuffled_spike_bin_counts = []
    shuffled_z_vals = []
    shuffled_p_vals = []
    surrogates_hd_fig = None
    overlay_surrogates_fig = None 
    emperical_p_val = None
    
    if SHUFFLE:
        # Shuffle spikes n times
        shuffled_spike_times =  shuffle_spikes(
                                            spikes,
                                            approach = shuffle_approach,
                                            n_shuffles = shuffle_n_surrogates
                                            )

        # Collect data for each shuffle
        for s_spikes in shuffled_spike_times:
            # Subset s_spikes during navigation
            shuffled_spikes_navigation = subset_period_event_time_data(s_spikes, navigation_start_times, navigation_end_times)
            shuffled_spike_hds_ix = get_spike_heading(shuffled_spikes_navigation, hd_times, hd_degrees)
            
            # Collect data
            shuffled_spike_hds.append(shuffled_spike_hds_ix)
            bin_edges, counts = bin_circular(shuffled_spike_hds_ix)
            shuffled_spike_bin_counts.append(counts)

            # Rayleigh
            shuff_z_val, shuff_p_val = circ_rayleigh(convert_angles(shuffled_spike_hds_ix))
            shuffled_z_vals.append(shuff_z_val)
            shuffled_p_vals.append(shuff_p_val)
        
        # Compare real with shuffle
        emperical_p_val = compute_empirical_pvalue(spike_hd_z_val, shuffled_z_vals)

        # Mean shuffled
        mean_shuffled_hd_counts =  np.mean(shuffled_spike_bin_counts, axis = 0)
        if PLOT:
            # Surrogates alone
            surrogates_hd_fig = plt.figure()
            ax = plt.subplot(111, polar=True)
            ax.bar(bin_edges[:-1], mean_shuffled_hd_counts)
            plt.title('Surrogates HD')
            plt.show()

            # Surrogates over original
            overlay_surrogates_fig = plt.figure() 
            ax = plot_polar_hist(spike_hds)
            ax.bar(bin_edges[:-1], mean_shuffled_hd_counts)
            
            plt.title('Overlay')
            plt.show()


    # -- RETURN --
    return_dict = {
        # General Head Direction
        'hd_degrees'            : hd_degrees,
        'hd_times'              : hd_times,
        'hd_z_val'              : hd_z_val,
        'hd_p_val'              : hd_p_val,
        'hd_fig'                : hd_fig,

        # Head Direction Spikes
        'spike_hds'             : spike_hds,
        'spike_hd_z_val'        : spike_hd_z_val,
        'spike_hd_p_val'        : spike_hd_p_val,

        # Raster
        'raster_fig'            : raster_fig,

        # Shuffle
        'shuffled_spike_hds'    : shuffled_spike_hds,
        'shuffled_z_vals'       : shuffled_z_vals, 
        'shuffled_p_vals'       : shuffled_p_vals,
        'surrogates_hd_fig'     : surrogates_hd_fig,
        'overlay_surrogates_fig': overlay_surrogates_fig,

        # Stat
        'emperical_p_val'       : emperical_p_val
    }

    return return_dict