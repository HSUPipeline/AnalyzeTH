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
from analyzeth.cmh.load_nwb import load_nwb
from analyzeth.analysis import get_spike_heading, bin_circular
from analyzeth.plts import plot_polar_hist

# Plots
import matplotlib.pyplot as plt 
import seaborn as sns 
import analyzeth.cmh.settings_plots as PLOTSETTINGS
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# Analysis settings
import analyzeth.cmh.settings_analysis as SETTINGS


def head_direction_cell_session(
    nwbfile = None,
    task = SETTINGS.TASK,
    subj = SETTINGS.SUBJECT,
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
    GETPLOT = False,
    SAVEFIG = False,
    VERBOSE = False
    ):

    """ Analyze Head Direction Cell for Single Trial

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
    if not GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time
    
    if GETPLOT:
        return spike_bin_counts, spikes_in_trial_time, bins_in_trial_time, fig, axs
    
    """

    # -------------------------------------------------------------------------------
    # -- LOAD & EXTRACT DATA --

    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subj, session, data_folder) 

    if VERBOSE:
        print('\n -- SUBJECT DATA --')
        print(experiment_label)
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
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    # Get spikes during navigation period 
    #   - note: this is listed as 'encoding period' in nwb
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    
    if len(navigation_start_times) != len(navigation_end_times):
        # I actually think I caught this in the parser but JIC
        msg = 'Different lengths of Navigation Start and End Times. The subject likely \
               stopped in the middle of a trial. Remove the last trial and try again'
        raise ValueError(msg)

    spikes_navigation = np.array([])
    for ix in range(len(navigation_start_times)):
        spikes_navigation = np.append(
                            spikes_navigation,
                            spikes[(spikes > navigation_start_times[ix]) \
                                 & (spikes < navigation_end_times[ix])],   # <= ?
                            axis = 0)
    n_spikes_navigation = len(spikes_navigation)

    if VERBOSE:
        print('\n -- UNIT DATA --')
        print('Chosen example unit: \t\t\t {}'.format(unit_ix))
        print('Total number of spikes: \t\t {}'.format(n_spikes_tot))
        print('Number of spikes within session: \t {}'.format(n_spikes_ses))  
        print('Number of spikes within navigation: \t {}'.format(n_spikes_navigation))  

    # -- HEAD DIRECTION DATA --
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    # hd_degrees = hd_degrees + -np.min(hd_degrees[hd_degrees<0]) 
        # The shift here is because there is a shift somehwere causing 
        # hd_degree vaulues to range from ex. [-0.5, 369.5].
        # By shifting all values up by the -minimum (the offset), the degree 
        # values maintain consistent relationship and fall in the correct 
        # range [0, 360].

    if VERBOSE:
        print('\n -- HEAD DIRECTION DATA --')
        print('Session | length of HD timestamps: \t {}'. format(len(hd_times)))
        print('Session | length of HD degree array \t {}'.format(len(hd_degrees)))
        print('Head direction degree range: \t\t [{}, {}]'.format(min(hd_degrees), max(hd_degrees)))


    # -------------------------------------------------------------------------------
    # -- PLOT RASTER --
    raster_fig = None
    if PLOT_RASTER:
        raster_fig = plt.figure()
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
                #print('{:1.2f}% of cells ({} / {})'.format(len(spike_hds) / len(spikes) * 100, len(spike_hds), len(spikes)))
                # @cmh this is not showing the right thing, seems to be showing the number of 
                # spikes that have an associated HD of all the spikes. 
                # 
                # Wwe need to find the interval at which HD is determined and find all spikes
                # within each interval that are associated with that time interval of HD...

    spike_hd_fig = None
    if PLOT:
        spike_hd_fig = plt.figure()
        hd_spike_ax = plot_polar_hist(spike_hds)
        plt.title('Spike HDs | Unit {}'.format(unit_ix))
        plt.show()


    # -------------------------------------------------------------------------------
    # -- STATISTICAL SHUFFLING --
    shuffled_spike_hds = []
    shuffled_spike_bin_counts = []
    shuffled_z_vals = []
    shuffled_p_vals = []
    
    if SHUFFLE:
        shuffled_spike_times =  shuffle_spikes(
                                            spikes,
                                            approach = shuffle_approach,
                                            n_shuffles = shuffle_n_surrogates
                                            )

        for s_spikes in shuffled_spike_times:
            shuffled_spikes_navigation = np.array([])
            for ix in range(len(navigation_start_times)):
                shuffled_spikes_navigation = np.append(
                                                    shuffled_spikes_navigation,
                                                    s_spikes[(s_spikes > navigation_start_times[ix]) \
                                                        & (s_spikes < navigation_end_times[ix])],   # <= ?
                                                    axis = 0
                                                    )
            
            shuffled_spike_hds_ix = get_spike_heading(shuffled_spikes_navigation, hd_times, hd_degrees)
            shuffled_spike_hds.append(shuffled_spike_hds_ix)
            bin_edges, counts = bin_circular(shuffled_spike_hds_ix)
            
            
            shuffled_spike_bin_counts.append(counts)
            shuff_z_val, shuff_p_val = circ_rayleigh(convert_angles(shuffled_spike_hds_ix))
            shuffled_z_vals.append(shuff_z_val)
            shuffled_p_vals.append(shuff_p_val)
        
        emperical_p_val = compute_empirical_pvalue(spike_hd_z_val, shuffled_z_vals)

        # Mean shuffled
        mean_shuffled_hd_counts =  np.mean(shuffled_spike_bin_counts, axis = 0)
        surrogates_hd_fig = None
        overlay_surrogates_fig = None 
        if PLOT:
            # Surrogates alone

            print('---------')
            print(len(mean_shuffled_hd_counts))
            print(len(shuffled_spike_bin_counts), len(shuffled_spike_bin_counts[0]))
            print(len(shuffled_spike_hds), len(shuffled_spike_hds[0]))

            print('---------')
            
            
            surrogates_hd_fig = plt.figure()
            ax = plt.subplot(111, polar=True)
            ax.bar(bin_edges[:-1], mean_shuffled_hd_counts)
            plt.title('Surrogates HD')
            plt.show()

            # Surrogates over original
            overlay_surrogates_fig = plt.figure() 
            ax = plot_polar_hist(spike_hds)

            print(len(mean_shuffled_hd_counts))
            
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
        nwbfile = load_nwb(task, subj, session, data_folder) 

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

