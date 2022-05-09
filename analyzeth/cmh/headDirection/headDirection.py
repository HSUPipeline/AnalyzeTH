
from random import shuffle
from analyzeth.cmh.utils import *
from analyzeth.cmh.headDirection.headDirectionUtils import *
from analyzeth.cmh.headDirection.headDirectionPlots import *

from analyzeth.analysis import get_spike_heading, bin_circular
import pandas as pd
import matplotlib.pyplot as plt



import scipy.stats as st
import numpy as np



# spiketools
from spiketools.stats.shuffle import shuffle_spikes

def nwb_headDirection_session(nwbfile, n_surrogates = 1000, plot=False):
    """"
    run and plot for each unit in session
    """
    
    occupancy = compute_hd_occupancy(nwbfile, return_hds = False, smooth = True, windowsize = 23, binsize = 1)
    if plot:
        fig = plt.figure(figsize=[10,10])
        plot_hd(occupancy)
        plt.title('Occupancy (sec)')
        plt.xlabel('')
        plt.ylabel('')
        plt.show()

    n_units = len(nwbfile.units)
    for unit_ix in range(n_units):
        print(f'Working on unit {unit_ix}...')
        nwb_headDirection_cell(nwbfile, unit_ix, occupancy, n_surrogates, plot)
    return 


def nwb_headDirection_cell(nwbfile, unit_ix, 
        occupancy = [], 
        n_surrogates = 1000, 
        binsize = 1, 
        windowsize = 23, 
        smooth = True, 
        plot = False
        ):
    """
    Run head direction analysis for unit and compare to surrogates
    """

    # Occupancy
    if occupancy == []:
        occupancy = compute_hd_occupancy(nwbfile, binsize, windowsize, smooth)
    
    # Head direction
    hd_hist = nwb_hd_cell_hist(nwbfile, unit_ix, binsize, windowsize, smooth)
    hd_score = compute_hd_score(hd_hist)
    hd_hist_norm = normalize_hd_hist_to_occupancy(hd_hist, occupancy)
    hd_norm_score = compute_hd_score(hd_hist)

    # Shuffle
    shuffled_spikes = nwb_shuffle_spikes_bincirc_navigation(nwbfile, unit_ix, n_surrogates, shift_range = [5e3, 20e3], verbose=False)
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    surrogate_hds = get_hd_shuffled_head_directions(shuffled_spikes, hd_times, hd_degrees)
    surrogate_histograms = get_hd_surrogate_histograms(surrogate_hds, binsize, windowsize, smooth)
    surrogates_norm = normalize_hd_surrogate_histograms_to_occupancy(surrogate_histograms, occupancy)
    surrogates_ci95 = compute_ci95_from_surrogates(surrogates_norm)


    if plot:
        fig = plt.figure(figsize = [10,10])
        ax = plt.subplot(111, polar=True)
        plot_hd(hd_hist_norm, ax=ax)
        plot_surrogates_95ci(surrogates_norm, ax=ax)
        plt.title('Unit {}'.format(unit_ix))
        plt.xlabel('')
        plt.ylabel('')
        plt.show()

    res = {
        'hd_score'                  : hd_score,
        'hd_score_norm'             : hd_norm_score,
        'hd_histogram'              : hd_hist,
        'hd_histogram_norm'         : hd_hist_norm,
        'surrogate_histograms'      : surrogate_histograms,
        'surrogate_histogams_norm'  : surrogates_norm,
        'surrogates_ci95'           : surrogates_ci95
    }
    return res
        


def get_hd_spike_headings (nwbfile, unit_ix):
    """
    Get spike headings in degrees during navigation periods
    """
    # Spike data - navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    spikes = subset_period_event_time_data(nwbfile.units.get_unit_spike_times(unit_ix), navigation_start_times, navigation_end_times)

    # Head Direction data 
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]

    # Histogram
    hd_spikes = np.array(get_spike_heading(spikes, hd_times, hd_degrees))

    return hd_spikes



def nwb_hd_cell_hist(nwbfile, unit_ix = 0, binsize = 1, windowsize = 23,  smooth = True):
    """ 
    Get head direction histogram 
    """
    # Spike data - navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    spikes = subset_period_event_time_data(nwbfile.units.get_unit_spike_times(unit_ix), navigation_start_times, navigation_end_times)

    # Head Direction data 
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    hd_spikes = get_spike_heading(spikes, hd_times, hd_degrees)

    # Histogram
    hd_histogram = get_hd_histogram(hd_spikes, binsize, windowsize, smooth)

    return hd_histogram

def normalize_hd_hist_to_occupancy(hd_hist, occupancy=[], nwbfile = None, binsize = 1, smooth= True, windowsize = 23):
    """
    Normalize histogram or set of histograms to occpancy. Occupancy is seconds spent in each bin, thus
    this generates a psuedo firing rate in Hz per bin
    """
    # Occupancy - seconds per bin
    if occupancy == []:
        occupancy = compute_hd_occupancy(nwbfile, binsize, smooth, windowsize)
    hd_firingRate =  hd_hist/ occupancy
    return hd_firingRate

def compute_hd_score(hd_hist):
    """
    Compute head direction score based on method from Gerlei 2020

    HD score should be >= 0.5 to be hd cell

    """
    n_bins = len(hd_hist)
    binsize = 360/n_bins
    degs = np.arange(0, 360, binsize)
    rads = np.radians(degs)
    dy = np.sin(rads)
    dx = np.cos(rads)
    xtotal = sum(dx * hd_hist)/sum(hd_hist)
    ytotal = sum(dy * hd_hist)/sum(hd_hist)
    hd_score = np.sqrt(xtotal**2 + ytotal**2)

    return hd_score

def shuffle_spikes_navigation(nwbfile, unit_ix, approach = 'ISI', n = 100):
    """     
    Genereate n shuffled spike trains 
    
    Shuffling will be done across the entire session. Navigation periods are then
    subset from the shuffled spike train. Build for statistical analysis of head 
    direction

    Parameters
    ----------
    nwbfile: .nwb
        NWB fiile containing session data

    unit_ix: int
        index of the unit to shuffle spikes for

    approach: str
        Method used for shuffling spikes
        'ISI'       shuffle interspike intervals (default)
        'CIRC'      shuffle all spikes circularly
        'BINCIRC'   select time bins and shuffle spikes circularly within time bins
    
    n: int
        number of shuffled spike trains (surrogates) to generate

    Returns
    -------
    shuffled_spikes: 2D arr
        Array of shuffled spike times during navigation periods
        x axis      spike times
        y axis      surrogate number

    """
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    shuffled_spikes = shuffle_spikes_epochs(spikes, navigation_start_times, navigation_end_times, approach, n)           
    return shuffled_spikes  

def shuffle_spikes_epochs(spikes, epoch_start_times, epoch_stop_times, approach = 'ISI', n = 100):
    """
    Generate n shuffled spike trains restricted to epochs of interest

    EX. Generate shuffled spikes during all task navigation periods

    Shuffling will be done across the entire session. Navigation periods are then
    subset from the shuffled spike train

    Parameters
    ----------
    spikes: 1D arr
        Array of spike times, data represents time at which each spike occured (ie. not a binary array)

    epoch_start_times: 1D arr
        Start times of epochs of interest 

    epoch_stop_times: 1D arr
        End times of epochs of interest

    approach: str
        Method used for shuffling spikes
        'ISI'       shuffle interspike intervals (default)
        'CIRC'      shuffle all spikes circularly
        'BINCIRC'   select time bins and shuffle spikes circularly within time bins
    
    n: int
        number of shuffled spike trains (surrogates) to generate

    Returns
    -------
    shuffled_spikes_epochs: list of 1D arr
        Each arraay contains shuffled spike times during epochs of interest for one surrogate
        
        Necessary to return list: method will generate shuffled spike trains of different
        length after epoch subsetting
        @cmh bincirc within navigation periods to avoid? 

    """
    shuffled_spikes_epochs = []
    shuffled_spikes = shuffle_spikes(spikes, approach, n_shuffles = n)
    for surrogate in shuffled_spikes:
        epoch_spikes = subset_period_event_time_data(surrogate, epoch_start_times, epoch_stop_times)
        shuffled_spikes_epochs.append(epoch_spikes)
    return shuffled_spikes_epochs


def nwb_shuffle_spikes_bincirc_navigation(nwbfile, unit_ix, n = 100, shift_range = [1e3, 10e3], verbose = False):
    """
    From nwb file, epochs are navigation periods
    
    Shuffle spikes circularly within each epoch period

    Maintains total number of spikes across all shuffles
    """    
    # Spike data - navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    spikes = subset_period_event_time_data(nwbfile.units.get_unit_spike_times(unit_ix), navigation_start_times, navigation_end_times)
    shuffled_spikes = shuffle_spikes_bincirc_epochs(spikes, navigation_start_times, navigation_end_times, n, shift_range, verbose)
    return shuffled_spikes


def shuffle_spikes_bincirc_epochs(spikes, epoch_start_times, epoch_end_times, n = 100, shift_range = [1e3, 10e3], verbose = False):
    """
    Shuffle spikes circularly within each epoch period

    Maintains total number of spikes across all shuffles
    """

    shuffled_spikes = np.zeros([n, len(spikes)])
    for ix in range(n):
        shuffle_circ_ammount = np.random.uniform(shift_range[0], shift_range[1], len(epoch_start_times))                                                      # ammount to circularly shuffle each bin (between 1-10 seconds)
        if verbose:
            print('Surrogate {} ---- Shuffle circ ammount {}'.format(ix, shuffle_circ_ammount))

        shuffled_spikes_n = [] 
        for le, re, shift in zip(epoch_start_times, epoch_end_times, shuffle_circ_ammount):
            spikes_bin = spikes[(le < spikes) & (spikes < re)]
            shuffled_spikes_bin = shuffle_roll_binned_times(spikes_bin, le, re, shift)
            shuffled_spikes_n.append(shuffled_spikes_bin)

            if verbose:
                print('le, re, shift: \t', le, re, shift)
                fig, ax = plt.subplots (figsize = [20,10])
                ax.eventplot([spikes_bin, shuffled_spikes_bin], linelengths = [0.9, 0.9], colors = ['g', 'b'])
                ax.set_yticks([0,1])
                ax.set_yticklabels(['Spikes Bin', 'Shuffled Spikes'])
                plt.show()

        shuffled_spikes_n = np.hstack(shuffled_spikes_n)
        shuffled_spikes[ix, :] = shuffled_spikes_n
    return shuffled_spikes

def shuffle_roll_binned_times (spikes, left_edge, right_edge, shift):
    """
    Cicularly roll spike times within bin by shift ammount
   
    Parameters
    ----------
    spikes: 1d arr
        array of spike times, must be only spikes within epoch n (ie. between left and right edge)
    
    left_edge: float
        time of epoch start

    right_edge: float
        time of epoch end
    
    shift: float
        amount to shift each spike
    
    Returns
    -------
    shuffled_spikes: 1d arr
        array containing the same number of spikes as original, circlarly shifted in time around the bin 
    """

    # Circular time shuffle
    shifted =  spikes + shift
    within_bin, right = shifted[shifted < right_edge],shifted[right_edge <= shifted]
    binsize = right_edge - left_edge
    rolled = right - binsize
    
    # Combine rolled spike times
    shuffled_spikes = np.sort(np.hstack([rolled, within_bin]))
    
    # check and fix if any are still ouside bin (i.e. due to shift larger than epoch size)
    if len(shuffled_spikes[right_edge <= shuffled_spikes]) > 0:
        shuffled_spikes = shuffle_roll_binned_times(shuffled_spikes, left_edge, right_edge, 0)
    return shuffled_spikes


def get_hd_shuffled_head_directions(shuffled_spikes, hd_times, hd_degrees):
    """
    Get head directions for each shuffled surrogate
    """
    shuffled_head_directions = np.zeros(shuffled_spikes.shape)
    for ix, s_spikes in enumerate(shuffled_spikes):
        hds_ix = get_spike_heading(s_spikes, hd_times, hd_degrees)
        shuffled_head_directions[ix,:] = hds_ix
    return shuffled_head_directions

def get_hd_surrogate_histograms(shuffled_head_directions, binsize = 1, windowsize = 23,  smooth = True):
    """"
    get histograms for each surrogate
    """
    surrogate_histograms = np.zeros([len(shuffled_head_directions), int(360/binsize)])
    for ix, surrogate in enumerate(shuffled_head_directions):
        hist = get_hd_histogram(surrogate, binsize, windowsize, smooth)
        surrogate_histograms[ix,:] = hist
    return surrogate_histograms

def normalize_hd_surrogate_histograms_to_occupancy (surrogate_histograms, occupancy=[], 
                                                    nwbfile = None, binsize = 1, smooth= True, windowsize = 23):
    """
    this is the same as the 1d version, can scrap
    """
    # Occupancy - seconds per bin
    if occupancy == []:
        occupancy = compute_hd_occupancy(nwbfile, binsize, smooth, windowsize)
    
    normalized_histograms = surrogate_histograms / occupancy
    return normalized_histograms
    

# def compute_hd_shuffle_rayleigh(shuffled_head_directions):
#     pvals, zvals = [], []

#     for surrogate in shuffled_head_directions:


def get_probability_histogram_from_hd_histogram(hd_histogram):
    """
    Turn normalized histogram of firing rates in each direction into a probability map of
    firing in each direction.

    This can then be used to ...
    """
    #this can be done with one hist or 2d array of hists
    probability_histogram = hd_histogram / sum(hd_histogram.T) 
    return probability_histogram
        



def compute_ci95_from_surrogates(surrogates):
    """
    For each bin compute the ci95, return as 2d array high over low for each bin
    """
    n_surrogates, n_bins = surrogates.shape[0], surrogates.shape[1]
    ci_low, ci_high = [], []
    for ix in range(n_bins):
        bin =  surrogates[:,ix]
        mean = np.mean(bin)
        s = 1.96 * np.std(bin)/np.sqrt(n_surrogates)
        ci_low.append(mean-s)
        ci_high.append(mean+s)
    ci95 = np.vstack([ci_high, ci_low])
    return ci95 

def compare_hd_histogram_to_surrogates(hd_histogram, surrogates):
    """
    For each bin, compare the actual firing rate to the firing rates determined 
    from shuffling for each. 
    
    Bins in which the firing rate of the real histogram are above the 95% confidence 
    interval of the surrogates are considered significant
    """
    

    ci95 = compute_ci95_from_surrogates(surrogates)
    

    


    


