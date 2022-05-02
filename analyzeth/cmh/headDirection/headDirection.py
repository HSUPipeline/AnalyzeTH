
from random import shuffle
from analyzeth.cmh.utils import *
from analyzeth.cmh.headDirection.headDirectionUtils import *
from analyzeth.cmh.headDirection.headDirectionPlots import *

from analyzeth.analysis import get_spike_heading, bin_circular
import pandas as pd
import matplotlib.pyplot as plt

# spiketools
from spiketools.stats.shuffle import shuffle_spikes

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
    hd_spikes = get_spike_heading(spikes, hd_times, hd_degrees)

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
    # if smooth:
    #     hd_hist = get_hd_histogram_smooth(hd_spikes, binsize, windowsize)
    # else:
    #     _, hd_hist = bin_circular(hd_spikes, binsize=binsize)
    return hd_histogram

def normalize_hd_hist_to_occupancy(hd_hist, occupancy=[], nwbfile = None, binsize = 1, smooth= True, windowsize = 23):
    # Occupancy - seconds per bin
    if occupancy == []:
        occupancy = compute_hd_occupancy(nwbfile, binsize, smooth, windowsize)
    hd_firingRate =  hd_hist/ occupancy
    return hd_firingRate

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

def shuffle_spikes_bincirc_navigation(spikes, navigation_start_times, navigation_end_times, n = 100):
    """
    Shuffle spikes circularly within each navigation period

    Maintains total number of spikes across all shuffles
    """

    shuffled_spikes = np.zeros([n, len(spikes)])
    print(shuffled_spikes.shape)

    for ix in range(n):
        shuffle_circ_ammount = np.random.uniform(1e3, 1e4, len(navigation_start_times))                                                      # ammount to circularly shuffle each bin (between 1-10 seconds)
        
        shuffled_spikes_n = [] 
        for le, re, shift in zip(navigation_start_times, navigation_end_times, shuffle_circ_ammount):
            spikes_bin = spikes[(le < spikes) & (spikes < re)]

            

            shuffled_spikes_bin = shuffle_roll_binned_times(spikes_bin, le, re, shift)
            shuffled_spikes_n.append(shuffled_spikes_bin)

        
        shuffled_spikes_n = np.hstack(shuffled_spikes_n)
        shuffled_spikes[ix, :] = shuffled_spikes_n

    return shuffled_spikes

def shuffle_roll_binned_times (spikes, left_edge, right_edge, shift):
    """
    Cicularly roll spike times within bin by shift ammount
    """

    # Subset spikes if not already
    spikes = spikes[(left_edge <= spikes) & (spikes <= right_edge)]

    # Circular time shuffle
    shuffled_temp =  spikes + shift
    shuffled_within_bin = shuffled_temp[shuffled_temp < right_edge]
    shuffled_right = shuffled_temp[right_edge <= shuffled_temp]
    shuffled_right -= right_edge
    shuffled_right += left_edge

    # Combine rolled spike times
    shuffled_spikes = np.hstack([shuffled_right, shuffled_within_bin])

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
    surrogate_histograms = np.zeros(len(shuffled_head_directions), 360/binsize)
    for ix, surrogate in shuffled_head_directions:
        hist = get_hd_histogram(surrogate, binsize, windowsize, smooth)
        surrogate_histograms[ix,:] = hist
    return surrogate_histograms

def normalize_hd_surrogates_to_occupancy (shuffled_head_directions, occpancy):
    """
    Normalize each surrogate to occupancy as with original data
    """
    normalized_histograms = 
    for surrogate in  shuffled_head_directions:
        

def compute_hd_shuffle_rayleigh(shuffled_head_directions):
    pvals, zvals = [], []

    for surrogate in shuffled_head_directions:


    
        

    

def plot_hd_occupancy_vs_spike_probability_overlay(hd_hist, occupancy_hist, ax = None, figsize = [10,10]):
    """ Plot overlay of two polar histograms"""
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar = True)
    
    
    hist1 = hist1/sum(hist1)
    hist2 = hist2/sum(hist2)
    plot_hd(hist1, ax=ax)
    plot_hd(hist2, ax=ax)
    plt.title('Overlay')
    plt.show()
    return ax

    


