import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analyzeth.cmh.headDirection import *



##################################################################
# Shuffling
##################################################################

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


##################################################################
# Surrogate functions
##################################################################

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
        

##################################################################
# CI95
##################################################################


def compute_std_ci95_from_surrogates(surrogates):
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



##################################################################
# Bootstrap equations
##################################################################

def draw_bs_replicates(data,func,size):
    """creates a bootstrap sample, computes replicates and returns replicates array"""
    # Create an empty array to store replicates
    bs_replicates = np.empty(size)
    
    # Create bootstrap replicates as much as size
    for i in range(size):
        # Create a bootstrap sample
        bs_sample = np.random.choice(data,size=len(data))
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample)
    
    return bs_replicates



def bootstrap_ci_from_surrogates(surrogates, func = np.mean, num_bootstraps = 10000, ci=95, verbose = False, plot_verbose = False):
    """
    Bootstrap for 95 % confidence intervals from surrogate data
    
    Parameters
    ----------
    surrogates: 2d arr
        array containing surrogates, 
            x = hz or firing count reading for each degreen bin (of arbitrary size - i.e. can be 1 or 
            90 degrees etc.), default 360 bins (one per degree)
            y = each surrogate, defualt 1000 surrogates
    
    func: function 
        estimator to use for bootstrapping, default np.mean, can do np.std etc

    num_bootstraps: int
        number of bootstraps to run, default 10,000
    
    ci: int
        confidence interval to calculate, default 95%
        will give upper and lower bounds, i.e. the 2.5% and 97,5% bounds for calculated metrics (default = np.mean)
    
    Returns
    -------
    boostrap_confidence_intervals: 2d arr
        2d array containing the (95%) confidence interval determined for each bin
    """


    # Setup 
    num_bins = surrogates.shape[1]
    ##df = pd.DataFrame(surrogates).melt()
    ##df.rename(columns={'variable' : 'bin', 'value': 'hz'}, inplace = True)

    # Bootstrap 
    bootstrap_ci95s = np.zeros([num_bins, 2]) 
    # print(f'bs array shape: \t {bootstrap_ci95s.shape}')

    for ix in range(num_bins):    
        #surrogates_bin = df[df['bin']==ix]
        surrogates_bin = surrogates[:,ix]
        bootstrap_replicates_bin = draw_bs_replicates(surrogates_bin, func, num_bootstraps)

        # Confidence interval
        ci_lower = (100 - ci)/2
        ci_upper = 100 - ci_lower
        conf_interval = np.percentile(bootstrap_replicates_bin,[ci_lower,ci_upper])
        bootstrap_ci95s[ix,:] = conf_interval

        # verbose
        if verbose and ix%100 ==0:
            print(f'Computing boostrap for bin {ix}')
            if plot_verbose:
                plot_bootstrap_replicates_PDF_hist(bootstrap_replicates_bin)

    print('in bs func -- ci95s shape', bootstrap_ci95s.shape)
    return bootstrap_ci95s 

def compute_significant_bins(hd_histogram, confidence_interval):
    """
    Check if sample from each bin is above ci95. If so
    record bin as significant only if the 5 bins on either side
    are also above the upper ci95 threshold

    Parameters
    ----------
    hd_histogram: 1d arr
        array of bin values (if normalized to occupancy then Hz) for each head direction

    confidence_interval: 2d arr
        array of lower [:,0] and upper [:,1] confidence bounds for each bin based on surrogates
        default is to use bootsrapping with np.mean for determining 95% confidence interval
    
    Returns
    -------
    significant_bins: 1d arr
        boolean array indicating which bins are found to be significant (above ci upper bound for
        bin + 5 bins on each side)

    """
    hd_mask = hd_histogram > confidence_interval[:,1]
    hd_mask_circle = np.hstack([hd_mask, hd_mask[:10]])

    num_bins = len(hd_histogram)
    significant_bins = np.zeros(num_bins)
    for ix in range(num_bins):
        if hd_mask_circle[ix] == True:
            sig = True
            for jx in range(ix-5,ix+6):
                if hd_mask_circle[jx] == False:
                    sig = False
            significant_bins[ix] = sig
    return significant_bins



