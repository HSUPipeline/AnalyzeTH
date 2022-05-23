
#from random import shuffle


#from analyzeth.cmh.headDirection.headDirectionStats import *
#from analyzeth.cmh.headDirection.headDirectionUtils import *
#from analyzeth.cmh.headDirection.headDirectionPlots import *
#from analyzeth.analysis import get_spike_heading, bin_circular


from analyzeth.cmh.utils import *
from analyzeth.cmh.headDirection import * 

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np






# spiketools
from spiketools.stats.shuffle import shuffle_spikes

def nwb_headDirection_session(nwbfile, n_surrogates = 1000, plot=False, subject='wv___'):
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
        nwb_headDirection_cell(nwbfile, unit_ix, occupancy, n_surrogates, plot = True, subject=subject)
    return 


def nwb_headDirection_cell(nwbfile, unit_ix, 
        occupancy = [], 
        n_surrogates = 1000, 
        binsize = 1, 
        windowsize = 23, 
        smooth = True, 
        plot = False,
        subject = 'wv____'
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
    #surrogates_ci95 = compute_std_ci95_from_surrogates(surrogates_norm)
    surrogates_ci95 = bootstrap_ci_from_surrogates(surrogates_norm)
    significant_bins = compute_significant_bins(hd_hist_norm, surrogates_ci95)

    print('inside hd func, surci95 size', surrogates_ci95.shape)

    if plot:
        title = subject + f' | Unit {unit_ix}'
        hd_ax = plot_hd_full(hd_hist_norm, surrogates_ci95, significant_bins)
        # fig = plt.figure(figsize = [10,10])
        # ax = plt.subplot(111, polar=True)
        # plot_hd(hd_hist_norm, ax=ax)
        # plot_surrogates_95ci(surrogates_norm, ax=ax)
        # plt.title('Unit {}'.format(unit_ix))
        # plt.xlabel('')
        # plt.ylabel('')
        # plt.show()

    res = {
        'hd_score'                  : hd_score,
        'hd_score_norm'             : hd_norm_score,
        'hd_histogram'              : hd_hist,
        'hd_histogram_norm'         : hd_hist_norm,

        'surrogate_hds'             : surrogate_hds,
        'surrogate_histograms'      : surrogate_histograms,
        'surrogate_histograms_norm' : surrogates_norm,
        'surrogates_ci95'           : surrogates_ci95,
        'significant_bins'          : significant_bins,

        'hd_plot'                   : hd_ax
    }
    return res
        












######################
# WIP
####################

def compute_bootstrap_ci95_from_surrogates(surrogates):
    """
    Compute ci95 from surrogates using seaborn bootstrapping method
    """

    num_bins = surrogates.shape[1]
    bisize = 360 / num_bins
    df = pd.DataFrame(surrogates).melt()


    df['variable'] = np.radians(df['variable']*binsize)
    #ax = sns.lineplot


def compare_hd_histogram_to_surrogates(hd_histogram, surrogates):
    """
    For each bin, compare the actual firing rate to the firing rates determined 
    from shuffling for each. 
    
    Bins in which the firing rate of the real histogram are above the 95% confidence 
    interval of the surrogates are considered significant
    """
    

    ci95 = compute_ci95_from_surrogates(surrogates)
    

    


    


