import itertools 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analyzeth.cmh.utils import *
from spiketools.utils import restrict_range


def cell_firing_rate(
        spikes,       #sec  
        epoch_start_time,
        epoch_stop_time, 
        window = 1,   #sec, must be 1 for Hz
        step = 0.1    #sec 
        ):

    """ Calculate firing rate across epoch 
    
    This function will take a 1D spike time array and calculate
    firing rate at each point in time (governed by step size) 
    with a rolling window (governed by window size)

    Edges are refelcted based on window size

    Parameters
    ----------
    spikes: 1D array
        time stamps of spikes in ms

    window: int 
        legth of window over which to calculate FR in ms

    step: int
        step size for moving window in ms

    Returns
    -------
    firing_rates: 1D array
        Hz (spikes per second)
        firing rates at each point in time
        length of array depends on step size

    """
    
    # Reflect edges by window size (1sec)
    #window = int(window) if ((int(window) % 2) == 0) else int(window) + 1     # make window even
    len_epoch = epoch_stop_time - epoch_start_time

    # Zero to start of epoch
    spikes = spikes - epoch_start_time

    # Start reflection (easy, * -1)
    start_reflection = -1 * spikes[spikes < window]
    
    # End reflection (harder, reflect based on distance to end)
    end_spikes = spikes[len_epoch - window < spikes]
    end_reflection = []
    if len(end_spikes) != 0:
        for spike in end_spikes:
            diff = len_epoch - spike
            reflected_spike = spike + 2*diff
            end_reflection.append(reflected_spike)
    spikes = np.concatenate([start_reflection, spikes, end_reflection])

    # Return to epoch time
    #spikes += epoch_start_time


    # Iter for bin FRs
    num_bins = int(np.ceil(len_epoch/step)) + 1 
    times = []
    FRs = []                                       
    for ix_win in range(num_bins):
        center = step*ix_win #epoch_start_time + step * ix_win
        left = center - window/2
        right = center + window/2
        spikes_bin = spikes[(left<spikes) & (spikes < right)]

        fr_bin = len(spikes_bin)/(window) # get FR in Hz 

        times.append(center)
        FRs.append(fr_bin)


    # returning times in epoch time s
    return FRs, times


####################
# FRs over time
####################

def compute_epochs_firing_rates_over_time (spikes, epoch_start_times, epoch_stop_times, window = 1, step = 0.1, return_means = False):
    """
    Compute mean firing rate at eacch step across all epochs of interest (i.e. navigation periods)
    """
    firing_rates_over_time = []
    fr_times = []

    num_epochs = len(epoch_start_times)
    for ix in range(num_epochs):
        len_epoch = epoch_stop_times[ix] - epoch_start_times[ix]
        spikes_epoch = restrict_range(spikes, epoch_start_times[ix], epoch_stop_times[ix])
        FRs, times =  cell_firing_rate(spikes_epoch, epoch_start_times[ix], epoch_stop_times[ix], window=1, step=0.1)
        if len(times) > len(fr_times):
            fr_times = times
        firing_rates_over_time.append(FRs)

    if return_means:
        # Find Mean of Non-zero bin counts
        firing_rate_bin_means = [np.nanmean(x) for x in itertools.zip_longest(*firing_rates_over_time, fillvalue=np.nan)]
        return firing_rates_over_time, firing_rate_bin_means
    return firing_rates_over_time

def nwb_compute_navigation_firing_rates_over_time (nwbfile, unit_ix, window=1, step =0.1, return_means = False):
    """
    Compute mean firing rate at eacch step across all navigation periods
    """

    # Get spikes
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)

    # Get epoch times
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]

    # Compute mean FRs at each step for each epoch
    if return_means:
        FRs, FR_means = compute_epochs_firing_rates_over_time(
            spikes, 
            navigation_start_times, 
            navigation_stop_times,
            window=1, 
            step =0.1, 
            return_means = True)
        return FRs, FR_means

    else: 
        FRs =  compute_epochs_firing_rates_over_time(
            spikes, 
            navigation_start_times, 
            navigation_stop_times,
            window=1, 
            step =0.1, 
            return_means = False)
        return FRs

#######################################
# FR over time normalized to trial len
#######################################

def nwb_norm_fr_over_time(nwbfile, unit_ix, num_bins = 10):
    """
    Normalize FR bins to trial length...
    """

    FRs = nwb_compute_navigation_firing_rates_over_time(nwbfile, unit_ix)

    binned_FRs = np.empty([len(FRs),num_bins])
    for ix, FR in enumerate(FRs):
        # print(ix, len(FR), FR)
        binned = np.array_split(FR, num_bins)
        means = [np.mean(bin) for bin in binned]
        binned_FRs[ix,:] = means
    return binned_FRs




#############
# Mean FR
#############

def compute_epochs_mean_firing_rate(spikes, epoch_start_times, epoch_stop_times, unit_ix =''):
    """
    Compute mean firing rate during epochs of interest
    """
    num_spikes = len(subset_period_event_time_data(spikes, epoch_start_times, epoch_stop_times))
    len_epochs = epoch_stop_times - epoch_start_times
    total_time = sum(len_epochs)
    # print(f'\nUnit {unit_ix}')
    # print(f'Num spikes: {num_spikes}')
    # print(f'Total time: {total_time}')
    # print(f'Mean FR: {num_spikes/total_time}')
    return num_spikes / total_time

def nwb_compute_navigation_mean_firing_rate(nwbfile, unit_ix, epoch_ixs = 'all' ):
    """
    Compute mean firing rate during navigation periods from nwbfile
    """

    # Get spikes
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)

    # Get epoch times
    if epoch_ixs == 'all':
        navigation_start_times = nwbfile.trials['navigation_start'][:]
        navigation_stop_times = nwbfile.trials['navigation_stop'][:]
    else:
        navigation_start_times = nwbfile.trials['navigation_start'][epoch_ixs[0]:epoch_ixs[1]]
        navigation_stop_times = nwbfile.trials['navigation_stop'][epoch_ixs[0]:epoch_ixs[1]]

    # Get mean FR
    mean_FR = compute_epochs_mean_firing_rate(spikes, navigation_start_times, navigation_stop_times, unit_ix)
    return mean_FR


############
# Plot
############

def plot_cell_firing_rate_over_time(FRs, mean_FR, step = 0.1, axs=None, unit_ix = '', ci = 95):
    """
    Plot mean FRs with bootstrapped 95ci and individual trial lengths
    """
    df = pd.DataFrame(FRs).melt()
    df['variable'] = df['variable'] * step
    
    # Plot
    if axs=='make':
        fig, axs = plt.subplots(2, figsize=[4,5], gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
    #fig, axs = plt.subplots(2, figsize=[4,5], gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
    
    xtick_locations = np.arange(0, max(df['variable']), 10);

    # FR plot
    ax = axs[0]
    sns.lineplot(ax = ax, data = df, estimator = np.mean, x='variable', y='value', ci = ci, color = 'grey')
    ax.set_ylabel('Firing Rate (Hz)', fontsize=14)
    ax.set_yticks([tick for tick in ax.get_yticks()[:-1] if tick % 1 == 0 and tick >=0])
    ax.set_xlabel('')
    ax.set_xlim(0, max(df['variable']))
    ax.set_xticks(xtick_locations)
    ax.set_xticklabels([])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_title(f'Unit {unit_ix} | Mean FR = {np.round(mean_FR,2)} Hz')

    # Epoch length plot
    ax = axs[1]
    lens = np.sort([np.round(len(x)*0.1,2) for x in FRs])[::-1]
    for ix, l in enumerate(lens):
        ax.hlines(-ix, 0, l, color = 'gray', lw=1, alpha=1)
    #ax.barh(np.arange(len(lens)), lens, align='center', height=1, color='k') #much slower?
    ax.set_ylabel(f'Trial \n (n={len(lens)})', fontsize=14)
    ax.set_yticks([1])
    ax.set_yticklabels([])
    ax.set_xlabel('Time (s)', fontsize = 16)
    ax.set_xticks(xtick_locations, rotation = 0);
    ax.set_xlim(0, max(df['variable']))
    ax.set_facecolor('white')
    ax.tick_params(axis=u'both', which=u'both',length=0)

    # Sup formatting 
    #plt.suptitle(f'Unit {unit_ix} | Mean FR = {np.round(mean_FR,2)} Hz')
    #plt.tight_layout()
    #plt.show()

    return axs

def plot_nwb_cell_firing_rate_over_time(nwbfile, unit_ix, step=0.1, axs=None, ci=95):
    """
    Plot mean FR over time from nwb file
    """
    #if not axs:
    #   fig, axs = plt.subplots(2, figsize=[4,5], gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})
        #plt.figure(figsize=[4,4])
        #ax=plt.subplot(111)

    FRs = nwb_compute_navigation_firing_rates_over_time(nwbfile, unit_ix, step = 0.1)
    mean_FR = nwb_compute_navigation_mean_firing_rate(nwbfile, unit_ix)
    
    axs = plot_cell_firing_rate_over_time(FRs, mean_FR, step=0.1, axs = axs, unit_ix = unit_ix, ci=ci)

    return axs