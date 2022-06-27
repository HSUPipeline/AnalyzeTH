""""Functions for spatial target analyses.

ToDo:
- Fix up interim updates
- Check different functions and merge common stuff
"""

from functools import partial
from collections import Counter

import numpy as np

from spiketools.spatial.occupancy import compute_nbins
from spiketools.stats.anova import create_dataframe, fit_anova
from spiketools.utils.data import restrict_range, get_value_by_time, get_value_by_time_range

# import local code
from analysis import get_spike_positions

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(target_bin)'
FEATURE = 'C(target_bin)'
COLUMNS = ['target_bin', 'fr']

# Create functions for target model
create_df_target = partial(create_dataframe, columns=COLUMNS)
fit_anova_target = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################

## TODO: THE TWO FUNCTIONS HERE SHOULD BE CONSOLIDATED

## ISSUE: THIS IS WRONG, BECAUSE OF HOW IT ADDS ACROSS TRIALS
def compute_spatial_target_bins(spikes, nav_starts, chest_openings, chest_trials,
                                ptimes, positions, chest_bins, ch_xbin, ch_ybin):
    """Compute the binned firing rate based on spatial target."""

    # Collect firing per chest location across all trials
    target_bins = np.zeros(chest_bins)
    for t_ind in range(len(nav_starts)):

        t_st = nav_starts[t_ind]
        ch_openings = chest_openings[t_ind]
        t_en = ch_openings[-1]

        t_mask = chest_trials == t_ind

        #t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        #ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        t_spikes = restrict_range(spikes, t_st, t_en)
        #t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

        #seg_times = np.diff(np.insert(ch_openings, 0, t_time[0]))
        seg_times = np.diff(np.insert(ch_openings, 0, t_st))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))

        frs = np.array(list(count.values())) / seg_times

        cur_ch_xbin = ch_xbin[t_mask]
        cur_ch_ybin = ch_ybin[t_mask]

        for fr, xbin, ybin in zip(frs, cur_ch_xbin, cur_ch_ybin):
            target_bins[xbin, ybin] = fr

    return target_bins

## ISSUE: THIS SHOULD PROBABLY SET ALL THE ZERO OBSERVATIONS TO BE NAN
def get_trial_target(spikes, navigations, bins, openings, chest_trials,
                     chest_xbin, chest_ybin, ptimes, positions):
    """Get the binned target firing, per trial."""

    n_trials = len(openings)
    n_bins = compute_nbins(bins)

    # Collect firing per chest location for each trial
    target_bins_all = np.zeros([n_trials, n_bins])
    for t_ind in range(n_trials):

        # Get chest and opening events of current trial
        t_openings = openings[t_ind]
        t_mask = chest_trials == t_ind

        # Get navigation start & end and restrict spikes to this range
        t_st = navigations[t_ind]
        t_en = t_openings[-1]
        t_spikes = restrict_range(spikes, t_st, t_en)

        # Select chest openings for the current trial
        #t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        #ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in t_openings]
        #t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

        # Compute firing rate per target bin per trial
        seg_times = np.diff(np.insert(t_openings, 0, t_st))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, t_openings))

        frs = np.array(list(count.values())) / seg_times

        cur_ch_xbin = chest_xbin[t_mask]
        cur_ch_ybin = chest_ybin[t_mask]

        target_bins = np.zeros(bins)
        for fr, xbin, ybin in zip(frs, cur_ch_xbin, cur_ch_ybin):
            target_bins[xbin, ybin] = fr

        target_bins_all[t_ind, :] = target_bins.flatten()

    return target_bins_all


def get_target_chest_location(target_bins, ch_xbin, ch_ybin, chest_xs, chest_ys):
    """compute the location of chests in a target bin"""
    
    bin_y, bin_x = np.where(target_bins==np.amax(target_bins))
    
    xbin = set(np.where(ch_xbin==bin_x)[0])
    ybin = set(np.where(ch_ybin==bin_y)[0])
    
    intersect = sorted(np.array(list(xbin.intersection(ybin))))
    
    chest_x = [chest_xs[i] for i in intersect]
    chest_y = [chest_ys[i] for i in intersect]
    
    return intersect, chest_x, chest_y

  
def get_chest_data_per_bin(intersect, chest_trial_number, ptimes, positions, spikes, ch_openings_all, nav_starts):
    """Get spikes and positions of all chest in a target bin"""
    
    t_pos_all = []
    t_all_xs = []
    t_all_ys = []
    
    for ind in intersect:
        # For the non-first chest position per trial
        if ind not in chest_trial_number[:,0]:
            t_time, t_pos = get_value_by_time_range(ptimes, positions, ch_openings_all[ind], ch_openings_all[ind+1])
            t_pos_all.append(t_pos)

            t_spikes = restrict_range(spikes, ch_openings_all[ind], ch_openings_all[ind+1])
            t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

            t_spike_xs = np.array(t_spike_xs).flatten()
            t_spike_ys = np.array(t_spike_ys).flatten()
            t_all_xs.append(t_spike_xs)
            t_all_ys.append(t_spike_ys) 
        # For the first chest position per trial
        else:          
            chest_trial = int(ind/4)
            t_time, t_pos = get_value_by_time_range(ptimes, positions, nav_starts[chest_trial], ch_openings_all[chest_trial])
            t_pos_all.append(t_pos)

            t_spikes = restrict_range(spikes, nav_starts[chest_trial], ch_openings_all[ind])
            t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

            t_spike_xs = np.array(t_spike_xs).flatten()
            t_spike_ys = np.array(t_spike_ys).flatten()
            t_all_xs.append(t_spike_xs)
            t_all_ys.append(t_spike_ys)
            
    t_all_xs = np.concatenate(t_all_xs).ravel()
    t_all_ys = np.concatenate(t_all_ys).ravel()   
    
    return t_pos_all, t_all_xs, t_all_ys
