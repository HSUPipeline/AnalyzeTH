""""Functions for spatial target analyses."""

from collections import Counter

import numpy as np

from spiketools.spatial.occupancy import compute_nbins
from spiketools.utils.data import restrict_range, get_value_by_time, get_value_by_time_range

from analysis import get_spike_positions

###################################################################################################
###################################################################################################

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

        t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        t_spikes = restrict_range(spikes, t_st, t_en)
        t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

        seg_times = np.diff(np.insert(ch_openings, 0, t_time[0]))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))

        frs = np.array(list(count.values())) / seg_times

        cur_ch_xbin = ch_xbin[t_mask]
        cur_ch_ybin = ch_ybin[t_mask]

        for fr, xbin, ybin in zip(frs, cur_ch_xbin, cur_ch_ybin):
            target_bins[xbin, ybin] = fr

    return target_bins


def get_trial_target(spikes, navigations, bins, openings, chest_trials,
                     chest_xbin, chest_ybin, ptimes, positions):
    """Get the binned target firing, per trial."""

    n_trials = len(openings)
    n_bins = compute_nbins(bins)

    # Collect firing per chest location for each trial
    target_bins_all = np.zeros([n_trials, n_bins])
    for t_ind in range(n_trials):

        # Get trial information
        t_st = navigations[t_ind]
        t_openings = openings[t_ind]
        t_en = t_openings[-1]

        t_mask = chest_trials == t_ind

        # Select chest openings for the current trial
        t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in t_openings]

        # Restrict spikes to the chest-opening period
        t_spikes = restrict_range(spikes, t_st, t_en)
        t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

        # compute firing rate per bin per trial
        seg_times = np.diff(np.insert(t_openings, 0, t_time[0]))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, t_openings))

        frs = np.array(list(count.values())) / seg_times

        cur_ch_xbin = chest_xbin[t_mask]
        cur_ch_ybin = chest_ybin[t_mask]

        target_bins = np.zeros(bins)
        for fr, xbin, ybin in zip(frs, cur_ch_xbin, cur_ch_ybin):
            target_bins[xbin, ybin] = fr

        target_bins_all[t_ind,:] = target_bins.flatten()

    return target_bins_all
